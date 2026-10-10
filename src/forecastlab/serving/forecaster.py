"""Production inference: load a registered model version and forecast every series from a given origin.

Feature builders are kept per store between calls; each night only the sales history is refreshed
(`set_sales`), so a nightly run is a few seconds per store.
"""
from __future__ import annotations

import json
import logging

import lightgbm as lgb
import numpy as np

from forecastlab.data.load import Dataset
from forecastlab.models import gbm
from forecastlab.models.explain import factors
from forecastlab.models.features import FeatureBuilder
from forecastlab.models.probabilistic import nb_dispersion, nb_quantiles
from forecastlab.models.registry import load_registry, version_dir

log = logging.getLogger(__name__)


class Forecaster:
    def __init__(self, version: str | None = None, role: str = "champion"):
        reg = load_registry()
        self.version = version or reg[role]
        d = version_dir(self.version)
        self.meta = json.loads((d / "meta.json").read_text())
        self.min_lag = self.meta["min_lag"]
        self.boosters = {s: lgb.Booster(model_file=str(d / f"{s}.txt")) for s in self.meta["stores"]}
        self._fb: dict[str, FeatureBuilder] = {}

    def _builder(self, ds: Dataset, store: str, rows: np.ndarray) -> FeatureBuilder:
        fb = self._fb.get(store)
        if fb is None or fb.ds.P is not ds.P:
            fb = self._fb[store] = FeatureBuilder(ds, rows, Y=ds.Y[rows])
        else:
            fb.set_sales(ds.Y[rows])
        return fb

    def forecast(self, ds: Dataset, origin: int, h: int) -> np.ndarray:
        """Mean forecast [n_series, h] for days origin+1 .. origin+h, using sales up to `origin` only."""
        if h > self.min_lag:
            raise ValueError(f"a direct model with min_lag {self.min_lag} cannot forecast {h} days ahead")
        F = np.zeros((len(ds.series), h), dtype=np.float32)
        days = np.arange(origin + 1, origin + 1 + h)
        for store, rows in gbm.store_rows(ds).items():
            fb = self._builder(ds, store, rows)
            X = fb.frame(days, self.min_lag)
            F[rows] = (np.clip(self.boosters[store].predict(X), 0, None) * fb.on_shelf(days)).reshape(len(rows), h)
        return F

    def quantiles(self, ds: Dataset, origin: int, F: np.ndarray, qs: list[float]) -> np.ndarray:
        return nb_quantiles(F, nb_dispersion(ds.Y[:, : origin + 1]), qs)

    def explain(self, ds: Dataset, row: int, origin: int, h: int):
        """Driver factors for one series over the forecast window (DataFrame, one row per day)."""
        store = ds.series["store_id"].iloc[row]
        rows = gbm.store_rows(ds)[store]
        fb = self._builder(ds, store, rows)   # store-wide builder: group features need the whole store
        days = np.arange(origin + 1, origin + 1 + h)
        i = int(np.searchsorted(rows, row))
        X = fb.frame(days, self.min_lag).iloc[i * h: (i + 1) * h]
        f = factors(self.boosters[store], X)
        f.index = ds.dates[days].date
        return f

"""Time-ordered evaluation design.

A forecast "origin" is the last day of data the model may see. Its test window is the `horizon` days after it.
Nothing from the test window (or later) may influence training, features, scaling or model selection.

    origin ─┐
    ████████│░░░░░░░░ test (h days)
    train   │
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from forecastlab.config import load_config
from forecastlab.data.load import Dataset, first_sale_day, model_start_day


@dataclass(frozen=True)
class Origin:
    name: str
    day: int          # index of last training day
    horizon: int

    @property
    def test(self) -> slice:
        return slice(self.day + 1, self.day + 1 + self.horizon)


def horizon() -> int:
    return load_config()["forecast"]["horizon"]


def backtest_origins(ds: Dataset) -> list[Origin]:
    v = load_config()["validation"]
    return [Origin(f"fold{i + 1} ({d})", ds.day(d), horizon()) for i, d in enumerate(v["backtest_origins"])]


def holdout_origin(ds: Dataset) -> Origin:
    d = load_config()["validation"]["holdout_origin"]
    return Origin(f"holdout ({d})", ds.day(d), horizon())


def sample_rows(ds: Dataset, n: int | None = None) -> np.ndarray:
    """Stratified sample of bottom-level series (store x category), each launched before the modelling period.

    The same sample is used by every per-series model so their numbers are comparable.
    """
    cfg = load_config()["sample"]
    n = n or cfg["n_series"]
    eligible = first_sale_day(ds.Y) < model_start_day(ds)
    df = ds.series.assign(row=np.arange(len(ds.series)))[eligible]
    rng = np.random.default_rng(cfg["random_state"])
    frac = n / len(df)
    rows = []
    for _, g in df.groupby(["store_id", "cat_id"]):
        k = max(1, round(len(g) * frac))
        rows.append(rng.choice(g["row"].to_numpy(), size=min(k, len(g)), replace=False))
    return np.sort(np.concatenate(rows))[:n]


def walk_forward(first_origin: int, n_folds: int, step: int, h: int) -> list[Origin]:
    """Expanding-window walk-forward origins: train on [0, o], test on (o, o+h], then move o by `step`."""
    return [Origin(f"wf{i + 1}", first_origin + i * step, h) for i in range(n_folds)]


def origin_dates(ds: Dataset, origins: list[Origin]) -> pd.DataFrame:
    return pd.DataFrame([{"origin": o.name, "train_end": ds.dates[o.day].date(),
                          "test_start": ds.dates[o.test.start].date(),
                          "test_end": ds.dates[o.test.stop - 1].date()} for o in origins])

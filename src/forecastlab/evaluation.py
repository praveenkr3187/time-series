"""One scorecard for every model, so the whole ladder can be compared like for like.

For a cached model key and an origin it reports, where available:
    WRMSSE         all 42,840 hierarchy series (needs forecasts for all 30,490 items)
    item_*         RMSSE / WAPE / bias on the shared stratified sample of item-store series
    agg_<level>    RMSSE on store x dept (70), store (10) and total (1): the model's own aggregate forecast
                   if it made one (classical models), else its item forecasts summed bottom-up
"""
from __future__ import annotations

import numpy as np

from forecastlab.data.hierarchy import m5_hierarchy
from forecastlab.data.load import Dataset
from forecastlab.metrics import WRMSSE, point_metrics
from forecastlab.store import load_forecast
from forecastlab.validation import Origin, sample_rows

AGG_LEVELS = {9: "store x dept", 3: "store", 1: "total"}
_W: dict[int, WRMSSE] = {}


def wrmsse(ds: Dataset, o: Origin) -> WRMSSE:
    if o.day not in _W:
        _W[o.day] = WRMSSE(ds, o.day, o.horizon)
    return _W[o.day]


def scorecard(ds: Dataset, key: str, o: Origin) -> dict[str, float]:
    out: dict[str, float] = {}
    F = load_forecast(key, o.day)
    hier = m5_hierarchy()
    full = F is not None and not np.isnan(F).any()
    if full:
        out["WRMSSE"] = wrmsse(ds, o).score(F)
    if F is not None:
        rows = sample_rows(ds)
        Fs = F[rows]
        if not np.isnan(Fs).any():
            m = point_metrics(ds.Y[rows, o.test], Fs, ds.Y[rows, : o.day + 1])
            out.update({"item RMSSE": m["rmsse"], "item WAPE": m["wape"], "item bias": m["bias"]})
    for level, name in AGG_LEVELS.items():
        Fa = load_forecast(f"{key} @ level{level}", o.day)
        if Fa is None and full:
            Fa = hier.aggregate(F, level)
        if Fa is not None:
            Ya = hier.aggregate(ds.Y[:, : o.test.stop], level)
            m = point_metrics(Ya[:, o.test], Fa, Ya[:, : o.day + 1])
            out[f"{name} RMSSE"] = m["rmsse"]
            if level == 3:
                out["store WAPE"] = m["wape"]
    return out


def paired_bootstrap(diff: np.ndarray, weights: np.ndarray | None = None, n_boot: int = 2000,
                     seed: int = 0) -> tuple[float, float, float]:
    """Mean of per-unit differences (e.g. per-series error A - B) with a 95% bootstrap CI over units."""
    rng = np.random.default_rng(seed)
    w = np.ones_like(diff) if weights is None else weights
    est = float(np.sum(w * diff) / np.sum(w))
    idx = rng.integers(0, len(diff), size=(n_boot, len(diff)))
    boots = (w[idx] * diff[idx]).sum(axis=1) / w[idx].sum(axis=1)
    return est, float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))


def diebold_mariano(e1: np.ndarray, e2: np.ndarray, h: int = 1) -> tuple[float, float]:
    """Diebold-Mariano test on squared-error loss differentials (HAC variance with h-1 lags).

    Returns (statistic, two-sided p-value); negative statistic = model 1 has lower loss.
    """
    from scipy import stats
    d = e1 ** 2 - e2 ** 2
    n = len(d)
    dc = d - d.mean()
    gamma = [np.sum(dc[k:] * dc[: n - k]) / n for k in range(h)]
    var = (gamma[0] + 2 * sum(gamma[1:])) / n
    dm = d.mean() / np.sqrt(max(var, 1e-12))
    return float(dm), float(2 * stats.t.sf(abs(dm), df=n - 1))

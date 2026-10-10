"""Forecast accuracy metrics.

Arrays are [n_series, horizon]: A = actuals, F = forecasts. `history` is each series' training data
(used only by the scaled metrics MASE / RMSSE, which compare the error to a naive forecast's in-sample error).

Point metrics
    MAE, RMSE        scale-dependent; fine within one series, meaningless summed across items of different volume
    MAPE             undefined when actual = 0 (68% of M5 cells) -> computed on non-zero actuals only, and reported
                     with the share of cells it had to drop
    sMAPE            bounded, but asymmetric and still unstable near zero
    WAPE             sum|error| / sum(actual): the volume-weighted error planners actually feel
    bias             sum(forecast - actual) / sum(actual): systematic over/under-forecasting
    MASE / RMSSE     error relative to the in-sample naive error -> comparable across series
    WRMSSE           M5's official metric: RMSSE on all 42,840 hierarchy series, weighted by recent dollar sales

Probabilistic metrics: pinball (quantile) loss, interval coverage, interval width.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from forecastlab.data.hierarchy import Hierarchy, m5_hierarchy
from forecastlab.data.load import Dataset


def _scale(history: np.ndarray, lag: int, power: int) -> np.ndarray:
    """Mean |y_t - y_{t-lag}|^power over each series' history, starting at its first non-zero value."""
    started = np.maximum.accumulate(history > 0, axis=1)
    d = np.abs(history[:, lag:] - history[:, :-lag]) ** power
    m = started[:, :-lag]  # both points after launch
    cnt = m.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        s = (d * m).sum(axis=1) / cnt
    s[(cnt == 0) | (s == 0)] = np.nan
    return s


def point_metrics(A: np.ndarray, F: np.ndarray, history: np.ndarray | None = None) -> dict[str, float]:
    A = np.asarray(A, dtype=np.float64)
    F = np.asarray(F, dtype=np.float64)
    E = F - A
    out = {
        "mae": float(np.mean(np.abs(E))),
        "rmse": float(np.sqrt(np.mean(E ** 2))),
        "wape": float(np.abs(E).sum() / max(A.sum(), 1e-9)),
        "bias": float(E.sum() / max(A.sum(), 1e-9)),
    }
    nz = A > 0
    out["mape"] = float(np.mean(np.abs(E[nz]) / A[nz])) if nz.any() else np.nan
    out["mape_cells_dropped"] = float(1 - nz.mean())
    den = np.abs(A) + np.abs(F)
    with np.errstate(invalid="ignore", divide="ignore"):
        out["smape"] = float(np.mean(np.where(den == 0, 0.0, 2 * np.abs(E) / den)))
    if history is not None:
        mase_s = _scale(history, 1, 1)
        rmsse_s = _scale(history, 1, 2)
        with np.errstate(invalid="ignore", divide="ignore"):
            out["mase"] = float(np.nanmean(np.mean(np.abs(E), axis=1) / mase_s))
            out["rmsse"] = float(np.nanmean(np.sqrt(np.mean(E ** 2, axis=1) / rmsse_s)))
    return out


def pinball(A: np.ndarray, Q: np.ndarray, q: float) -> float:
    d = A - Q
    return float(np.mean(np.maximum(q * d, (q - 1) * d)))


def interval_metrics(A: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> dict[str, float]:
    return {"coverage": float(np.mean((A >= lo) & (A <= hi))), "mean_width": float(np.mean(hi - lo))}


class WRMSSE:
    """M5 Weighted Root Mean Squared Scaled Error for one forecast origin.

    origin = index of the last training day; the test window is origin+1 .. origin+horizon.
    """

    def __init__(self, ds: Dataset, origin: int, horizon: int = 28, hier: Hierarchy | None = None):
        self.hier = hier or m5_hierarchy()
        self.origin, self.horizon = origin, horizon
        S = self.hier.S
        hist = np.asarray(S @ ds.Y[:, : origin + 1])
        self.scale = _scale(hist, 1, 2)
        dollars = np.nan_to_num(ds.Y[:, origin - 27: origin + 1] * ds.P[:, origin - 27: origin + 1]).sum(axis=1)
        agg = np.asarray(S @ dollars).ravel()
        lvl = self.hier.keys["level"].to_numpy()
        tot = pd.Series(agg).groupby(lvl).transform("sum").to_numpy()
        self.weights = agg / tot / 12.0
        self.weights[np.isnan(self.scale)] = 0.0
        self.levels = lvl
        self.actual = np.asarray(S @ ds.Y[:, origin + 1: origin + 1 + horizon])

    def rmsse(self, F_bottom: np.ndarray) -> np.ndarray:
        Fa = np.asarray(self.hier.S @ np.asarray(F_bottom, dtype=np.float32))
        with np.errstate(invalid="ignore", divide="ignore"):
            r = np.sqrt(np.mean((Fa - self.actual) ** 2, axis=1) / self.scale)
        return np.nan_to_num(r)

    def score(self, F_bottom: np.ndarray) -> float:
        return float(np.sum(self.weights * self.rmsse(F_bottom)))

    def by_level(self, F_bottom: np.ndarray) -> pd.Series:
        """WRMSSE contribution of each level, rescaled so each value is that level's own weighted RMSSE."""
        contrib = pd.Series(self.weights * self.rmsse(F_bottom)).groupby(self.levels).sum() * 12
        contrib.index = [self.hier.keys.drop_duplicates("level").set_index("level").loc[i, "level_name"]
                         for i in contrib.index]
        return contrib

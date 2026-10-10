"""Turning a mean forecast into a predictive distribution (P10 / P50 / P90).

    negative binomial   mean from the champion + a per-series dispersion estimated from history.
                        Counts, zero-heavy, variance >= mean: NB is the natural distribution for retail units.
    conformal           distribution-free: quantiles of past *scaled* residuals of the same model, by horizon week
                        and volume band, taken from earlier backtest windows only.

Quantiles of a sum are not sums of quantiles: store-level P90 must come from the store's distribution, never by
adding up item P90s (Phase 15 shows how wrong that is).
"""
from __future__ import annotations

import numpy as np
from scipy import stats

VOLUME_BANDS = np.array([0.2, 0.5, 1, 2, 5, 10])


def nb_dispersion(H: np.ndarray, window: int = 365, smooth: int = 28) -> np.ndarray:
    """Per-series NB dispersion alpha (Var = mu + alpha mu^2) by method of moments around a rolling mean."""
    X = H[:, -window - smooth:].astype(np.float64)
    C = np.cumsum(np.pad(X, ((0, 0), (1, 0))), axis=1)
    mu = (C[:, smooth:-1] - C[:, :-smooth - 1]) / smooth          # trailing mean before each day
    y = X[:, smooth:]
    started = mu > 0
    with np.errstate(invalid="ignore", divide="ignore"):
        num = np.where(started, (y - mu) ** 2 - mu, 0).sum(axis=1)
        den = np.where(started, mu ** 2, 0).sum(axis=1)
        a = num / den
    return np.clip(np.nan_to_num(a, nan=0.5), 0.01, 10.0)


def nb_quantiles(mu: np.ndarray, alpha: np.ndarray, qs: list[float]) -> np.ndarray:
    """Quantiles [len(qs), n, h] of NB(mean mu, dispersion alpha per row)."""
    mu = np.maximum(np.asarray(mu, dtype=np.float64), 1e-6)
    r = 1.0 / alpha[:, None]
    p = r / (r + mu)
    return np.stack([stats.nbinom.ppf(q, r, p) for q in qs]).astype(np.float32)


def nb_sum_quantiles(mu: np.ndarray, alpha: np.ndarray, qs: list[float], axis: int = 1) -> np.ndarray:
    """Quantiles of a sum of independent NB variables (e.g. a lead-time total), moment-matched to one NB."""
    m = mu.sum(axis=axis)
    v = (mu + alpha[:, None] * mu ** 2).sum(axis=axis) if mu.ndim == 2 else mu + alpha * mu ** 2
    a = np.clip((v - m) / np.maximum(m, 1e-6) ** 2, 1e-4, None)
    r = 1.0 / a
    p = r / (r + np.maximum(m, 1e-6))
    return np.stack([stats.nbinom.ppf(q, r, p) for q in qs]).astype(np.float32)


class Conformal:
    """Scaled-residual conformal intervals, calibrated per (horizon week, volume band)."""

    def __init__(self, qs: list[float]):
        self.qs = qs
        self.table: dict[tuple[int, int], np.ndarray] = {}

    @staticmethod
    def _keys(F: np.ndarray, level: np.ndarray):
        week = np.minimum(np.arange(F.shape[1]) // 7, 3)[None, :].repeat(len(F), 0)
        band = np.digitize(level, VOLUME_BANDS)[:, None].repeat(F.shape[1], 1)
        return week, band

    def fit(self, pairs: list[tuple[np.ndarray, np.ndarray, np.ndarray]]) -> "Conformal":
        """pairs: (actual [n,h], forecast [n,h], recent level [n]) from earlier windows."""
        res, wk, bd = [], [], []
        for A, F, level in pairs:
            s = (A - F) / np.sqrt(F + 0.25)
            w, b = self._keys(F, level)
            res.append(s.ravel()); wk.append(w.ravel()); bd.append(b.ravel())
        res, wk, bd = map(np.concatenate, (res, wk, bd))
        for w in range(4):
            for b in range(len(VOLUME_BANDS) + 1):
                m = (wk == w) & (bd == b)
                if m.sum() > 50:
                    self.table[(w, b)] = np.quantile(res[m], self.qs)
        self.fallback = np.quantile(res, self.qs)
        return self

    def predict(self, F: np.ndarray, level: np.ndarray) -> np.ndarray:
        w, b = self._keys(F, level)
        out = np.zeros((len(self.qs), *F.shape), dtype=np.float32)
        filled = np.zeros(F.shape, dtype=bool)
        for (wi, bi), qv in self.table.items():
            m = (w == wi) & (b == bi)
            filled |= m
            for j, q in enumerate(qv):
                out[j][m] = F[m] + q * np.sqrt(F[m] + 0.25)
        for j, q in enumerate(self.fallback):
            out[j][~filled] = F[~filled] + q * np.sqrt(F[~filled] + 0.25)
        return np.clip(np.sort(out, axis=0), 0, None)


def recent_level(H: np.ndarray, days: int = 28) -> np.ndarray:
    return H[:, -days:].mean(axis=1)

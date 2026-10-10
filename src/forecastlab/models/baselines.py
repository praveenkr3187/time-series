"""Vectorised local models: every function maps a history matrix H [n_series, T] to forecasts F [n_series, h].

They loop over time (never over series), so all 30,490 M5 series are forecast in seconds. These are the
first rungs of the ladder - every later model has to beat them to justify its cost.

    naive              F = last value
    seasonal_naive     F = value on the same weekday last week
    moving_average     F = mean of the last k days
    weekday_profile    F = 28-day level x each weekday's share of demand (seasonal moving average)
    ses                simple exponential smoothing, alpha chosen per series on in-sample one-step error
    croston / sba      intermittent demand: smooth non-zero sizes and the gaps between them separately
    tsb                Teunter-Syntetos-Babai: smooth the probability of a sale instead of the gap (handles obsolescence)
"""
from __future__ import annotations

import numpy as np

ALPHAS = np.array([0.02, 0.05, 0.1, 0.2, 0.3, 0.5], dtype=np.float32)


def naive(H: np.ndarray, h: int) -> np.ndarray:
    return np.repeat(H[:, -1:], h, axis=1)


def seasonal_naive(H: np.ndarray, h: int, m: int = 7) -> np.ndarray:
    return np.tile(H[:, -m:], int(np.ceil(h / m)))[:, :h]


def moving_average(H: np.ndarray, h: int, k: int = 28) -> np.ndarray:
    return np.repeat(H[:, -k:].mean(axis=1, keepdims=True), h, axis=1)


def weekday_profile(H: np.ndarray, h: int, k: int = 28, m: int = 7, profile_weeks: int = 26) -> np.ndarray:
    """Level from the last k days, weekday shape from the last `profile_weeks` weeks (shrunk towards flat)."""
    level = H[:, -k:].mean(axis=1, keepdims=True)
    P = H[:, -profile_weeks * m:].reshape(len(H), profile_weeks, m).sum(axis=1)
    tot = P.sum(axis=1, keepdims=True)
    shape = np.where(tot > 0, (P + 1) / (tot + m) * m, 1.0)   # +1 smoothing: sparse series stay near flat
    return level * np.tile(shape, int(np.ceil(h / m)))[:, :h]


def ses(H: np.ndarray, h: int, window: int = 365, alphas: np.ndarray = ALPHAS) -> np.ndarray:
    X = H[:, -window:].astype(np.float32)
    level = np.repeat(X[:, :28].mean(axis=1, keepdims=True), len(alphas), axis=1)
    sse = np.zeros_like(level)
    for t in range(X.shape[1]):
        e = X[:, t:t + 1] - level
        if t >= 28:
            sse += e ** 2
        level += alphas * e
    best = sse.argmin(axis=1)
    return np.repeat(level[np.arange(len(X)), best][:, None], h, axis=1)


def _intermittent(H: np.ndarray, h: int, alpha: float, kind: str, window: int, beta: float = 0.05) -> np.ndarray:
    X = H[:, -window:].astype(np.float32)
    nz = X > 0
    cnt = nz.sum(axis=1)
    z = np.where(cnt > 0, X.sum(axis=1) / np.maximum(cnt, 1), 0.0)          # mean demand size
    p = np.where(cnt > 0, X.shape[1] / np.maximum(cnt, 1), X.shape[1])       # mean gap between sales
    pr = cnt / X.shape[1]                                                    # probability of a sale
    q = np.ones(len(X), dtype=np.float32)                                    # periods since last sale
    for t in range(X.shape[1]):
        y, s = X[:, t], nz[:, t]
        z = np.where(s, z + alpha * (y - z), z)
        if kind == "tsb":
            pr = pr + beta * (s - pr)
        else:
            p = np.where(s, p + alpha * (q - p), p)
            q = np.where(s, 1.0, q + 1.0)
    if kind == "tsb":
        f = pr * z
    else:
        f = z / np.maximum(p, 1.0)
        if kind == "sba":
            f = f * (1 - alpha / 2)
    return np.repeat(f[:, None], h, axis=1)


def croston(H: np.ndarray, h: int, alpha: float = 0.1, window: int = 365) -> np.ndarray:
    return _intermittent(H, h, alpha, "croston", window)


def sba(H: np.ndarray, h: int, alpha: float = 0.1, window: int = 365) -> np.ndarray:
    return _intermittent(H, h, alpha, "sba", window)


def tsb(H: np.ndarray, h: int, alpha: float = 0.1, beta: float = 0.05, window: int = 365) -> np.ndarray:
    return _intermittent(H, h, alpha, "tsb", window, beta)


LADDER = {
    "Naive": naive,
    "Seasonal naive (7)": seasonal_naive,
    "Moving average (7)": lambda H, h: moving_average(H, h, 7),
    "Moving average (28)": moving_average,
    "SES (alpha per series)": ses,
    "Croston": croston,
    "SBA": sba,
    "TSB": tsb,
    "Weekday profile x MA(28)": weekday_profile,
}


def syntetos_boylan_class(H: np.ndarray) -> np.ndarray:
    """Demand pattern classes (Syntetos & Boylan 2005) from each series' history after launch.

    ADI  = average interval between non-zero demands   (cut-off 1.32)
    CV^2 = squared coefficient of variation of non-zero demand sizes (cut-off 0.49)
    """
    started = np.maximum.accumulate(H > 0, axis=1)
    nz = H > 0
    n_days = started.sum(axis=1)
    n_nz = nz.sum(axis=1)
    adi = np.where(n_nz > 0, n_days / np.maximum(n_nz, 1), np.inf)
    sizes = np.where(nz, H, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        cv2 = (np.nanstd(sizes, axis=1) / np.nanmean(sizes, axis=1)) ** 2
    cv2 = np.nan_to_num(cv2)
    return np.select([(adi < 1.32) & (cv2 < 0.49), (adi < 1.32), (cv2 < 0.49)],
                     ["smooth", "erratic", "intermittent"], "lumpy")

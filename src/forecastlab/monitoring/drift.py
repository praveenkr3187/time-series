"""Population Stability Index: has a distribution moved away from its reference?

PSI = sum_b (p_b - q_b) ln(p_b / q_b)      < 0.1 stable, 0.1-0.2 moderate, > 0.2 significant shift
"""
from __future__ import annotations

import numpy as np

UNIT_BINS = np.array([0.5, 1.5, 2.5, 4.5, 9.5, 19.5])   # 0 | 1 | 2 | 3-4 | 5-9 | 10-19 | 20+ units


def distribution(x: np.ndarray, bins: np.ndarray = UNIT_BINS) -> np.ndarray:
    counts = np.bincount(np.digitize(np.asarray(x).ravel(), bins), minlength=len(bins) + 1).astype(float)
    return counts / max(counts.sum(), 1)


def psi(expected: np.ndarray, actual: np.ndarray, eps: float = 1e-4) -> float:
    p = np.clip(actual, eps, None)
    q = np.clip(expected, eps, None)
    return float(np.sum((p - q) * np.log(p / q)))

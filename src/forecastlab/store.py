"""Forecast cache: every model's forecasts for every origin, so later phases compare without refitting.

    artifacts/forecasts/<model_key>/<origin_day>.npy     [n_series, horizon] (NaN = series not forecast)
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from forecastlab.config import artifacts_dir


def _path(model: str, origin: int) -> Path:
    key = re.sub(r"[^a-z0-9]+", "_", model.lower()).strip("_")
    d = artifacts_dir() / "forecasts" / key
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{origin}.npy"


def save_forecast(model: str, origin: int, F: np.ndarray) -> None:
    np.save(_path(model, origin), np.asarray(F, dtype=np.float32))


def load_forecast(model: str, origin: int) -> np.ndarray | None:
    p = _path(model, origin)
    return np.load(p) if p.exists() else None


def cached(model: str, origin: int, fn) -> np.ndarray:
    F = load_forecast(model, origin)
    if F is None:
        F = fn()
        save_forecast(model, origin, F)
    return F

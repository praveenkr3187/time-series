"""Global gradient-boosted forecasters (one LightGBM per store, shared across that store's 3,049 items).

"Global" = one model learns from many series at once, instead of one model per series. It shares
patterns (weekday shape, price response, event lift) across items, which is why it beats per-series
statistical models on sparse retail data - and it scales: 10 models instead of 30,490.

Multi-horizon strategies (Phase 11):
    direct      one model with min_lag = h_max; forecasts every horizon day at once
    bucketed    several direct models, min_lag 7/14/21/28, each serving its own horizon bucket
    recursive   one lag-1 model applied 28 times, feeding its predictions back as lags
"""
from __future__ import annotations

import logging
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from forecastlab.config import load_config
from forecastlab.data.load import Dataset, model_start_day
from forecastlab.models.features import CATEGORICAL, FeatureBuilder

log = logging.getLogger(__name__)


def lgbm_params(**overrides) -> dict:
    p = dict(load_config()["model"]["lgbm"])
    p.update(overrides)
    return p


def store_rows(ds: Dataset) -> dict[str, np.ndarray]:
    return {s: np.flatnonzero(ds.series["store_id"].to_numpy() == s) for s in sorted(ds.series["store_id"].unique())}


def training_data(fb: FeatureBuilder, origin: int, min_lag: int, start: int, row_frac: float = 1.0, seed: int = 0,
                  exclude: np.ndarray | None = None) -> tuple[pd.DataFrame, np.ndarray]:
    """Training rows for target days start..origin. `exclude` [n_rows, n_days] drops cells (e.g. stock-out days)."""
    days = np.arange(start, origin + 1)
    X = fb.frame(days, min_lag)
    y = fb.target(days)
    keep = fb.on_shelf(days)
    if exclude is not None:
        keep &= ~exclude[:, days].ravel()
    if row_frac < 1:
        keep &= np.random.default_rng(seed).random(len(keep)) < row_frac
    return X[keep].reset_index(drop=True), y[keep]


def fit(X: pd.DataFrame, y: np.ndarray, sample_weight: np.ndarray | None = None, **overrides) -> lgb.LGBMRegressor:
    params = lgbm_params(**overrides)
    model = lgb.LGBMRegressor(**params, random_state=load_config()["model"]["random_state"], n_jobs=-1, verbose=-1)
    cat = [c for c in CATEGORICAL if c in X.columns]
    model.fit(X, y, sample_weight=sample_weight, categorical_feature=cat)
    return model


def predict_days(model, fb: FeatureBuilder, days: np.ndarray, min_lag: int) -> np.ndarray:
    X = fb.frame(days, min_lag)
    p = np.clip(model.predict(X), 0, None) * fb.on_shelf(days)
    return p.reshape(len(fb.rows), len(days))


BUCKETS = {"direct": [(28, 1, 28)], "bucketed": [(7, 1, 7), (14, 8, 14), (21, 15, 21), (28, 22, 28)]}


def fit_xgb(X: pd.DataFrame, y: np.ndarray):
    """XGBoost with the same Tweedie loss; categorical codes are used as plain numbers."""
    import xgboost as xgb
    p = lgbm_params()
    m = xgb.XGBRegressor(objective="reg:tweedie", tweedie_variance_power=p["tweedie_variance_power"],
                         n_estimators=p["n_estimators"], learning_rate=p["learning_rate"], max_depth=8,
                         subsample=p["subsample"], colsample_bytree=p["colsample_bytree"], tree_method="hist",
                         max_bin=p["max_bin"], min_child_weight=10, n_jobs=-1,
                         random_state=load_config()["model"]["random_state"])
    m.fit(X, y)
    return m


def forecast_store(ds: Dataset, rows: np.ndarray, origin: int, h: int = 28, strategy: str = "direct",
                   start: int | None = None, row_frac: float = 1.0, return_models: bool = False,
                   exclude: np.ndarray | None = None, learner: str = "lgbm", **overrides):
    """Train on data up to `origin` and forecast days origin+1 .. origin+h for one group of series."""
    start = model_start_day(ds) if start is None else start
    fb = FeatureBuilder(ds, rows)
    F = np.zeros((len(rows), h), dtype=np.float32)
    models = {}
    if strategy == "recursive":
        X, y = training_data(fb, origin, 1, start, row_frac, exclude=exclude)
        m = fit(X, y, **overrides) if learner == "lgbm" else fit_xgb(X, y)
        models[1] = m
        Yb = fb.Y.copy()
        Yb[:, origin + 1:] = np.nan  # make sure nothing from the future survives
        for s in range(1, h + 1):
            t = origin + s
            fb.set_sales(np.nan_to_num(Yb))
            F[:, s - 1] = predict_days(m, fb, np.array([t]), 1)[:, 0]
            Yb[:, t] = F[:, s - 1]
    else:
        for min_lag, h0, h1 in BUCKETS[strategy]:
            if h0 > h:
                break
            X, y = training_data(fb, origin, min_lag, start, row_frac, exclude=exclude)
            m = fit(X, y, **overrides) if learner == "lgbm" else fit_xgb(X, y)
            models[min_lag] = m
            days = np.arange(origin + h0, origin + min(h1, h) + 1)
            F[:, h0 - 1: min(h1, h)] = predict_days(m, fb, days, min_lag)
    return (F, models) if return_models else F


def forecast_all(ds: Dataset, origin: int, h: int = 28, strategy: str = "direct", stores: list[str] | None = None,
                 **kw) -> np.ndarray:
    """Per-store global models for every store -> F [n_series, h] (rows outside `stores` stay NaN)."""
    F = np.full((len(ds.series), h), np.nan, dtype=np.float32)
    for s, rows in store_rows(ds).items():
        if stores and s not in stores:
            continue
        t0 = time.time()
        F[rows] = forecast_store(ds, rows, origin, h, strategy, **kw)
        log.info("  %s %s done in %.0fs", strategy, s, time.time() - t0)
    return F

"""Explaining LightGBM forecasts with exact TreeSHAP contributions, grouped into business drivers.

The Tweedie model predicts log(expected units); SHAP contributions add up in that log space, so each driver is a
*multiplicative factor* on the forecast:

    forecast = exp(base) x exp(c_level) x exp(c_weekday) x exp(c_event) x ...

A factor of 1.25 for "events" reads: "the event effect raises this day's forecast by 25%".
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DRIVERS = {
    "recent sales level": ["mean_", "std_", "lag_", "same_wday_mean", "zero_share", "days_since_sale",
                           "store_dept_mean", "mean_28_vs", "mean_7_vs"],
    "day of week": ["wday"],
    "events & holidays": ["event_name_1", "event_type_1", "event_name_2", "days_to_event", "days_since_event"],
    "SNAP (food stamps)": ["snap"],
    "price & promotion": ["price", "price_vs_max", "price_vs_365d_mean", "price_change_7d", "price_vs_item_mean"],
    "season (month / week)": ["month", "week", "mday"],
    "item & store identity": ["item_id", "dept_id", "cat_id", "store_id", "state_id"],
    "time on shelf": ["days_on_shelf"],
}


def driver_of(feature: str) -> str:
    for d, keys in DRIVERS.items():
        if any(feature == k or feature.startswith(k) for k in keys):
            return d
    return "other"


def contributions(booster, X: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Per-row contributions grouped by driver (log space) and the base value per row."""
    C = booster.predict(X, pred_contrib=True)
    feats = list(X.columns)
    base = C[:, -1]
    groups = pd.Series([driver_of(f) for f in feats])
    df = pd.DataFrame(C[:, :-1], columns=feats).T.groupby(groups.to_numpy()).sum().T
    return df, base


def factors(booster, X: pd.DataFrame) -> pd.DataFrame:
    """Multiplicative factor per driver per row (exp of grouped log contributions) + implied baseline."""
    df, base = contributions(booster, X)
    out = np.exp(df)
    out.insert(0, "baseline units", np.exp(base))
    return out

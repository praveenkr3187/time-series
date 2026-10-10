"""Phase 10 - Feature engineering for global ML forecasters.

    python -m forecastlab.pipelines.features

Feature catalogue, the automated look-ahead guard, and an ablation: which feature groups earn their place?
"""
from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from forecastlab import viz
from forecastlab.data.load import load_dataset
from forecastlab.metrics import point_metrics
from forecastlab.models import gbm
from forecastlab.models.features import (FEATURE_GROUPS, FeatureBuilder, assert_no_lookahead, feature_group,
                                         lags_for)
from forecastlab.report import Report
from forecastlab.validation import backtest_origins

log = logging.getLogger(__name__)
STORE = "TX_3"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ds = load_dataset()
    r = Report("10_features", "Phase 10 - Feature Engineering")
    rows = gbm.store_rows(ds)[STORE]
    fb = FeatureBuilder(ds, rows)
    o = backtest_origins(ds)[-1]
    h = o.horizon
    X = fb.frame(np.array([o.day + 1]), h)
    cols = list(X.columns)

    r.h2("The rule: a feature for day t may only see sales up to t - L")
    r.p(f"""L (the minimum lag) is the gap between the last observed day and the day being forecast. For a **direct
28-day model** L = 28, so every sales feature of day t ends at t-28 - no matter whether t is 1 or 28 days ahead.
Calendar, events, SNAP and the price schedule are **known-future** covariates and are read *at* day t.

Lags used for L = 28: {lags_for(28)} (recent lags L..L+6 plus same-weekday lags). For L = 1: {lags_for(1)}.""")
    cat = pd.DataFrame({"feature": cols, "group": [feature_group(c) for c in cols]})
    cat["reads"] = np.where(cat["group"] == "sales history (lags, windows)", "sales up to t-L", "known at t")
    r.table(cat.groupby(["group", "reads"])["feature"].apply(lambda s: ", ".join(s)).to_frame())

    r.h2("Automated look-ahead guard")
    for L in [1, 7, 28]:
        assert_no_lookahead(fb, np.array([o.day + 1]), L)
    r.p("""`assert_no_lookahead` scrambles every sales value after t - L and recomputes the features of day t; any
column that changes is reading data the forecaster cannot have. It passes for L = 1, 7 and 28 and runs in the unit
tests, so a new feature that peeks (e.g. a centred window, or `rolling().mean()` without a shift) fails CI
instead of silently inflating backtest accuracy (Phase 4 showed by how much).""")

    r.h2(f"Ablation: which feature groups matter? (store {STORE}, 28-day direct LightGBM)")
    groups = {
        "identity + calendar": FEATURE_GROUPS["identity"] + FEATURE_GROUPS["calendar & events"],
    }
    hist_cols = [c for c in cols if feature_group(c) == "sales history (lags, windows)"]
    groups["+ sales history"] = groups["identity + calendar"] + hist_cols
    groups["+ price (all features)"] = cols
    groups["all except calendar & events"] = [c for c in cols if c not in FEATURE_GROUPS["calendar & events"]]
    Xtr, ytr = gbm.training_data(fb, o.day, h, ds.day("2014-01-01"))
    test_days = np.arange(o.test.start, o.test.stop)
    Xte = fb.frame(test_days, h)
    shelf = fb.on_shelf(test_days)
    A = fb.Y[:, test_days]
    hist = fb.Y[:, : o.day + 1]
    res, full_model = {}, None
    for name, cs in groups.items():
        m = gbm.fit(Xtr[cs], ytr)
        F = (np.clip(m.predict(Xte[cs]), 0, None) * shelf).reshape(len(rows), h)
        pm = point_metrics(A, F, hist)
        agg = point_metrics(A.sum(axis=0, keepdims=True), F.sum(axis=0, keepdims=True))
        res[name] = {"n_features": len(cs), "item RMSSE": pm["rmsse"], "item WAPE": pm["wape"], "bias": pm["bias"],
                     "store-total WAPE": agg["wape"]}
        if cs is cols:
            full_model = m
        log.info("%s done", name)
    t = pd.DataFrame(res).T
    r.table(t, floatfmt=".3f")
    r.p(f"""Test window {ds.dates[o.test.start].date()} .. {ds.dates[o.test.stop - 1].date()} (backtest fold 3), {len(rows):,} series.
"store-total WAPE" sums the item forecasts to the store's daily total.

* Identity + calendar alone is already a model: it learns each item's average level and weekday shape.
* Sales history adds the *current* level - items drift, launch and get delisted.
* Price features help most where prices move (promotions); removing calendar & events hurts the store total, where
  weekday and event effects dominate.""")

    imp = pd.Series(full_model.booster_.feature_importance("gain"), index=cols)
    imp = (imp / imp.sum()).sort_values(ascending=False)
    gi = imp.groupby([feature_group(c) for c in imp.index]).sum().sort_values(ascending=False)
    r.h2("Feature importance (gain share, full model)")
    r.table(gi.rename("gain share").to_frame(), floatfmt=".3f")
    fig, ax = plt.subplots(figsize=(7, 5.5))
    top = imp.head(20)
    viz.bar(ax, list(top.index), top.to_numpy(), fmt="{:.1%}", horizontal=True)
    ax.set(title="Top 20 features by gain share", xlabel="share of total gain")
    r.figure(viz.save(fig, r.dir / "importance.png"), "Feature importance")
    r.p("Gain importance says what the trees *used*, not what *causes* demand; Phase 18 uses SHAP to explain "
        "individual forecasts.")
    r.metrics = {"ablation": t.to_dict(orient="index"), "group_importance": gi.to_dict(),
                 "top_features": imp.head(20).to_dict()}
    r.save()
    print(t.round(3).to_string())


if __name__ == "__main__":
    main()

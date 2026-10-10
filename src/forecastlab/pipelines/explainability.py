"""Phase 18 - Explaining forecasts: SHAP drivers as multiplicative factors.

    python -m forecastlab.pipelines.explainability     (needs the production model: `make train-prod`)
"""
from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from forecastlab import viz
from forecastlab.config import load_config
from forecastlab.data.load import load_dataset
from forecastlab.models import gbm
from forecastlab.models.explain import contributions
from forecastlab.models.registry import load_registry
from forecastlab.report import Report
from forecastlab.serving.forecaster import Forecaster

log = logging.getLogger(__name__)
STORE = "CA_3"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ds = load_dataset()
    r = Report("18_explainability", "Phase 18 - Explaining Forecasts")
    reg = load_registry()
    version = "lgbm_v1" if "lgbm_v1" in reg["models"] else reg["champion"]
    fc = Forecaster(version)
    origin = ds.day(load_config()["validation"]["production_start"]) - 1
    h = load_config()["forecast"]["horizon"]
    rows = gbm.store_rows(ds)[STORE]
    fb = fc._builder(ds, STORE, rows)
    days = np.arange(origin + 1, origin + 1 + h)
    X = fb.frame(days, fc.min_lag)
    shelf = fb.on_shelf(days)
    C, base = contributions(fc.boosters[STORE], X[shelf])
    r.p(f"""Model `{version}`, store {STORE}, forecasts made on {ds.dates[origin].date()} for the next {h} days
({shelf.sum():,} item-days). LightGBM computes exact TreeSHAP contributions natively
(`predict(pred_contrib=True)`). With the Tweedie loss the model works in log space, so contributions add in log space
and **multiply** on the forecast: `forecast = baseline x factor_1 x factor_2 x ...`. Features are grouped into
business drivers (`models/explain.py`).""")
    imp = C.abs().mean().sort_values(ascending=False)
    r.h2("Which drivers move forecasts most? (mean |log factor|)")
    t = pd.DataFrame({"mean |log contribution|": imp, "typical factor range": [
        f"x{np.exp(-v):.2f} .. x{np.exp(v):.2f}" for v in imp]})
    r.table(t, floatfmt=".3f")
    fig, ax = plt.subplots(figsize=(7, 3.6))
    viz.bar(ax, list(imp.index), imp.to_numpy(), fmt="{:.3f}", horizontal=True)
    ax.set(title=f"Driver importance, {STORE} forecasts", xlabel="mean |log contribution|")
    r.figure(viz.save(fig, r.dir / "drivers.png"), "Driver importance")

    # an example FOODS item: daily factor lines
    meta = ds.series.iloc[rows]
    vol = ds.Y[rows, origin - 27: origin + 1].mean(axis=1)
    cand = np.flatnonzero((meta["cat_id"].to_numpy() == "FOODS") & (vol > 5))
    j = int(cand[np.argsort(vol[cand])[len(cand) // 2]])
    Xi = X.iloc[j * h: (j + 1) * h]
    Ci, bi = contributions(fc.boosters[STORE], Xi)
    F = np.exp(Ci)
    fig, ax = plt.subplots(figsize=(10, 3.6))
    for k, c in enumerate(["day of week", "SNAP (food stamps)", "events & holidays", "price & promotion",
                           "recent sales level"]):
        if c in F:
            ax.plot(ds.dates[days], F[c], marker="o", markersize=3, color=viz.SERIES[k], label=c)
    ax.axhline(1, color=viz.MUTED, linewidth=1)
    ax.set(title=f"{meta['id'].iloc[j]}: driver factors for each forecast day", ylabel="multiplicative factor")
    ax.legend(fontsize=8, ncol=3)
    r.figure(viz.save(fig, r.dir / "item_factors.png"), "Item factors")
    pred = np.exp(bi + Ci.sum(axis=1))
    dec = pd.DataFrame({"date": ds.dates[days].date, "forecast": pred, "baseline": np.exp(bi), **{c: F[c] for c in F}})
    snap_col = "SNAP (food stamps)"
    pick = [int(np.argmax(dec[snap_col])), int(np.argmin(dec[snap_col]))] if snap_col in dec else [0, 1]
    r.h2(f"One item, two days: {meta['id'].iloc[j]}")
    r.table(dec.iloc[pick].set_index("date").T, floatfmt=".3f")
    r.p("""Read a column top to bottom: baseline x each factor = forecast. This is the form planners can use ("Saturday
+25%, SNAP day +8%, no promotion"), and it is exactly what the GenAI analyst (Phase 23) receives as evidence.
SHAP explains the **model**, not the world: a large "recent sales level" factor means the model leans on history,
not that history causes demand.""")
    r.metrics = {"driver_importance": imp.to_dict(), "example_item": meta["id"].iloc[j]}
    r.save()
    print(t.to_string())


if __name__ == "__main__":
    main()

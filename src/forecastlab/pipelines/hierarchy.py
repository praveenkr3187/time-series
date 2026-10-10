"""Phase 16 - Hierarchical forecasting and reconciliation.

    python -m forecastlab.pipelines.hierarchy

Hierarchy: total -> state -> store, crossed with category -> department; bottom = store x department (70),
154 nodes in total. Base forecasts: AutoETS on every node (independent, incoherent) and the champion LightGBM
summed up from items (coherent by construction). Reconciliation methods are compared level by level.
"""
from __future__ import annotations

import logging
import time

import numpy as np
import pandas as pd

from forecastlab.data.hierarchy import LEVELS, build_hierarchy, m5_hierarchy
from forecastlab.data.load import load_dataset
from forecastlab.metrics import point_metrics
from forecastlab.models import classical
from forecastlab.models.reconcile import is_coherent, reconcile
from forecastlab.report import Report
from forecastlab.store import cached, load_forecast
from forecastlab.validation import backtest_origins, holdout_origin

log = logging.getLogger(__name__)
CHAMP = "LightGBM (per-store, direct)"
HISTORY = 730
METHODS = ["bottom-up", "top-down (historical shares)", "OLS", "WLS (structural)", "WLS (variance)", "MinT (shrink)"]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ds = load_dataset()
    r = Report("16_hierarchy", "Phase 16 - Hierarchical Forecasting and Reconciliation")
    m5 = m5_hierarchy()
    bottom_rows = m5.level_rows(9)
    keys = m5.keys.loc[bottom_rows, "key"].str.split("/", expand=True)
    meta = pd.DataFrame({"store_id": keys[0], "dept_id": keys[1]})
    meta["state_id"] = meta["store_id"].str[:2]
    meta["cat_id"] = meta["dept_id"].str.rsplit("_", n=1).str[0]
    levels = LEVELS[:9]
    hier = build_hierarchy(meta, levels)
    S = hier.S.toarray()
    lvl = hier.keys["level"].to_numpy()
    names = dict(enumerate([n for n, _ in levels], start=1))
    Yb_all = np.asarray(m5.S[bottom_rows] @ ds.Y)          # [70, T]
    Y_all = S @ Yb_all                                         # [154, T]
    origins = backtest_origins(ds) + [holdout_origin(ds)]
    rows = []
    for o in origins:
        s0 = o.day - HISTORY + 1
        dates = ds.dates[s0: o.day + 1]
        hist = classical.fill_closures(Y_all[:, s0: o.day + 1], dates)
        t0 = time.time()
        res = {}

        def ets(o=o, hist=hist, dates=dates):
            F, fitted = classical.forecast_with_fitted(hist, dates, o.horizon, "AutoETS")
            res["resid"] = hist - fitted
            return F
        base = cached("hier AutoETS 154", o.day, ets)
        resid = cached("hier AutoETS 154 residuals", o.day, lambda: res["resid"]) if "resid" in res else \
            load_forecast("hier AutoETS 154 residuals", o.day)
        resid = np.nan_to_num(resid[:, 7:])
        log.info("%s base ETS ready (%.0fs)", o.name, time.time() - t0)
        A = Y_all[:, o.test]
        H = Y_all[:, : o.day + 1]
        cands = {"AutoETS base (incoherent)": base}
        for m in METHODS:
            cands[f"AutoETS + {m}"] = reconcile(base, S, m, resid, hist[-S.shape[1]:])
        lgb_bottom = np.asarray(m5.S[bottom_rows] @ load_forecast(CHAMP, o.day))
        cands["LightGBM items, bottom-up"] = S @ lgb_bottom
        mixed = base.copy()
        mixed[-S.shape[1]:] = lgb_bottom
        cands["ETS upper + LightGBM bottom, WLS (structural)"] = reconcile(mixed, S, "WLS (structural)")
        cands["ETS upper + LightGBM bottom, MinT (shrink)"] = reconcile(mixed, S, "MinT (shrink)", resid)
        for name, F in cands.items():
            for L in np.unique(lvl):
                m = lvl == L
                pm = point_metrics(A[m], F[m], H[m])
                rows.append({"origin": o.name, "split": "holdout" if o is origins[-1] else "backtest",
                             "method": name, "level": names[L], "rmsse": pm["rmsse"],
                             "coherent": is_coherent(F, S, 1e-4)})
    df = pd.DataFrame(rows)
    bt = df[df["split"] == "backtest"].pivot_table(index="method", columns="level", values="rmsse", sort=False)
    bt = bt[[n for n, _ in levels]]
    bt["mean over levels"] = bt.mean(axis=1)
    ho = df[df["split"] == "holdout"].pivot_table(index="method", columns="level", values="rmsse").mean(axis=1)
    bt["holdout mean"] = ho
    coh = df.groupby("method")["coherent"].all()
    bt.insert(0, "coherent", coh.reindex(bt.index).map({True: "yes", False: "no"}))
    r.h2("RMSSE by level (mean of 3 backtest origins)")
    r.table(bt, floatfmt=".3f")
    best = bt["mean over levels"].astype(float).idxmin()
    ets_rows = [i for i in bt.index if i.startswith("AutoETS")]
    best_ets = bt.loc[ets_rows, "mean over levels"].astype(float).idxmin()
    r.p(f"""154 nodes; bottom level = store x department. Base AutoETS forecasts are fitted independently per node and
do **not** add up (`coherent = no`): the business would receive a store total that differs from the sum of its
departments.

* Best reconciliation of the ETS forecasts: **{best_ets}** (mean RMSSE {bt.loc[best_ets, 'mean over levels']:.3f} vs
  {bt.loc['AutoETS base (incoherent)', 'mean over levels']:.3f} for the incoherent base). Reconciliation is not just
  book-keeping - combining information across levels usually *improves* accuracy, because the top levels see a smooth
  signal and the bottom levels see local detail.
* Top-down with fixed historical shares is the weakest at the bottom: it cannot see that one department grows while
  another shrinks.
* Best overall: **{best}**. The item-level LightGBM summed bottom-up is already coherent and strong at every level;
  combining it with ETS at the upper levels via reconciliation tests whether the two sources add information.

**Production rule**: one coherent set of numbers. The platform serves item x store forecasts from the champion and
derives every aggregate by summation (bottom-up), unless a reconciled combination wins the backtest by a margin.""")
    r.metrics = {"rmsse_by_level": bt.drop(columns="coherent").to_dict(orient="index"), "best": best}
    r.save()
    print(bt.round(3).to_string())


if __name__ == "__main__":
    main()

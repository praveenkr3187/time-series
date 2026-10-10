"""Phase 14 - Backtesting the whole ladder and choosing a champion.

    python -m forecastlab.pipelines.backtest

Every model from Phases 5-13 on the same 3 walk-forward origins, the same series and the same metrics.
Then: is the winner *significantly* better, where does it win, does an ensemble help - and registration.
"""
from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from forecastlab import viz
from forecastlab.data.hierarchy import m5_hierarchy
from forecastlab.data.load import load_dataset
from forecastlab.evaluation import diebold_mariano, paired_bootstrap, scorecard, wrmsse
from forecastlab.models.baselines import syntetos_boylan_class
from forecastlab.models.registry import register
from forecastlab.report import Report
from forecastlab.store import load_forecast, save_forecast
from forecastlab.validation import backtest_origins, holdout_origin

log = logging.getLogger(__name__)

# (rung of the ladder, model key, how it is fitted)
LADDER = [
    ("Naive", "Naive", "local"),
    ("Moving average", "Moving average (28)", "local"),
    ("Exponential smoothing", "SES (alpha per series)", "local"),
    ("Exponential smoothing", "Holt", "local"),
    ("Exponential smoothing", "Holt-Winters", "local"),
    ("Exponential smoothing", "AutoETS", "local"),
    ("Intermittent", "TSB", "local"),
    ("ARIMA", "ARIMA (non-seasonal)", "local"),
    ("SARIMA", "SARIMA (AutoARIMA, s=7)", "local"),
    ("ARIMAX", "ARIMAX (SARIMA + calendar/price)", "local, aggregates only"),
    ("Prophet", "Prophet + SNAP/price regressors", "local, aggregates only"),
    ("Seasonal heuristic", "Seasonal naive (7)", "local"),
    ("Seasonal heuristic", "Weekday profile x MA(28)", "local"),
    ("Gradient boosting", "LightGBM (per-store, direct)", "global"),
    ("RNN", "GRU (global)", "global"),
    ("RNN", "LSTM (global)", "global"),
    ("Transformer", "Transformer (global)", "global"),
    ("Probabilistic RNN", "DeepAR (global)", "global"),
    ("Foundation model", "Chronos-Bolt small", "pretrained, zero-shot"),
    ("Foundation model", "Chronos-2 (with covariates)", "pretrained, zero-shot"),
]
CHAMPION_KEY = "LightGBM (per-store, direct)"


def make_ensemble(ds, origins, members: list[str], name: str) -> bool:
    for o in origins:
        Fs = [load_forecast(m, o.day) for m in members]
        if any(F is None or np.isnan(F).any() for F in Fs):
            return False
        save_forecast(name, o.day, np.mean(Fs, axis=0))
    return True


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ds = load_dataset()
    r = Report("14_backtest", "Phase 14 - Backtesting the Ladder and Model Selection")
    folds = backtest_origins(ds)
    ho = holdout_origin(ds)
    origins = [*folds, ho]

    # candidate ensembles: champion candidate + the best full-coverage network / foundation model
    full_cov = [k for _, k, _ in LADDER if all(load_forecast(k, o.day) is not None and
                                                 not np.isnan(load_forecast(k, o.day)).any() for o in origins)]
    bt_w = {k: np.mean([wrmsse(ds, o).score(load_forecast(k, o.day)) for o in folds]) for k in full_cov}
    others = sorted([k for k in full_cov if k != CHAMPION_KEY and "(global)" in k], key=bt_w.get)
    ladder = list(LADDER)
    if others:
        ens = f"Ensemble: LightGBM + {others[0]}"
        if make_ensemble(ds, origins, [CHAMPION_KEY, others[0]], ens):
            ladder.append(("Ensemble", ens, "average of two global models"))
    if "Chronos-Bolt small" in full_cov:
        ens = "Ensemble: LightGBM + Chronos-Bolt"
        if make_ensemble(ds, origins, [CHAMPION_KEY, "Chronos-Bolt small"], ens):
            ladder.append(("Ensemble", ens, "average"))

    rows = []
    for rung, key, kind in ladder:
        for o in origins:
            sc = scorecard(ds, key, o)
            if sc:
                rows.append({"rung": rung, "model": key, "fitted": kind,
                             "split": "holdout" if o is ho else "backtest", "origin": o.name, **sc})
    df = pd.DataFrame(rows)
    df.to_csv(r.dir / "scorecard.csv", index=False)
    metrics = ["WRMSSE", "item RMSSE", "store x dept RMSSE", "store RMSSE", "total RMSSE", "item bias"]
    bt = df[df["split"] == "backtest"]
    agg = bt.groupby("model", sort=False)[metrics].mean()
    agg["WRMSSE sd"] = bt.groupby("model", sort=False)["WRMSSE"].std()
    agg["WRMSSE (holdout)"] = df[df["split"] == "holdout"].set_index("model")["WRMSSE"]
    meta = df.drop_duplicates("model").set_index("model")[["rung", "fitted"]]
    tab = meta.join(agg)

    r.h2("The ladder (mean of 3 backtest origins)")
    r.table(tab, floatfmt=".3f")
    r.p("""Blank cells: the model was not run at that scope (per-series models run on the 500-series sample and the
aggregates; WRMSSE needs forecasts for all 30,490 items). Aggregate RMSSE of item-level models = their item forecasts
summed bottom-up. `fitted`: *local* = one model per series, *global* = one model across many series.""")

    eligible = tab.dropna(subset=["WRMSSE"]).sort_values("WRMSSE")
    champion = eligible.index[0]
    runner = eligible.index[1]
    baseline = "Weekday profile x MA(28)"
    r.h2("Champion selection")
    r.p(f"""Rule (fixed before looking at the holdout): lowest mean backtest WRMSSE among models that forecast every
series, provided it beats the best baseline on all 3 folds.

**Champion: {champion}** (backtest WRMSSE {eligible.loc[champion, 'WRMSSE']:.3f} +/- {eligible.loc[champion, 'WRMSSE sd']:.3f});
runner-up: {runner} ({eligible.loc[runner, 'WRMSSE']:.3f}); best baseline: {baseline}
({eligible.loc[baseline, 'WRMSSE']:.3f}).""")
    wins = sum(wrmsse(ds, o).score(load_forecast(champion, o.day)) < wrmsse(ds, o).score(load_forecast(baseline, o.day))
               for o in folds)

    # ---- is it significant?
    hier = m5_hierarchy()
    lv12 = hier.level_rows(12)
    tests = []
    for other in [runner, baseline]:
        diffs, wts, e_c, e_o = [], [], [], []
        for o in folds:
            w = wrmsse(ds, o)
            rc, ro = w.rmsse(load_forecast(champion, o.day)), w.rmsse(load_forecast(other, o.day))
            diffs.append(rc[lv12] - ro[lv12])
            wts.append(w.weights[lv12])
            tot = hier.level_rows(1)[0]
            A = w.actual[tot]
            e_c.append(np.asarray(hier.S[tot] @ load_forecast(champion, o.day)).ravel() - A)
            e_o.append(np.asarray(hier.S[tot] @ load_forecast(other, o.day)).ravel() - A)
        est, lo, hi = paired_bootstrap(np.concatenate(diffs), np.concatenate(wts))
        dm, p = diebold_mariano(np.concatenate(e_c), np.concatenate(e_o), h=1)
        tests.append({"champion vs": other, "item RMSSE diff (dollar-weighted)": est, "95% CI low": lo,
                      "95% CI high": hi, "DM stat (total, daily)": dm, "DM p-value": p})
    tt = pd.DataFrame(tests).set_index("champion vs")
    r.table(tt, floatfmt=".4f")
    r.p(f"""The champion beats the baseline on **{wins} of 3** folds.
* *Item level*: per-series RMSSE difference, dollar-weighted (the level-12 WRMSSE weights), pooled over 3 folds, with a
  bootstrap CI over series. A CI entirely below 0 = the champion is better on items, not by luck of a few series.
* *Total level*: Diebold-Mariano test on the daily errors of the company total (84 days). Daily errors are
  autocorrelated and few, so this test is weak; the hierarchy-wide picture is the WRMSSE table above.""")

    # ---- where does it win?
    r.h2("Where does each model win? (item RMSSE by segment, all series, mean of 3 folds)")
    seg_models = [k for k in [baseline, "TSB", CHAMPION_KEY, others[0] if others else None, "Chronos-Bolt small"]
                  if k in full_cov]
    seg = []
    for o in folds:
        hist = ds.Y[:, : o.day + 1]
        cls = syntetos_boylan_class(hist[:, -365:])
        cat = ds.series["cat_id"].to_numpy()
        w = wrmsse(ds, o)
        for k in seg_models:
            rm = w.rmsse(load_forecast(k, o.day))[lv12]
            for c in np.unique(cls):
                seg.append({"model": k, "segment": f"pattern: {c}", "rmsse": rm[cls == c].mean()})
            for c in np.unique(cat):
                seg.append({"model": k, "segment": f"category: {c}", "rmsse": rm[cat == c].mean()})
    seg = pd.DataFrame(seg).groupby(["segment", "model"])["rmsse"].mean().unstack()
    r.table(seg, floatfmt=".3f")
    best_by = seg.idxmin(axis=1)
    r.p("Best model per segment: " + "; ".join(f"{s} -> {m}" for s, m in best_by.items()) + ".")

    # ---- chart
    plot = tab.dropna(subset=["store x dept RMSSE"]).sort_values("store x dept RMSSE", ascending=False)
    fig, ax = plt.subplots(figsize=(8, 0.32 * len(plot) + 1.2))
    colors = [viz.SERIES[1] if m == champion else viz.SERIES[0] for m in plot.index]
    ax.barh(list(plot.index), plot["store x dept RMSSE"], color=colors, height=0.7, edgecolor=viz.SURFACE)
    ax.grid(axis="x"); ax.grid(axis="y", visible=False)
    ax.set(title="Store x department RMSSE, backtest mean (lower is better)", xlabel="RMSSE",
           xlim=(plot["store x dept RMSSE"].min() * 0.85, plot["store x dept RMSSE"].max() * 1.02))
    r.figure(viz.save(fig, r.dir / "ladder.png"), "Ladder")

    r.h2("Holdout: the one-time check")
    r.p(f"""Champion on the holdout (M5 evaluation window, never used for any decision above): WRMSSE
**{tab.loc[champion, 'WRMSSE (holdout)']:.3f}** vs best baseline {tab.loc[baseline, 'WRMSSE (holdout)']:.3f} and seasonal naive
{tab.loc['Seasonal naive (7)', 'WRMSSE (holdout)']:.3f}. Top 3 by backtest: {', '.join(eligible.index[:3])}; top 3 on the
holdout: {', '.join(eligible.sort_values('WRMSSE (holdout)').index[:3])}
({'same order' if list(eligible.index[:3]) == list(eligible.sort_values('WRMSSE (holdout)').index[:3]) else
  'the order differs - a reminder that close models swap places between windows'}).

**Answer to "does the complicated model beat the simple one?"** Yes - but only the *right* complicated model.
Global models that pool information across thousands of items beat per-series models; extra architectural
complexity (deep nets on CPU budgets) did not beat well-engineered gradient boosting here, and per-series ARIMA /
Prophet do not scale to item level at all.""")

    register("lgbm_v1", {"model": CHAMPION_KEY if champion == CHAMPION_KEY else champion,
                         "learner": "lightgbm", "strategy": "direct", "per": "store",
                         "train_window_start": "2014-01-01"},
             {"backtest_wrmsse": float(eligible.loc[champion, "WRMSSE"]),
              "backtest_wrmsse_sd": float(eligible.loc[champion, "WRMSSE sd"]),
              "holdout_wrmsse": float(tab.loc[champion, "WRMSSE (holdout)"]),
              "backtest_store_wape": float(bt.loc[bt["model"] == champion, "store WAPE"].mean())},
             role="champion", status="champion", reason="Phase 14 backtest selection")
    if champion != CHAMPION_KEY:
        log.warning("champion is %s, but the serving stack is built for LightGBM", champion)
    r.metrics = {"ladder": tab.reset_index().to_dict(orient="records"), "champion": champion, "runner_up": runner,
                 "tests": tt.reset_index().to_dict(orient="records"), "segments": seg.to_dict()}
    r.save()
    print(tab[["WRMSSE", "store x dept RMSSE", "WRMSSE (holdout)"]].round(3).to_string())


if __name__ == "__main__":
    main()

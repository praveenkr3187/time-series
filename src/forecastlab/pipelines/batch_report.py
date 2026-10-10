"""Phases 19 & 20 - Reports on the simulated production period (run `make nightly` first).

    python -m forecastlab.pipelines.batch_report

19_batch_forecasting : what the nightly job did - runs, timings, blocked nights, catch-up, published forecasts,
                       forecast stability across runs
20_monitoring        : live accuracy by horizon, bias, interval coverage, drift (PSI), alerts
"""
from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from forecastlab import viz
from forecastlab.config import load_config
from forecastlab.data.load import load_dataset
from forecastlab.models.registry import load_registry
from forecastlab.monitoring.accuracy import weekly_summary
from forecastlab.report import Report
from forecastlab.serving.db import ForecastDB

log = logging.getLogger(__name__)


def batch_report(db: ForecastDB, ds) -> None:
    r = Report("19_batch_forecasting", "Phase 19 - Nightly Batch Forecasting")
    runs = db.runs()
    issues = db.query("SELECT * FROM issues")
    r.p(f"""The nightly job (`serving/nightly.py`) was replayed for every night from {runs['run_date'].min()} to
{runs['run_date'].max()}: each night the previous day's POS file lands in the data lake, is validated, appended to the
curated history, and the champion model ({', '.join(runs['model_version'].unique())}) publishes a 28-day forecast
(mean, P10, P90) for all 30,490 item-stores plus 154 aggregate series. Two incidents were injected: a POS file with a
whole store missing, and a file loaded twice.""")
    r.h2("Runs")
    s = runs.groupby("status").agg(runs=("run_id", "size"), mean_duration_s=("duration_s", "mean"))
    r.table(s, floatfmt=".1f")
    blocked = runs[runs["status"] != "SUCCESS"]
    if len(blocked):
        bl = blocked[["run_date", "status"]].merge(issues[issues["severity"] == "error"],
                                                    left_on="run_date", right_on=issues["run_id"].str[4:], how="left")
        r.table(bl[["run_date", "status", "check_name", "detail"]].set_index("run_date"))
        nxt = [runs.loc[runs.index[runs.index.get_loc(i) + 1]] for i in blocked.index if runs.index.get_loc(i) + 1 < len(runs)]
        r.p("On a BLOCKED night nothing is published: planners keep the previous night's forecast (one day staler), the "
            "data team is alerted, and the corrected file is re-delivered. The next night ingests **both** days before "
            "forecasting (catch-up): " + ", ".join(f"{n['run_date']} -> {n['status']}" for n in nxt) + ".")
    warn = issues[issues["severity"] == "warning"].groupby("check_name").size().rename("nights with warning")
    if len(warn):
        r.p("Warnings that did not block a run:")
        r.table(warn.to_frame())
    fig, ax = plt.subplots(figsize=(9, 3))
    ok = runs["status"] == "SUCCESS"
    ax.bar(pd.to_datetime(runs["run_date"]), runs["duration_s"], color=np.where(ok, viz.SERIES[0], viz.STATUS["critical"]))
    ax.set(title="Nightly run duration (red = blocked)", ylabel="seconds")
    r.figure(viz.save(fig, r.dir / "durations.png"), "Run durations")

    # published store forecast with interval, from one run
    run = runs[ok].iloc[len(runs[ok]) // 2]
    agg = db.query("SELECT * FROM aggregates WHERE run_id = ? AND level = 'store' AND key = 'CA_3'", (run["run_id"],))
    agg["target_date"] = pd.to_datetime(agg["target_date"])
    o = ds.day(run["origin_date"])
    st = ds.series["store_id"].to_numpy() == "CA_3"
    hist = pd.Series(ds.Y[st].sum(axis=0), index=ds.dates)
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.plot(hist.index[o - 41: o + 29], hist.values[o - 41: o + 29], color=viz.INK_2, linewidth=1.2, label="actual")
    ax.fill_between(agg["target_date"], agg["p10"], agg["p90"], color=viz.SERIES[0], alpha=0.18, label="P10-P90")
    ax.plot(agg["target_date"], agg["mean"], color=viz.SERIES[0], label="forecast")
    ax.axvline(ds.dates[o], color=viz.MUTED, linestyle=":", linewidth=1)
    ax.set(title=f"Store CA_3 as published by {run['run_id']}", ylabel="units")
    ax.legend(loc="upper left", fontsize=8)
    r.figure(viz.save(fig, r.dir / "published_store.png"), "Published store forecast")
    inside = agg.merge(pd.DataFrame({"target_date": hist.index, "a": hist.values}), on="target_date")
    r.p(f"Published interval coverage for this store and run: {((inside['a'] >= inside['p10']) & (inside['a'] <= inside['p90'])).mean():.0%} "
        "of the 28 days inside P10-P90 (target 80%).")

    # forecast stability: how often does the forecast for a fixed day change?
    r.h2("Forecast stability across runs")
    tot = db.query("SELECT run_id, target_date, horizon, mean FROM aggregates WHERE level = 'total'")
    piv = tot.pivot_table(index="target_date", columns="horizon", values="mean")
    full = piv.dropna()
    if len(full):
        rev = (full.max(axis=1) - full.min(axis=1)) / full.mean(axis=1)
        r.p(f"""For the {len(full)} target days forecast by all 28 horizons, the company-total forecast moved by
{rev.median():.1%} (median, max-min over the 28 runs). Planners dislike "nervous" forecasts: a purchase order placed on
day-21's forecast should not be contradicted by day-7's. A direct model whose features end 28 days before the target
changes only as its 28-day-old inputs roll forward - stability is a design property, not luck.""")
    r.metrics = {"runs": s.to_dict(orient="index"), "blocked": blocked["run_date"].tolist()}
    r.save()


def monitoring_report(db: ForecastDB, ds) -> None:
    r = Report("20_monitoring", "Phase 20 - Production Monitoring")
    mcfg = load_config()["monitoring"]
    reg = load_registry()
    champ = reg["champion"]
    base = reg["models"].get(champ, {}).get("metrics", {}).get("backtest_store_wape")
    ws = weekly_summary(db, "store")
    wi = weekly_summary(db, "items")
    r.p("""As each day's actuals arrive, every forecast that targeted that day (from up to 28 earlier runs) is scored.
Accuracy is therefore known by **horizon**: day-1 forecasts are scored the next morning, day-28 forecasts four weeks
later. Store scope = error of store totals; item scope = summed item-level absolute errors.""")
    if len(ws):
        t = ws.pivot(index="week", columns="horizon_bucket", values="wape")
        t.index = t.index.date
        r.h2("Store-level WAPE by week and horizon")
        r.table(t, floatfmt=".3f")
        ti = wi.pivot(index="week", columns="horizon_bucket", values="wape")
        ti.index = ti.index.date
        r.h2("Item-level WAPE by week and horizon")
        r.table(ti, floatfmt=".3f")
        b = ws.groupby("week")[["abs_error", "actual", "forecast"]].sum()
        b["bias"] = b["forecast"] / b["actual"] - 1
        b["wape"] = b["abs_error"] / b["actual"]
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
        axes[0].plot(b.index, b["wape"], marker="o", color=viz.SERIES[0], label="live store WAPE")
        if base:
            axes[0].axhline(base, color=viz.MUTED, linestyle=":", label=f"backtest {base:.1%}")
            axes[0].axhline(base + mcfg["wape_alert"], color=viz.STATUS["critical"], linestyle="--", linewidth=1,
                            label="alert threshold")
        axes[0].set(title="Store WAPE (all horizons) by week")
        axes[0].legend(fontsize=8)
        axes[1].bar(b.index, b["bias"], width=4, color=[viz.SERIES[1] if v < 0 else viz.SERIES[0] for v in b["bias"]])
        axes[1].axhline(mcfg["bias_alert"], color=viz.STATUS["critical"], linestyle="--", linewidth=1)
        axes[1].axhline(-mcfg["bias_alert"], color=viz.STATUS["critical"], linestyle="--", linewidth=1)
        axes[1].set(title="Bias by week (forecast / actual - 1)")
        r.figure(viz.save(fig, r.dir / "live_accuracy.png"), "Live accuracy")
        r.p(f"""Live store WAPE over the period: {b['abs_error'].sum() / b['actual'].sum():.1%} (backtest expectation
{base:.1%} if known). Bias: {b['forecast'].sum() / b['actual'].sum() - 1:+.1%}. Partial weeks at the edges contain only
short or only long horizons - compare like with like (same horizon bucket).""" if base else "")
    cov = db.query("SELECT target_date, AVG(in_interval) AS coverage FROM accuracy WHERE scope='items' GROUP BY target_date")
    if len(cov):
        r.p(f"Item-level P10-P90 coverage of live forecasts: {cov['coverage'].mean():.1%} (target 80%; >80% expected "
            "for discrete low counts, see Phase 15).")
    drift = db.query("SELECT * FROM drift")
    if len(drift):
        d = drift.pivot_table(index="run_date", columns="metric", values="value")
        r.h2("Drift")
        r.table(d[[c for c in d.columns if c.startswith("psi")]].describe().loc[["mean", "max"]], floatfmt=".4f")
        r.p(f"""PSI compares the last 7 days of item-level unit sales (binned 0, 1, 2, 3-4, 5-9, 10-19, 20+) with the 28
days before production started, and the distribution of each run's first-week forecasts with the first run's.
Alert threshold {mcfg['psi_alert']}. Input drift alone is not an accuracy problem - it is an early warning to look
at accuracy and bias, which arrive later.""")
    alerts = db.query("SELECT * FROM alerts ORDER BY run_date")
    r.h2("Alerts raised")
    if len(alerts):
        r.table(alerts.groupby(["kind", "severity"]).agg(n=("message", "size"), first=("run_date", "min"),
                                                         last=("run_date", "max"), example=("message", "first")))
    else:
        r.p("No alerts.")
    r.p("""**What is monitored, and why**

| signal | latency | catches |
|---|---|---|
| data-quality gate | before the run | broken / missing / duplicated files |
| pipeline status & duration | same night | failures, slow-downs, stale forecasts |
| input drift (PSI) | 1-7 days | demand distribution changes, new assortment |
| prediction drift | same night | model behaving differently (bad features, bad model version) |
| bias | 7 days | systematic over/under-forecasting (concept drift, promotions, competitor) |
| accuracy by horizon | 1-28 days | everything, but late |

Phase 22 injects a real regime change and shows which signal fires first.""")
    r.metrics = {"alerts": alerts.groupby("kind").size().to_dict() if len(alerts) else {}}
    r.save()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    db = ForecastDB()
    ds = load_dataset()
    if db.runs().empty:
        raise SystemExit("no runs recorded - run `make nightly` first")
    batch_report(db, ds)
    monitoring_report(db, ds)


if __name__ == "__main__":
    main()

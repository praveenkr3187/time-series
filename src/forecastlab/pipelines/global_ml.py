"""Phase 11 - Global gradient boosting: the model that won M5.

    python -m forecastlab.pipelines.global_ml

1. The candidate champion: per-store global LightGBM (Tweedie), direct 28-day, all 30,490 series,
   every backtest origin + the holdout.
2. Design experiments on two stores (backtest fold 3 only - the holdout stays untouched):
   loss function, multi-horizon strategy, stock-out masking, XGBoost, per-store vs pooled.
"""
from __future__ import annotations

import logging
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from forecastlab import viz
from forecastlab.data.hierarchy import build_hierarchy
from forecastlab.data.load import data_dir, load_dataset
from forecastlab.data.quality import STOCKOUT
from forecastlab.metrics import WRMSSE, point_metrics
from forecastlab.models import gbm
from forecastlab.report import Report
from forecastlab.store import cached, load_forecast
from forecastlab.validation import backtest_origins, holdout_origin

log = logging.getLogger(__name__)
CHAMPION = "LightGBM (per-store, direct)"
EXP_STORES = ["CA_1", "TX_2"]


def champion_forecasts(ds, origins) -> dict[int, np.ndarray]:
    out = {}
    for o in origins:
        t0 = time.time()
        out[o.day] = cached(CHAMPION, o.day, lambda o=o: gbm.forecast_all(ds, o.day, o.horizon, "direct"))
        log.info("champion %s ready (%.0fs)", o.name, time.time() - t0)
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ds = load_dataset()
    r = Report("11_global_ml", "Phase 11 - Global Gradient Boosting")
    origins = backtest_origins(ds) + [holdout_origin(ds)]
    champ = champion_forecasts(ds, origins)

    # ---------------------------------------------------------------- 1. champion candidate vs baselines
    rows = []
    for o in origins:
        w = WRMSSE(ds, o.day, o.horizon)
        for name in ["Seasonal naive (7)", "Weekday profile x MA(28)", CHAMPION]:
            F = champ[o.day] if name == CHAMPION else load_forecast(name, o.day)
            pm = point_metrics(ds.Y[:, o.test], F)
            rows.append({"model": name, "origin": o.name, "WRMSSE": w.score(F), "WAPE": pm["wape"], "bias": pm["bias"]})
    df = pd.DataFrame(rows)
    t = df.pivot(index="model", columns="origin", values="WRMSSE")
    folds = [o.name for o in origins[:-1]]
    t["backtest mean"] = t[folds].mean(axis=1)
    t = t[[*folds, "backtest mean", origins[-1].name]].sort_values("backtest mean")
    r.h2("Per-store global LightGBM vs the best baselines (WRMSSE, all 30,490 series)")
    r.table(t, floatfmt=".3f")
    ho = origins[-1].name
    gain = 1 - t.loc[CHAMPION, "backtest mean"] / t.loc["Weekday profile x MA(28)", "backtest mean"]
    r.p(f"""One LightGBM per store (10 models), each learning from all ~3,049 items of its store, with the Phase 10
features and a **Tweedie** loss; direct strategy (every feature ends 28 days before the target), trained on
2014-01-01 .. origin with no early stopping (no test data is peeked at).

* Backtest WRMSSE improves **{gain:.0%}** over the best baseline, and every fold improves.
* Holdout (M5 evaluation window): **{t.loc[CHAMPION, ho]:.3f}**. For context, the M5 winning entry scored 0.520 on this
  window with a large ensemble; seasonal naive scores 0.847.""")
    wl = WRMSSE(ds, origins[-1].day)
    lvl = pd.DataFrame({n: wl.by_level(champ[origins[-1].day] if n == CHAMPION else load_forecast(n, origins[-1].day))
                        for n in ["Weekday profile x MA(28)", CHAMPION]})
    r.table(lvl, floatfmt=".3f")
    r.p("""Holdout WRMSSE by level. The gain is largest at the aggregate levels: the global model gets the level and the
weekday/event/SNAP shape right *jointly* across items, so item errors cancel instead of adding up.""")
    bias = df.pivot(index="model", columns="origin", values="bias")[[*folds, ho]]
    r.p("Bias (sum forecast / sum actual - 1):")
    r.table(bias, floatfmt="+.3f")
    r.metrics["champion"] = t.to_dict(orient="index")

    # ---------------------------------------------------------------- 2. experiments
    o = origins[-2]  # backtest fold 3
    sr = gbm.store_rows(ds)
    rows2 = np.concatenate([sr[s] for s in EXP_STORES])
    meta = ds.series.iloc[rows2].reset_index(drop=True)
    hier = build_hierarchy(meta, [("store", ["store_id"]), ("store x dept", ["store_id", "dept_id"]),
                                  ("item x store", ["item_id", "store_id"])])
    A = ds.Y[rows2, o.test]
    hist = ds.Y[rows2, : o.day + 1]
    labels = np.load(data_dir() / "zero_run_labels.npy") if (data_dir() / "zero_run_labels.npy").exists() else None

    def per_store(**kw):
        return np.concatenate([gbm.forecast_store(ds, sr[s], o.day, o.horizon, **kw) for s in EXP_STORES])

    variants = {
        "LightGBM Tweedie, direct (champion setup)": lambda: champ[o.day][rows2],
        "LightGBM Poisson": lambda: per_store(objective="poisson"),
        "LightGBM L2 (squared error)": lambda: per_store(objective="regression"),
        "Bucketed direct (7/14/21/28)": lambda: per_store(strategy="bucketed"),
        "Recursive (lag-1, fed back 28x)": lambda: per_store(strategy="recursive"),
        "Stock-out days removed from training": lambda: np.concatenate([
            gbm.forecast_store(ds, sr[s], o.day, o.horizon, exclude=labels[sr[s]] == STOCKOUT) for s in EXP_STORES]),
        "Stock-out + not-ranged days removed": lambda: np.concatenate([
            gbm.forecast_store(ds, sr[s], o.day, o.horizon, exclude=labels[sr[s]] > 0) for s in EXP_STORES]),
        "XGBoost Tweedie, direct": lambda: per_store(learner="xgb"),
        "One model pooled over both stores": lambda: gbm.forecast_store(ds, rows2, o.day, o.horizon),
    }
    res = {}
    for name, fn in variants.items():
        t0 = time.time()
        full = cached(f"exp11 {name}", o.day, lambda fn=fn: _embed(ds, rows2, fn()))
        secs = time.time() - t0
        F = full[rows2]
        pm = point_metrics(A, F, hist)
        agg = {}
        for lv, nm in [(1, "store"), (2, "store x dept")]:
            Aa, Fa, Ha = (hier.aggregate(X, lv) for X in (A, F, hist))
            agg[f"{nm} RMSSE"] = point_metrics(Aa, Fa, Ha)["rmsse"]
        res[name] = {"item RMSSE": pm["rmsse"], **agg, "WAPE": pm["wape"], "bias": pm["bias"]}
        log.info("%s done (%.0fs)", name, secs)
    ex = pd.DataFrame(res).T
    r.h2(f"Design experiments (stores {', '.join(EXP_STORES)}, backtest fold 3)")
    r.table(ex, floatfmt=".3f")
    base = ex.iloc[0]
    better = ex.index[(ex["store x dept RMSSE"] < base["store x dept RMSSE"]) & (ex.index != ex.index[0])].tolist()
    r.p(f"""Test window {ds.dates[o.test.start].date()} .. {ds.dates[o.test.stop - 1].date()}, {len(rows2):,} series.
Variants that beat the champion setup at store x dept level: {', '.join(better) if better else 'none'}.

* **Loss**: Tweedie and Poisson model a non-negative, zero-heavy count whose variance grows with its mean; squared
  error treats a miss of 3 units on a slow mover like one on a best-seller and its predictions go negative (clipped).
* **Strategy**: direct models never see their own errors; recursive models compound them (and are 28x slower to
  predict). Bucketed models use fresher lags for the first weeks at the cost of 4x training.
* **Stock-outs**: removing censored days stops the model learning "this item sells zero" from empty shelves; the
  effect on accuracy *measured against sales* is small because the test window contains stock-outs too - the benefit
  is in inventory decisions (Phase 17), where under-forecasting causes the next stock-out.
* **XGBoost vs LightGBM**: same loss and features; differences are tuning-level. LightGBM is kept for speed and
  native categorical handling.
* **Pooled vs per-store**: one model on more data vs specialised models; the M5 winners used per-store (or
  per-store-category) models for exactly this memory/accuracy trade-off.

Single fold, two stores: differences of ~0.01 are within noise (Phase 4). Anything adopted must also win the
3-fold backtest (Phase 14).""")
    r.metrics["experiments"] = ex.to_dict(orient="index")

    fig, ax = plt.subplots(figsize=(8, 4.2))
    viz.bar(ax, list(ex.index), ex["store x dept RMSSE"].to_numpy(), fmt="{:.3f}", horizontal=True)
    ax.set(title="Experiments: store x dept RMSSE (fold 3)", xlabel="RMSSE")
    ax.set_xlim(left=min(ex["store x dept RMSSE"]) * 0.9)
    r.figure(viz.save(fig, r.dir / "experiments.png"), "Experiments")
    r.save()
    print(t.round(3).to_string())
    print(ex.round(3).to_string())


def _embed(ds, rows, F) -> np.ndarray:
    full = np.full((len(ds.series), F.shape[1]), np.nan, dtype=np.float32)
    full[rows] = F
    return full


if __name__ == "__main__":
    main()

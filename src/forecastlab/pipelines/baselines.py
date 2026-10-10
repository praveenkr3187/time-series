"""Phase 5 - Baselines and forecast metrics.

    python -m forecastlab.pipelines.baselines

The first rungs of the ladder, on all 30,490 series, every backtest origin and the holdout. Also: what each
accuracy metric rewards, and why a metric can make a useless forecast look good.
"""
from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from forecastlab import viz
from forecastlab.data.load import load_dataset
from forecastlab.metrics import WRMSSE, point_metrics
from forecastlab.models.baselines import LADDER
from forecastlab.report import Report
from forecastlab.store import save_forecast
from forecastlab.validation import backtest_origins, holdout_origin, origin_dates

log = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ds = load_dataset()
    r = Report("05_baselines", "Phase 5 - Baselines and Metrics")
    origins = backtest_origins(ds) + [holdout_origin(ds)]
    r.p("Evaluation windows used from here on (all 30,490 series):")
    r.table(origin_dates(ds, origins).set_index("origin"))

    rows, preds = [], {}
    for o in origins:
        w = WRMSSE(ds, o.day, o.horizon)
        H = ds.Y[:, : o.day + 1]
        for name, fn in LADDER.items():
            F = fn(H, o.horizon) * ds.on_shelf[:, o.test]
            save_forecast(name, o.day, F)
            rows.append({"model": name, "origin": o.name, "WRMSSE": w.score(F)})
            preds[(name, o.name)] = F
        log.info("%s done", o.name)
    res = pd.DataFrame(rows).pivot(index="model", columns="origin", values="WRMSSE")
    folds = [o.name for o in origins[:-1]]
    res["backtest mean"] = res[folds].mean(axis=1)
    res = res[[*folds, "backtest mean", origins[-1].name]].sort_values("backtest mean", ascending=False)

    r.h2("The baseline ladder (WRMSSE, lower is better)")
    r.table(res, floatfmt=".3f")
    best = res.index[-1]
    r.p(f"""Our seasonal naive and naive reproduce the official M5 benchmarks on the holdout exactly
(**{res.loc['Seasonal naive (7)', origins[-1].name]:.3f}** and **{res.loc['Naive', origins[-1].name]:.3f}**; published: 0.847
and 1.752), which validates the WRMSSE implementation.

* **Naive** (tomorrow = today) is terrible here: it copies one random day of a weekly cycle for 28 days.
* **Seasonal naive** copies last week - simple and strong because weekly seasonality dominates (Phase 3).
* Flat forecasts (moving averages, SES, Croston family) lose the weekly shape; at item level they are fine,
  at aggregate levels the missing weekday pattern costs a lot (see by-level table).
* **{best}** - a level from the last 28 days times a weekday shape from the last 26 weeks - is the best
  baseline. Every model from here on has to beat **{res.loc[best, 'backtest mean']:.3f}** backtest WRMSSE to earn its
  complexity.""")

    lvl = pd.DataFrame({name: WRMSSE(ds, origins[-1].day).by_level(preds[(name, origins[-1].name)])
                        for name in ["Seasonal naive (7)", "TSB", best]})
    r.h2("Where the error comes from (holdout, WRMSSE per hierarchy level)")
    r.table(lvl, floatfmt=".3f")
    r.p("""Item-level series (levels 10-12) are noisy and score ~1 for everything - roughly as good as a naive
forecast's in-sample error. Aggregates (levels 1-9) separate the models: the weekly shape matters there.
WRMSSE weights the 12 levels equally, so a forecast must be good **both** for an item on a shelf and for a store's
total - the business needs both.""")

    # ---- metric zoo
    r.h2("What does each metric reward?")
    ho = origins[-1]
    A = ds.Y[:, ho.test]
    hist = ds.Y[:, : ho.day + 1]
    zoo = {name: preds[(name, ho.name)] for name in ["Seasonal naive (7)", "Moving average (28)", best]}
    zoo["Always zero"] = np.zeros_like(A)
    under = f"{best} x 0.7 (under-forecast)"
    zoo[under] = preds[(best, ho.name)] * 0.7
    mt = pd.DataFrame({k: point_metrics(A, F, hist) for k, F in zoo.items()}).T
    mt["WRMSSE"] = [WRMSSE(ds, ho.day).score(F) for F in zoo.values()]
    mt = mt[["mae", "rmse", "wape", "bias", "mape", "smape", "mase", "rmsse", "WRMSSE"]]
    r.table(mt, floatfmt=".3f")
    r.p(f"""Holdout, all 30,490 item-store series ({(A == 0).mean():.0%} of actual cells are zero; MAPE ignores those cells).

* **The 30% under-forecast wins on MAE, WAPE and MASE** ({mt.loc[under, 'mae']:.3f} / {mt.loc[under, 'wape']:.3f} /
  {mt.loc[under, 'mase']:.3f} vs {mt.loc[best, 'mae']:.3f} / {mt.loc[best, 'wape']:.3f} / {mt.loc[best, 'mase']:.3f} for the
  unbiased forecast). Absolute-error metrics are minimised by the *median*, and for intermittent demand the median
  sits below the mean - they quietly reward under-forecasting, i.e. empty shelves.
* **"Always zero"** - a forecast that stocks nothing - beats seasonal naive on MASE ({mt.loc['Always zero', 'mase']:.3f} vs
  {mt.loc['Seasonal naive (7)', 'mase']:.3f}) and RMSSE, and beats the moving average on sMAPE.
* **MAPE** cannot separate the unbiased and the under-forecast ({mt.loc[best, 'mape']:.3f} vs {mt.loc[under, 'mape']:.3f}), and
  it silently dropped {(A == 0).mean():.0%} of cells. Never use MAPE on SKU-level retail data.
* **Bias** exposes the under-forecast immediately ({mt.loc[under, 'bias']:+.0%}), and **WRMSSE** punishes it hard
  ({mt.loc[under, 'WRMSSE']:.3f} vs {mt.loc[best, 'WRMSSE']:.3f}) because aggregation turns a consistent bias into large
  errors at store and total level. Every report from here on shows bias next to accuracy.
* **RMSE / RMSSE / WRMSSE** are minimised by the *mean*, the quantity inventory planning needs for expected demand;
  that is why M5 (and this project) select models on WRMSSE, and report WAPE + bias for the business.

Worked example from the brief (actual 100, forecast 90): error 10, MAE 10, MSE 100, RMSE 10, MAPE 10%. With actual
0 and forecast 1, MAPE is undefined (division by zero) while MAE is 1 - the whole story of this table.""")

    fig, ax = plt.subplots(figsize=(8, 4))
    order = res.index[::-1]
    viz.bar(ax, list(order), res.loc[order, "backtest mean"].to_numpy(), fmt="{:.3f}", horizontal=True)
    ax.set(title="Baselines: mean backtest WRMSSE (3 folds)", xlabel="WRMSSE")
    r.figure(viz.save(fig, r.dir / "ladder.png"), "Baseline ladder")
    r.metrics = {"wrmsse": res.to_dict(orient="index"), "metric_zoo": mt.to_dict(orient="index")}
    r.save()
    print(res.round(3).to_string())


if __name__ == "__main__":
    main()

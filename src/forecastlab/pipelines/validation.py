"""Phase 4 - Validation design and leakage: why you must never shuffle a time series.

    python -m forecastlab.pipelines.validation

Same model, same data, five ways to evaluate it. Only the last two are what production will experience.
Then: why one backtest window is not enough (walk-forward).
"""
from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from forecastlab import viz
from forecastlab.data.load import load_dataset
from forecastlab.metrics import WRMSSE
from forecastlab.models import gbm
from forecastlab.models.baselines import moving_average, seasonal_naive, weekday_profile
from forecastlab.models.features import FeatureBuilder
from forecastlab.report import Report
from forecastlab.validation import walk_forward

log = logging.getLogger(__name__)
STORE = "CA_3"
FAST = dict(n_estimators=300, learning_rate=0.1)


def wape(a, f) -> float:
    return float(np.abs(np.asarray(f) - a).sum() / a.sum())


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ds = load_dataset()
    r = Report("04_validation", "Phase 4 - Validation Design and Leakage")
    rows = gbm.store_rows(ds)[STORE]
    fb = FeatureBuilder(ds, rows)
    start, origin, h = ds.day("2015-01-01"), ds.day("2016-03-27"), 28
    test_days = np.arange(origin + 1, origin + 1 + h)
    A = fb.Y[:, test_days]
    res = {}

    r.p("""**Question: if we randomly shuffle the rows before training a forecasting model, why is that a problem?**

Because a forecast is always made *at a point in time* about *later* days. A random split puts days from the future
in the training set and days from the past in the test set, so:
1. the model is trained on information it will never have when it forecasts (tomorrow's sales, next month's level);
2. with lag features, a test row's target literally appears as a *feature* of the neighbouring training rows
   (`lag_1` of Tuesday = Monday's sales), so the model can interpolate instead of forecast;
3. the score measures interpolation inside the known period, not extrapolation into the unknown one - so it is
   optimistic, and it ranks models wrongly (complex models interpolate best).

Below, the same LightGBM on store """ + STORE + """ (3,049 items, trained 2015-01-01 onwards) is evaluated five ways.""")

    # 1) random split, lag-1 features
    days = np.arange(start, origin + 1 + h)
    X1, y1 = fb.frame(days, 1), fb.target(days)
    shelf = fb.on_shelf(days)
    rng = np.random.default_rng(0)
    test_mask = (rng.random(len(y1)) < 0.2) & shelf
    m = gbm.fit(X1[~test_mask & shelf], y1[~test_mask & shelf], **FAST)
    res["1. random 80/20 split, lag-1 features"] = wape(y1[test_mask], np.clip(m.predict(X1[test_mask]), 0, None))
    log.info("random split done")

    # 2) temporal split, lag-1 features, evaluated one step ahead with true lags
    m1 = gbm.fit(*gbm.training_data(fb, origin, 1, start), **FAST)
    F2 = gbm.predict_days(m1, fb, test_days, 1)
    res["2. time split, lag-1 features, one-step-ahead (true lags every day)"] = wape(A, F2)

    # 3) leaky feature: centred 7-day mean (uses the target's own future)
    Xl, yl = fb.frame(days, 1), fb.target(days)
    Yp = np.pad(fb.Y, ((0, 0), (3, 3)))
    centred = np.stack([Yp[:, d: d + 7].mean(axis=1) for d in days], axis=1).ravel()
    Xl["centred_mean_7"] = centred.astype(np.float32)
    past = np.tile(days <= origin, len(rows))  # rows are series-major: row = series * len(days) + day
    tr, te = past & shelf, ~past
    ml = gbm.fit(Xl[tr], yl[tr], **FAST)
    res["3. time split + leaky centred moving average"] = wape(yl[te], np.clip(ml.predict(Xl[te]), 0, None) * shelf[te])
    log.info("leaky done")

    # 4) honest recursive and 5) honest direct
    F4 = gbm.forecast_store(ds, rows, origin, h, "recursive", start=start, **FAST)
    res["4. time split, recursive 28-day forecast (honest)"] = wape(A, F4)
    F5 = gbm.forecast_store(ds, rows, origin, h, "direct", start=start, **FAST)
    res["5. time split, direct 28-day forecast (honest)"] = wape(A, F5)
    log.info("honest done")

    t = pd.Series(res, name="WAPE").to_frame()
    t["vs honest direct"] = t["WAPE"] / res["5. time split, direct 28-day forecast (honest)"] - 1
    r.h2("Five evaluations of the same model")
    r.table(pd.DataFrame({"WAPE": t["WAPE"].map("{:.1%}".format),
                          "vs honest direct": t["vs honest direct"].map("{:+.1%}".format)}))
    r.p(f"""Test window {ds.dates[test_days[0]].date()} .. {ds.dates[test_days[-1]].date()}. WAPE = sum|error| / sum(actual).

* **(1) Random split** looks {-t.iloc[0, 1]:.0%} better than reality. It is the number a notebook produces when
  `train_test_split(shuffle=True)` is applied to a sales table.
* **(2) One-step-ahead** evaluation is honest about time but not about the *use*: the planner needs day 28 today,
  when days 1-27 are still unknown. Reporting one-step accuracy for a 28-day plan is a common, subtle leak.
* **(3) A centred moving average** uses days t+1..t+3 - a feature that does not exist at forecast time. A
  backtest built on the full table happily computes it; production cannot.
* **(4) and (5)** are what the business will experience. Their gap to (1) is the size of the illusion.""")

    # error by horizon
    r.h2("Error grows with the horizon")
    eh = pd.DataFrame({"one-step (true lags)": [wape(A[:, k], F2[:, k]) for k in range(h)],
                       "recursive": [wape(A[:, k], F4[:, k]) for k in range(h)],
                       "direct": [wape(A[:, k], F5[:, k]) for k in range(h)]}, index=np.arange(1, h + 1))
    fig, ax = plt.subplots(figsize=(8, 3.4))
    for i, c in enumerate(eh.columns):
        ax.plot(eh.index, eh[c].rolling(7, min_periods=1).mean(), color=viz.SERIES[i], label=c)
    ax.set(title=f"WAPE by days ahead ({STORE}, 7-day rolling)", xlabel="days ahead", ylabel="WAPE")
    ax.legend()
    r.figure(viz.save(fig, r.dir / "error_by_horizon.png"), "Error by horizon")
    r.p(f"""Week 1 vs week 4 WAPE: recursive {eh['recursive'].iloc[:7].mean():.1%} -> {eh['recursive'].iloc[21:].mean():.1%},
direct {eh['direct'].iloc[:7].mean():.1%} -> {eh['direct'].iloc[21:].mean():.1%}. Accuracy must always be reported
**per horizon** - "the model is 25% WAPE" means nothing without "at how many days ahead".""")

    # walk-forward variability
    r.h2("One backtest window is not enough: walk-forward")
    origins = walk_forward(ds.day("2015-06-28"), 12, 28, h)
    models = {"Seasonal naive": seasonal_naive, "Moving average (28)": moving_average,
              "Weekday profile x MA(28)": weekday_profile}
    rows_ = []
    for o in origins:
        w = WRMSSE(ds, o.day, h)
        Hh = ds.Y[:, : o.day + 1]
        for name, fn in models.items():
            rows_.append({"origin": ds.dates[o.day].date(), "model": name, "WRMSSE": w.score(fn(Hh, h))})
    wf = pd.DataFrame(rows_).pivot(index="origin", columns="model", values="WRMSSE")
    r.table(wf, floatfmt=".3f")
    best = wf.idxmin(axis=1).value_counts()
    wp_wins = int((wf["Weekday profile x MA(28)"] < wf["Seasonal naive"]).sum())
    spread = wf["Weekday profile x MA(28)"]
    worst = wf.mean(axis=1).idxmax()
    r.p(f"""Walk-forward (expanding window): train on everything up to the origin, forecast 28 days, move the origin by
28 days, repeat.

* The **same model** scores between {spread.min():.3f} and {spread.max():.3f} depending on the window - the window
  matters as much as the model. The worst window for every model starts {worst} (Christmas closures fall inside it).
* The weekday-profile model beats seasonal naive in {wp_wins} of {len(origins)} windows. Had we picked a single
  window we might have chosen the wrong model ({len(origins) - wp_wins} window(s) say the opposite), and for close
  pairs (Phase 14) one window is a coin flip.

Model selection therefore uses **3 walk-forward origins** (Phase 14) and reports the spread; the final **holdout**
(M5's evaluation window) is touched once.

```
fold 1   train ███████████████████████░░░░ test
fold 2   train ███████████████████████████░░░░ test
fold 3   train ███████████████████████████████░░░░ test
holdout  train ███████████████████████████████████████░░░░ (once, at the end)
```""")
    fig, ax = plt.subplots(figsize=(9, 3.4))
    for i, c in enumerate(wf.columns):
        ax.plot(pd.to_datetime(wf.index), wf[c], marker="o", color=viz.SERIES[i], label=c)
    ax.set(title="WRMSSE per walk-forward origin", ylabel="WRMSSE")
    ax.legend()
    r.figure(viz.save(fig, r.dir / "walk_forward.png"), "Walk-forward")
    r.metrics = {"evaluations": res, "walk_forward_best_counts": best.to_dict(),
                 "wape_by_horizon": eh.to_dict(orient="list")}
    r.save()
    print(t)


if __name__ == "__main__":
    main()

"""Phases 6-9 - Classical statistical models.

    python -m forecastlab.pipelines.classical ets            # Phase 6: SES, Holt, Holt-Winters, AutoETS
    python -m forecastlab.pipelines.classical arima          # Phase 7: ARIMA, SARIMA, ARIMAX
    python -m forecastlab.pipelines.classical prophet        # Phase 8: Prophet
    python -m forecastlab.pipelines.classical intermittent   # Phase 9: Croston family, ADIDA, IMAPA

Two arenas, the same for every family:
* **item level**: a stratified sample of item-store series (one model per series, last year of history);
* **aggregate level**: store x department (70), store (10) and total (1) series, forecast directly.
Every number is compared with the Phase 5 baselines on exactly the same series and windows.
"""
from __future__ import annotations

import argparse
import logging
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from forecastlab import viz
from forecastlab.config import load_config
from forecastlab.data.hierarchy import m5_hierarchy
from forecastlab.data.load import Dataset, load_dataset
from forecastlab.metrics import point_metrics
from forecastlab.models import classical
from forecastlab.models.baselines import seasonal_naive, syntetos_boylan_class, weekday_profile
from forecastlab.report import Report
from forecastlab.store import cached, load_forecast
from forecastlab.validation import Origin, backtest_origins, holdout_origin, sample_rows

log = logging.getLogger(__name__)
AGG_LEVELS = {9: "store x dept", 3: "store", 1: "total"}
AGG_HISTORY = 730
FAMILIES = {
    "ets": ("06_exponential_smoothing", "Phase 6 - Exponential Smoothing",
            ["SES", "Holt", "Holt-Winters", "AutoETS"], True),
    "arima": ("07_arima", "Phase 7 - ARIMA, SARIMA, ARIMAX",
              ["ARIMA (non-seasonal)", "SARIMA (AutoARIMA, s=7)"], True),
    "prophet": ("08_prophet", "Phase 8 - Prophet", [], False),
    "intermittent": ("09_intermittent", "Phase 9 - Intermittent Demand Methods",
                     ["Croston (classic)", "Croston (optimised)", "SBA", "TSB", "ADIDA", "IMAPA"], False),
}
BASELINES = ["Seasonal naive (7)", "Weekday profile x MA(28)"]


# --------------------------------------------------------------------------- shared helpers
def item_forecasts(ds: Dataset, o: Origin, rows: np.ndarray, names: list[str]) -> dict[str, np.ndarray]:
    """Per-series models on the sample, cached as full [n_series, h] matrices (NaN outside the sample)."""
    todo = [n for n in names if load_forecast(n, o.day) is None]
    if todo:
        L = load_config()["sample"]["history_days"]
        s0 = o.day - L + 1
        Y = classical.fill_closures(ds.Y[rows, s0: o.day + 1], ds.dates[s0: o.day + 1])
        t0 = time.time()
        out = classical.forecast(Y, ds.dates[s0: o.day + 1], o.horizon, todo)
        log.info("  %s: %s in %.0fs", o.name, todo, time.time() - t0)
        for n, F in out.items():
            full = np.full((len(ds.series), o.horizon), np.nan, dtype=np.float32)
            full[rows] = F * ds.on_shelf[rows, o.test]
            cached(n, o.day, lambda full=full: full)
    return {n: load_forecast(n, o.day)[rows] for n in names}


def agg_history(ds: Dataset, level: int, o: Origin) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    hier = m5_hierarchy()
    s0 = o.day - AGG_HISTORY + 1
    Yfull = hier.aggregate(ds.Y[:, : o.test.stop], level)
    dates = ds.dates[s0: o.day + 1]
    return classical.fill_closures(Yfull[:, s0: o.day + 1], dates), Yfull, dates


def exog(ds: Dataset, level: int) -> np.ndarray:
    """Known-future regressors per aggregate series and day: SNAP share, event day, relative price index."""
    hier = m5_hierarchy()
    rows = hier.level_rows(level)
    S = hier.S[rows]
    n_mem = np.asarray(S.sum(axis=1))
    states = ds.series["state_id"].to_numpy()
    snap = np.stack([ds.cal[f"snap_{s}"].to_numpy() for s in ["CA", "TX", "WI"]])[
        pd.Categorical(states, categories=["CA", "TX", "WI"]).codes].astype(np.float32)
    snap_agg = np.asarray(S @ snap) / n_mem
    event = np.broadcast_to((ds.cal["event_name_1"] != "").to_numpy(np.float32), snap_agg.shape)
    P = pd.DataFrame(ds.P.T).ffill()
    rel = (P / P.rolling(365, min_periods=1).mean()).to_numpy().T
    on = ~np.isnan(rel)
    price_idx = np.asarray(S @ np.nan_to_num(rel)) / np.maximum(np.asarray(S @ on.astype(np.float32)), 1)
    return np.stack([snap_agg, event, price_idx], axis=2)


def agg_forecast(ds: Dataset, level: int, o: Origin, name: str, fn) -> np.ndarray:
    return cached(f"{name} @ level{level}", o.day, fn)


def agg_baselines(ds: Dataset, level: int, o: Origin) -> dict[str, np.ndarray]:
    Yh, _, _ = agg_history(ds, level, o)
    hier = m5_hierarchy()
    bu = load_forecast("Weekday profile x MA(28)", o.day)
    return {"Seasonal naive (7)": seasonal_naive(Yh, o.horizon),
            "Weekday profile x MA(28)": weekday_profile(Yh, o.horizon),
            "Weekday profile, bottom-up from items": hier.aggregate(bu, level)}


def score(A: np.ndarray, F: np.ndarray, hist: np.ndarray) -> dict[str, float]:
    m = point_metrics(A, F, hist)
    return {"wape": m["wape"], "bias": m["bias"], "rmsse": m["rmsse"]}


def summarise(rows: list[dict], origins: list[Origin], metric: str) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    t = df.pivot_table(index="model", columns="origin", values=metric)
    folds = [o.name for o in origins[:-1]]
    t["backtest mean"] = t[folds].mean(axis=1)
    return t[[*folds, "backtest mean", origins[-1].name]].sort_values("backtest mean")


# --------------------------------------------------------------------------- per-family extras
def arima_orders(ds: Dataset, o: Origin) -> pd.DataFrame:
    from statsforecast.models import AutoARIMA
    hier = m5_hierarchy()
    out = []
    for level, keys in [(1, ["total"]), (3, ["CA_3", "TX_2", "WI_1"]),
                        (9, ["TX_2/FOODS_3", "CA_1/HOBBIES_1", "WI_2/HOUSEHOLD_1"])]:
        Yh, _, _ = agg_history(ds, level, o)
        k = hier.keys.loc[hier.level_rows(level), "key"].to_numpy()
        for key in keys:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m = AutoARIMA(season_length=7).fit(Yh[list(k).index(key)].astype(np.float64))
            p, q, P, Q, s, d, D = m.model_["arma"]
            out.append({"series": key, "order (p,d,q)(P,D,Q)s": f"({p},{d},{q})({P},{D},{Q})_{s}",
                        "AICc": m.model_.get("aicc", np.nan)})
    return pd.DataFrame(out).set_index("series")


def prophet_holidays(ds: Dataset) -> pd.DataFrame:
    c = ds.cal[ds.cal["event_name_1"] != ""]
    return pd.DataFrame({"holiday": c["event_name_1"].to_numpy(), "ds": c["date"].to_numpy(),
                         "lower_window": 0, "upper_window": 1})


# --------------------------------------------------------------------------- main
def run(family: str) -> None:
    stage, title, item_models, agg_too = FAMILIES[family]
    ds = load_dataset()
    hier = m5_hierarchy()
    origins = backtest_origins(ds) + [holdout_origin(ds)]
    rows = sample_rows(ds)
    r = Report(stage, title)
    item_rows, agg_rows = [], []
    agg_models: dict[str, callable] = {}
    if family in ("ets", "arima"):
        for n in item_models:
            agg_models[n] = n
    if family == "arima":
        agg_models["ARIMAX (SARIMA + calendar/price)"] = "ARIMAX (SARIMA + calendar/price)"
    if family == "prophet":
        agg_models["Prophet"] = "prophet"
        agg_models["Prophet + SNAP/price regressors"] = "prophet_x"

    for o in origins:
        A = ds.Y[rows, o.test]
        hist = ds.Y[rows, : o.day + 1]
        fc = item_forecasts(ds, o, rows, item_models) if item_models else {}
        for b in BASELINES:
            fc[b] = load_forecast(b, o.day)[rows]
        for n, F in fc.items():
            item_rows.append({"model": n, "origin": o.name, **score(A, F, hist)})
        if agg_models:
            for level in AGG_LEVELS:
                Yh, Yfull, dates = agg_history(ds, level, o)
                Aa, ha = Yfull[:, o.test], Yfull[:, : o.day + 1]
                fa = agg_baselines(ds, level, o)
                for n, kind in agg_models.items():
                    if kind == "prophet" or kind == "prophet_x":
                        def fn(kind=kind):
                            reg = regf = None
                            if kind == "prophet_x":
                                ex = exog(ds, level)
                                reg = ex[:, o.day - AGG_HISTORY + 1: o.day + 1][:, :, [0, 2]]
                                regf = ex[:, o.test][:, :, [0, 2]]
                            return classical.prophet_forecast(Yh, dates, o.horizon, prophet_holidays(ds), reg, regf)
                    elif n.startswith("ARIMAX"):
                        def fn(n=n):
                            ex = exog(ds, level)
                            return classical.forecast(Yh, dates, o.horizon, [n],
                                                      ex[:, o.day - AGG_HISTORY + 1: o.day + 1],
                                                      ex[:, o.test])[n]
                    else:
                        def fn(n=n):
                            return classical.forecast(Yh, dates, o.horizon, [n])[n]
                    t0 = time.time()
                    fa[n] = agg_forecast(ds, level, o, n, fn)
                    log.info("  %s level %d %s %.0fs", o.name, level, n, time.time() - t0)
                for n, F in fa.items():
                    agg_rows.append({"model": n, "origin": o.name, "level": AGG_LEVELS[level], **score(Aa, F, ha)})
        log.info("%s done", o.name)


    r.h2(f"Item level: {len(rows)} sampled item-store series")
    if item_models:
        r.p(f"""Each model is fitted separately on every series, using the last
{load_config()['sample']['history_days']} days before each origin (Christmas closures interpolated). RMSSE = per-series
RMSE scaled by that series' naive in-sample error, averaged; WAPE and bias are volume-weighted.""")
    t_rmsse = summarise(item_rows, origins, "rmsse")
    t_wape = summarise(item_rows, origins, "wape")
    t_bias = summarise(item_rows, origins, "bias")
    comb = pd.DataFrame({"RMSSE (backtest)": t_rmsse["backtest mean"], "RMSSE (holdout)": t_rmsse.iloc[:, -1],
                         "WAPE (backtest)": t_wape["backtest mean"], "bias (backtest)": t_bias["backtest mean"]})
    r.table(comb.sort_values("RMSSE (backtest)"), floatfmt=".3f")
    r.metrics["item_level"] = comb.to_dict(orient="index")

    if family == "intermittent":
        cls = syntetos_boylan_class(ds.Y[rows, : origins[-1].day + 1][:, -365:])
        seg = []
        for o in origins:
            A = ds.Y[rows, o.test]
            hist = ds.Y[rows, : o.day + 1]
            for n in item_models + BASELINES + ["SES (alpha per series)"]:
                F = load_forecast(n, o.day)[rows]
                for c in np.unique(cls):
                    m = cls == c
                    seg.append({"model": n, "class": c, "origin": o.name, "rmsse": score(A[m], F[m], hist[m])["rmsse"]})
        seg = pd.DataFrame(seg).groupby(["model", "class"])["rmsse"].mean().unstack()
        counts = pd.Series(cls).value_counts()
        r.h2("By demand pattern (mean RMSSE over all 4 windows)")
        r.table(seg, floatfmt=".3f")
        col = seg["intermittent"]
        ties = col[col <= col.min() + 0.002].index.tolist()
        r.p(f"""Syntetos-Boylan class from the last year before the holdout ({", ".join(f"{k}: {v}" for k, v in counts.items())}
series). On **intermittent** series the best RMSSE ({col.min():.3f}) is shared by {", ".join(ties)} - within 0.002 of
each other; the weekday-profile baseline is at {col['Weekday profile x MA(28)']:.3f}.

* Croston-type methods separate *how often* an item sells from *how much* it sells when it does, and forecast a flat
  rate ("0.4 units/day"): the right expected value, but no weekday shape. Phase 5 showed what that costs once
  forecasts are added up the hierarchy (Croston WRMSSE vs the weekday profile).
* **Croston's bias is visible**: classic {comb.loc['Croston (classic)', 'bias (backtest)']:+.1%} and optimised
  {comb.loc['Croston (optimised)', 'bias (backtest)']:+.1%} over-forecast. Dividing size by interval is biased upwards;
  **SBA** multiplies by (1 - alpha/2) to correct it ({comb.loc['SBA', 'bias (backtest)']:+.1%}).
* **TSB** updates the *probability* of a sale every day, so it decays towards zero when an item stops selling
  (obsolescence); Croston only updates when a sale happens and keeps forecasting the old rate forever.
* **ADIDA / IMAPA** aggregate the series into buckets (where it is no longer intermittent), forecast, and
  disaggregate - temporal aggregation, the same idea as forecasting weekly for purchasing.

**Lesson:** specialised intermittent methods are a sensible per-series baseline for slow movers, but they do not
model drivers (price, SNAP, events) or weekly shape. Global ML models with a Tweedie/Poisson loss (Phase 11)
handle intermittency *and* drivers in one model.""")
        r.metrics["by_class"] = seg.to_dict()

    if agg_rows:
        r.h2("Aggregate levels (forecast directly at that level)")
        for lvl in AGG_LEVELS.values():
            sub = [x for x in agg_rows if x["level"] == lvl]
            tr, tw, tb = (summarise(sub, origins, k) for k in ["rmsse", "wape", "bias"])
            t = pd.DataFrame({"RMSSE (backtest)": tr["backtest mean"], "RMSSE (holdout)": tr.iloc[:, -1],
                              "WAPE (backtest)": tw["backtest mean"], "bias (backtest)": tb["backtest mean"]})
            r.p(f"**{lvl}** ({len(hier.level_rows([k for k, v in AGG_LEVELS.items() if v == lvl][0]))} series)")
            r.table(t.sort_values("RMSSE (backtest)"), floatfmt=".3f")
            r.metrics[f"agg_{lvl}"] = t.to_dict(orient="index")
        r.p("""*Weekday profile, bottom-up from items* = the item-level baseline forecasts summed up the hierarchy - the
cheapest way to get coherent aggregate forecasts (Phase 16 compares this with proper reconciliation).""")

    if family == "arima":
        r.h2("Which orders does AutoARIMA choose?")
        orders = arima_orders(ds, origins[-1])
        r.table(orders, floatfmt=".1f")
        r.p("""AutoARIMA (Hyndman-Khandakar): unit-root tests choose d and D, then a stepwise search over p, q, P, Q
minimises AICc. Compare with the ACF/PACF reading in Phase 3. ARIMAX adds regression on known-future
covariates (share of members on SNAP, event day, relative price index) with SARIMA errors.""")
        r.metrics["orders"] = orders.reset_index().to_dict(orient="records")

    if family in ("ets", "arima", "prophet"):
        o = origins[-1]
        _, Yfull, _ = agg_history(ds, 3, o)
        k = list(hier.keys.loc[hier.level_rows(3), "key"]).index("TX_2")
        fig, ax = plt.subplots(figsize=(10, 3.6))
        d = ds.dates[o.day - 55: o.test.stop]
        ax.plot(d, Yfull[k, o.day - 55: o.test.stop], color=viz.INK_2, linewidth=1.2, label="actual")
        for i, n in enumerate(agg_models):
            ax.plot(ds.dates[o.test], load_forecast(f"{n} @ level3", o.day)[k], color=viz.SERIES[i], label=n)
        ax.axvline(ds.dates[o.day], color=viz.MUTED, linestyle=":", linewidth=1)
        ax.set(title="Store TX_2: last 8 weeks and the 28-day holdout forecast", ylabel="units")
        ax.legend(fontsize=8, loc="upper left")
        r.figure(viz.save(fig, r.dir / "store_forecast.png"), "Store forecast")

    r.p(LESSONS[family](comb, r.metrics))
    r.save()
    print(comb.round(3).to_string())


def _ets_lesson(comb: pd.DataFrame, m: dict) -> str:
    sd, st, tot = (pd.DataFrame(m[f"agg_{k}"]).T for k in ["store x dept", "store", "total"])
    wp = "Weekday profile x MA(28)"

    def cmp(t: pd.DataFrame, col: str) -> str:
        return f"AutoETS {t.loc['AutoETS', col]:.3f} vs baseline {t.loc[wp, col]:.3f}"
    return f"""**Reading the results**

* **No seasonal component, no chance at aggregate level.** SES and Holt forecast a flat line / straight line; at
  store x dept their backtest RMSSE is {sd.loc['SES', 'RMSSE (backtest)']:.3f} / {sd.loc['Holt', 'RMSSE (backtest)']:.3f}
  vs {sd.loc['Holt-Winters', 'RMSSE (backtest)']:.3f} for Holt-Winters. The weekly cycle is most of the signal (Phase 3).
* **Item level is a dead heat**: every model lands at RMSSE {comb['RMSSE (backtest)'].drop('Seasonal naive (7)').min():.3f} -
  {comb['RMSSE (backtest)'].drop('Seasonal naive (7)').max():.3f}. At 0-3 units a day, noise dominates and the seasonal
  component barely matters; seasonal naive is the outlier ({comb.loc['Seasonal naive (7)', 'RMSSE (backtest)']:.3f})
  because copying one noisy week amplifies noise.
* **AutoETS vs the weekday-profile baseline flips with the window**: backtest store x dept {cmp(sd, 'RMSSE (backtest)')},
  store {cmp(st, 'RMSSE (backtest)')}, total {cmp(tot, 'RMSSE (backtest)')}; on the holdout store x dept
  {cmp(sd, 'RMSSE (holdout)')}. A textbook model is not automatically better than a well-built heuristic - and
  single-window comparisons would have told either story (Phase 4).
* **Bias**: within {comb['bias (backtest)'].abs().max():.1%} at item level for every model; at store x dept the seasonal
  models under-forecast by {-sd.loc['Holt-Winters', 'bias (backtest)']:.1%} (Holt-Winters) and
  {-sd.loc['AutoETS', 'bias (backtest)']:.1%} (AutoETS) in the backtest windows - a period of growing sales.

ETS = weighted averages of the past with exponentially decaying weights, extended with level, trend and season
states (AutoETS picks additive/multiplicative/none forms by AICc). Fast, robust, and still the default in many
supply-chain tools."""


def _arima_lesson(comb: pd.DataFrame, m: dict) -> str:
    agg = pd.DataFrame(m["agg_store x dept"])
    best = agg.loc["RMSSE (backtest)"].idxmin()
    return f"""**Reading the results**

* Non-seasonal ARIMA cannot reproduce the weekly cycle beyond a few days - its forecasts converge to the mean.
  Seasonal differencing + seasonal MA (SARIMA) is what makes ARIMA competitive.
* At store x dept level the best model is **{best}** (backtest RMSSE {agg.loc['RMSSE (backtest)', best]:.3f}).
  Exogenous regressors let ARIMA react to SNAP days and price changes that pure history cannot anticipate.
* Cost: AutoARIMA's order search is the slowest per-series model in this project (~0.6 s per series per origin on
  a year of daily data). For 30,490 series and nightly runs that is hours of compute; this is the scaling argument
  for global models."""


def _prophet_lesson(comb: pd.DataFrame, m: dict) -> str:
    agg = pd.DataFrame(m["agg_store x dept"])
    return f"""**Reading the results**

Prophet is a curve-fitting model: piecewise-linear trend + Fourier weekly/yearly seasonality + holiday effects
(+ optional regressors), fitted per series. Its strengths are interpretability and holiday handling; it is not
designed for intermittent counts, which is why it is evaluated on aggregates only.

* Best Prophet variant at store x dept level: backtest RMSSE
  {min(agg.loc['RMSSE (backtest)', c] for c in agg.columns if c.startswith('Prophet')):.3f} vs weekday-profile baseline
  {agg.loc['RMSSE (backtest)', 'Weekday profile x MA(28)']:.3f}.
* A flexible trend fitted to two years can drift over a 28-day horizon - check its bias column.
* In published comparisons (including M5 analyses) Prophet is rarely the most accurate model; it is popular because it
  is easy to explain and tune, which is a legitimate production criterion."""


def _intermittent_lesson(comb: pd.DataFrame, m: dict) -> str:
    return ""


LESSONS = {"ets": _ets_lesson, "arima": _arima_lesson, "prophet": _prophet_lesson, "intermittent": _intermittent_lesson}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("family", choices=list(FAMILIES))
    run(ap.parse_args().family)


if __name__ == "__main__":
    main()

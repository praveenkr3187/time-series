"""Phase 3 - Decomposition, stationarity, autocorrelation.

    python -m forecastlab.pipelines.decomposition

The vocabulary that classical models (ARIMA/ETS) are built on, applied to our sales:
trend + seasonality + remainder (STL), stationarity tests (ADF / KPSS), differencing, ACF / PACF.
"""
from __future__ import annotations

import logging
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import acf, adfuller, kpss, pacf

from forecastlab import viz
from forecastlab.data.hierarchy import m5_hierarchy
from forecastlab.data.load import load_dataset
from forecastlab.report import Report

log = logging.getLogger(__name__)


def stationarity(x: np.ndarray) -> dict:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        adf_p = adfuller(x, autolag="AIC")[1]
        kpss_p = kpss(x, regression="c", nlags="auto")[1]
    verdict = ("stationary" if adf_p < 0.05 and kpss_p > 0.05 else
               "non-stationary" if adf_p >= 0.05 and kpss_p <= 0.05 else "inconclusive")
    return {"ADF p (H0: unit root)": adf_p, "KPSS p (H0: stationary)": kpss_p, "verdict": verdict}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ds = load_dataset()
    hier = m5_hierarchy()
    r = Report("03_decomposition", "Phase 3 - Decomposition and Stationarity")
    keep = ds.dates >= "2013-01-01"
    dates = ds.dates[keep]
    xmas = (dates.month == 12) & (dates.day == 25)

    def agg(level: int, key: str) -> pd.Series:
        rows = hier.level_rows(level)
        i = rows[hier.keys.loc[rows, "key"].to_numpy() == key][0]
        y = np.asarray(hier.S[i] @ ds.Y).ravel()[keep]
        s = pd.Series(y, index=dates)
        s[xmas] = np.nan  # closed day: interpolate so it is not read as a demand shock
        return s.interpolate()

    total = agg(1, "total")
    r.h2("STL decomposition of total sales")
    stl = STL(total, period=7, seasonal=13, robust=True).fit()
    yearly = STL(stl.trend, period=365, robust=True).fit()
    fig, axes = plt.subplots(4, 1, figsize=(10, 7.5), sharex=True)
    for ax, (name, s) in zip(axes, [("observed", total), ("trend (incl. yearly cycle)", stl.trend),
                                    ("weekly seasonal", stl.seasonal), ("remainder", stl.resid)]):
        ax.plot(s.index, s.values, color=viz.SERIES[0], linewidth=0.8)
        ax.set_title(name, fontsize=10)
    r.figure(viz.save(fig, r.dir / "stl_total.png"), "STL decomposition")
    var = total.var()
    shares = {"trend + yearly": stl.trend.var() / var, "weekly": stl.seasonal.var() / var, "remainder": stl.resid.var() / var}
    r.p(f"""`y_t = T_t + S_t + R_t` (STL, robust, weekly period; Christmas closures interpolated first).

Variance shares of the total series 2013+: trend+yearly **{shares['trend + yearly']:.0%}**, weekly
**{shares['weekly']:.0%}**, remainder **{shares['remainder']:.0%}** (shares need not sum to 1 - components covary).
The "trend" is not smooth: it wiggles with a period of about a month - the **SNAP cycle** (food-stamp days fall in
the first half of each month, Phase 1). A weekly-period STL cannot separate it; a model needs a SNAP covariate.
A second STL on the trend with period 365 separates the yearly cycle (amplitude
{(yearly.seasonal.max() - yearly.seasonal.min()):,.0f} units/day vs weekly amplitude {(stl.seasonal.max() - stl.seasonal.min()):,.0f}).""")

    r.h2("Is it stationary? ADF and KPSS, before and after differencing")
    series = {"total": total, "store CA_3": agg(3, "CA_3"), "store x dept TX_2/FOODS_3": agg(9, "TX_2/FOODS_3"),
              "store x dept WI_1/HOBBIES_2": agg(9, "WI_1/HOBBIES_2")}
    rows = []
    for name, s in series.items():
        x = s.to_numpy()
        for tname, tx in [("level", x), ("first difference", np.diff(x)), ("seasonal difference (7)", x[7:] - x[:-7]),
                          ("log, seasonal difference", np.diff(np.log1p(x), 7))]:
            rows.append({"series": name, "transform": tname, **stationarity(tx)})
    st = pd.DataFrame(rows).set_index(["series", "transform"])
    r.table(st, floatfmt=".3f")
    r.p("""ADF's null is "unit root" (non-stationary); KPSS's null is "stationary". They agree only when the evidence is
clear. Levels of growing aggregates are non-stationary (trend); a seasonal difference removes most of it. This is
what the `d` and `D` in SARIMA(p,d,q)(P,D,Q)_7 do, and why AutoARIMA runs these tests to choose them (Phase 7).

Stationarity is a modelling assumption of ARIMA, **not** a requirement of forecasting in general: tree models and
neural nets instead need the *relationship between features and target* to be stable, which is why they get
level-free features (ratios, recent means) and why Phase 22 studies drift.""")

    r.h2("Autocorrelation: what does the past say about tomorrow?")
    x = series["store x dept TX_2/FOODS_3"].to_numpy()
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
    for ax, (name, v, fn) in zip(axes, [("ACF, level", x, acf), ("ACF, seasonal diff", x[7:] - x[:-7], acf),
                                        ("PACF, seasonal diff", x[7:] - x[:-7], pacf)]):
        vals = fn(v, nlags=35)
        ax.bar(range(len(vals)), vals, color=viz.SERIES[0], width=0.6)
        ax.axhline(1.96 / np.sqrt(len(v)), color=viz.MUTED, linestyle=":", linewidth=1)
        ax.axhline(-1.96 / np.sqrt(len(v)), color=viz.MUTED, linestyle=":", linewidth=1)
        ax.set(title=f"{name}", xlabel="lag (days)", ylim=(-0.6, 1.05))
    r.figure(viz.save(fig, r.dir / "acf.png"), "ACF / PACF")
    a = acf(x, nlags=35)
    r.p(f"""TX_2 / FOODS_3 (store x department). On levels the ACF decays slowly with spikes at 7, 14, 21, 28
(lag-7 autocorrelation {a[7]:.2f}, lag-1 {a[1]:.2f}) - the signature of trend + weekly seasonality.
After seasonal differencing the ACF spikes negative at lag 7 ({acf(x[7:] - x[:-7], nlags=7)[7]:.2f}) and the PACF
decays at lags 7, 14, 21 - the textbook signature of a **seasonal MA(1)** term, with small positive AR at lags 1-2.
That is the SARIMA(p,0,q)(0,1,1)_7 shape Phase 7's AutoARIMA usually lands on.

**Why this matters for features**: lags 7, 14, 21, 28 carry the weekly signal. A model that must forecast 28
days ahead cannot use lag 1 for day 28 - Phase 10 builds lags that respect the forecast horizon.""")

    r.h2("Item level: the textbook picture breaks")
    q = ds.Y[:, -365:]
    rng = np.random.default_rng(0)
    on = ds.on_shelf[:, -365:].all(axis=1) & (q.mean(axis=1) > 0.2)
    sample = rng.choice(np.flatnonzero(on), 300, replace=False)
    verdicts = pd.Series([stationarity(q[i])["verdict"] for i in sample]).value_counts(normalize=True)
    lag7 = np.array([acf(q[i], nlags=7)[7] for i in sample])
    r.p(f"""300 random item-store series (last year): {verdicts.get('stationary', 0):.0%} test stationary,
{verdicts.get('non-stationary', 0):.0%} non-stationary, {verdicts.get('inconclusive', 0):.0%} inconclusive. Median lag-7
autocorrelation is only **{np.median(lag7):.2f}** (total series: {acf(total.to_numpy(), nlags=7)[7]:.2f}): at item level
the weekly pattern is buried in Poisson-like noise. Aggregates are smooth and predictable; items are noisy - one
reason forecasts are reconciled across the hierarchy (Phase 16) and models are trained globally (Phase 11).""")
    r.metrics = {"variance_shares_total": shares, "stationarity": st.reset_index().to_dict(orient="records"),
                 "item_median_lag7_acf": float(np.median(lag7))}
    r.save()


if __name__ == "__main__":
    main()

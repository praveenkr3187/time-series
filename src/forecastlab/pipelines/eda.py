"""Phase 1 - Know the demand: structure, seasonality, intermittency, drivers.

    python -m forecastlab.pipelines.eda
"""
from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from forecastlab import viz
from forecastlab.data.hierarchy import m5_hierarchy
from forecastlab.data.load import first_sale_day, load_dataset
from forecastlab.models.baselines import syntetos_boylan_class
from forecastlab.report import Report

log = logging.getLogger(__name__)
WDAY = ["Sat", "Sun", "Mon", "Tue", "Wed", "Thu", "Fri"]  # M5 wday 1 = Saturday


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ds = load_dataset()
    Y, cal, meta = ds.Y, ds.cal, ds.series
    hier = m5_hierarchy()
    r = Report("01_eda", "Phase 1 - Exploratory Data Analysis")

    r.h2("The forecasting object")
    n, T = Y.shape
    r.p(f"""**{n:,} daily series** (3,049 products x 10 stores in California, Texas and Wisconsin), **{T:,} days**
({ds.dates[0].date()} -> {ds.dates[-1].date()}), {Y.sum():,.0f} units sold. Each series is one product in one store:
`(date, store_id, item_id) -> units_sold`, plus weekly sell prices, a calendar of 30 events and SNAP
(food-stamp) days per state.

The series form a 12-level hierarchy of **{len(hier.keys):,} series** - the business needs forecasts at every level
(item-store for replenishment, store-department for staffing, category-state for purchasing):""")
    r.table(hier.keys.groupby(["level", "level_name"]).size().rename("series").reset_index().set_index("level"))

    # --- total demand over time
    total = pd.Series(Y.sum(axis=0), index=ds.dates)
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.plot(total.index, total.values, color=viz.SERIES[0], linewidth=0.6, alpha=0.6, label="daily")
    ax.plot(total.index, total.rolling(28, center=True).mean(), color=viz.SERIES[1], linewidth=2, label="28-day mean")
    ax.set(title="Total units sold per day, all 10 stores", ylabel="units")
    ax.legend(loc="upper left")
    r.figure(viz.save(fig, r.dir / "total_sales.png"), "Total daily sales")
    xmas = total[(total.index.month == 12) & (total.index.day == 25)]
    r.p(f"""* **Trend:** the 28-day mean grows from {total.iloc[:28].mean():,.0f} to {total.iloc[-28:].mean():,.0f}
  units/day - partly real growth, partly **new products launching** (see below).
* **Christmas:** sales drop to {xmas.max():,.0f} units on every 25 December - the stores are closed. A model
  that does not know this calls it a "demand collapse"; a data-quality check that does not know it raises a false alarm.
* The dense band is the **weekly cycle**.""")

    # --- launches: series join over time
    fs = first_sale_day(Y)
    on = pd.Series(np.bincount(np.minimum(fs, T - 1), minlength=T).cumsum(), index=ds.dates)
    r.h2("Product launches: the panel is not rectangular")
    r.p(f"""Only {(fs == 0).sum():,} series sell on day 1; the rest start later ({(fs > T // 2).sum():,} launch in the
second half of the data). Before its first sale a series is **not on the shelf** - those leading zeros are not
"zero demand" and must not be used for training or for scaling errors. We use the price file to know whether a
product is on the shelf on a given day (no price = not stocked): {ds.on_shelf.mean():.1%} of all cells are on-shelf.""")
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(on.index, on.values, color=viz.SERIES[0])
    ax.set(title="Series that have recorded at least one sale", ylabel="series")
    r.figure(viz.save(fig, r.dir / "launches.png"), "Launches")

    # --- seasonality
    r.h2("Seasonality")
    d = cal.assign(**{c: Y[meta["cat_id"].to_numpy() == c].sum(axis=0) for c in sorted(meta["cat_id"].unique())})
    d = d[d["date"] >= "2014-01-01"]
    cats = sorted(meta["cat_id"].unique())
    wd = d.groupby("wday")[cats].mean()
    wd = wd / wd.mean()
    mo = d.groupby("month")[cats].mean()
    mo = mo / mo.mean()
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    for i, c in enumerate(cats):
        axes[0].plot(WDAY, wd[c].values, marker="o", color=viz.SERIES[i], label=c)
        axes[1].plot(mo.index, mo[c].values, marker="o", color=viz.SERIES[i], label=c)
    axes[0].set(title="Weekly profile (index, 1 = average day)")
    axes[1].set(title="Monthly profile (index)", xticks=range(1, 13))
    axes[0].legend()
    r.figure(viz.save(fig, r.dir / "seasonality.png"), "Weekly and monthly profiles")
    r.p(f"""Weekend days sell {wd.loc[[1, 2]].mean().mean() / wd.loc[[3, 4, 5, 6]].mean().mean() - 1:.0%} more than
Monday-Thursday (2014+). **The weekly cycle is the dominant seasonality** (weekday index range
{(wd.max() - wd.min()).mean():.2f}); the yearly cycle is weaker (monthly index range {(mo.max() - mo.min()).mean():.2f}; peak
months: {", ".join(f"{c} {mo[c].idxmax()}" for c in cats)}; December and January are the low months). That is why
the seasonal-naive forecast with period 7 is the baseline to beat, and why every model gets day-of-week information.""")
    r.table(wd.set_axis(WDAY).T, floatfmt=".2f")

    # --- intermittency
    r.h2("Intermittent demand")
    H = Y[:, -365:]
    cls = syntetos_boylan_class(H)
    zero_share = (H == 0).mean(axis=1)
    mean_d = H.mean(axis=1)
    r.p(f"""Over the last year, **{(H == 0).mean():.0%} of item-store-days have zero sales**; the median series sells
{np.median(mean_d):.2f} units/day. Most retail SKUs are *slow movers*. Syntetos-Boylan classification
(ADI = average gap between sales, CV^2 = variability of non-zero sizes):""")
    t = pd.DataFrame({"class": cls, "zero_share": zero_share, "mean_daily": mean_d})
    tab = t.groupby("class").agg(series=("mean_daily", "size"), zero_share=("zero_share", "mean"),
                                 mean_daily_units=("mean_daily", "mean"))
    tab["share_of_units"] = t.groupby("class")["mean_daily"].sum() / t["mean_daily"].sum()
    r.table(tab.loc[[c for c in ["smooth", "erratic", "intermittent", "lumpy"] if c in tab.index]], floatfmt=".3f")
    r.p("""* **Smooth** series behave like textbook time series; ARIMA/ETS are fine there.
* **Intermittent / lumpy** series are mostly zeros. A point forecast of "0.4 units/day" is never observed, yet it is
  the right *expected* demand. Metrics like MAPE are undefined on zero days; Croston-type methods and Tweedie /
  Poisson losses are designed for this.
* Few series carry most volume -> volume-weighted metrics (WAPE, WRMSSE) and per-segment reporting.""")
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.hist(np.log10(mean_d[mean_d > 0]), bins=60, color=viz.SERIES[0])
    ax.set(title="Mean daily units per series (last 365 days, log10)", xlabel="log10(units/day)", ylabel="series")
    r.figure(viz.save(fig, r.dir / "volume_hist.png"), "Volume distribution")

    # --- SNAP
    r.h2("Demand drivers")
    rows = []
    for st in ["CA", "TX", "WI"]:
        for c in cats:
            m = (meta["state_id"].to_numpy() == st) & (meta["cat_id"].to_numpy() == c)
            s = pd.Series(Y[m].sum(axis=0), index=ds.dates)
            snap = cal[f"snap_{st}"].to_numpy() == 1
            sel = ds.dates >= "2014-01-01"
            # compare within weekday to avoid confounding with the weekly cycle
            df = pd.DataFrame({"y": s.values, "snap": snap, "wday": cal["wday"]})[sel]
            lift = df.groupby(["wday", "snap"])["y"].mean().unstack()
            rows.append({"state": st, "category": c, "snap_lift": (lift[True] / lift[False]).mean() - 1})
    snap = pd.DataFrame(rows).pivot(index="category", columns="state", values="snap_lift")
    r.p("**SNAP days** (US food-assistance payment days, 10 per month, different per state): average lift in "
        "units vs non-SNAP days of the same weekday, 2014+.")
    r.table(snap, floatfmt="+.1%")

    ev = cal.assign(y=Y.sum(axis=0))
    ev = ev[(ev["date"] >= "2012-01-01") & (ev["date"].dt.strftime("%m-%d") != "12-25")]
    ev["lift"] = ev["y"] / ev.groupby("wday")["y"].transform(
        lambda s: s.shift(1).rolling(4, min_periods=1).median()) - 1
    evt = ev[ev["event_name_1"] != ""].groupby("event_name_1")["lift"].mean().sort_values()
    r.p("**Events** - total sales on the event day vs the median of the same weekday in the previous 4 weeks "
        "(most and least positive):")
    r.table(pd.concat([evt.head(5), evt.tail(5)]).rename("lift_vs_recent_same_weekday").to_frame(), floatfmt="+.1%")

    # --- price
    p = ds.P[:, -365:]
    changes = (np.abs(np.diff(p, axis=1)) > 1e-6) & ~np.isnan(p[:, 1:])
    rel = ds.P / np.nanmax(ds.P, axis=1, keepdims=True)
    disc = (rel < 0.9) & (rel > 0)
    lifts = []
    sel = slice(Y.shape[1] - 730, Y.shape[1])
    Ys, Ds = Y[:, sel], disc[:, sel]
    for c in cats:
        m = meta["cat_id"].to_numpy() == c
        a, b = Ys[m][Ds[m]].mean(), Ys[m][~Ds[m] & ds.on_shelf[m][:, sel]].mean()
        lifts.append({"category": c, "share_of_days_discounted": Ds[m].mean(), "units_when_discounted": a,
                      "units_otherwise": b})
    r.p(f"""**Prices** change for {(changes.any(axis=1)).mean():.0%} of series within the last year. Days on which an
item sells >= 10% below its own maximum price (a proxy for a promotion - M5 has no promotion flag):""")
    r.table(pd.DataFrame(lifts).set_index("category"), floatfmt=".3f")
    r.p("""This is correlation, not elasticity: discounted items may also be different items. The model still benefits
from price-relative-to-normal features, which is what Phase 10 builds.

**What this means for the platform**
1. Weekly seasonality + events + SNAP + price are the drivers -> calendar and price are *known-future covariates*.
2. Thousands of sparse, related series -> **global** models that share information beat one model per series.
3. Zeros are the norm -> loss functions (Tweedie) and metrics (WRMSSE, WAPE) that survive zeros.
4. Launches and Christmas closures -> data-quality rules (Phase 2) before anything is trained.""")
    r.metrics = {"n_series": n, "n_days": T, "zero_share_last_year": float((H == 0).mean()),
                 "intermittency_classes": t["class"].value_counts().to_dict(),
                 "weekend_uplift": float(wd.loc[[1, 2]].mean().mean() / wd.loc[[3, 4, 5, 6]].mean().mean() - 1),
                 "snap_lift": snap.to_dict()}
    r.save()
    log.info("EDA written to %s", r.dir)


if __name__ == "__main__":
    main()

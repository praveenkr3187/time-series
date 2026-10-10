"""Phase 2 - Data quality: audit the history, and prove the nightly validator catches broken files.

    python -m forecastlab.pipelines.data_quality

Writes artifacts/data/zero_run_labels.npy (suspected stock-out / not-ranged days), used by later phases.
"""
from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from forecastlab import viz
from forecastlab.data.lake import day_batch
from forecastlab.data.load import data_dir, first_sale_day, load_dataset
from forecastlab.data.quality import INACTIVE, STOCKOUT, check_batch, make_reference, store_closures, suspected_stockouts
from forecastlab.report import Report

log = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ds = load_dataset()
    Y, P, meta = ds.Y, ds.P, ds.series
    r = Report("02_data_quality", "Phase 2 - Data Quality")

    r.h2("Structural checks on the full history")
    dates = ds.dates
    fs = first_sale_day(Y)
    checks = {
        "calendar has no gaps": bool((np.diff(dates.values).astype("timedelta64[D]").astype(int) == 1).all()),
        "one row per item x store": bool(not meta.duplicated(["item_id", "store_id"]).any()),
        "no negative sales": bool((Y >= 0).all()),
        "no sales without a shelf price": bool(not (Y[~ds.on_shelf] > 0).any()),
        "all prices > 0": bool((P[ds.on_shelf] > 0).all()),
        "item belongs to exactly one dept": bool(meta.groupby("item_id")["dept_id"].nunique().max() == 1),
        "store belongs to exactly one state": bool(meta.groupby("store_id")["state_id"].nunique().max() == 1),
    }
    r.table(pd.Series(checks, name="passed").to_frame())
    priced_before_sale = (np.argmax(ds.on_shelf, axis=1) < fs).mean()
    r.p(f"""All pass - M5 is a cleaned competition dataset. Real POS feeds fail these regularly (double-loaded files,
returns netted into negative sales, ids re-used). The value of this phase is the **behavioural** checks below,
which find problems no schema check can.

Note: {priced_before_sale:.0%} of series have a shelf price *before* their first sale - "launched" (price exists)
and "first sold" are different events.""")

    r.h2("Store closures and outages")
    cl = store_closures(Y, meta, dates)
    cl["md"] = cl["date"].dt.strftime("%m-%d")
    xmas = cl["md"] == "12-25"
    r.p(f"""{len(cl)} store-days have a total below 10% of the store's usual volume. {int(xmas.sum())} of them are
Christmas (stores closed - expected). The rest are **unexplained**: a nightly validator must flag them, because a
model trained on them learns a fake demand collapse.""")
    other = cl[~xmas].sort_values("date")
    if len(other):
        other["date"] = other["date"].dt.date
        r.table(other.drop(columns="md").head(15).set_index("date"), floatfmt=".0f")

    r.h2("Suspected stock-outs (censored demand)")
    lab = suspected_stockouts(Y, ds.on_shelf)
    np.save(data_dir() / "zero_run_labels.npy", lab)
    mask = lab == STOCKOUT
    inactive = lab == INACTIVE
    on = ds.on_shelf & (np.arange(Y.shape[1]) >= fs[:, None])
    share, share_inactive = mask.sum() / on.sum(), inactive.sum() / on.sum()
    per = mask.any(axis=1)
    r.p(f"""Implausible runs come in two very different kinds:

* **Short runs (< 28 days) = suspected stock-outs: {share:.2%} of on-shelf days**, affecting {per.mean():.0%} of series
  at least once. Demand existed; the shelf was empty. Sales are *censored*.
* **Long runs (28+ days) = item not ranged: {share_inactive:.2%} of on-shelf days**, {inactive.any(axis=1).mean():.0%} of
  series. The price file still lists the item but it is not selling at all - delisted, seasonal range, or moved.
  These would dominate the count if we did not separate them.

Neither is "zero demand". Phase 11 tests training with these days removed; Phase 17's inventory simulation
treats stock-outs as lost sales. Labels are saved to `artifacts/data/zero_run_labels.npy`.""")
    by = pd.DataFrame({"cat": meta["cat_id"], "stockout_days": mask.sum(axis=1), "not_ranged_days": inactive.sum(axis=1),
                       "on_shelf_days": on.sum(axis=1)})
    byc = by.groupby("cat").sum()
    byc["stockout_share"] = byc["stockout_days"] / byc["on_shelf_days"]
    byc["not_ranged_share"] = byc["not_ranged_days"] / byc["on_shelf_days"]
    r.table(byc, floatfmt=".4f")

    # example series with the most flagged days among sizeable sellers
    vol = Y[:, -365:].mean(axis=1)
    cand = np.flatnonzero((vol > 3) & (mask[:, -365:].sum(axis=1) > 0))
    i = cand[np.argmax(mask[cand, -365:].sum(axis=1))] if len(cand) else int(np.argmax(mask.sum(axis=1)))
    sl = slice(Y.shape[1] - 365, Y.shape[1])
    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.plot(dates[sl], Y[i, sl], color=viz.SERIES[0], linewidth=0.9)
    m = mask[i, sl]
    ax.scatter(dates[sl][m], np.zeros(m.sum()), color=viz.STATUS["critical"], s=10, zorder=3, label="suspected stock-out")
    ax.set(title=f"{meta['id'][i]}: daily units, last year", ylabel="units")
    ax.legend(loc="upper left")
    r.figure(viz.save(fig, r.dir / "stockout_example.png"), "Stock-out example")

    r.h2("Nightly batch validation: does the gate catch broken files?")
    day = ds.day("2016-04-25")
    ref = make_reference(Y, P, meta, day)
    good = day_batch(ds, day)
    scenarios: dict[str, pd.DataFrame] = {"clean file": good}
    b = pd.concat([good, good.sample(500, random_state=0)])
    scenarios["500 rows loaded twice"] = b
    b = good.copy(); b.loc[b.sample(40, random_state=1).index, "units"] = -3
    scenarios["returns netted as negative units"] = b
    scenarios["store TX_2 missing"] = good[good["store_id"] != "TX_2"]
    b = good.copy(); idx = b[b["store_id"] == "WI_1"].index
    b.loc[idx, "units"] = (b.loc[idx, "units"] * 0.3).round()
    scenarios["WI_1 partial upload (30% of rows' units)"] = b
    b = good.copy(); b.loc[b.sample(5, random_state=2).index, "units"] = 900
    scenarios["5 fat-finger quantities (900 units)"] = b
    b = good.copy(); b.loc[b.sample(30, random_state=3).index, "sell_price"] *= 100
    scenarios["30 prices in cents instead of dollars"] = b
    b = good.copy(); b["date"] = ds.dates[day - 1]
    scenarios["yesterday's file re-sent"] = b
    b = pd.concat([good, pd.DataFrame({"date": ds.dates[day], "store_id": "CA_1",
                                       "item_id": [f"FOODS_9_{k:03d}" for k in range(3)], "units": 4,
                                       "sell_price": 2.5})])
    scenarios["3 unknown new items"] = b
    rows = []
    for name, batch in scenarios.items():
        iss = check_batch(batch, ds.dates[day], ref)
        rows.append({"scenario": name, "verdict": "BLOCK" if any(i.severity == "error" for i in iss) else
                     ("WARN" if iss else "PASS"), "checks fired": ", ".join(f"{i.check}({i.n_rows})" for i in iss) or "-"})
    res = pd.DataFrame(rows).set_index("scenario")
    r.table(res)
    clean = check_batch(good, ds.dates[day], ref)
    if clean:
        r.p("The clean file is not silent: " + "; ".join(f"`{i.check}` - {i.detail}" for i in clean) +
            ". Real data contains genuine outliers; that is why spikes are warnings, not blockers.")
    r.p("""`BLOCK` stops the nightly run: yesterday's forecast stays live and the data team is paged - a stale forecast
is better than a forecast built on a broken file. `WARN` lets the run proceed and records the issue for monitoring.
The same `check_batch` function runs inside the nightly job (`serving/nightly.py`).""")

    r.metrics = {"structural_checks": checks, "store_closure_days": int(len(cl)), "christmas_closures": int(xmas.sum()),
                 "stockout_share_of_on_shelf_days": float(share), "not_ranged_share": float(share_inactive), "series_with_stockout": float(per.mean()),
                 "validation_scenarios": res["verdict"].to_dict()}
    r.save()
    log.info("done: %s", res["verdict"].to_dict())


if __name__ == "__main__":
    main()

"""Phase 17 - From forecast to inventory decision: what is accuracy worth?

    python -m forecastlab.pipelines.inventory

Replays the holdout window: every item-store is replenished daily with an order-up-to policy driven by a
forecast. Same policy, different forecasts -> the difference in money is the business value of the model.
"""
from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from forecastlab import viz
from forecastlab.config import load_config
from forecastlab.data.load import load_dataset
from forecastlab.inventory.sim import Costs, order_up_to, simulate, summarise
from forecastlab.models.probabilistic import nb_dispersion
from forecastlab.report import Report
from forecastlab.store import load_forecast
from forecastlab.validation import holdout_origin

log = logging.getLogger(__name__)
SOURCES = ["Seasonal naive (7)", "Weekday profile x MA(28)", "LightGBM (per-store, direct)"]
GRID = [None, 0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ds = load_dataset()
    b = load_config()["business"]
    costs = Costs(b["lead_time_days"], b["holding_cost_rate"], b["stockout_margin_rate"])
    qstar = costs.critical_fractile()
    r = Report("17_inventory", "Phase 17 - Inventory Simulation: The Business Value of a Forecast")
    o = holdout_origin(ds)
    D = ds.Y[:, o.test]
    price = ds.P[:, o.test]
    alpha = nb_dispersion(ds.Y[:, : o.day + 1])
    rows = []
    for src in SOURCES:
        F = load_forecast(src, o.day)
        for q in [*GRID, qstar]:
            S = order_up_to(F, alpha, q, costs.lead_time)
            sim = simulate(D, S, price, costs)
            rows.append({"forecast": src, "service quantile": "mean (no safety stock)" if q is None else
                         ("q* = %.4f (cost-optimal)" % q if q == qstar else f"{q:.2f}"), "q": q if q else np.nan,
                         **summarise(sim, D)})
        log.info("%s done", src)
    df = pd.DataFrame(rows)

    r.p(f"""Window {ds.dates[o.test.start].date()} .. {ds.dates[o.test.stop - 1].date()}, all 30,490 item-stores.
Policy: daily review, order-up-to, lead time {costs.lead_time} days, lost sales. Costs: holding
{costs.holding_rate:.0%} of price per year, a lost sale costs {costs.margin_rate:.0%} of price (gross margin). Demand is
modelled as negative binomial around each forecast (Phase 15), so every forecast source gets the same
safety-stock logic and differs only in its mean.""")

    r.h2("Same policy, different forecasts (target service quantile 0.90)")
    t90 = df[df["service quantile"] == "0.90"].set_index("forecast").drop(columns=["service quantile", "q"])
    r.table(t90, floatfmt=",.3f")
    base, champ = SOURCES[0], SOURCES[-1]
    saving = t90.loc[base, "total cost $"] - t90.loc[champ, "total cost $"]
    r.p(f"""Over 28 days the champion forecast lowers total cost by **${saving:,.0f}** vs seasonal naive
({saving / t90.loc[base, 'total cost $']:.0%}) with fill rate {t90.loc[champ, 'fill rate']:.1%} vs {t90.loc[base, 'fill rate']:.1%}
and average inventory ${t90.loc[champ, 'avg inventory value $']:,.0f} vs ${t90.loc[base, 'avg inventory value $']:,.0f}.
Annualised for these 10 stores: ~${saving * 365 / 28:,.0f} - this is the number a business case is built on, not WRMSSE.""")

    r.h2("Choosing the service level: the critical fractile")
    r.p(f"""Holding one unit for one more day costs {costs.holding_rate / 365:.4%} of its price; losing a sale costs
{costs.margin_rate:.0%}. The cost-minimising quantile is q* = c_u / (c_u + c_o) = **{qstar:.4f}**. With cheap holding and
expensive lost sales, a high service level is rational - the opposite of the "forecast the median" instinct.""")
    tq = df[df["forecast"] == champ].set_index("service quantile").drop(columns=["forecast", "q"])
    r.table(tq, floatfmt=",.3f")
    best_q = tq["total cost $"].idxmin()
    r.p(f"""Lowest total cost for the champion: **{best_q}**. Using the plain mean forecast as the order-up-to level (no
safety stock) leaves fill rate at {tq.loc['mean (no safety stock)', 'fill rate']:.1%}.""")

    fig, ax = plt.subplots(figsize=(8, 4.2))
    for i, src in enumerate(SOURCES):
        d = df[(df["forecast"] == src) & df["q"].notna()].sort_values("q")
        ax.plot(d["avg inventory value $"] / 1e3, d["fill rate"], marker="o", color=viz.SERIES[i], label=src)
    ax.set(title="Fill rate vs inventory held (service quantile 0.5 -> 0.99)", xlabel="average inventory value ($k)",
           ylabel="fill rate")
    ax.legend(loc="lower right", fontsize=8)
    r.figure(viz.save(fig, r.dir / "frontier.png"), "Efficient frontier")
    r.p("""Each curve is one forecast with increasing safety stock. A better forecast moves the curve **up and left**:
the same fill rate with less inventory, or a higher fill rate with the same inventory. That is the business meaning
of "more accurate".

**Caveats.** Demand here is *recorded sales*: on stock-out days (Phase 2) true demand was higher, so every policy's
fill rate is overstated a little. Lead times, case-pack sizes, shelf capacity and supplier minimums are ignored.""")

    r.h2("Business-weighted accuracy")
    rev_w = np.nan_to_num(price)
    acc = {}
    for src in SOURCES:
        F = load_forecast(src, o.day)
        e = np.abs(F - D)
        acc[src] = {"unit WAPE": e.sum() / D.sum(), "revenue-weighted WAPE": (e * rev_w).sum() / (D * rev_w).sum(),
                    "bias": (F - D).sum() / D.sum()}
    r.table(pd.DataFrame(acc).T, floatfmt=".3f")
    r.p("""Weighting errors by price measures the error in **dollars**: an error on a $20 item matters 20x more than on a
$1 item. Planners and finance read this number; WRMSSE remains the model-selection metric.""")
    r.metrics = {"at_0.90": t90.to_dict(orient="index"), "champion_by_quantile": tq.to_dict(orient="index"),
                 "critical_fractile": qstar}
    r.save()
    print(t90.round(3).to_string())


if __name__ == "__main__":
    main()

"""Inventory simulation: what a forecast is worth on the shelf.

Daily periodic review, order-up-to policy, lead time L >= 1: an order placed at the end of day t is on the shelf
at the start of day t+L. Unmet demand is lost (retail shoppers do not backorder).

    S[:, t] = quantile of demand over days t .. t+L      (review period 1 + lead time L)
    end of day t: order max(0, S[:, t+1] - inventory position),   position = on hand + on order
    the shelf starts the window with S[:, 0] on hand

Costs per unit: holding = price x annual rate / 365 per day on hand; lost sale = price x gross-margin rate.
The cost-optimal service quantile is the critical fractile  q* = c_under / (c_under + c_over).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from forecastlab.models.probabilistic import nb_sum_quantiles


@dataclass
class Costs:
    lead_time: int
    holding_rate: float      # annual, share of price
    margin_rate: float       # lost margin per unit of unmet demand, share of price

    def critical_fractile(self) -> float:
        cu = self.margin_rate
        co = self.holding_rate / 365
        return cu / (cu + co)


def order_up_to(F: np.ndarray, alpha: np.ndarray, q: float | None, L: int) -> np.ndarray:
    """S [n, h+1]: q-quantile (None = mean) of NB demand summed over days t..t+L.

    Days past the forecast window repeat its last week (the next nightly run would supply them).
    """
    n, h = F.shape
    need = h + 1 + L
    ext = np.concatenate([F, np.tile(F[:, -7:], int(np.ceil((need - h) / 7)))], axis=1)[:, :need]
    S = np.zeros((n, h + 1), dtype=np.float32)
    for t in range(h + 1):
        win = ext[:, t: t + L + 1]
        S[:, t] = win.sum(axis=1) if q is None else nb_sum_quantiles(win, alpha, [q])[0]
    return np.ceil(S)


def simulate(D: np.ndarray, S: np.ndarray, price: np.ndarray, costs: Costs) -> dict[str, np.ndarray]:
    """Vectorised over series. D actual demand [n, h], S order-up-to levels [n, h+1], price [n, h]."""
    n, h = D.shape
    L = costs.lead_time
    assert L >= 1
    on_hand = S[:, 0].astype(np.float64).copy()
    arriving = np.zeros((n, h + L + 1))
    sold = np.zeros((n, h)); lost = np.zeros((n, h)); inv = np.zeros((n, h))
    for t in range(h):
        on_hand += arriving[:, t]
        s = np.minimum(on_hand, D[:, t])
        sold[:, t], lost[:, t] = s, D[:, t] - s
        on_hand -= s
        inv[:, t] = on_hand
        position = on_hand + arriving[:, t + 1:].sum(axis=1)
        arriving[:, t + L] += np.maximum(0, S[:, t + 1] - position)
    p = np.nan_to_num(price)
    return {
        "sold": sold, "lost": lost, "inventory": inv,
        "holding_cost": inv * p * costs.holding_rate / 365,
        "lost_margin": lost * p * costs.margin_rate,
        "inventory_value": inv * p,
    }


def summarise(sim: dict[str, np.ndarray], D: np.ndarray) -> dict[str, float]:
    demand = D.sum()
    return {
        "fill rate": float(sim["sold"].sum() / max(demand, 1)),
        "stock-out item-days": float(((sim["lost"] > 0)).mean()),
        "avg inventory value $": float(sim["inventory_value"].sum(axis=0).mean()),
        "holding cost $": float(sim["holding_cost"].sum()),
        "lost margin $": float(sim["lost_margin"].sum()),
        "total cost $": float(sim["holding_cost"].sum() + sim["lost_margin"].sum()),
    }

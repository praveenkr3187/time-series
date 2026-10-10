"""Data-quality checks, used offline (Phase 2 audit) and by the nightly ingestion job before anything is forecast.

Two families:
* batch checks on one day's POS file - schema, duplicates, impossible values, unknown / missing series,
  store totals out of line (an outage or a closure), price jumps;
* history audits - store closures and suspected stock-outs (zero runs too long to be bad luck).

A stock-out matters for forecasting because sales are *censored demand*: zero sales on an empty shelf is not
zero demand. Training on those zeros teaches the model to under-forecast exactly the items that sell out.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

BATCH_COLUMNS = {"date": "datetime64[ns]", "store_id": "object", "item_id": "object", "units": "number",
                 "sell_price": "number"}
KNOWN_CLOSURES = {"12-25"}  # stores closed on Christmas day


@dataclass
class Issue:
    check: str
    severity: str        # "error" blocks the nightly run, "warning" is logged and monitored
    detail: str
    n_rows: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Reference:
    """What a healthy day looks like, computed from recent history (no look-ahead)."""
    series: pd.DataFrame          # item_id, store_id of every expected series
    on_shelf: np.ndarray          # bool per series: has a price for this day
    store_baseline: pd.Series     # median units of the same weekday over the last 4 weeks, per store
    unit_cap: np.ndarray          # per series: max plausible units (robust upper bound from history)
    last_price: np.ndarray        # per series: last known price


def make_reference(Y: np.ndarray, P: np.ndarray, series: pd.DataFrame, day: int) -> Reference:
    """Reference for validating day `day` using only data up to day-1."""
    stores = series["store_id"].to_numpy()
    st = pd.DataFrame(Y[:, day - 28: day]).groupby(stores).sum()
    same_wday = st.iloc[:, [i for i in range(28) if (28 - i) % 7 == 0]]
    H = Y[:, max(0, day - 365): day]
    q = np.quantile(H, 0.995, axis=1)
    cap = np.maximum(10.0, 3 * np.maximum(q, H.max(axis=1) / 2))
    last_price = pd.DataFrame(P[:, :day].T).ffill().iloc[-1].to_numpy()
    return Reference(series[["item_id", "store_id"]], ~np.isnan(P[:, day]), same_wday.median(axis=1), cap, last_price)


def check_batch(batch: pd.DataFrame, expected_date: pd.Timestamp, ref: Reference) -> list[Issue]:
    issues: list[Issue] = []
    missing_cols = [c for c in BATCH_COLUMNS if c not in batch.columns]
    if missing_cols:
        return [Issue("schema", "error", f"missing columns {missing_cols}", len(batch))]
    if batch.empty:
        return [Issue("empty_file", "error", "no rows received")]

    dates = pd.to_datetime(batch["date"]).dt.normalize().unique()
    if len(dates) != 1 or dates[0] != expected_date:
        issues.append(Issue("date", "error", f"expected {expected_date.date()}, got {[str(d)[:10] for d in dates]}",
                            len(batch)))
    dup = batch.duplicated(["store_id", "item_id"], keep=False)
    if dup.any():
        issues.append(Issue("duplicates", "error", "same item/store reported more than once (double-loaded file?)",
                            int(dup.sum())))
    neg = batch["units"] < 0
    if neg.any():
        issues.append(Issue("negative_units", "error", "negative unit sales (returns must not be netted here)",
                            int(neg.sum())))
    nulls = batch[["units", "store_id", "item_id"]].isna().any(axis=1)
    if nulls.any():
        issues.append(Issue("nulls", "error", "null units / ids", int(nulls.sum())))

    key = ref.series.reset_index(drop=True).reset_index().rename(columns={"index": "row"})
    m = batch.merge(key, on=["item_id", "store_id"], how="left", indicator=True)
    unknown = m["_merge"] == "left_only"
    if unknown.any():
        issues.append(Issue("unknown_series", "warning", "item/store not in the product master (new listing?)",
                            int(unknown.sum())))
    m = m[~unknown].drop_duplicates(["item_id", "store_id"])
    rows = m["row"].to_numpy().astype(int)
    got = np.zeros(len(key), dtype=bool)
    got[rows] = True
    missing = ref.on_shelf & ~got
    if missing.any():
        stores = key.loc[missing, "store_id"].value_counts()
        full = stores[stores >= 0.9 * ref.series["store_id"].value_counts().reindex(stores.index)]
        sev = "error" if len(full) else "warning"
        detail = (f"stores with no data: {list(full.index)}" if len(full) else
                  f"{int(missing.sum())} on-shelf series absent (treated as zero sales)")
        issues.append(Issue("missing_series", sev, detail, int(missing.sum())))

    units = m["units"].to_numpy(dtype=float)
    spike = units > ref.unit_cap[rows]
    if spike.any():
        top = m.loc[spike].nlargest(3, "units")
        issues.append(Issue("unit_spike", "warning", "units far above the series' history, e.g. " + ", ".join(
            f"{r.item_id}/{r.store_id}={r.units:.0f}" for r in top.itertuples()), int(spike.sum())))

    price = m["sell_price"].to_numpy(dtype=float)
    prev = ref.last_price[rows]
    with np.errstate(invalid="ignore", divide="ignore"):
        jump = (price <= 0) | (np.abs(price / prev - 1) > 0.5)
    jump &= ~np.isnan(price) & ~np.isnan(prev)
    if jump.any():
        issues.append(Issue("price_jump", "warning", "price <= 0 or moved more than 50% day-on-day", int(jump.sum())))

    if expected_date.strftime("%m-%d") not in KNOWN_CLOSURES:
        totals = m.groupby("store_id")["units"].sum().reindex(ref.store_baseline.index).fillna(0)
        ratio = totals / ref.store_baseline.clip(lower=1)
        low = ratio[ratio < 0.5]
        if len(low):
            issues.append(Issue("store_total_drop", "error", "store total < 50% of its usual same-weekday volume: " +
                                ", ".join(f"{s} {v:.0%}" for s, v in low.items()), len(low)))
    return issues


def store_closures(Y: np.ndarray, series: pd.DataFrame, dates: pd.DatetimeIndex, threshold: float = 0.1) -> pd.DataFrame:
    """Store-days whose total is below `threshold` x the store's median over the surrounding 8 weeks."""
    st = pd.DataFrame(Y).groupby(series["store_id"].to_numpy()).sum().T
    st.index = dates
    base = st.rolling(57, center=True, min_periods=15).median()
    low = (st < threshold * base).stack()
    out = low[low].reset_index()
    out.columns = ["date", "store_id", "_"]
    out["units"] = [st.loc[d, s] for d, s in zip(out["date"], out["store_id"])]
    out["usual"] = [base.loc[d, s] for d, s in zip(out["date"], out["store_id"])]
    return out.drop(columns="_")


STOCKOUT, INACTIVE = 1, 2


def suspected_stockouts(Y: np.ndarray, on_shelf: np.ndarray, window: int = 56, p_max: float = 1e-4,
                        min_run: int = 3, long_run: int = 28) -> np.ndarray:
    """Label zero runs that are implausible given how often the series normally sells.

    Returns int8 [n, T]: 0 = normal, STOCKOUT = implausible run shorter than `long_run` days (empty shelf),
    INACTIVE = implausible run of `long_run`+ days (item delisted / not ranged even though a price exists).

    For a run of k zero days starting at t, p0 = smoothed share of zero days in the `window` on-shelf days
    before t, and P(k zeros in a row) = p0^k. Using the empirical zero share (not a Poisson rate) keeps the test
    honest for over-dispersed retail demand. Runs with probability < p_max are flagged. Vectorised over all series.
    """
    n, T = Y.shape
    z = (Y == 0) & on_shelf
    pad = np.zeros((n, 1), dtype=bool)
    d = np.diff(np.concatenate([pad, z, pad], axis=1).astype(np.int8), axis=1)
    sr, sc = np.nonzero(d == 1)     # run starts (row-major order)
    er, ec = np.nonzero(d == -1)    # run ends, exclusive (same order: runs never overlap within a row)
    k = ec - sc
    CZ = np.concatenate([np.zeros((n, 1)), np.cumsum(z, axis=1)], axis=1)
    CS = np.concatenate([np.zeros((n, 1)), np.cumsum(on_shelf, axis=1)], axis=1)
    lo = np.maximum(sc - window, 0)
    days = CS[sr, sc] - CS[sr, lo]
    p0 = (CZ[sr, sc] - CZ[sr, lo] + 1) / (days + 2)
    flag = (k >= min_run) & (days >= window // 2) & (k * np.log(p0) < np.log(p_max))
    mask = np.zeros((n, T), dtype=np.int8)
    for r, s, e in zip(sr[flag], sc[flag], ec[flag]):
        mask[r, s:e] = STOCKOUT if e - s < long_run else INACTIVE
    return mask

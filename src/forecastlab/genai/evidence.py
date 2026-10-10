"""Evidence the forecast analyst may use - every number comes from the platform, never from the LLM.

All views are "as of" the latest successful nightly run: the curated history it saw, the forecast it published,
the model that produced it, and what monitoring knows so far.
"""
from __future__ import annotations

import json
from functools import cached_property

import numpy as np
import pandas as pd

from forecastlab.data.lake import Curated
from forecastlab.data.load import data_dir, load_dataset
from forecastlab.data.quality import INACTIVE, STOCKOUT
from forecastlab.serving.db import ForecastDB
from forecastlab.serving.forecaster import Forecaster


def dumps(x) -> str:
    return json.dumps(x, default=lambda o: o.item() if hasattr(o, "item") else str(o), indent=1)


def _r(x, nd: int = 1):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), nd)


class EvidenceService:
    def __init__(self, run_id: str | None = None):
        self.ds = load_dataset()
        self.db = ForecastDB()
        run = (self.db.query("SELECT * FROM runs WHERE run_id = ?", (run_id,)).iloc[0].to_dict() if run_id
               else self.db.latest_run())
        if run is None:
            raise RuntimeError("no successful nightly run yet - run `make nightly`")
        self.run = run
        self.origin = self.ds.day(run["origin_date"])
        cur = Curated.load()
        Y = cur.Y.copy()
        Y[:, self.origin + 1:] = 0            # exactly what the run could see
        self.Y = Y
        self.items = self.db.read_items(run["run_id"])
        self.h = self.items["mean"].shape[1]

    # ------------------------------------------------------------------ lookups
    def row(self, store_id: str, item_id: str) -> int:
        s = self.ds.series
        m = np.flatnonzero((s["store_id"].to_numpy() == store_id) & (s["item_id"].to_numpy() == item_id))
        if not len(m):
            raise KeyError(f"unknown item/store {item_id}/{store_id}")
        return int(m[0])

    @cached_property
    def forecaster(self) -> Forecaster:
        return Forecaster(self.run["model_version"])

    def _dates(self, a: int, b: int) -> list[str]:
        return [str(d.date()) for d in self.ds.dates[a:b]]

    # ------------------------------------------------------------------ evidence
    def forecast(self, store_id: str, item_id: str) -> dict:
        i = self.row(store_id, item_id)
        f, lo, hi = (self.items[k][i] for k in ("mean", "p10", "p90"))
        last28 = self.Y[i, self.origin - 27: self.origin + 1]
        ly = self.Y[i, self.origin + 1 - 364: self.origin + 1 - 364 + self.h]
        price = self.ds.P[i, self.origin]
        return {
            "series": f"{item_id} @ {store_id}", "category": self.ds.series["cat_id"].iloc[i],
            "department": self.ds.series["dept_id"].iloc[i], "run_id": self.run["run_id"],
            "model_version": self.run["model_version"], "data_through": self.run["origin_date"],
            "next_28_days_total": {"forecast": _r(f.sum()), "p10_daily_sum": _r(lo.sum()), "p90_daily_sum": _r(hi.sum())},
            "last_28_days_actual_total": _r(last28.sum()),
            "same_28_days_last_year_actual_total": _r(ly.sum()),
            "change_vs_last_28_days_pct": _r(100 * (f.sum() / max(last28.sum(), 1e-9) - 1)),
            "current_price": _r(price, 2),
            "daily": [{"date": d, "forecast": _r(a, 2), "p10": _r(b, 0), "p90": _r(c, 0)}
                      for d, a, b, c in zip(self._dates(self.origin + 1, self.origin + 1 + self.h), f, lo, hi)],
            "note": "p10/p90 are per-day quantiles; summing them over days overstates the spread of the total.",
        }

    def sales_history(self, store_id: str, item_id: str, weeks: int = 8) -> dict:
        i = self.row(store_id, item_id)
        y = pd.Series(self.Y[i, self.origin - 7 * weeks + 1: self.origin + 1],
                      index=self.ds.dates[self.origin - 7 * weeks + 1: self.origin + 1])
        wk = y.resample("W-SUN").sum()
        dow = y.groupby(y.index.day_name()).mean().round(2).to_dict()
        lab = np.load(data_dir() / "zero_run_labels.npy", mmap_mode="r")[i, self.origin - 7 * weeks + 1: self.origin + 1] \
            if (data_dir() / "zero_run_labels.npy").exists() else np.zeros(len(y))
        return {"weekly_units": {str(k.date()): _r(v, 0) for k, v in wk.items()},
                "average_by_weekday": dow, "days_with_zero_sales": int((y == 0).sum()), "days": len(y),
                "suspected_stockout_days": int((np.asarray(lab) == STOCKOUT).sum()),
                "suspected_not_ranged_days": int((np.asarray(lab) == INACTIVE).sum())}

    def calendar(self) -> dict:
        cal = self.ds.cal.iloc[self.origin + 1: self.origin + 1 + self.h]
        ev = cal[cal["event_name_1"] != ""]
        return {"events": [{"date": str(d.date()), "event": n, "type": t}
                           for d, n, t in zip(ev["date"], ev["event_name_1"], ev["event_type_1"])],
                "snap_days": {s: [str(d.date()) for d in cal.loc[cal[f"snap_{s}"] == 1, "date"]]
                              for s in ["CA", "TX", "WI"]}}

    def price_plan(self, store_id: str, item_id: str) -> dict:
        i = self.row(store_id, item_id)
        p = self.ds.P[i, self.origin - 90: self.origin + 1 + self.h]
        d = self.ds.dates[self.origin - 90: self.origin + 1 + self.h]
        s = pd.Series(p, index=d)
        changes = s[s.diff().abs() > 1e-6]
        return {"price_today": _r(s.iloc[90], 2), "max_price_last_90_days": _r(np.nanmax(p[:91]), 2),
                "planned_changes": [{"date": str(k.date()), "new_price": _r(v, 2)} for k, v in changes.items()],
                "discount_vs_90d_max_pct_in_forecast_window": _r(100 * (1 - np.nanmin(p[91:]) / np.nanmax(p[:91])))}

    def drivers(self, store_id: str, item_id: str) -> dict:
        i = self.row(store_id, item_id)
        view = Curated(self.Y, self.origin).view(self.ds)
        f = self.forecaster.explain(view, i, self.origin, self.h)
        drv = f.drop(columns="baseline units")
        avg = np.exp(np.log(drv).mean()).sort_values()
        top_days = {}
        for c in drv.columns:
            k = drv[c].sub(1).abs().idxmax()
            top_days[c] = {"date": str(k), "factor": _r(drv.loc[k, c], 3)}
        return {"explanation": "multiplicative factors from TreeSHAP on the production LightGBM; 1.10 = +10%",
                "baseline_units_per_day": _r(f["baseline units"].iloc[0], 3),
                "average_factor_over_window": {k: _r(v, 3) for k, v in avg.items()},
                "strongest_day_per_driver": top_days}

    def accuracy(self, store_id: str, item_id: str | None = None) -> dict:
        acc = self.db.query("SELECT * FROM accuracy WHERE scope='store' AND key=? AND target_date > ?",
                            (store_id, str(self.ds.dates[self.origin - 28].date())))
        out = {"store": store_id, "scored_store_days_last_28": int(len(acc))}
        if len(acc):
            out["store_wape_last_28_days"] = _r(acc["abs_error"].sum() / acc["actual"].sum(), 3)
            out["store_bias_last_28_days"] = _r(acc["forecast"].sum() / acc["actual"].sum() - 1, 3)
        if item_id:
            i = self.row(store_id, item_id)
            errs = []
            runs = self.db.query("SELECT run_id, origin_date FROM runs WHERE status='SUCCESS' AND origin_date < ? "
                                 "ORDER BY origin_date DESC LIMIT 14", (self.run["origin_date"],))
            for r in runs.itertuples():
                o = self.ds.day(r.origin_date)
                it = self.db.read_items(r.run_id)
                if it is not None and o + 1 <= self.origin:
                    errs.append((it["mean"][i, 0], self.Y[i, o + 1]))
            if errs:
                f, a = np.array(errs).T
                out["item_one_day_ahead_last_14_runs"] = {"forecast_total": _r(f.sum()), "actual_total": _r(a.sum()),
                                                          "abs_error_total": _r(np.abs(f - a).sum())}
        alerts = self.db.query("SELECT * FROM alerts WHERE run_date <= ? ORDER BY run_date DESC LIMIT 5",
                               (self.run["run_date"],))
        out["recent_alerts"] = alerts.to_dict(orient="records")
        return out

    def aggregate(self, level: str, key: str) -> dict:
        df = self.db.query("SELECT target_date, mean, p10, p90 FROM aggregates WHERE run_id=? AND level=? AND key=?",
                           (self.run["run_id"], level, key))
        if df.empty:
            raise KeyError(f"no aggregate {level}={key}")
        return {"level": level, "key": key, "next_28_days_total": _r(df["mean"].sum()),
                "daily": df.round(1).to_dict(orient="records")}

    def full_pack(self, store_id: str, item_id: str) -> dict:
        return {"forecast": self.forecast(store_id, item_id), "history": self.sales_history(store_id, item_id),
                "calendar": self.calendar(), "price": self.price_plan(store_id, item_id),
                "drivers": self.drivers(store_id, item_id), "accuracy": self.accuracy(store_id, item_id)}

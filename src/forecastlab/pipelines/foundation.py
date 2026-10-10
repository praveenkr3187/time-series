"""Phase 13 - Time-series foundation models, zero-shot.

    python -m forecastlab.pipelines.foundation

Chronos-Bolt (univariate) and Chronos-2 (with covariates) are pretrained on large public time-series corpora and
used here without any training on M5. The question: how close does "download and predict" get to a tuned
domain model?
"""
from __future__ import annotations

import logging
import time

import numpy as np
import pandas as pd

from forecastlab.config import load_config
from forecastlab.data.hierarchy import m5_hierarchy
from forecastlab.data.load import load_dataset
from forecastlab.evaluation import AGG_LEVELS, scorecard
from forecastlab.models import foundation
from forecastlab.pipelines.classical import exog
from forecastlab.report import Report
from forecastlab.store import load_forecast, save_forecast
from forecastlab.validation import backtest_origins, holdout_origin, sample_rows

log = logging.getLogger(__name__)
CONTEXT = 512
REF = ["Weekday profile x MA(28)", "AutoETS", "SARIMA (AutoARIMA, s=7)", "LightGBM (per-store, direct)"]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ds = load_dataset()
    hier = m5_hierarchy()
    r = Report("13_foundation_models", "Phase 13 - Foundation Models (zero-shot)")
    origins = backtest_origins(ds) + [holdout_origin(ds)]
    qs = load_config()["forecast"]["quantiles"]
    rows = sample_rows(ds)
    timing = {}
    ex = {lvl: exog(ds, lvl) for lvl in AGG_LEVELS}
    for o in origins:
        ctx = slice(o.day - CONTEXT + 1, o.day + 1)
        # Chronos-Bolt: every item series (univariate)
        name = "Chronos-Bolt small"
        if load_forecast(name, o.day) is None:
            t0 = time.time()
            F, Q = foundation.forecast(name, ds.Y[:, ctx], o.horizon, qs)
            F *= ds.on_shelf[:, o.test]
            save_forecast(name, o.day, F)
            for q, Fq in zip(qs, Q):
                save_forecast(f"{name} q{q}", o.day, Fq * ds.on_shelf[:, o.test])
            timing[name] = time.time() - t0
            for lvl in AGG_LEVELS:
                Ya = hier.aggregate(ds.Y[:, ctx], lvl)
                save_forecast(f"{name} @ level{lvl}", o.day, foundation.forecast(name, Ya, o.horizon, qs)[0])
        # Chronos-2: sample items + aggregates, with known-future covariates
        name = "Chronos-2 (with covariates)"
        if load_forecast(name, o.day) is None:
            t0 = time.time()
            full = np.full((len(ds.series), o.horizon), np.nan, dtype=np.float32)
            item_ex = _item_cov(ds, rows)
            pc = {k: v[:, ctx] for k, v in item_ex.items()}
            fc = {k: v[:, o.test] for k, v in item_ex.items()}
            full[rows] = foundation.forecast("Chronos-2", ds.Y[rows, ctx], o.horizon, qs, pc, fc)[0] * \
                ds.on_shelf[rows, o.test]
            save_forecast(name, o.day, full)
            timing[name] = time.time() - t0
            for lvl in AGG_LEVELS:
                Ya = hier.aggregate(ds.Y[:, ctx], lvl)
                e = ex[lvl]
                pc = {k: e[:, ctx, j] for j, k in enumerate(["snap", "event", "price"])}
                fc = {k: e[:, o.test, j] for j, k in enumerate(["snap", "event", "price"])}
                save_forecast(f"{name} @ level{lvl}", o.day,
                              foundation.forecast("Chronos-2", Ya, o.horizon, qs, pc, fc)[0])
        log.info("%s done", o.name)

    res = []
    for o in origins:
        for k in ["Chronos-Bolt small", "Chronos-2 (with covariates)", *REF]:
            if load_forecast(k, o.day) is not None or load_forecast(f"{k} @ level9", o.day) is not None:
                res.append({"model": k, "split": "holdout" if o is origins[-1] else "backtest", **scorecard(ds, k, o)})
    df = pd.DataFrame(res)
    cols = ["WRMSSE", "item RMSSE", "item bias", "store x dept RMSSE", "store RMSSE", "total RMSSE"]
    bt = df[df["split"] == "backtest"].groupby("model")[cols].mean()
    ho = df[df["split"] == "holdout"].groupby("model")[["WRMSSE"]].mean().rename(columns={"WRMSSE": "WRMSSE (holdout)"})
    tab = bt.join(ho).sort_values("store x dept RMSSE")
    r.h2("Zero-shot vs trained models (backtest mean; holdout WRMSSE)")
    r.table(tab, floatfmt=".3f")
    r.p(f"""Context: last {CONTEXT} days. Chronos-Bolt forecasts all 30,490 items (univariate) and each aggregate series
directly. Chronos-2 forecasts the {len(rows)}-series sample and the aggregates with known-future covariates
(SNAP share, event day, relative price index). Neither model was trained or fine-tuned on M5. WRMSSE needs all items,
so it is reported for Chronos-Bolt only; aggregate rows for LightGBM are its item forecasts summed bottom-up.

* Zero-shot foundation models are **strong baselines with zero training cost**: compare them with AutoETS / SARIMA
  at the aggregate levels, which needed per-series fitting.
* They do not know M5-specific drivers unless given as covariates, and they do not learn from the cross-section
  of 30,490 related items the way the global LightGBM does.
* Operational trade-off: no training pipeline to maintain, but GPU/CPU inference cost per run and a model you cannot
  easily inspect or adapt; fine-tuning on own data is the usual next step.""")
    if timing:
        r.p("Inference time (CPU, one origin): " + ", ".join(f"{k} {v / 60:.1f} min" for k, v in timing.items()) + ".")
    r.metrics = {"results": tab.to_dict(orient="index"), "timing_secs": timing}
    r.save()
    print(tab.round(3).to_string())


def _item_cov(ds, rows: np.ndarray) -> dict[str, np.ndarray]:
    states = pd.Categorical(ds.series["state_id"].iloc[rows], categories=["CA", "TX", "WI"]).codes
    snap = np.stack([ds.cal[f"snap_{s}"].to_numpy(np.float32) for s in ["CA", "TX", "WI"]])[states]
    event = np.broadcast_to((ds.cal["event_name_1"] != "").to_numpy(np.float32), snap.shape)
    P = pd.DataFrame(ds.P[rows].T).ffill().bfill()
    price = (P / P.rolling(365, min_periods=1).mean()).to_numpy().T.astype(np.float32)
    return {"snap": snap, "event": np.ascontiguousarray(event), "price": np.nan_to_num(price, nan=1.0)}


if __name__ == "__main__":
    main()

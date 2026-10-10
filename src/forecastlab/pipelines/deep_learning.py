"""Phase 12 - Deep learning: LSTM, GRU, Transformer, DeepAR.

    python -m forecastlab.pipelines.deep_learning

Global networks trained on random windows from all 30,490 series (CPU, a few minutes each), evaluated on
every backtest origin and the holdout with the same scorecard as everything else.
"""
from __future__ import annotations

import logging
import time

import pandas as pd

from forecastlab.config import load_config
from forecastlab.data.load import load_dataset
from forecastlab.evaluation import scorecard
from forecastlab.models import deep
from forecastlab.report import Report
from forecastlab.store import load_forecast, save_forecast
from forecastlab.validation import backtest_origins, holdout_origin

log = logging.getLogger(__name__)
KINDS = ["GRU", "LSTM", "Transformer", "DeepAR"]
STEPS = {"GRU": 2500, "LSTM": 2500, "Transformer": 2500, "DeepAR": 2500}
REF = ["Weekday profile x MA(28)", "LightGBM (per-store, direct)"]


def key(kind: str) -> str:
    return f"{kind} (global)"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ds = load_dataset()
    r = Report("12_deep_learning", "Phase 12 - Deep Learning Forecasters")
    origins = backtest_origins(ds) + [holdout_origin(ds)]
    t = deep.prepare(ds)
    start = ds.day(load_config()["data"]["model_start"])
    qs = load_config()["forecast"]["quantiles"]
    info = {}
    for o in origins:
        for kind in KINDS:
            if load_forecast(key(kind), o.day) is not None:
                continue
            t0 = time.time()
            model = deep.train(kind, t, o.day, o.horizon, start, steps=STEPS[kind])
            secs = time.time() - t0
            t0 = time.time()
            if kind == "DeepAR":
                F, Q = deep.predict(model, t, o.day, o.horizon, quantiles=tuple(qs))
                for q, Fq in zip(qs, Q):
                    save_forecast(f"{key(kind)} q{q}", o.day, Fq)
            else:
                F = deep.predict(model, t, o.day, o.horizon)
            save_forecast(key(kind), o.day, F)
            info[kind] = {"parameters": deep.n_params(model), "train_secs": secs, "predict_secs": time.time() - t0}
            log.info("%s %s: train %.0fs, predict %.0fs", o.name, kind, secs, info[kind]["predict_secs"])

    rows = []
    for o in origins:
        for k in [key(x) for x in KINDS] + REF:
            if load_forecast(k, o.day) is not None:
                rows.append({"model": k, "origin": o.name, "split": "holdout" if o is origins[-1] else "backtest",
                             **scorecard(ds, k, o)})
    df = pd.DataFrame(rows)
    cols = ["WRMSSE", "item RMSSE", "store x dept RMSSE", "store RMSSE", "item bias"]
    bt = df[df["split"] == "backtest"].groupby("model")[cols].mean()
    ho = df[df["split"] == "holdout"].set_index("model")[cols]
    tab = bt.add_suffix(" (backtest)").join(ho[["WRMSSE"]].rename(columns={"WRMSSE": "WRMSSE (holdout)"}))
    tab = tab.sort_values("WRMSSE (backtest)")
    r.h2("Results (mean of 3 backtest origins; holdout WRMSSE)")
    r.table(tab, floatfmt=".3f")
    if info:
        r.p("Model size and CPU cost (last origin trained in this run):")
        r.table(pd.DataFrame(info).T, floatfmt=",.0f")
    best = tab.index[0]
    lgb = REF[1]
    nn_best = tab.drop(REF, errors="ignore").index[0]
    r.p(f"""Every network sees a 112-day context of scaled sales and covariates (weekday, SNAP, events, relative
price, on-shelf flag) plus learned embeddings for store / dept / category / item, and the known-future covariates
of the 28 forecast days. Training: {STEPS['GRU']} steps of 512 random windows (Poisson loss; DeepAR:
negative-binomial likelihood), windows ending before the origin.

* Best overall: **{best}**. Best network: **{nn_best}** (backtest WRMSSE {tab.loc[nn_best, 'WRMSSE (backtest)']:.3f} vs
  LightGBM {tab.loc[lgb, 'WRMSSE (backtest)']:.3f}).
* The direct LSTM/GRU heads ("MQ-RNN" style) emit all 28 days at once from the encoder state plus the future
  covariates - the same direct strategy as the champion LightGBM; DeepAR forecasts one day at a time and feeds its
  own samples back, which is how it produces a full predictive distribution (used in Phase 15).
* On a laptop CPU the networks were trained for minutes, not the hours/GPU-days used by M5 neural entries; tabular
  boosting with good features is the stronger *and* cheaper choice at this scale - consistent with the M5 results,
  where the top entries were LightGBM-based and neural nets mostly helped inside ensembles.

**Does the complicated model beat the simple one?** Compare every row with the weekday-profile baseline above: a
model that cannot beat it on the backtest does not go to production, however modern it is.""")
    r.metrics = {"results": tab.to_dict(orient="index"), "cost": info}
    r.save()
    print(tab.round(3).to_string())


if __name__ == "__main__":
    main()

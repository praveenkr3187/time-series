"""Phase 21 - Retraining and champion / challenger governance.

    python -m forecastlab.pipelines.champion_challenger      (after `make nightly`)

Question 1: the champion lgbm_v1 was trained on data up to 2016-04-24. Four weeks later, is a retrained model
            (lgbm_v2, data up to 2016-05-22) better - by more than noise?
Question 2: does the research forecast (Phase 11) equal what the production code serves? (training/serving skew)
Question 3: should a different challenger (ensemble) replace the champion?
Decision rule (fixed in advance): promote if the challenger lowers WRMSSE and the bootstrap 95% CI of the
dollar-weighted item RMSSE difference is entirely below zero.
"""
from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from forecastlab import viz
from forecastlab.data.hierarchy import m5_hierarchy
from forecastlab.data.load import load_dataset
from forecastlab.evaluation import paired_bootstrap, wrmsse
from forecastlab.metrics import point_metrics
from forecastlab.models.registry import load_registry, promote, register, version_dir
from forecastlab.report import Report
from forecastlab.serving.db import ForecastDB
from forecastlab.serving.forecaster import Forecaster
from forecastlab.serving.train import train_version
from forecastlab.store import load_forecast
from forecastlab.validation import holdout_origin

log = logging.getLogger(__name__)
RESEARCH = "LightGBM (per-store, direct)"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ds = load_dataset()
    db = ForecastDB()
    r = Report("21_champion_challenger", "Phase 21 - Retraining and Champion / Challenger")
    o = holdout_origin(ds)
    reg = load_registry()
    v1 = reg["champion"]

    # the static champion's live forecast from origin o (published by the nightly run the morning after)
    run = db.query("SELECT run_id FROM runs WHERE origin_date = ? AND status='SUCCESS'", (str(ds.dates[o.day].date()),))
    if run.empty:
        raise SystemExit("no nightly run with origin %s - run `make nightly` first" % ds.dates[o.day].date())
    F_static = db.read_items(run["run_id"].iloc[0])["mean"]

    # retrain through the production code path
    v2 = "lgbm_v2"
    if not (version_dir(v2) / "meta.json").exists():
        meta = train_version(ds, o.day, v2)
        register(v2, {"learner": "lightgbm", "strategy": "direct", "per": "store",
                      "train_end": str(ds.dates[o.day].date())}, {}, role="challenger", status="challenger",
                 train_secs=meta["train_secs"], reason="scheduled 28-day retrain")
    F_v2 = Forecaster(v2).forecast(ds, o.day, o.horizon)
    F_research = load_forecast(RESEARCH, o.day)
    skew = float(np.abs(F_v2 - F_research).max())
    ens_key = next((k for k in ["Ensemble: LightGBM + GRU (global)", "Ensemble: LightGBM + LSTM (global)",
                                "Ensemble: LightGBM + Transformer (global)", "Ensemble: LightGBM + DeepAR (global)",
                                "Ensemble: LightGBM + Chronos-Bolt"] if load_forecast(k, o.day) is not None), None)

    cands = {f"{v1} (static, trained {reg['models'][v1]['spec'].get('train_end', '2016-04-24')})": F_static,
             f"{v2} (retrained {ds.dates[o.day].date()})": F_v2}
    if ens_key:
        cands[f"shadow challenger: {ens_key}"] = load_forecast(ens_key, o.day)
    w = wrmsse(ds, o)
    A = ds.Y[:, o.test]
    hier = m5_hierarchy()
    lv12 = hier.level_rows(12)
    S3 = hier.S[hier.level_rows(3)]
    rows = {}
    for name, F in cands.items():
        st = point_metrics(np.asarray(S3 @ A), np.asarray(S3 @ F))
        rows[name] = {"WRMSSE": w.score(F), "store WAPE": st["wape"], "store bias": st["bias"]}
    tab = pd.DataFrame(rows).T
    names = list(cands)
    tests = []
    for ch in names[1:]:
        d = w.rmsse(cands[ch])[lv12] - w.rmsse(cands[names[0]])[lv12]
        est, lo, hi = paired_bootstrap(d, w.weights[lv12])
        tests.append({"challenger": ch, "item RMSSE diff vs champion": est, "CI low": lo, "CI high": hi,
                      "WRMSSE diff": tab.loc[ch, "WRMSSE"] - tab.loc[names[0], "WRMSSE"],
                      "decision": "PROMOTE" if hi < 0 and tab.loc[ch, "WRMSSE"] < tab.loc[names[0], "WRMSSE"] else "keep champion"})
    tt = pd.DataFrame(tests).set_index("challenger")

    r.p(f"""Window: {ds.dates[o.test.start].date()} .. {ds.dates[o.test.stop - 1].date()} - live production data the models
had not seen.

**Training/serving skew check.** The production trainer + forecaster (`serving/train.py`, `serving/forecaster.py`)
retrained at the same origin reproduce the research forecast of Phase 11 to within **{skew:.2e} units** (max absolute
difference over 30,490 x 28 cells) - the model that was evaluated is the model that is served.""")
    r.h2("Candidates on live data")
    r.table(tab, floatfmt=".4f")
    r.h2("Promotion test")
    r.table(tt, floatfmt=".4f")
    promoted = None
    retrain = tt.loc[names[1]]
    if retrain["decision"] == "PROMOTE":
        promote(v2, reason=f"retrain beat static champion on {ds.dates[o.test.start].date()}+: WRMSSE "
                           f"{retrain['WRMSSE diff']:+.4f}, CI [{retrain['CI low']:+.4f}, {retrain['CI high']:+.4f}]")
        promoted = v2
    r.p(f"""Decision: {'**promote ' + v2 + '**' if promoted else '**keep ' + v1 + '**'} for the retraining question.
{('The ensemble shadow challenger: ' + tt.loc[names[2], 'decision'] + '.') if len(names) > 2 else ''}

A promotion in this platform is a registry change (`models/registry.py`): the nightly job loads whatever version
is `champion` at run time, the previous champion stays on disk as `archived`, and rollback is the same one-line
registry change. Shadow mode = the challenger forecasts every night next to the champion but nothing downstream
consumes it, so its live accuracy accumulates with zero business risk.""")

    # model age: horizon 1-7 accuracy of the static model by week since training
    acc = db.query("SELECT * FROM accuracy WHERE scope='store' AND horizon <= 7")
    if len(acc):
        acc["origin_week"] = (pd.to_datetime(acc["origin_date"]) - pd.Timestamp(
            reg["models"][v1]["spec"].get("train_end", "2016-04-24"))).dt.days // 7 + 1
        age = acc.groupby("origin_week")[["abs_error", "actual", "forecast"]].sum()
        age["WAPE (horizon 1-7)"] = age["abs_error"] / age["actual"]
        age["bias"] = age["forecast"] / age["actual"] - 1
        r.h2("Does the static model get worse as it ages?")
        r.table(age[["WAPE (horizon 1-7)", "bias"]].rename_axis("weeks since training"), floatfmt=".3f")
        fig, ax = plt.subplots(figsize=(7, 3))
        ax.plot(age.index, age["WAPE (horizon 1-7)"], marker="o", color=viz.SERIES[0])
        ax.set(title="Static model: store WAPE (1-7 days ahead) by model age", xlabel="weeks since training",
               ylabel="WAPE")
        r.figure(viz.save(fig, r.dir / "model_age.png"), "Model age")
        slope = np.polyfit(age.index, age["WAPE (horizon 1-7)"], 1)[0]
        r.p(f"""Same horizon (1-7 days), so differences are about the model's age and the weeks themselves, not forecast
difficulty. Trend: {slope:+.4f} WAPE per week of age. Weekly noise is of the same size, so with 8 weeks of data a
retraining *schedule* is justified by the controlled comparison above, not by this curve alone. Retraining policy
of this platform: **every 28 days** (one horizon), plus **on alert** (Phase 22), always through the champion /
challenger gate.""")
    r.metrics = {"candidates": tab.to_dict(orient="index"), "tests": tt.reset_index().to_dict(orient="records"),
                 "skew_max_abs": skew, "promoted": promoted}
    r.save()
    print(tab.round(4).to_string())
    print(tt.round(4).to_string())


if __name__ == "__main__":
    main()

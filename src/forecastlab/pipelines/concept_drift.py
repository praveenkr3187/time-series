"""Phase 22 - Concept drift: a competitor opens next to store CA_3.

    python -m forecastlab.pipelines.concept_drift

From 2016-05-09, CA_3's FOODS demand drops to 60% (simulated by binomial thinning of real sales, so the noise
stays realistic). The production model (trained 2016-04-24) does not know. We replay the nights for CA_3 and ask:

1. which monitoring signal notices first - and how many days of bad forecasts did that cost?
2. how do we adapt? nothing / online level correction / retrain / retrain with recency weights
3. why does a 28-day direct model need ~4 weeks to "see" a level shift through its features?
"""
from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from forecastlab import viz
from forecastlab.data.load import Dataset, load_dataset
from forecastlab.models import gbm
from forecastlab.models.features import FeatureBuilder
from forecastlab.models.registry import load_registry
from forecastlab.monitoring.drift import distribution, psi
from forecastlab.report import Report
from forecastlab.serving.forecaster import Forecaster

log = logging.getLogger(__name__)
STORE, CAT, FACTOR, SHIFT = "CA_3", "FOODS", 0.6, "2016-05-09"
H = 7  # evaluate the forecasts planners act on first: 1-7 days ahead


def shifted(ds: Dataset, shift_day: int) -> np.ndarray:
    rng = np.random.default_rng(0)
    Y = ds.Y.copy()
    m = (ds.series["store_id"].to_numpy() == STORE) & (ds.series["cat_id"].to_numpy() == CAT)
    Y[np.ix_(m, np.arange(shift_day, Y.shape[1]))] = rng.binomial(
        Y[np.ix_(m, np.arange(shift_day, Y.shape[1]))].astype(int), FACTOR)
    return Y


def cusum(z: np.ndarray, k: float = 0.5, h: float = 4.0) -> int | None:
    """Two-sided Page CUSUM on standardised errors; index of the first alarm."""
    hi = lo = 0.0
    for i, v in enumerate(z):
        hi, lo = max(0.0, hi + v - k), max(0.0, lo - v - k)
        if hi > h or lo > h:
            return i
    return None


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ds = load_dataset()
    r = Report("22_concept_drift", "Phase 22 - Concept Drift: Detect and Adapt")
    shift = ds.day(SHIFT)
    Y = shifted(ds, shift)
    dsm = Dataset(Y, ds.P, ds.series, ds.cal)
    rows = gbm.store_rows(ds)[STORE]
    foods = ds.series["cat_id"].to_numpy()[rows] == CAT
    reg = load_registry()
    version = "lgbm_v1" if "lgbm_v1" in reg["models"] else reg["champion"]   # the model trained before the shift
    fc = Forecaster(version)
    booster = fc.boosters[STORE]
    start, end = ds.day("2016-04-25"), ds.day("2016-06-12")   # origins (last day of data)

    fb = FeatureBuilder(dsm, rows, Y=Y[rows])
    nights = []
    for o in range(start, end + 1):
        visible = Y[rows].copy()
        visible[:, o + 1:] = 0
        fb.set_sales(visible)
        days = np.arange(o + 1, o + 1 + H)
        F = (np.clip(booster.predict(fb.frame(days, 28)), 0, None) * fb.on_shelf(days)).reshape(len(rows), H)
        nights.append({"origin": o, "F": F})
    log.info("replayed %d nights", len(nights))

    # --- monitoring signals on the stream of 1-day-ahead forecasts (what is known each morning)
    t_days = np.array([n["origin"] + 1 for n in nights])
    f1 = np.array([n["F"][foods, 0].sum() for n in nights])
    a1 = np.array([Y[rows][foods, t].sum() for t in t_days])
    err = (a1 - f1) / np.sqrt(np.maximum(f1, 1))
    pre = t_days < shift
    sd = err[pre].std()
    z = (err - err[pre].mean()) / sd
    bias7 = pd.Series(f1).rolling(7).sum() / pd.Series(a1).rolling(7).sum() - 1
    post_idx = np.flatnonzero(~pre)
    c_alarm = cusum(z[post_idx])
    b_alarm = next((i for i in post_idx if abs(bias7.iloc[i]) > 0.05), None)
    ref = distribution(ds.Y[rows][:, shift - 35: shift - 7])
    psi7 = [psi(ref, distribution(Y[rows][:, t - 6: t + 1])) for t in t_days]
    p_alarm = next((i for i in post_idx if psi7[i] > 0.1), None)
    det = pd.DataFrame({
        "signal": ["CUSUM on standardised daily errors (k=0.5, h=4)", "7-day bias > 5% (production rule)",
                   "PSI of 7-day unit distribution > 0.1"],
        "alarm date": [str(ds.dates[t_days[post_idx[c_alarm]]].date()) if c_alarm is not None else "none",
                       str(ds.dates[t_days[b_alarm]].date()) if b_alarm is not None else "none",
                       str(ds.dates[t_days[p_alarm]].date()) if p_alarm is not None else "none"],
        "days after shift": [c_alarm + 1 if c_alarm is not None else np.nan,
                             (b_alarm - post_idx[0] + 1) if b_alarm is not None else np.nan,
                             (p_alarm - post_idx[0] + 1) if p_alarm is not None else np.nan]}).set_index("signal")
    r.p(f"""Scenario: from **{SHIFT}** store {STORE}'s {CAT} demand falls to {FACTOR:.0%} (a competitor opens). The
production model `{version}` (trained before the shift) keeps forecasting the old level. Nights replayed
{ds.dates[start].date()} .. {ds.dates[end].date()}, CA_3 only.""")
    r.h2("1. Detection: which signal fires first?")
    r.table(det, floatfmt=".0f")
    r.p("""CUSUM accumulates small standardised errors and fires after a few days of consistent under-performance; the
7-day bias rule needs a week of data by construction; PSI looks at inputs only and sees the shift as soon as
enough low-sales days enter its window - but it would fire on harmless shifts too.""")

    # --- adaptation
    detect_day = t_days[post_idx[c_alarm]] if c_alarm is not None else shift + 7
    eval_origins = [n for n in nights if n["origin"] >= detect_day]

    def wape_on(get_F) -> float:
        e = a = 0.0
        for n in eval_origins:
            F = get_F(n)
            A = Y[rows][:, n["origin"] + 1: n["origin"] + 1 + H]
            e += np.abs(F[foods] - A[foods]).sum()
            a += A[foods].sum()
        return e / a

    def bias_on(get_F) -> float:
        f = a = 0.0
        for n in eval_origins:
            F = get_F(n)
            f += F[foods].sum()
            a += Y[rows][foods, n["origin"] + 1: n["origin"] + 1 + H].sum()
        return f / a - 1

    # (b) online level correction: scale each department by actual / forecast over the last 7 scored days
    depts = ds.series["dept_id"].to_numpy()[rows]
    by_t = {n["origin"] + 1: n["F"][:, 0] for n in nights}

    def corrected(n):
        o = n["origin"]
        ts = [t for t in range(o - 6, o + 1) if t in by_t]
        F = n["F"].copy()
        for d in np.unique(depts):
            m = depts == d
            fa = sum(by_t[t][m].sum() for t in ts)
            aa = sum(Y[rows][m, t].sum() for t in ts)
            F[m] *= np.clip(aa / max(fa, 1e-6), 0.3, 2.0) if fa > 0 else 1.0
        return F

    # (c)/(d) retrain at the detection day, without / with recency weights
    retrained = {}
    for label, weight in [("retrain at detection", 1.0), ("retrain at detection, last 56 days weighted x5", 5.0)]:
        fbt = FeatureBuilder(dsm, rows, Y=np.where(np.arange(Y.shape[1]) <= detect_day, Y[rows], 0))
        X, y = gbm.training_data(fbt, detect_day, 28, ds.day("2014-01-01"))
        w = None
        if weight != 1.0:
            dd = np.tile(np.arange(ds.day("2014-01-01"), detect_day + 1), len(rows))[
                fbt.on_shelf(np.arange(ds.day("2014-01-01"), detect_day + 1))]
            w = np.where(dd > detect_day - 56, weight, 1.0)
        retrained[label] = gbm.fit(X, y, sample_weight=w)
        log.info("%s trained", label)

    def with_model(m):
        def get(n):
            visible = Y[rows].copy()
            visible[:, n["origin"] + 1:] = 0
            fb.set_sales(visible)
            days = np.arange(n["origin"] + 1, n["origin"] + 1 + H)
            return (np.clip(m.predict(fb.frame(days, 28)), 0, None) * fb.on_shelf(days)).reshape(len(rows), H)
        return get

    strategies = {"do nothing (static model)": lambda n: n["F"], "online level correction (7-day ratio)": corrected}
    for label, m in retrained.items():
        cache = {}
        g = with_model(m)
        strategies[label] = lambda n, g=g, cache=cache: cache.setdefault(n["origin"], g(n))
    res = pd.DataFrame({k: {"WAPE (1-7 days ahead)": wape_on(f), "bias": bias_on(f)} for k, f in strategies.items()}).T
    r.h2(f"2. Adaptation (CA_3 FOODS items, origins {ds.dates[detect_day].date()} onwards)")
    r.table(res, floatfmt=".3f")
    r.p("""* **Retraining alone barely helps**: the direct model's sales features end 28 days before each target, so for
4 weeks after the shift they still describe the *old* world - the new trees see the same inputs. A model is only as
adaptive as its most recent input.
* **Online level correction** (multiply each department by its recent actual/forecast ratio) reacts within days and
removes most of the bias - cheap, transparent, and standard practice as a guard-rail. It must be bounded (here 0.3-2x)
and switched off once the model catches up, otherwise it fights the next genuine change.
* Recency weighting tells the trees that recent rows matter more, but cannot invent post-shift lag features either.
* The structural fix for fast adaptation is short-lag information: the bucketed strategy of Phase 11 (a min-lag-7
model for days 1-7) or a recursive model sees the new level after one week.""")

    fig, ax = plt.subplots(figsize=(10, 3.6))
    d = ds.dates[t_days]
    ax.plot(d, a1, color=viz.INK_2, linewidth=1.2, label="actual (CA_3 FOODS)")
    ax.plot(d, f1, color=viz.SERIES[0], label="static model, 1 day ahead")
    ax.plot(d, [corrected(n)[foods, 0].sum() for n in nights], color=viz.SERIES[2], label="with level correction")
    ax.axvline(ds.dates[shift], color=viz.STATUS["critical"], linestyle="--", linewidth=1, label="shift")
    ax.axvline(ds.dates[detect_day], color=viz.STATUS["warning"], linestyle=":", linewidth=1.5, label="CUSUM alarm")
    ax.set(title="Competitor opening: CA_3 FOODS units per day", ylabel="units")
    ax.legend(fontsize=8, loc="lower left")
    r.figure(viz.save(fig, r.dir / "drift.png"), "Drift")
    r.metrics = {"detection": det.reset_index().to_dict(orient="records"), "adaptation": res.to_dict(orient="index")}
    r.save()
    print(det.to_string())
    print(res.round(3).to_string())


if __name__ == "__main__":
    main()

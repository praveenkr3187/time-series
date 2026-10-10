"""Phase 15 - Probabilistic forecasts: P10 / P50 / P90 and prediction intervals.

    python -m forecastlab.pipelines.probabilistic

A planner needs "expected 82, likely between 65 and 105", not just 82. Methods compared on all 30,490 series:
negative-binomial around the champion mean, conformal intervals, DeepAR samples, Chronos quantiles; plus quantile
LightGBM on two stores. Conformal calibration only ever uses windows *before* the one being evaluated.
"""
from __future__ import annotations

import json
import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from forecastlab import viz
from forecastlab.config import artifacts_dir, load_config
from forecastlab.data.hierarchy import m5_hierarchy
from forecastlab.data.load import load_dataset
from forecastlab.metrics import interval_metrics, pinball
from forecastlab.models import gbm
from forecastlab.models.probabilistic import Conformal, nb_dispersion, nb_quantiles, recent_level
from forecastlab.report import Report
from forecastlab.store import cached, load_forecast, save_forecast
from forecastlab.validation import backtest_origins, holdout_origin

log = logging.getLogger(__name__)
CHAMP = "LightGBM (per-store, direct)"
QSTORES = ["CA_1", "TX_2"]


def score(A: np.ndarray, Q: np.ndarray, qs: list[float]) -> dict[str, float]:
    im = interval_metrics(A, Q[0], Q[-1])
    return {"coverage P10-P90 (target 0.80)": im["coverage"],
            "share below P10 (target 0.10)": float(np.mean(A < Q[0])),
            "share above P90 (target 0.10)": float(np.mean(A > Q[-1])),
            "mean width": im["mean_width"],
            "mean pinball loss": float(np.mean([pinball(A, Q[j], q) for j, q in enumerate(qs)]))}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ds = load_dataset()
    qs = load_config()["forecast"]["quantiles"]
    r = Report("15_probabilistic", "Phase 15 - Probabilistic Forecasts")
    folds = backtest_origins(ds)
    ho = holdout_origin(ds)
    evals = [(folds[-1], folds[:-1]), (ho, folds)]   # (window evaluated, windows used for calibration)
    res, store_res = [], []
    for o, calib in evals:
        A = ds.Y[:, o.test]
        H = ds.Y[:, : o.day + 1]
        F = load_forecast(CHAMP, o.day)
        alpha = nb_dispersion(H)
        Qnb = nb_quantiles(F, alpha, qs)
        for q, Qj in zip(qs, Qnb):
            save_forecast(f"{CHAMP} NB q{q}", o.day, Qj)
        conf = Conformal(qs).fit([(ds.Y[:, c.test], load_forecast(CHAMP, c.day), recent_level(ds.Y[:, : c.day + 1]))
                                  for c in calib])
        Qcf = conf.predict(F, recent_level(H))
        methods = {"Negative binomial around LightGBM mean": Qnb, "Conformal around LightGBM mean": Qcf}
        for name in ["DeepAR (global)", "Chronos-Bolt small"]:
            Qm = [load_forecast(f"{name} q{q}", o.day) for q in qs]
            if all(x is not None for x in Qm):
                methods[f"{name} quantiles"] = np.stack(Qm)
        for name, Q in methods.items():
            res.append({"window": o.name, "method": name, **score(A, Q, qs)})
        log.info("%s item-level done", o.name)

        # ---- store level: quantiles of a sum
        hier = m5_hierarchy()
        S = hier.S[hier.level_rows(3)]
        As, Fs = np.asarray(S @ A), np.asarray(S @ F)
        mu_var = F + alpha[:, None] * F ** 2
        sd = np.sqrt(np.asarray(S @ mu_var))
        from scipy import stats
        z = stats.norm.ppf(qs)
        Q_indep = np.stack([Fs + zq * sd for zq in z])
        Q_sum = np.stack([np.asarray(S @ Qnb[j]) for j in range(len(qs))])
        pairs = [(np.asarray(S @ ds.Y[:, c.test]), np.asarray(S @ load_forecast(CHAMP, c.day)),
                  np.full(S.shape[0], 1e6)) for c in calib]
        Q_conf = Conformal(qs).fit(pairs).predict(Fs, np.full(S.shape[0], 1e6))
        for name, Q in {"sum of item P10/P90 (wrong)": Q_sum,
                        "NB items assumed independent": Q_indep,
                        "conformal on store totals": Q_conf}.items():
            store_res.append({"window": o.name, "method": name, **score(As, Q, qs)})

    t = pd.DataFrame(res).set_index(["window", "method"])
    r.h2("Item level: 30,490 series x 28 days")
    r.table(t, floatfmt=".3f")
    hb = t.loc[ho.name]
    best = hb["mean pinball loss"].idxmin()
    r.p(f"""Fold 3 is calibrated on folds 1-2; the holdout on folds 1-3. **Pinball loss** is the proper score for a
quantile (lower is better, rewards both calibration and sharpness); **coverage** checks calibration alone.

* Best mean pinball loss on the holdout: **{best}**.
* Intervals for counts are discrete: with P10 = 0 for most slow movers, "share below P10" is necessarily < 10%
  (you cannot sell fewer than 0) - coverage above 80% at item level is expected, not a bug.
* The NB intervals cost nothing to compute nightly (one dispersion per series), which is why serving uses them.""")

    st = pd.DataFrame(store_res).set_index(["window", "method"])
    r.h2("Store level: the quantile of a sum is not the sum of quantiles")
    r.table(st, floatfmt=".3f")
    r.p(f"""Adding up 3,049 item P90s gives a store "P90" that almost never gets exceeded
(share above P90 = {st.loc[(ho.name, 'sum of item P10/P90 (wrong)'), 'share above P90 (target 0.10)']:.1%}) - it assumes every item
has a bad day at once. Assuming independence goes the other way: item errors are *positively correlated* (weather,
traffic, events hit all items together), so the independent-sum interval is too narrow. Calibrating directly on
store totals (conformal) is the honest fix; a full solution samples joint paths or reconciles quantiles (Phase 16).""")

    # ---- calibration used by the nightly job: how much wider than "independent items" must aggregates be?
    hier = m5_hierarchy()
    kappa = {}
    for level in range(1, 10):
        rows_l = hier.level_rows(level)
        S = hier.S[rows_l]
        num = den = 0.0
        for c in folds:
            Fc = load_forecast(CHAMP, c.day)
            al = nb_dispersion(ds.Y[:, : c.day + 1])
            num += float(((np.asarray(S @ ds.Y[:, c.test]) - np.asarray(S @ Fc)) ** 2).sum())
            den += float(np.asarray(S @ (Fc + al[:, None] * Fc ** 2)).sum())
        kappa[hier.keys["level_name"].iloc[rows_l[0]]] = num / den
    kappa["default"] = 1.0
    (artifacts_dir() / "models").mkdir(parents=True, exist_ok=True)
    (artifacts_dir() / "models" / "interval_calibration.json").write_text(json.dumps(kappa, indent=2))
    r.h2("Calibrating aggregate intervals for serving")
    r.table(pd.Series(kappa, name="variance inflation vs independent items").drop("default").to_frame(), floatfmt=".2f")
    r.p("""Ratio of the observed squared error of each aggregate (backtest folds 1-3) to the variance implied by
independent negative-binomial items. Values well above 1 quantify the positive correlation between items' errors.
The nightly job multiplies the independent variance by this factor to publish calibrated P10/P90 for stores,
departments and totals (`artifacts/models/interval_calibration.json`).""")
    r.metrics["aggregate_variance_inflation"] = kappa

    # ---- quantile LightGBM on two stores
    o = ho
    rows = np.concatenate([gbm.store_rows(ds)[s] for s in QSTORES])
    Ql = []
    for q in qs:
        def fit_q(q=q):
            full = np.full((len(ds.series), o.horizon), np.nan, dtype=np.float32)
            for s in QSTORES:
                sr = gbm.store_rows(ds)[s]
                full[sr] = gbm.forecast_store(ds, sr, o.day, o.horizon, objective="quantile", alpha=q,
                                              n_estimators=400)
            return full
        Ql.append(cached(f"LightGBM quantile q{q}", o.day, fit_q)[rows])
    Ql = np.sort(np.stack(Ql), axis=0)
    A = ds.Y[rows, o.test]
    Qnb = np.stack([load_forecast(f"{CHAMP} NB q{q}", o.day)[rows] for q in qs])
    qt = pd.DataFrame({"quantile LightGBM (3 models per store)": score(A, Ql, qs),
                       "NB around LightGBM mean": score(A, Qnb, qs)}).T
    r.h2(f"Quantile regression vs NB (stores {', '.join(QSTORES)}, holdout)")
    r.table(qt, floatfmt=".3f")
    r.p("""Quantile LightGBM learns each quantile directly with the pinball loss - no distributional assumption, but one
model per quantile (3x training), quantiles can cross (sorted here), and P10 of an intermittent item is usually
exactly 0. The NB route needs one model and gives *any* quantile on demand - the inventory phase asks for P95/P99.""")

    # ---- chart: one item
    o = ho
    F = load_forecast(CHAMP, o.day)
    Q = np.stack([load_forecast(f"{CHAMP} NB q{q}", o.day) for q in qs])
    i = int(np.argsort(ds.Y[:, o.day - 27: o.day + 1].mean(axis=1))[-200])
    fig, ax = plt.subplots(figsize=(9, 3.4))
    d0 = o.day - 41
    ax.plot(ds.dates[d0: o.test.stop], ds.Y[i, d0: o.test.stop], color=viz.INK_2, linewidth=1.2, label="actual")
    ax.fill_between(ds.dates[o.test], Q[0, i], Q[-1, i], color=viz.SERIES[0], alpha=0.18, label="P10-P90")
    ax.plot(ds.dates[o.test], F[i], color=viz.SERIES[0], label="mean forecast")
    ax.axvline(ds.dates[o.day], color=viz.MUTED, linestyle=":", linewidth=1)
    ax.set(title=f"{ds.series['id'][i]}: holdout forecast with 80% interval", ylabel="units")
    ax.legend(loc="upper left", fontsize=8)
    r.figure(viz.save(fig, r.dir / "interval_example.png"), "Interval example")
    r.metrics = {"item": t.reset_index().to_dict(orient="records"),
                 "store": st.reset_index().to_dict(orient="records"),
                 "quantile_lgbm": qt.to_dict(orient="index")}
    r.save()
    print(t.round(3).to_string())
    print(st.round(3).to_string())


if __name__ == "__main__":
    main()

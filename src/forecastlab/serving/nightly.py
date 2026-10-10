"""Phase 19 - The nightly batch forecasting job.

    python -m forecastlab.serving.nightly --simulate            # replay the production period night by night
    python -m forecastlab.serving.nightly --date 2016-05-02     # one night (expects the landing file to exist)

Every night, for run date D (yesterday = D-1):

    1. ingest   read every landed POS file not yet in the curated history (catch-up after a blocked night)
    2. validate data-quality gate (data/quality.py); any error -> run BLOCKED, yesterday's forecast stays live
    3. curate   append validated days to the curated history
    4. score    actuals for D-1 arrived -> score every earlier forecast that targeted D-1 (monitoring)
    5. forecast champion model, origin D-1, 28 days, mean + P10/P90 for every item-store
    6. publish  item forecasts (.npz) + coherent aggregates for 9 hierarchy levels (SQLite)
    7. monitor  input / prediction drift (PSI), accuracy and bias checks -> alerts
"""
from __future__ import annotations

import argparse
import json
import logging
import time

import numpy as np
import pandas as pd
from scipy import stats

from forecastlab.config import artifacts_dir, load_config
from forecastlab.data.hierarchy import m5_hierarchy
from forecastlab.data.lake import Curated, day_batch, land, landing_path, read_landing
from forecastlab.data.load import Dataset, load_dataset
from forecastlab.data.quality import check_batch, make_reference
from forecastlab.models.probabilistic import nb_dispersion, nb_quantiles
from forecastlab.models.registry import load_registry, version_dir
from forecastlab.monitoring.accuracy import score_target_day
from forecastlab.monitoring.drift import distribution, psi
from forecastlab.serving.db import ForecastDB
from forecastlab.serving.forecaster import Forecaster

log = logging.getLogger(__name__)
AGG_LEVELS = range(1, 10)  # total .. store x dept


def interval_calibration() -> dict[str, float]:
    p = artifacts_dir() / "models" / "interval_calibration.json"
    return json.loads(p.read_text()) if p.exists() else {}


class NightlyJob:
    def __init__(self, ds: Dataset | None = None, db: ForecastDB | None = None):
        self.ds = ds or load_dataset()
        self.db = db or ForecastDB()
        self.hier = m5_hierarchy()
        self.cfg = load_config()
        self.h = self.cfg["forecast"]["horizon"]
        self._forecaster: Forecaster | None = None
        self.kappa = interval_calibration()

    def forecaster(self) -> Forecaster:
        champion = load_registry()["champion"]
        if self._forecaster is None or self._forecaster.version != champion:
            self._forecaster = Forecaster(champion)
            log.info("loaded model %s", champion)
        return self._forecaster

    # ------------------------------------------------------------------ one night
    def run(self, run_date: pd.Timestamp) -> dict:
        ds, t0 = self.ds, time.time()
        run_day = ds.day(run_date)
        origin = run_day - 1
        run_id = f"run_{run_date.date()}"
        cur = Curated.load()
        issues: list[dict] = []
        status = "SUCCESS"
        # 1-3 ingest + validate + curate (catch up every missing day)
        for day in range(cur.last_day + 1, origin + 1):
            batch = read_landing(ds.dates[day])
            if batch is None:
                issues.append({"check": "missing_file", "severity": "error", "n_rows": 0,
                               "detail": f"no POS file landed for {ds.dates[day].date()}"})
                status = "BLOCKED"
                break
            ref = make_reference(cur.Y, ds.P, ds.series, day)
            found = [i.to_dict() for i in check_batch(batch, ds.dates[day], ref)]
            for i in found:
                i["detail"] = f"[{ds.dates[day].date()}] {i['detail']}"
            issues += found
            if any(i["severity"] == "error" for i in found):
                status = "BLOCKED"
                break
            cur.append(day, batch, ds.series)
            # 4 score every earlier forecast that targeted this day
            self.db.append("accuracy", score_target_day(self.db, cur.Y, ds.series, ds.dates, day, self.h))
        version = load_registry()["champion"]
        if status != "SUCCESS":
            run = {"run_id": run_id, "run_date": str(run_date.date()), "origin_date": str(ds.dates[origin].date()),
                   "model_version": version, "status": status, "n_series": 0, "duration_s": time.time() - t0}
            self.db.write_run(run, issues)
            self.db.append("alerts", pd.DataFrame([{"run_date": run["run_date"], "kind": "pipeline", "severity": "critical",
                                                    "message": "nightly run BLOCKED by data quality: " +
                                                    "; ".join(i["detail"] for i in issues if i["severity"] == "error")}]))
            log.warning("%s BLOCKED: %s", run_id, [i["check"] for i in issues if i["severity"] == "error"])
            return run

        # 5 forecast
        view = cur.view(ds)
        fc = self.forecaster()
        F = fc.forecast(view, origin, self.h)
        alpha = nb_dispersion(view.Y[:, : origin + 1])
        Q = nb_quantiles(F, alpha, [0.1, 0.9])
        # 6 publish
        self.db.write_items(run_id, F, Q[0], Q[1])
        self.db.write_aggregates(run_id, self._aggregates(F, alpha, origin))
        run = {"run_id": run_id, "run_date": str(run_date.date()), "origin_date": str(ds.dates[origin].date()),
               "model_version": fc.version, "status": status, "n_series": int(len(F)), "duration_s": time.time() - t0}
        self.db.write_run(run, issues)
        # 7 monitor
        self._monitor(run, view, origin, F)
        log.info("%s %s in %.0fs (%d warnings)", run_id, status, run["duration_s"], len(issues))
        return run

    def _aggregates(self, F: np.ndarray, alpha: np.ndarray, origin: int) -> pd.DataFrame:
        var = F + alpha[:, None] * F ** 2
        z = stats.norm.ppf(0.9)
        frames = []
        for level in AGG_LEVELS:
            rows = self.hier.level_rows(level)
            name = self.hier.keys["level_name"].iloc[rows[0]]
            S = self.hier.S[rows]
            m = np.asarray(S @ F)
            sd = np.sqrt(np.asarray(S @ var) * self.kappa.get(name, self.kappa.get("default", 1.0)))
            keys = self.hier.keys["key"].iloc[rows].to_numpy()
            for k in range(self.h):
                frames.append(pd.DataFrame({"level": name, "key": keys,
                                            "target_date": str(self.ds.dates[origin + 1 + k].date()), "horizon": k + 1,
                                            "mean": m[:, k], "p10": np.clip(m[:, k] - z * sd[:, k], 0, None),
                                            "p90": m[:, k] + z * sd[:, k]}))
        return pd.concat(frames, ignore_index=True)

    def _monitor(self, run: dict, view: Dataset, origin: int, F: np.ndarray) -> None:
        mcfg = self.cfg["monitoring"]
        start = self.ds.day(self.cfg["validation"]["production_start"])
        ref = distribution(view.Y[:, start - 28: start][view.on_shelf[:, start - 28: start]])
        cur = distribution(view.Y[:, origin - 6: origin + 1][view.on_shelf[:, origin - 6: origin + 1]])
        first = self.db.query("SELECT run_id FROM runs WHERE status='SUCCESS' ORDER BY origin_date LIMIT 1")
        first_items = self.db.read_items(first["run_id"].iloc[0]) if len(first) else None
        pred_ref = distribution(first_items["mean"][:, :7]) if first_items is not None else distribution(F[:, :7])
        rows = [{"run_date": run["run_date"], "metric": "psi_sales_7d_vs_reference", "value": psi(ref, cur)},
                {"run_date": run["run_date"], "metric": "psi_forecast_vs_first_run", "value": psi(pred_ref,
                                                                                               distribution(F[:, :7]))}]
        self.db.append("drift", pd.DataFrame(rows))
        alerts = []
        for r_ in rows:
            if r_["value"] > mcfg["psi_alert"]:
                alerts.append({"kind": "drift", "severity": "warning", "message": f"{r_['metric']} = {r_['value']:.3f}"})
        acc = self.db.query("SELECT * FROM accuracy WHERE scope='store' AND target_date > ?",
                            (str(self.ds.dates[origin - 7].date()),))
        if len(acc) and acc["actual"].sum() > 0:
            wape = acc["abs_error"].sum() / acc["actual"].sum()
            bias = acc["forecast"].sum() / acc["actual"].sum() - 1
            base = load_registry()["models"].get(run["model_version"], {}).get("metrics", {}) \
                .get("backtest_store_wape")
            self.db.append("drift", pd.DataFrame([
                {"run_date": run["run_date"], "metric": "store_wape_last7d", "value": wape},
                {"run_date": run["run_date"], "metric": "bias_last7d", "value": bias}]))
            if base is not None and wape > base + mcfg["wape_alert"]:
                alerts.append({"kind": "accuracy", "severity": "warning",
                               "message": f"store WAPE last 7 days {wape:.1%} vs backtest {base:.1%}"})
            if abs(bias) > mcfg["bias_alert"]:
                alerts.append({"kind": "bias", "severity": "warning", "message": f"bias last 7 days {bias:+.1%}"})
        if alerts:
            self.db.append("alerts", pd.DataFrame([{"run_date": run["run_date"], **a} for a in alerts]))


# ---------------------------------------------------------------------- simulation of the production period
def corrupt(batch: pd.DataFrame, scenario: str) -> pd.DataFrame:
    if scenario == "store_missing":
        return batch[batch["store_id"] != "TX_2"]
    if scenario == "double_load":
        return pd.concat([batch, batch.sample(2000, random_state=0)])
    raise ValueError(scenario)


def setup(ds: Dataset, start: int, db: ForecastDB) -> None:
    Curated.initialise(ds, start - 1)
    db.reset()
    for p in (artifacts_dir() / "lake" / "landing").rglob("pos.parquet"):
        p.unlink()


def simulate(ds: Dataset, start: str, end: str, incidents: dict[str, str]) -> pd.DataFrame:
    """Run every night from `start` to `end`. incidents: {file date: scenario} - that file arrives broken, and the
    data team re-delivers a correct file before the next night."""
    db = ForecastDB()
    s, e = ds.day(start), ds.day(end)
    setup(ds, s, db)
    job = NightlyJob(ds, db)
    if not (version_dir(load_registry()["champion"]) / "meta.json").exists():
        raise SystemExit("champion model files missing - run `make train-prod` first")
    out = []
    for run_day in range(s, e + 1):
        file_day = run_day - 1
        batch = day_batch(ds, file_day)
        scenario = incidents.get(str(ds.dates[file_day].date()))
        land(corrupt(batch, scenario) if scenario else batch, ds.dates[file_day])
        out.append(job.run(ds.dates[run_day]))
        if scenario:   # the data team fixes the file during the day
            land(batch, ds.dates[file_day])
            assert landing_path(ds.dates[file_day]).exists()
    return pd.DataFrame(out)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--simulate", action="store_true")
    ap.add_argument("--date")
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default="2016-06-19")
    a = ap.parse_args()
    ds = load_dataset()
    if a.simulate:
        start = a.start or load_config()["validation"]["production_start"]
        runs = simulate(ds, start, a.end, {"2016-05-09": "store_missing", "2016-06-01": "double_load"})
        print(runs[["run_id", "status", "model_version", "duration_s"]].to_string())
    elif a.date:
        print(NightlyJob(ds).run(pd.Timestamp(a.date)))


if __name__ == "__main__":
    main()

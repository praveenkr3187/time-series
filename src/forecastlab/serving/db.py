"""Forecast store: SQLite for run metadata, aggregates and monitoring; one compressed .npz per run for item level.

    runs        one row per nightly run (status SUCCESS / BLOCKED / FAILED, model version, timings)
    issues      data-quality findings per run
    aggregates  forecasts (mean, P10, P90) for store / dept / category / state / total, every run
    accuracy    errors of past forecasts once actuals arrive (by run origin, horizon, scope)
    drift       PSI of inputs and predictions per run
    alerts      monitoring alerts
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from forecastlab.config import ROOT, artifacts_dir, load_config

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (run_id TEXT PRIMARY KEY, run_date TEXT, origin_date TEXT, model_version TEXT,
    status TEXT, n_series INTEGER, n_issues INTEGER, duration_s REAL, created_at TEXT DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE IF NOT EXISTS issues (run_id TEXT, check_name TEXT, severity TEXT, detail TEXT, n_rows INTEGER);
CREATE TABLE IF NOT EXISTS aggregates (run_id TEXT, level TEXT, key TEXT, target_date TEXT, horizon INTEGER,
    mean REAL, p10 REAL, p90 REAL);
CREATE INDEX IF NOT EXISTS agg_idx ON aggregates (level, key, run_id);
CREATE TABLE IF NOT EXISTS accuracy (target_date TEXT, origin_date TEXT, horizon INTEGER, scope TEXT, key TEXT,
    abs_error REAL, actual REAL, forecast REAL, in_interval REAL, n INTEGER);
CREATE INDEX IF NOT EXISTS acc_idx ON accuracy (scope, target_date);
CREATE TABLE IF NOT EXISTS drift (run_date TEXT, metric TEXT, value REAL);
CREATE TABLE IF NOT EXISTS alerts (run_date TEXT, kind TEXT, severity TEXT, message TEXT);
"""


def db_path() -> Path:
    p = ROOT / load_config()["serving"]["forecast_db"]
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def runs_dir() -> Path:
    d = artifacts_dir() / "serving" / "runs"
    d.mkdir(parents=True, exist_ok=True)
    return d


class ForecastDB:
    def __init__(self, path: Path | None = None):
        self.path = path or db_path()
        self.con = sqlite3.connect(self.path, check_same_thread=False)
        self.con.executescript(SCHEMA)

    def reset(self) -> None:
        for t in ["runs", "issues", "aggregates", "accuracy", "drift", "alerts"]:
            self.con.execute(f"DELETE FROM {t}")
        self.con.commit()
        for f in runs_dir().glob("*.npz"):
            f.unlink()

    def write_run(self, run: dict, issues: list[dict]) -> None:
        self.con.execute("DELETE FROM runs WHERE run_id = ?", (run["run_id"],))
        self.con.execute("DELETE FROM issues WHERE run_id = ?", (run["run_id"],))
        self.con.execute("INSERT INTO runs (run_id, run_date, origin_date, model_version, status, n_series, n_issues, "
                         "duration_s) VALUES (?,?,?,?,?,?,?,?)",
                         (run["run_id"], run["run_date"], run["origin_date"], run["model_version"], run["status"],
                          run["n_series"], len(issues), run["duration_s"]))
        self.con.executemany("INSERT INTO issues VALUES (?,?,?,?,?)",
                             [(run["run_id"], i["check"], i["severity"], i["detail"], i["n_rows"]) for i in issues])
        self.con.commit()

    def write_items(self, run_id: str, mean: np.ndarray, p10: np.ndarray, p90: np.ndarray) -> None:
        np.savez_compressed(runs_dir() / f"{run_id}.npz", mean=mean.astype(np.float32),
                            p10=p10.astype(np.float32), p90=p90.astype(np.float32))

    def read_items(self, run_id: str) -> dict[str, np.ndarray] | None:
        p = runs_dir() / f"{run_id}.npz"
        if not p.exists():
            return None
        with np.load(p) as z:
            return {k: z[k] for k in z.files}

    def write_aggregates(self, run_id: str, df: pd.DataFrame) -> None:
        self.con.execute("DELETE FROM aggregates WHERE run_id = ?", (run_id,))
        df.assign(run_id=run_id)[["run_id", "level", "key", "target_date", "horizon", "mean", "p10", "p90"]] \
            .to_sql("aggregates", self.con, if_exists="append", index=False)
        self.con.commit()

    def append(self, table: str, df: pd.DataFrame) -> None:
        if len(df):
            df.to_sql(table, self.con, if_exists="append", index=False)
            self.con.commit()

    def query(self, sql: str, params: tuple = ()) -> pd.DataFrame:
        return pd.read_sql_query(sql, self.con, params=params)

    def latest_run(self, status: str = "SUCCESS") -> dict | None:
        df = self.query("SELECT * FROM runs WHERE status = ? ORDER BY origin_date DESC LIMIT 1", (status,))
        return None if df.empty else df.iloc[0].to_dict()

    def runs(self) -> pd.DataFrame:
        return self.query("SELECT * FROM runs ORDER BY run_date")


def dumps(x) -> str:
    return json.dumps(x, default=float)

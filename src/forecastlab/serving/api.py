"""Forecast API - serves what the nightly job published.

    uvicorn forecastlab.serving.api:app --port 8000

GET  /health                                  service + latest run + champion model
GET  /forecast/{store_id}/{item_id}           28-day item forecast (mean, P10, P90) from the latest (or given) run
GET  /forecast/aggregate?level=store&key=CA_3 aggregate forecast (store, department, category, state, total, ...)
GET  /runs                                    nightly run history with status
GET  /runs/{run_id}/issues                    data-quality findings of a run
GET  /monitoring                              live accuracy by week/horizon, drift, alerts
GET  /models                                  model registry (champion, challenger, history)
POST /explain/{store_id}/{item_id}            GenAI analyst explanation (optional ?question= for agentic mode)

Forecasts are precomputed in batch; the API never runs a model at request time, so latency is a file/DB read.
"""
from __future__ import annotations

import logging

import numpy as np
from fastapi import FastAPI, HTTPException, Query

from forecastlab.data.load import load_dataset
from forecastlab.models.registry import load_registry
from forecastlab.monitoring.accuracy import weekly_summary
from forecastlab.serving.db import ForecastDB

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
app = FastAPI(title="Retail Demand Forecasting Platform", version="1.0")
_state: dict = {}


def db() -> ForecastDB:
    if "db" not in _state:
        _state["db"] = ForecastDB()
    return _state["db"]


def ds():
    if "ds" not in _state:
        _state["ds"] = load_dataset()
    return _state["ds"]


def _run(run_id: str | None) -> dict:
    if run_id:
        df = db().query("SELECT * FROM runs WHERE run_id = ?", (run_id,))
        if df.empty:
            raise HTTPException(404, f"unknown run {run_id}")
        return df.iloc[0].to_dict()
    run = db().latest_run()
    if run is None:
        raise HTTPException(503, "no successful forecast run yet")
    return run


@app.get("/health")
def health():
    run = db().latest_run()
    reg = load_registry()
    return {"status": "ok" if run else "no_forecast", "latest_run": run and run["run_id"],
            "data_through": run and run["origin_date"], "champion": reg.get("champion"),
            "challenger": reg.get("challenger")}


@app.get("/forecast/aggregate")
def forecast_aggregate(level: str = Query("store"), key: str = Query("CA_1"), run_id: str | None = None):
    run = _run(run_id)
    df = db().query("SELECT target_date, horizon, mean, p10, p90 FROM aggregates WHERE run_id=? AND level=? AND key=? "
                    "ORDER BY horizon", (run["run_id"], level, key))
    if df.empty:
        raise HTTPException(404, f"no forecast for {level}={key}")
    return {"run_id": run["run_id"], "model_version": run["model_version"], "level": level, "key": key,
            "total": float(df["mean"].sum()), "forecast": df.round(2).to_dict(orient="records")}


@app.get("/forecast/{store_id}/{item_id}")
def forecast_item(store_id: str, item_id: str, run_id: str | None = None):
    run = _run(run_id)
    s = ds().series
    rows = np.flatnonzero((s["store_id"].to_numpy() == store_id) & (s["item_id"].to_numpy() == item_id))
    if not len(rows):
        raise HTTPException(404, f"unknown item/store {item_id}/{store_id}")
    items = db().read_items(run["run_id"])
    if items is None:
        raise HTTPException(410, f"item-level forecast for {run['run_id']} no longer retained")
    i = int(rows[0])
    o = ds().day(run["origin_date"])
    dates = ds().dates[o + 1: o + 1 + items["mean"].shape[1]]
    return {"run_id": run["run_id"], "model_version": run["model_version"], "data_through": run["origin_date"],
            "store_id": store_id, "item_id": item_id, "total": float(items["mean"][i].sum()),
            "forecast": [{"date": str(d.date()), "horizon": k + 1, "mean": round(float(items["mean"][i, k]), 3),
                          "p10": float(items["p10"][i, k]), "p90": float(items["p90"][i, k])}
                         for k, d in enumerate(dates)]}


@app.get("/runs")
def runs(limit: int = 60):
    return db().runs().tail(limit).to_dict(orient="records")


@app.get("/runs/{run_id}/issues")
def run_issues(run_id: str):
    return db().query("SELECT * FROM issues WHERE run_id = ?", (run_id,)).to_dict(orient="records")


@app.get("/monitoring")
def monitoring():
    ws = weekly_summary(db(), "store")
    if len(ws):
        ws["week"] = ws["week"].astype(str)
    return {"store_accuracy_by_week": ws.round(4).to_dict(orient="records") if len(ws) else [],
            "drift": db().query("SELECT * FROM drift ORDER BY run_date DESC LIMIT 60").to_dict(orient="records"),
            "alerts": db().query("SELECT * FROM alerts ORDER BY run_date DESC LIMIT 50").to_dict(orient="records")}


@app.get("/models")
def models():
    return load_registry()


@app.post("/explain/{store_id}/{item_id}")
def explain(store_id: str, item_id: str, question: str | None = None):
    from forecastlab.genai.analyst import ask, explain as explain_
    try:
        return ask(question, store_id, item_id) if question else explain_(store_id, item_id)
    except KeyError as e:
        raise HTTPException(404, str(e))

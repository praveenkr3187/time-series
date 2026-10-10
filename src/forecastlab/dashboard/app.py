"""Planner dashboard.

    streamlit run src/forecastlab/dashboard/app.py

Pages: forecast explorer (item x store), aggregates, monitoring, nightly runs & data quality, model registry,
AI analyst. Reads the forecast store written by the nightly job.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from forecastlab.data.load import load_dataset
from forecastlab.models.registry import load_registry
from forecastlab.monitoring.accuracy import weekly_summary
from forecastlab.serving.db import ForecastDB

st.set_page_config(page_title="Demand Forecasting", layout="wide")
BLUE, ORANGE, INK = "#2a78d6", "#eb6834", "#52514e"


@st.cache_resource
def resources():
    return load_dataset(), ForecastDB()


ds, db = resources()
runs = db.runs()
if runs.empty:
    st.warning("No nightly runs yet - run `make nightly`.")
    st.stop()
ok = runs[runs["status"] == "SUCCESS"]
page = st.sidebar.radio("Page", ["Forecast explorer", "Aggregates", "Monitoring", "Runs & data quality", "Models",
                                 "AI analyst"])
run_id = st.sidebar.selectbox("Forecast run", ok["run_id"].tolist()[::-1])
run = ok.set_index("run_id").loc[run_id]
origin = ds.day(run["origin_date"])
st.sidebar.caption(f"data through {run['origin_date']} - model {run['model_version']}")


def band(fig, x, lo, hi, name="P10-P90"):
    fig.add_trace(go.Scatter(x=list(x) + list(x)[::-1], y=list(hi) + list(lo)[::-1], fill="toself",
                             fillcolor="rgba(42,120,214,0.15)", line=dict(width=0), name=name, hoverinfo="skip"))


if page == "Forecast explorer":
    c1, c2 = st.columns(2)
    store = c1.selectbox("Store", sorted(ds.series["store_id"].unique()), index=2)
    items = ds.series.loc[ds.series["store_id"] == store, "item_id"]
    vol = ds.Y[items.index, origin - 27: origin + 1].sum(axis=1)
    item = c2.selectbox("Item (sorted by recent volume)", items.iloc[np.argsort(-vol)].tolist())
    i = int(ds.series.index[(ds.series["store_id"] == store) & (ds.series["item_id"] == item)][0])
    it = db.read_items(run_id)
    h = it["mean"].shape[1]
    hist = slice(origin - 55, origin + 1)
    fut = ds.dates[origin + 1: origin + 1 + h]
    fig = go.Figure()
    band(fig, fut, it["p10"][i], it["p90"][i])
    fig.add_trace(go.Scatter(x=ds.dates[hist], y=ds.Y[i, hist], name="sales", line=dict(color=INK)))
    fig.add_trace(go.Scatter(x=fut, y=it["mean"][i], name="forecast", line=dict(color=BLUE)))
    from forecastlab.data.lake import Curated
    known = min(origin + 1 + h, Curated.load().last_day + 1)   # actuals that have arrived since this run
    if known > origin + 1:
        fig.add_trace(go.Scatter(x=ds.dates[origin + 1: known], y=ds.Y[i, origin + 1: known], name="actual (later)",
                                 line=dict(color=ORANGE, dash="dot")))
    fig.update_layout(height=420, margin=dict(t=30), title=f"{item} @ {store}")
    st.plotly_chart(fig, use_container_width=True)
    m = st.columns(4)
    m[0].metric("Next 28 days", f"{it['mean'][i].sum():,.0f} units")
    m[1].metric("Last 28 days", f"{ds.Y[i, origin - 27: origin + 1].sum():,.0f} units")
    m[2].metric("Price", f"${ds.P[i, origin]:.2f}")
    m[3].metric("Zero-sales days (8 wk)", int((ds.Y[i, hist] == 0).sum()))
    if st.checkbox("Show model drivers (SHAP)"):
        from forecastlab.data.lake import Curated
        from forecastlab.serving.forecaster import Forecaster
        Y = Curated.load().Y.copy()
        Y[:, origin + 1:] = 0
        f = Forecaster(run["model_version"]).explain(Curated(Y, origin).view(ds), i, origin, h)
        st.caption("Multiplicative driver factors per day (1.10 = driver raises that day's forecast by 10%).")
        st.dataframe(f.round(3), use_container_width=True)

elif page == "Aggregates":
    levels = db.query("SELECT DISTINCT level FROM aggregates WHERE run_id=?", (run_id,))["level"].tolist()
    c1, c2 = st.columns(2)
    level = c1.selectbox("Level", levels, index=levels.index("store") if "store" in levels else 0)
    keys = db.query("SELECT DISTINCT key FROM aggregates WHERE run_id=? AND level=?", (run_id, level))["key"].tolist()
    key = c2.selectbox("Key", keys)
    df = db.query("SELECT * FROM aggregates WHERE run_id=? AND level=? AND key=? ORDER BY horizon", (run_id, level, key))
    df["target_date"] = pd.to_datetime(df["target_date"])
    fig = go.Figure()
    band(fig, df["target_date"], df["p10"], df["p90"])
    fig.add_trace(go.Scatter(x=df["target_date"], y=df["mean"], name="forecast", line=dict(color=BLUE)))
    fig.update_layout(height=420, margin=dict(t=30), title=f"{level}: {key}")
    st.plotly_chart(fig, use_container_width=True)
    st.metric("Next 28 days", f"{df['mean'].sum():,.0f} units")

elif page == "Monitoring":
    ws = weekly_summary(db, "store")
    if ws.empty:
        st.info("No actuals scored yet.")
    else:
        st.subheader("Store-level WAPE by week and horizon")
        st.dataframe(ws.pivot(index="week", columns="horizon_bucket", values="wape").style.format("{:.1%}"))
        b = ws.groupby("week")[["abs_error", "actual", "forecast"]].sum()
        fig = go.Figure(go.Bar(x=b.index, y=b["forecast"] / b["actual"] - 1, marker_color=BLUE))
        fig.update_layout(title="Bias by week", height=300, yaxis_tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)
    d = db.query("SELECT * FROM drift")
    if len(d):
        piv = d.pivot_table(index="run_date", columns="metric", values="value")
        st.subheader("Drift and rolling accuracy")
        st.line_chart(piv)
    st.subheader("Alerts")
    st.dataframe(db.query("SELECT * FROM alerts ORDER BY run_date DESC"), use_container_width=True)

elif page == "Runs & data quality":
    st.dataframe(runs, use_container_width=True)
    sel = st.selectbox("Issues for run", runs["run_id"].tolist()[::-1])
    st.dataframe(db.query("SELECT * FROM issues WHERE run_id=?", (sel,)), use_container_width=True)

elif page == "Models":
    reg = load_registry()
    st.write(f"**Champion:** {reg.get('champion')} - **Challenger:** {reg.get('challenger')}")
    st.dataframe(pd.DataFrame(reg["models"]).T, use_container_width=True)
    st.subheader("History")
    st.dataframe(pd.DataFrame(reg.get("history", [])), use_container_width=True)

elif page == "AI analyst":
    from forecastlab.genai.analyst import ask, explain
    from forecastlab.genai.evidence import EvidenceService
    c1, c2 = st.columns(2)
    store = c1.selectbox("Store", sorted(ds.series["store_id"].unique()), index=2)
    item = c2.text_input("Item", "FOODS_3_090")
    q = st.text_input("Question (optional - lets Claude choose which evidence to look up)", "")
    if st.button("Explain"):
        with st.spinner("Gathering evidence..."):
            ev = EvidenceService(run_id)
            out = ask(q, store, item, ev) if q else explain(store, item, ev)
        st.subheader(out["headline"])
        st.write(out["summary"])
        st.write("**Drivers**")
        st.table(pd.DataFrame(out["drivers"]))
        st.write("**Risks**")
        for x in out["risks"]:
            st.write("- " + x)
        st.write("**Recommended actions**")
        for x in out["recommended_actions"]:
            st.write("- " + x)
        st.caption(f"confidence {out['confidence']} - {out.get('generated_by', '')}")

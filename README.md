# Retail Demand Forecasting Platform

An end-to-end, production-style demand forecasting platform built on the **M5 / Walmart** dataset: 30,490 daily
item x store series (3,049 products x 10 stores in CA, TX, WI), 2011-01-29 .. 2016-06-19, with weekly prices, an
event calendar and SNAP (food-stamp) days.

The business question, every night: **how many units of each product will each store sell over the next 28 days**,
with a P10-P90 range, coherent totals for every store / department / category / state, monitored as actuals arrive.

EDA -> data quality -> decomposition & stationarity -> validation design & leakage -> baselines & metrics ->
exponential smoothing -> ARIMA/SARIMA/ARIMAX -> Prophet -> intermittent-demand methods -> feature engineering ->
global LightGBM/XGBoost -> LSTM/GRU/Transformer/DeepAR -> foundation models (Chronos) -> backtesting & model
selection -> probabilistic forecasts -> hierarchical reconciliation -> inventory simulation -> explainability ->
nightly batch pipeline -> monitoring -> retraining & champion/challenger -> concept drift -> GenAI analyst.

Every phase is a runnable pipeline that writes a report with real numbers to `reports/<phase>/README.md`.
**Start with [`reports/RESULTS.md`](reports/RESULTS.md)** for the headline results.

```
   POS files (daily) ──► landing zone ──► data-quality gate ──BLOCK──► alert, keep yesterday's forecast
                                               │ pass
                                               ▼
                                     curated sales history ──► score yesterday's forecasts (monitoring)
                                               │
                     ┌─────────────────────────▼──────────────────────────┐
                     │  champion model (registry)                          │
                     │  10 per-store global LightGBMs, Tweedie, direct 28d │
                     │  horizon-aware features (sales <= t-28,             │
                     │  calendar / events / SNAP / price plan at t)        │
                     │  NB intervals (P10/P90), calibrated aggregates      │
                     └─────────────────────────┬──────────────────────────┘
                                               ▼
                 forecast store: item .npz per run + SQLite (runs, aggregates, accuracy, drift, alerts)
                          │                     │                         │
                     Forecast API          Planner dashboard        GenAI analyst (Claude)
                          │
               retrain every 28 days / on alert ──► challenger ──► bootstrap gate ──► promote / keep
```

## The learning rule

One production problem is the backdrop for every concept: stationarity is tested on these sales, SARIMA is fitted
to these stores, LightGBM and the LSTM forecast the same items, and MLOps deploys this same system. Each model is
asked the same question: **does the complicated model actually beat the simple one?** (Phase 14).

## Data

Download M5 (`calendar.csv`, `sell_prices.csv`, `sales_train_evaluation.csv`, `sales_test_evaluation.csv`) into
`m5-forecasting/`. The Kaggle competition files work; this repo was built from the public mirror used by Nixtla:

```bash
mkdir -p m5-forecasting && cd m5-forecasting
curl -L -o m5.zip https://github.com/Nixtla/m5-forecasts/raw/main/datasets/m5.zip && unzip m5.zip && rm m5.zip
```

| Brief | M5 |
|---|---|
| `store_id`, `product_id` | `store_id` (10), `item_id` (3,049) + dept / category / state hierarchy |
| `units_sold` | daily unit sales |
| `price` | weekly `sell_price` (known in advance = planned price schedule); no price = not on the shelf |
| `promotion` / `discount` | not recorded - proxied by price relative to its own recent maximum |
| `holiday` | 30 named events (sporting, cultural, national, religious) |
| external factors | SNAP days per state; no weather / competitor data |
| `inventory` | not recorded - stock-outs are *inferred* from implausible zero runs (Phase 2) |

## Running it

Requirements: Python 3.11 via [uv](https://docs.astral.sh/uv/); on Intel macOS `brew install libomp`.
Docker only for the serving stack. 16 GB RAM is enough.

```bash
make setup            # uv sync (+ OpenMP fix for torch + LightGBM on macOS)
make all              # data -> every research phase -> production simulation (~5-6 h on a 6-core laptop)
make test             # unit tests (synthetic data) + integration tests (real artifacts, API round trip)
```

Individual phases: `make eda | quality | decomposition | validation | baselines | ets | arima | prophet |
intermittent | features | global_ml | deep | foundation | backtest | probabilistic | hierarchy | inventory |
train-prod | explain | nightly | batch-report | challenger | drift | genai` (see `Makefile`). Every model's
forecasts are cached in `artifacts/forecasts/`, so re-running a report is fast.

### Serving

```bash
make nightly          # replay every night 2016-04-25 .. 2016-06-19 (two broken POS files are injected)
make api              # FastAPI on :8000 - docs at http://localhost:8000/docs
make dashboard        # planner dashboard on :8501
```

```bash
curl -s localhost:8000/forecast/CA_3/FOODS_3_090 | jq '.forecast[:3]'
curl -s 'localhost:8000/forecast/aggregate?level=store%20x%20department&key=CA_3/FOODS_3' | jq .total
curl -s -X POST 'localhost:8000/explain/TX_2/FOODS_3_090?question=Should%20I%20order%20extra%20for%20the%20events%3F'
```

Endpoints: `GET /health`, `GET /forecast/{store}/{item}`, `GET /forecast/aggregate`, `GET /runs`,
`GET /runs/{id}/issues`, `GET /monitoring`, `GET /models`, `POST /explain/{store}/{item}`.

Docker: `make stack-up` starts the API, the dashboard and a scheduler that runs the nightly job (demo mode: one
simulated night per minute).

### GenAI analyst

Set `ANTHROPIC_API_KEY` (or log in with the `ant` CLI). Without credentials the analyst returns a deterministic
template explanation built from the same evidence.

```bash
uv run python -m forecastlab.genai.analyst CA_3 FOODS_3_090
uv run python -m forecastlab.genai.analyst TX_2 FOODS_3_090 --ask "Should I order extra for the events?"
```

The model defaults to `claude-opus-5-5` (`FORECASTLAB_LLM_MODEL` to change) with structured JSON output and
server-side refusal fallback enabled (`fallbacks: "default"`).

## Evaluation design

| Window | Dates (train end -> test) | Used for |
|---|---|---|
| backtest folds 1-3 | origins 2016-01-31, 2016-02-28, 2016-03-27; 28 days each | every model comparison and choice |
| holdout | origin 2016-05-22 -> 2016-05-23 .. 2016-06-19 (M5 evaluation window) | reported once, never used to choose |
| production | nightly runs 2016-04-25 .. 2016-06-19 | monitoring, retraining, drift |

Primary metric: M5's **WRMSSE** (all 42,840 hierarchy series, dollar-weighted); our implementation reproduces the
official M5 benchmark scores exactly (seasonal naive 0.847, naive 1.752). Business reporting: WAPE, bias, fill rate,
inventory cost.

## Code map

```
src/forecastlab/
  data/        load.py (CSV -> wide arrays), hierarchy.py (12-level summing matrix), quality.py (gate, stock-outs),
               lake.py (landing zone + curated history)
  metrics.py   MAE/RMSE/WAPE/bias/MAPE/sMAPE/MASE/RMSSE, pinball, WRMSSE
  validation.py, evaluation.py   origins, walk-forward, sample, scorecard, bootstrap / Diebold-Mariano
  models/      baselines.py (vectorised naive..TSB), classical.py (statsforecast, Prophet), features.py (horizon-aware,
               look-ahead guard), gbm.py (LightGBM/XGBoost, direct/bucketed/recursive), deep.py (LSTM/GRU/Transformer/
               DeepAR), foundation.py (Chronos), probabilistic.py (NB, conformal), reconcile.py (BU/TD/OLS/WLS/MinT),
               explain.py (SHAP drivers), registry.py
  inventory/   sim.py (order-up-to simulation, critical fractile)
  serving/     train.py, forecaster.py, nightly.py, db.py, api.py, scheduler.py
  monitoring/  accuracy.py, drift.py (PSI)
  genai/       evidence.py, analyst.py (single call + tool-runner agent)
  dashboard/   app.py (Streamlit)
  pipelines/   one module per phase -> reports/<phase>/README.md
```

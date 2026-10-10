PY := uv run python -m

.PHONY: setup data eda quality decomposition validation baselines ets arima prophet intermittent features global_ml \
        deep foundation backtest probabilistic hierarchy inventory train-prod explain nightly batch-report \
        challenger drift genai results research production all api dashboard test stack-up stack-down

setup:            ## install dependencies (Intel macOS also needs: brew install libomp)
	uv sync
	./scripts/fix_macos_openmp.sh

# ---- Stage 1: data, EDA, quality, time-series basics
data:             ## M5 CSVs -> artifacts/data (wide numpy matrices)
	$(PY) forecastlab.data.load
eda:              ## Phase 1
	$(PY) forecastlab.pipelines.eda
quality:          ## Phase 2 (writes stock-out labels)
	$(PY) forecastlab.pipelines.data_quality
decomposition:    ## Phase 3
	$(PY) forecastlab.pipelines.decomposition
validation:       ## Phase 4: why you never shuffle a time series
	$(PY) forecastlab.pipelines.validation

# ---- Stage 2: the model ladder
baselines:        ## Phase 5
	$(PY) forecastlab.pipelines.baselines
ets:              ## Phase 6
	$(PY) forecastlab.pipelines.classical ets
arima:            ## Phase 7 (slow: ~1 h, per-series AutoARIMA)
	$(PY) forecastlab.pipelines.classical arima
prophet:          ## Phase 8
	$(PY) forecastlab.pipelines.classical prophet
intermittent:     ## Phase 9
	$(PY) forecastlab.pipelines.classical intermittent
features:         ## Phase 10
	$(PY) forecastlab.pipelines.features
global_ml:        ## Phase 11 (LightGBM per store, 4 origins: ~1.5 h)
	$(PY) forecastlab.pipelines.global_ml
deep:             ## Phase 12
	$(PY) forecastlab.pipelines.deep_learning
foundation:       ## Phase 13 (downloads Chronos models from Hugging Face)
	$(PY) forecastlab.pipelines.foundation
backtest:         ## Phase 14: compare the ladder, choose + register the champion
	$(PY) forecastlab.pipelines.backtest

# ---- Stage 3: uncertainty, hierarchy, business value, explainability
probabilistic:    ## Phase 15
	$(PY) forecastlab.pipelines.probabilistic
hierarchy:        ## Phase 16
	$(PY) forecastlab.pipelines.hierarchy
inventory:        ## Phase 17
	$(PY) forecastlab.pipelines.inventory
train-prod:       ## train the production champion lgbm_v1 on data up to the day before production starts
	$(PY) forecastlab.serving.train --version lgbm_v1 --origin 2016-04-24 --role champion
explain:          ## Phase 18
	$(PY) forecastlab.pipelines.explainability

# ---- Stage 4: production
nightly:          ## Phase 19: replay every night 2016-04-25 .. 2016-06-19 (land -> validate -> forecast -> monitor)
	$(PY) forecastlab.serving.nightly --simulate
batch-report:     ## Phases 19-20 reports
	$(PY) forecastlab.pipelines.batch_report
challenger:       ## Phase 21: retrain + champion/challenger gate
	$(PY) forecastlab.pipelines.champion_challenger
drift:            ## Phase 22
	$(PY) forecastlab.pipelines.concept_drift
genai:            ## Phase 23
	$(PY) forecastlab.pipelines.genai

results:          ## reports/RESULTS.md from every phase's metrics
	$(PY) forecastlab.pipelines.results

research: eda quality decomposition validation baselines ets intermittent arima prophet features global_ml deep \
          foundation backtest probabilistic hierarchy inventory
production: train-prod explain nightly batch-report challenger drift genai results
all: data research production

api:              ## forecast API on :8000
	uv run uvicorn forecastlab.serving.api:app --port 8000
dashboard:        ## planner dashboard on :8501
	uv run streamlit run src/forecastlab/dashboard/app.py

stack-up:         ## API + dashboard + nightly scheduler in Docker (needs ./artifacts from `make all`)
	docker compose up -d --build
stack-down:
	docker compose down

test:
	uv run pytest -q

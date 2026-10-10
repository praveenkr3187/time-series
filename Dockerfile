FROM python:3.11-slim

# libgomp: OpenMP runtime for LightGBM / XGBoost
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.8 /uv /bin/uv

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
COPY configs ./configs
RUN uv sync --frozen --no-dev

ENV PATH=/app/.venv/bin:$PATH FORECASTLAB_ROOT=/app PYTHONUNBUFFERED=1
# Data arrays, models, the data lake and the forecast DB are mounted at /app/artifacts (built by `make all`).
EXPOSE 8000 8501
CMD ["uvicorn", "forecastlab.serving.api:app", "--host", "0.0.0.0", "--port", "8000"]

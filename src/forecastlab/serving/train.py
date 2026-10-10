"""Train a production model version: one LightGBM booster per store, direct 28-day strategy.

    python -m forecastlab.serving.train --version lgbm_v1 --origin 2016-04-24
"""
from __future__ import annotations

import argparse
import json
import logging
import time

import numpy as np

from forecastlab.data.load import Dataset, load_dataset, model_start_day
from forecastlab.models import gbm
from forecastlab.models.features import FeatureBuilder
from forecastlab.models.registry import register, version_dir

log = logging.getLogger(__name__)
MIN_LAG = 28


def train_version(ds: Dataset, origin: int, version: str, stores: list[str] | None = None,
                  recency_weight: float = 1.0, recency_days: int = 56, **overrides) -> dict:
    """Train on model_start .. origin. recency_weight > 1 up-weights the last `recency_days` (drift response)."""
    out = version_dir(version)
    meta = {"version": version, "train_end": str(ds.dates[origin].date()), "min_lag": MIN_LAG, "stores": {},
            "recency_weight": recency_weight}
    t_all = time.time()
    for store, rows in gbm.store_rows(ds).items():
        if stores and store not in stores:
            continue
        t0 = time.time()
        fb = FeatureBuilder(ds, rows)
        start = model_start_day(ds)
        X, y = gbm.training_data(fb, origin, MIN_LAG, start)
        w = None
        if recency_weight != 1.0:
            days = np.tile(np.arange(start, origin + 1), len(rows))[fb.on_shelf(np.arange(start, origin + 1))]
            w = np.where(days > origin - recency_days, recency_weight, 1.0)
        m = gbm.fit(X, y, sample_weight=w, **overrides)
        m.booster_.save_model(str(out / f"{store}.txt"))
        meta["stores"][store] = {"rows": int(len(y)), "secs": round(time.time() - t0, 1)}
        log.info("  %s %s trained on %d rows (%.0fs)", version, store, len(y), time.time() - t0)
    meta["features"] = list(X.columns)
    meta["train_secs"] = round(time.time() - t_all, 1)
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", required=True)
    ap.add_argument("--origin", required=True, help="last training day, YYYY-MM-DD")
    ap.add_argument("--role", default=None, choices=[None, "champion", "challenger"])
    a = ap.parse_args()
    ds = load_dataset()
    meta = train_version(ds, ds.day(a.origin), a.version)
    register(a.version, {"learner": "lightgbm", "strategy": "direct", "per": "store", "train_end": a.origin},
             {}, role=a.role, status=a.role or "candidate", train_secs=meta["train_secs"])


if __name__ == "__main__":
    main()

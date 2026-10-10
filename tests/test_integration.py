"""Integration tests against the real artifacts (skipped until `make all` has produced them)."""


import numpy as np
import pytest

from forecastlab.config import ROOT

ART = ROOT / "artifacts"
pytestmark = pytest.mark.skipif(not (ART / "data" / "sales.npy").exists(), reason="run `make data` first")


def test_wrmsse_reproduces_official_m5_benchmarks():
    from forecastlab.data.load import load_dataset
    from forecastlab.metrics import WRMSSE
    from forecastlab.models.baselines import naive, seasonal_naive
    ds = load_dataset()
    o = ds.day("2016-05-22")
    w = WRMSSE(ds, o)
    H = ds.Y[:, : o + 1]
    assert w.score(seasonal_naive(H, 28)) == pytest.approx(0.847, abs=0.0005)
    assert w.score(naive(H, 28)) == pytest.approx(1.752, abs=0.0005)


@pytest.mark.skipif(not (ART / "serving" / "forecasts.sqlite").exists(), reason="run `make nightly` first")
def test_api_round_trip():
    from fastapi.testclient import TestClient

    from forecastlab.serving.api import app
    c = TestClient(app)
    h = c.get("/health").json()
    assert h["status"] == "ok" and h["latest_run"]
    f = c.get("/forecast/CA_3/FOODS_3_090").json()
    assert len(f["forecast"]) == 28 and all(d["p10"] <= d["mean"] + 1 and d["mean"] <= d["p90"] + 1e-6 or d["mean"] < 1
                                            for d in f["forecast"])
    a = c.get("/forecast/aggregate", params={"level": "store", "key": "CA_3"}).json()
    assert len(a["forecast"]) == 28 and a["total"] > 0
    assert c.get("/forecast/CA_3/NOT_AN_ITEM").status_code == 404
    runs = c.get("/runs").json()
    assert {"SUCCESS", "BLOCKED"} <= {r["status"] for r in runs}


@pytest.mark.skipif(not (ART / "serving" / "forecasts.sqlite").exists(), reason="run `make nightly` first")
def test_published_aggregates_are_coherent_with_items():
    from forecastlab.data.load import load_dataset
    from forecastlab.serving.db import ForecastDB
    ds, db = load_dataset(), ForecastDB()
    run = db.latest_run()
    items = db.read_items(run["run_id"])
    agg = db.query("SELECT key, horizon, mean FROM aggregates WHERE run_id=? AND level='store'", (run["run_id"],))
    for store, g in agg.groupby("key"):
        m = ds.series["store_id"].to_numpy() == store
        assert np.allclose(g.sort_values("horizon")["mean"].to_numpy(), items["mean"][m].sum(axis=0), rtol=1e-4)


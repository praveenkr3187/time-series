import numpy as np
import pandas as pd
import pytest

from forecastlab.data.load import Dataset


@pytest.fixture
def tiny() -> Dataset:
    """4 items x 2 stores, 500 days, weekly seasonality + Poisson noise, one late launch."""
    rng = np.random.default_rng(0)
    T = 500
    dates = pd.date_range("2014-01-04", periods=T, freq="D")
    cal = pd.DataFrame({"date": dates, "wm_yr_wk": 11400 + np.arange(T) // 7, "wday": (np.arange(T) % 7) + 1,
                        "month": dates.month, "year": dates.year, "event_name_1": "", "event_type_1": "",
                        "event_name_2": "", "event_type_2": "", "snap_CA": (dates.day <= 10).astype(int),
                        "snap_TX": (dates.day <= 10).astype(int), "snap_WI": 0, "d": np.arange(T)})
    cal.loc[100, ["event_name_1", "event_type_1"]] = ["SuperBowl", "Sporting"]
    rows = []
    for store, state in [("CA_1", "CA"), ("TX_1", "TX")]:
        for k, (cat, dept) in enumerate([("FOODS", "FOODS_1"), ("FOODS", "FOODS_1"), ("HOBBIES", "HOBBIES_1"),
                                          ("HOBBIES", "HOBBIES_1")]):
            item = f"{dept}_{k:03d}"
            rows.append({"id": f"{item}_{store}", "item_id": item, "dept_id": dept, "cat_id": cat,
                         "store_id": store, "state_id": state})
    series = pd.DataFrame(rows)
    level = np.array([5, 2, 1, 0.3] * 2)[:, None]
    week = np.array([1.3, 1.2, 0.9, 0.8, 0.8, 0.9, 1.1])[np.arange(T) % 7][None, :]
    Y = rng.poisson(level * week).astype(np.float32)
    P = np.full(Y.shape, 2.5, dtype=np.float32)
    P[3, :200] = np.nan
    Y[3, :200] = 0
    return Dataset(Y, P, series, cal)

"""Fast unit tests on synthetic data - no artifacts needed."""
import numpy as np
import pandas as pd
import pytest

from forecastlab.data.hierarchy import build_hierarchy
from forecastlab.data.quality import STOCKOUT, check_batch, make_reference, suspected_stockouts
from forecastlab.inventory.sim import Costs, order_up_to, simulate, summarise
from forecastlab.metrics import WRMSSE, pinball, point_metrics
from forecastlab.models.baselines import croston, seasonal_naive, ses, syntetos_boylan_class, tsb, weekday_profile
from forecastlab.models.features import FeatureBuilder, assert_no_lookahead, lags_for
from forecastlab.models.probabilistic import Conformal, nb_quantiles, nb_sum_quantiles
from forecastlab.models.reconcile import is_coherent, reconcile
from forecastlab.monitoring.drift import distribution, psi


def test_point_metrics_worked_example():
    m = point_metrics(np.array([[100.0]]), np.array([[90.0]]))
    assert m["mae"] == 10 and m["rmse"] == 10 and m["mape"] == pytest.approx(0.1) and m["bias"] == pytest.approx(-0.1)


def test_mape_drops_zero_actuals():
    m = point_metrics(np.array([[0.0, 2.0]]), np.array([[1.0, 2.0]]))
    assert m["mape"] == 0 and m["mape_cells_dropped"] == 0.5


def test_pinball_is_asymmetric():
    assert pinball(np.array([10.0]), np.array([8.0]), 0.9) == pytest.approx(1.8)
    assert pinball(np.array([10.0]), np.array([12.0]), 0.9) == pytest.approx(0.2)


def test_wrmsse_perfect_forecast_and_weights(tiny):
    hier = build_hierarchy(tiny.series)
    w = WRMSSE(tiny, origin=400, horizon=28, hier=hier)
    assert w.weights.sum() == pytest.approx(1.0, abs=1e-5)
    assert w.score(tiny.Y[:, 401:429]) == pytest.approx(0.0)
    assert w.score(np.zeros((8, 28))) > 0.5


def test_hierarchy_sums(tiny):
    hier = build_hierarchy(tiny.series)
    tot = hier.aggregate(tiny.Y, 1)
    assert np.allclose(tot[0], tiny.Y.sum(axis=0))
    assert hier.S.shape == (1 + 2 + 2 + 2 + 2 + 4 + 4 + 4 + 4 + 4 + 8 + 8, 8)


def test_baselines_shapes_and_semantics(tiny):
    H = tiny.Y[:, :400]
    assert np.array_equal(seasonal_naive(H, 14), np.tile(H[:, -7:], 2))
    const = np.full((2, 100), 3.0)
    assert np.allclose(ses(const, 5), 3) and np.allclose(croston(const, 5), 3, atol=1e-5)
    assert np.allclose(tsb(np.zeros((1, 100)) + np.r_[np.ones(50), np.zeros(50)][None], 3)[0, 0] < 1, True)
    wp = weekday_profile(H, 7)
    assert wp.shape == (8, 7) and (wp >= 0).all()


def test_syntetos_boylan_classes():
    smooth = np.full((1, 200), 5.0) + np.random.default_rng(0).integers(0, 2, (1, 200))
    sparse = np.zeros((1, 200))
    sparse[0, ::10] = 1
    assert syntetos_boylan_class(np.vstack([smooth, sparse])).tolist() == ["smooth", "intermittent"]


def test_lags_respect_min_lag():
    for L in [1, 7, 28]:
        assert min(lags_for(L)) == L


def test_features_do_not_look_ahead(tiny):
    fb = FeatureBuilder(tiny, np.arange(8))
    for L in [1, 7, 28]:
        assert_no_lookahead(fb, np.array([450]), L)


def test_lookahead_guard_catches_a_peeking_feature(tiny):
    fb = FeatureBuilder(tiny, np.arange(8))
    orig = fb.frame

    def peeking(days, L):
        X = orig(days, L)
        X["centred_mean"] = np.repeat(fb.Y[:, days[0] - 1: days[0] + 2].mean(axis=1), len(days))
        return X
    fb.frame = peeking
    with pytest.raises(AssertionError):
        assert_no_lookahead(fb, np.array([450]), 28)


def test_quality_gate(tiny):
    day = 450
    ref = make_reference(tiny.Y, tiny.P, tiny.series, day)
    good = pd.DataFrame({"date": tiny.dates[day], "store_id": tiny.series["store_id"], "item_id": tiny.series["item_id"],
                         "units": tiny.Y[:, day], "sell_price": tiny.P[:, day]})
    assert not [i for i in check_batch(good, tiny.dates[day], ref) if i.severity == "error"]
    dup = pd.concat([good, good.head(2)])
    assert any(i.check == "duplicates" for i in check_batch(dup, tiny.dates[day], ref))
    neg = good.assign(units=-1)
    assert any(i.check == "negative_units" for i in check_batch(neg, tiny.dates[day], ref))
    wrong_day = good.assign(date=tiny.dates[day - 1])
    assert any(i.check == "date" and i.severity == "error" for i in check_batch(wrong_day, tiny.dates[day], ref))
    missing = good[good["store_id"] != "TX_1"]
    assert any(i.check == "missing_series" and i.severity == "error"
               for i in check_batch(missing, tiny.dates[day], ref))


def test_stockout_detection():
    Y = np.full((1, 200), 5.0)
    Y[0, 150:156] = 0
    lab = suspected_stockouts(Y, np.ones_like(Y, dtype=bool))
    assert (lab[0, 150:156] == STOCKOUT).all() and lab[0, :150].sum() == 0


def test_nb_quantiles():
    mu = np.array([[0.5, 5.0, 50.0]])
    q = nb_quantiles(mu, np.array([0.3]), [0.1, 0.5, 0.9])
    assert (np.diff(q, axis=0) >= 0).all()
    s = nb_sum_quantiles(np.full((1, 3), 2.0), np.array([0.01]), [0.5])
    assert 5 <= s[0, 0] <= 7


def test_conformal_coverage():
    rng = np.random.default_rng(1)
    F = np.full((2000, 28), 5.0)
    pairs = [(rng.poisson(5, F.shape).astype(float), F, np.full(2000, 5.0)) for _ in range(2)]
    Q = Conformal([0.1, 0.9]).fit(pairs).predict(F, np.full(2000, 5.0))
    A = rng.poisson(5, F.shape)
    cov = np.mean((A >= Q[0]) & (A <= Q[1]))
    assert 0.75 < cov < 0.95


def test_reconciliation_is_coherent():
    meta = pd.DataFrame({"store_id": ["A", "A", "B", "B"], "dept_id": ["x", "y", "x", "y"]})
    h = build_hierarchy(meta, [("total", []), ("store", ["store_id"]), ("store x dept", ["store_id", "dept_id"])])
    S = h.S.toarray()
    rng = np.random.default_rng(0)
    base = S @ rng.poisson(10, (4, 7)) + rng.normal(0, 3, (S.shape[0], 7))
    assert not is_coherent(base, S)
    resid = rng.normal(0, 1, (S.shape[0], 200))
    for m in ["bottom-up", "OLS", "WLS (structural)", "WLS (variance)", "MinT (shrink)"]:
        assert is_coherent(reconcile(base, S, m, resid), S), m
    perfect = S @ rng.poisson(10, (4, 7)).astype(float)
    assert np.allclose(reconcile(perfect, S, "OLS"), perfect)


def test_inventory_simulation():
    rng = np.random.default_rng(0)
    D = rng.poisson(4, (500, 28)).astype(float)
    F = np.full_like(D, 4.0)
    costs = Costs(2, 0.25, 0.3)
    alpha = np.full(500, 0.01)
    lo = summarise(simulate(D, order_up_to(F, alpha, 0.5, 2), np.ones_like(D), costs), D)
    hi = summarise(simulate(D, order_up_to(F, alpha, 0.99, 2), np.ones_like(D), costs), D)
    assert hi["fill rate"] > lo["fill rate"] and hi["avg inventory value $"] > lo["avg inventory value $"]
    assert hi["fill rate"] > 0.97
    assert 0.99 < costs.critical_fractile() < 1


def test_psi():
    p = distribution(np.random.default_rng(0).poisson(2, 10000))
    assert psi(p, p) == pytest.approx(0)
    q = distribution(np.random.default_rng(0).poisson(6, 10000))
    assert psi(p, q) > 0.2

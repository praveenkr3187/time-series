"""Per-series statistical models (one model fitted per series) via statsforecast, plus Prophet.

These are "local" models: each series learns only from its own past. They are the textbook workhorses
(ETS, ARIMA) and work well on smooth aggregates, but must be fitted 30,490 times at item level - which is
why they are evaluated on a sample and on the aggregate levels, and why global models (Phase 11) exist.
"""
from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def to_long(Y: np.ndarray, dates: pd.DatetimeIndex, ids: list[str] | None = None) -> pd.DataFrame:
    ids = ids or [str(i) for i in range(len(Y))]
    return pd.DataFrame({"unique_id": np.repeat(ids, Y.shape[1]), "ds": np.tile(dates, len(Y)),
                         "y": Y.ravel().astype(np.float64)})


def fill_closures(Y: np.ndarray, dates: pd.DatetimeIndex) -> np.ndarray:
    """Replace Christmas-day closures by the mean of the neighbouring days (closures are known in advance)."""
    Y = Y.copy()
    for i in np.flatnonzero((dates.month == 12) & (dates.day == 25)):
        if 0 < i < Y.shape[1] - 1:
            Y[:, i] = (Y[:, i - 1] + Y[:, i + 1]) / 2
    return Y


def statsforecast_models(names: list[str], season: int = 7) -> dict:
    from statsforecast.models import (ADIDA, IMAPA, TSB, AutoARIMA, AutoETS, CrostonClassic, CrostonOptimized,
                                      CrostonSBA, Holt, HoltWinters,
                                      SimpleExponentialSmoothingOptimized)
    factory = {
        "SES": lambda: SimpleExponentialSmoothingOptimized(alias="SES"),
        "Holt": lambda: Holt(season_length=season, alias="Holt"),
        "Holt-Winters": lambda: HoltWinters(season_length=season, alias="Holt-Winters"),
        "AutoETS": lambda: AutoETS(season_length=season, alias="AutoETS"),
        "ARIMA (non-seasonal)": lambda: AutoARIMA(season_length=1, alias="ARIMA (non-seasonal)"),
        "SARIMA (AutoARIMA, s=7)": lambda: AutoARIMA(season_length=season, alias="SARIMA (AutoARIMA, s=7)"),
        "ARIMAX (SARIMA + calendar/price)": lambda: AutoARIMA(season_length=season,
                                                               alias="ARIMAX (SARIMA + calendar/price)"),
        "Croston (classic)": lambda: CrostonClassic(alias="Croston (classic)"),
        "Croston (optimised)": lambda: CrostonOptimized(alias="Croston (optimised)"),
        "SBA": lambda: CrostonSBA(alias="SBA"),
        "TSB": lambda: TSB(alpha_d=0.1, alpha_p=0.05, alias="TSB"),
        "ADIDA": lambda: ADIDA(alias="ADIDA"),
        "IMAPA": lambda: IMAPA(alias="IMAPA"),
    }
    return {n: factory[n]() for n in names}


def forecast(Y: np.ndarray, dates: pd.DatetimeIndex, h: int, names: list[str], X: np.ndarray | None = None,
             X_future: np.ndarray | None = None, season: int = 7, n_jobs: int = -1) -> dict[str, np.ndarray]:
    """Fit each named model on every row of Y (history ending at dates[-1]); return {name: F [n, h]}.

    X / X_future: optional exogenous regressors [n, T, k] / [n, h, k] (used by the ARIMAX model only).
    """
    from statsforecast import StatsForecast
    mods = statsforecast_models(names, season)
    exo = [n for n in names if n.startswith("ARIMAX")]
    plain = [m for n, m in mods.items() if n not in exo]
    out: dict[str, np.ndarray] = {}
    df = to_long(Y, dates)
    ids = [str(i) for i in range(len(Y))]
    future_dates = pd.date_range(dates[-1] + pd.Timedelta(days=1), periods=h, freq="D")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if plain:
            sf = StatsForecast(models=plain, freq="D", n_jobs=n_jobs, fallback_model=None)
            fc = sf.forecast(df=df, h=h)
            for n in names:
                if n not in exo:
                    out[n] = _wide(fc, n, ids, h)
        if exo:
            k = X.shape[2]
            cols = [f"x{j}" for j in range(k)]
            dfx = df.assign(**{c: X[:, :, j].ravel() for j, c in enumerate(cols)})
            Xf = pd.DataFrame({"unique_id": np.repeat(ids, h), "ds": np.tile(future_dates, len(ids)),
                               **{c: X_future[:, :, j].ravel() for j, c in enumerate(cols)}})
            sf = StatsForecast(models=[mods[n] for n in exo], freq="D", n_jobs=n_jobs)
            fc = sf.forecast(df=dfx, h=h, X_df=Xf)
            for n in exo:
                out[n] = _wide(fc, n, ids, h)
    return {n: np.clip(F, 0, None) for n, F in out.items()}


def _wide(fc: pd.DataFrame, col: str, ids: list[str], h: int) -> np.ndarray:
    fc = fc.reset_index() if "unique_id" not in fc.columns else fc
    piv = fc.pivot(index="unique_id", columns="ds", values=col)
    return piv.loc[ids].to_numpy()[:, :h]


def prophet_forecast(Y: np.ndarray, dates: pd.DatetimeIndex, h: int, holidays: pd.DataFrame,
                     regressors: np.ndarray | None = None, regressors_future: np.ndarray | None = None) -> np.ndarray:
    """One Prophet model per row (weekly + yearly seasonality, M5 events as holidays, optional regressors)."""
    from prophet import Prophet
    logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
    logging.getLogger("prophet").setLevel(logging.WARNING)
    F = np.zeros((len(Y), h))
    future_dates = pd.date_range(dates[-1] + pd.Timedelta(days=1), periods=h, freq="D")
    for i in range(len(Y)):
        m = Prophet(holidays=holidays, weekly_seasonality=True, yearly_seasonality=True, daily_seasonality=False,
                    seasonality_mode="multiplicative")
        df = pd.DataFrame({"ds": dates, "y": Y[i]})
        fut = pd.DataFrame({"ds": future_dates})
        if regressors is not None:
            for j in range(regressors.shape[2]):
                m.add_regressor(f"r{j}")
                df[f"r{j}"] = regressors[i, :, j]
                fut[f"r{j}"] = regressors_future[i, :, j]
        m.fit(df)
        F[i] = m.predict(fut)["yhat"].to_numpy()
    return np.clip(F, 0, None)


def forecast_with_fitted(Y: np.ndarray, dates: pd.DatetimeIndex, h: int, name: str,
                         season: int = 7) -> tuple[np.ndarray, np.ndarray]:
    """Forecasts [n, h] and in-sample one-step fitted values [n, T] (for reconciliation residuals)."""
    from statsforecast import StatsForecast
    ids = [str(i) for i in range(len(Y))]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sf = StatsForecast(models=list(statsforecast_models([name], season).values()), freq="D", n_jobs=-1)
        fc = sf.forecast(df=to_long(Y, dates), h=h, fitted=True)
        fit = sf.forecast_fitted_values()
    fit = fit.reset_index() if "unique_id" not in fit.columns else fit
    fitted = fit.pivot(index="unique_id", columns="ds", values=name).loc[ids].to_numpy()
    return np.clip(_wide(fc, name, ids, h), 0, None), fitted

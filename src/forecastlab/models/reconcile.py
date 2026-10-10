"""Forecast reconciliation: make forecasts add up across the hierarchy.

Independent forecasts per level are *incoherent*: the store forecasts do not sum to the total forecast, so
purchasing (category x state) and replenishment (store x dept) plan against different numbers.
Linear reconciliation maps base forecasts y_hat (all levels) to coherent ones:

    y_tilde = S P y_hat,     P = (S' W^-1 S)^-1 S' W^-1

    bottom-up       P picks the bottom rows only
    top-down        bottom = historical shares x total forecast
    OLS             W = I
    WLS structural  W = diag(number of bottom series under each node)
    WLS variance    W = diag(in-sample residual variance)
    MinT (shrink)   W = shrunk covariance of in-sample residuals (Wickramasuriya, Athanasopoulos & Hyndman 2019)
"""
from __future__ import annotations

import numpy as np


def projection(S: np.ndarray, W: np.ndarray) -> np.ndarray:
    Wi = np.linalg.pinv(W)
    return np.linalg.solve(S.T @ Wi @ S, S.T @ Wi)


def reconcile(Yhat: np.ndarray, S: np.ndarray, method: str, residuals: np.ndarray | None = None,
              history_bottom: np.ndarray | None = None) -> np.ndarray:
    """Yhat [n_all, h] base forecasts for every node (rows ordered like S). Returns coherent [n_all, h]."""
    n_all, n_b = S.shape
    if method == "bottom-up":
        return S @ Yhat[-n_b:]
    if method == "top-down (historical shares)":
        shares = history_bottom.sum(axis=1) / history_bottom.sum()
        return S @ (shares[:, None] * Yhat[0][None, :])
    if method == "OLS":
        W = np.eye(n_all)
    elif method == "WLS (structural)":
        W = np.diag(S.sum(axis=1))
    elif method == "WLS (variance)":
        W = np.diag(np.var(residuals, axis=1) + 1e-6)
    elif method == "MinT (shrink)":
        W = shrunk_covariance(residuals)
    else:
        raise ValueError(method)
    return S @ (projection(S, W) @ Yhat)


def shrunk_covariance(E: np.ndarray) -> np.ndarray:
    """Schafer-Strimmer shrinkage of the residual covariance towards its diagonal. E [n_series, T]."""
    E = E - E.mean(axis=1, keepdims=True)
    n, T = E.shape
    cov = E @ E.T / T
    sd = np.sqrt(np.diag(cov))
    Z = E / np.maximum(sd[:, None], 1e-9)
    corr = Z @ Z.T / T
    W2 = (Z ** 2) @ (Z ** 2).T / T
    var_r = (W2 - corr ** 2) * T / (T - 1) ** 2
    off = ~np.eye(n, dtype=bool)
    lam = np.clip(var_r[off].sum() / max((corr[off] ** 2).sum(), 1e-12), 0, 1)
    shrunk = (1 - lam) * corr
    np.fill_diagonal(shrunk, 1.0)
    return shrunk * np.outer(sd, sd)


def is_coherent(Y: np.ndarray, S: np.ndarray, tol: float = 1e-6) -> bool:
    n_b = S.shape[1]
    return bool(np.allclose(S @ Y[-n_b:], Y, atol=tol * max(1.0, np.abs(Y).max())))

"""Goodness-of-fit metrics."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def mse(y: npt.ArrayLike, y_pred: npt.ArrayLike) -> float:
    y = np.asarray(y, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean((y - y_pred) ** 2))


def weighted_mse(
    y: npt.ArrayLike,
    y_pred: npt.ArrayLike,
    sigma: npt.ArrayLike,
) -> float:
    y = np.asarray(y, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    sigma = np.maximum(sigma, 1e-12)
    w = 1.0 / sigma**2
    return float(np.sum(w * (y - y_pred) ** 2) / np.sum(w))


def chi_square(
    y: npt.ArrayLike,
    y_pred: npt.ArrayLike,
    sigma: npt.ArrayLike,
) -> float:
    y = np.asarray(y, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    sigma = np.maximum(sigma, 1e-12)
    return float(np.sum(((y - y_pred) / sigma) ** 2))


def reduced_chi_square(chi2: float, n_data: int, n_params: int) -> float:
    dof = max(n_data - n_params, 1)
    return chi2 / dof


def aic(chi2: float, n_params: int) -> float:
    return chi2 + 2.0 * n_params


def bic(chi2: float, n_data: int, n_params: int) -> float:
    n_data = max(n_data, 1)
    return chi2 + n_params * np.log(n_data)


def percent_improvement(baseline: float, candidate: float) -> float:
    """Percent reduction in metric (e.g. MSE): 100 * (baseline - candidate) / baseline."""
    if baseline == 0:
        return 0.0 if candidate == 0 else float("inf")
    return 100.0 * (baseline - candidate) / baseline

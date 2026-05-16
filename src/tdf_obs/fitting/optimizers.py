"""Optimizer helpers (extensible for future channels)."""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.optimize import curve_fit, least_squares


def fit_curve(
    model: Callable[..., np.ndarray],
    x: np.ndarray,
    y: np.ndarray,
    p0: tuple[float, ...],
    bounds: tuple[list[float], list[float]],
    sigma: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Thin wrapper around scipy.optimize.curve_fit."""
    popt, pcov = curve_fit(
        model,
        x,
        y,
        p0=p0,
        bounds=bounds,
        sigma=sigma,
        absolute_sigma=True,
        maxfev=20_000,
    )
    return popt, pcov


def fit_least_squares(
    residual_fn: Callable[[np.ndarray], np.ndarray],
    p0: np.ndarray,
    bounds: tuple[np.ndarray, np.ndarray],
) -> np.ndarray:
    """Thin wrapper around scipy.optimize.least_squares."""
    result = least_squares(residual_fn, p0, bounds=bounds)
    return result.x

"""Phenomenological dark-matter halo rotation baselines (Phase 3)."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

# Minimum x = r/r_s for stable evaluation of f(x) = [ln(1+x) - x/(1+x)] / x
_X_MIN: float = 1e-8


def _nfw_enclosed_factor(x: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    """
    f(x) = [ln(1+x) - x/(1+x)] / x  with stable limit f(0) -> 0.

    x = r / r_s (dimensionless).
    """
    x = np.asarray(x, dtype=float)
    x_safe = np.maximum(x, _X_MIN)
    numer = np.log1p(x_safe) - x_safe / (1.0 + x_safe)
    factor = numer / x_safe
    # At x -> 0: v_DM^2 -> v_baryon^2 (no divergent halo term)
    factor = np.where(x < _X_MIN, 0.0, factor)
    return factor


def v2_nfw_halo_only(
    r: npt.ArrayLike,
    Vh2: float,
    r_s: float,
) -> npt.NDArray[np.floating]:
    """NFW halo contribution only: Vh2 * f(r/r_s) [km^2/s^2], excluding baryons."""
    r = np.asarray(r, dtype=float)
    r_s = max(float(r_s), _X_MIN)
    x = r / r_s
    return float(Vh2) * _nfw_enclosed_factor(x)


def v2_nfw_simple(
    r: npt.ArrayLike,
    v_baryon: npt.ArrayLike,
    Vh2: float,
    r_s: float,
) -> npt.NDArray[np.floating]:
    """
    Simplified NFW-like circular-speed squared [km^2/s^2].

    v_DM^2(r) = v_baryon^2(r) + Vh2 * f(r/r_s)

    with f(x) = [ln(1+x) - x/(1+x)] / x.

    Parameters
    ----------
    r, r_s : kpc
    v_baryon : km/s
    Vh2 : km^2/s^2 (halo velocity-squared scale)
    """
    r = np.asarray(r, dtype=float)
    v_baryon = np.asarray(v_baryon, dtype=float)
    r_s = max(float(r_s), _X_MIN)
    x = r / r_s
    return v_baryon**2 + v2_nfw_halo_only(r, Vh2, r_s)


def v_nfw_simple(
    r: npt.ArrayLike,
    v_baryon: npt.ArrayLike,
    Vh2: float,
    r_s: float,
) -> npt.NDArray[np.floating]:
    """Circular speed [km/s] from v2_nfw_simple."""
    v2 = v2_nfw_simple(r, v_baryon, Vh2, r_s)
    return np.sqrt(np.maximum(v2, 0.0))

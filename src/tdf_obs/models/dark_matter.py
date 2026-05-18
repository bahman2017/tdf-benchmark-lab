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


def _burkert_mass_factor(x: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    """
    m(x)/x for Burkert enclosed mass with x = r/r_core.

    m(x) = 0.5 [ln(1+x²) + 2 arctan(x) - ln(1+x)]; v²_halo ∝ m(x)/x.
    Stable at x → 0 (limit m/x → 0.5).
    """
    x = np.asarray(x, dtype=float)
    xs = np.maximum(x, 1e-10)
    m = 0.5 * (np.log1p(xs**2) + 2.0 * np.arctan(xs) - np.log1p(xs))
    ratio = m / xs
    small = xs < 1e-8
    if np.any(small):
        ratio = np.where(small, 0.5 + 0.25 * xs, ratio)
    return np.maximum(ratio, 0.0)


def v2_burkert_halo_only(
    r: npt.ArrayLike,
    v_core: float,
    r_core: float,
) -> npt.NDArray[np.floating]:
    """
    Burkert halo contribution only [km²/s²].

    Normalized so v_halo(r_core) = v_core. Parameters: v_core [km/s], r_core [kpc].
    """
    r = np.asarray(r, dtype=float)
    r_core = max(float(r_core), _X_MIN)
    v_core = max(float(v_core), 0.0)
    x = r / r_core
    norm = 0.25 * np.pi  # m(1)/1 with m(1) = π/4
    return (v_core**2) * _burkert_mass_factor(x) / norm


def v2_pseudo_isothermal_halo_only(
    r: npt.ArrayLike,
    v_inf: float,
    r_core: float,
) -> npt.NDArray[np.floating]:
    """
    Pseudo-isothermal halo only: v² = V_∞² [1 - (r_c/r) arctan(r/r_c)].

    Finite at r → 0; approaches V_∞² at large r.
    """
    r = np.asarray(r, dtype=float)
    r_core = max(float(r_core), _X_MIN)
    v_inf = max(float(v_inf), 0.0)
    rr = np.maximum(r, 1e-10)
    x = rr / r_core
    arct = np.arctan(x)
    bracket = 1.0 - (r_core / rr) * arct
    small = r < 0.01 * r_core
    bracket = np.where(small, (r / r_core) ** 2 / 3.0, bracket)
    return (v_inf**2) * np.maximum(bracket, 0.0)

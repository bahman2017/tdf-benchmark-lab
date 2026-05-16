"""Black-hole phenomenological formulas (TDF v0.8.1 ansatz level)."""

from __future__ import annotations

import warnings

import numpy as np

from tdf_obs.constants import C, G, HBAR, K_B


def schwarzschild_radius(M: float) -> float:
    """r_s = 2 G M / c^2 [m]."""
    return 2.0 * G * M / C**2


def non_return_radius_tdf(M: float, rc: float) -> float:
    """
    r_nr_TDF = sqrt(r_s^2 - r_c^2).

    Returns np.nan if rc > r_s (unphysical for this ansatz).
    """
    rs = schwarzschild_radius(M)
    rc = float(rc)
    if rc > rs:
        warnings.warn(
            f"rc ({rc:.3e} m) > r_s ({rs:.3e} m): non_return_radius_tdf undefined; returning nan.",
            stacklevel=2,
        )
        return float(np.nan)
    if rc == rs:
        return 0.0
    return float(np.sqrt(rs**2 - rc**2))


def hawking_temperature(M: float) -> float:
    """T_H = hbar c^3 / (8 pi G M k_B) [K]."""
    return HBAR * C**3 / (8.0 * np.pi * G * M * K_B)


def tdf_temperature(M: float, rc: float) -> float:
    """
    T_TDF = T_H * sqrt(1 - r_c^2 / r_s^2).

    Returns 0 when rc -> r_s; approaches T_H when rc << r_s.
    """
    rs = schwarzschild_radius(M)
    rc = float(rc)
    if rc > rs:
        warnings.warn("rc > r_s: tdf_temperature undefined; returning nan.", stacklevel=2)
        return float(np.nan)
    ratio = (rc / rs) ** 2
    return hawking_temperature(M) * np.sqrt(max(0.0, 1.0 - ratio))


def remnant_mass(rc: float) -> float:
    """
    Phenomenological remnant mass scale from core radius (placeholder).

    M_rem ~ c^2 rc / (2G)  (order-of-magnitude ansatz; not a derived identity).
    """
    rc = float(rc)
    if rc <= 0:
        return 0.0
    return C**2 * rc / (2.0 * G)

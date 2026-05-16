"""Tau-bar profiles and derivatives (TDF v0.8.1 weak-field ansatz)."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from tdf_obs.constants import G


def tau_log_profile(
    r: npt.ArrayLike,
    A: float,
    r0: float,
) -> npt.NDArray[np.floating]:
    """
    Logarithmic tau-bar profile: tau_bar_l(r) = A * log(1 + r / r0).

    Parameters
    ----------
    r : array
        Radius [kpc] (consistent with rotation module).
    A, r0 : float
        Profile amplitude and scale radius [kpc].
    """
    r = np.asarray(r, dtype=float)
    r0 = max(float(r0), 1e-12)
    return A * np.log1p(r / r0)


def d_tau_log_dr(
    r: npt.ArrayLike,
    A: float,
    r0: float,
) -> npt.NDArray[np.floating]:
    """d/dr [A * log(1 + r/r0)] = A / (r + r0)."""
    r = np.asarray(r, dtype=float)
    r0 = max(float(r0), 1e-12)
    return A / (r + r0)


def tau_core_potential(
    r: npt.ArrayLike,
    M: float,
    rc: float,
) -> npt.NDArray[np.floating]:
    """
    Phenomenological BH core potential (SI): Phi_tau_core = -G M / sqrt(r^2 + rc^2).

    Parameters
    ----------
    r, rc : meters; M : kg.
    """
    r = np.asarray(r, dtype=float)
    rc = max(float(rc), 0.0)
    return -G * M / np.sqrt(r**2 + rc**2)


def d_tau_core_potential_dr(
    r: npt.ArrayLike,
    M: float,
    rc: float,
) -> npt.NDArray[np.floating]:
    """Radial derivative of tau_core_potential."""
    r = np.asarray(r, dtype=float)
    rc = max(float(rc), 0.0)
    denom = (r**2 + rc**2) ** 1.5
    return G * M * r / denom

"""Rotation-curve model from TDF v0.8.1 simple ansatz."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def v2_tdf_simple(
    r: npt.ArrayLike,
    v_baryon: npt.ArrayLike,
    B: float,
    r0: float,
) -> npt.NDArray[np.floating]:
    """
    Squared circular speed including tau correction.

    v_model^2(r) = v_baryon^2(r) + B * r / (r + r0)

    with tau_bar_l = A log(1 + r/r0) and B = K_tau * A.

    Units
    -----
    r : kpc
    v_baryon : km/s
    B : km^2/s^2
    r0 : kpc
  """
    r = np.asarray(r, dtype=float)
    v_baryon = np.asarray(v_baryon, dtype=float)
    r0 = max(float(r0), 1e-12)
    v2_b = v_baryon**2
    correction = B * r / (r + r0)
    return v2_b + correction


def v_tdf_simple(
    r: npt.ArrayLike,
    v_baryon: npt.ArrayLike,
    B: float,
    r0: float,
) -> npt.NDArray[np.floating]:
    """Circular speed [km/s] from v2_tdf_simple."""
    v2 = v2_tdf_simple(r, v_baryon, B, r0)
    return np.sqrt(np.maximum(v2, 0.0))


def baryon_only_model(v_baryon: npt.ArrayLike) -> npt.NDArray[np.floating]:
    """Reference model: v = v_baryon (km/s)."""
    return np.asarray(v_baryon, dtype=float)

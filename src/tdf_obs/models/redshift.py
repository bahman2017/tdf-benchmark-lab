"""Doppler / redshift tau correction."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from tdf_obs.constants import C


def z_tau(delta_tau: npt.ArrayLike, K_tau: float) -> npt.NDArray[np.floating]:
    """
    Phenomenological redshift correction: z_tau = K_tau * Delta(tau_bar_l) / c^2.

    delta_tau : difference in tau-bar potential [SI units consistent with K_tau].
    K_tau : coupling constant [1/s^2] when delta_tau has units of m^2/s^2 or as documented.
    """
    delta_tau = np.asarray(delta_tau, dtype=float)
    return K_tau * delta_tau / C**2


def redshift_residual_placeholder(
    z_obs: npt.ArrayLike,
    z_kin: npt.ArrayLike,
    z_baryon: npt.ArrayLike,
    z_tau_pred: npt.ArrayLike,
) -> dict[str, npt.NDArray[np.floating]]:
    """
    Placeholder observational residual test (not a full pipeline).

    residual = z_obs - z_kin - z_baryon - z_tau_pred
    """
    z_obs = np.asarray(z_obs, dtype=float)
    z_kin = np.asarray(z_kin, dtype=float)
    z_baryon = np.asarray(z_baryon, dtype=float)
    z_tau_pred = np.asarray(z_tau_pred, dtype=float)
    residual = z_obs - z_kin - z_baryon - z_tau_pred
    return {
        "residual": residual,
        "status": np.array(["not_yet_tested"] * len(residual), dtype=object),
    }

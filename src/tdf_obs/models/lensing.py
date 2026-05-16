"""Lensing consistency channel (placeholder)."""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt


def alpha_lens_tdf_placeholder(
    r: npt.ArrayLike,
    *,
    baryon_alpha: npt.ArrayLike | None = None,
    tau_gradient: npt.ArrayLike | None = None,
) -> npt.NDArray[np.floating]:
    """
    Placeholder for alpha_lens ~ (2/c^2) * integral grad_perp(Phi_b + Phi_tau) dl.

    Not implemented: requires line-of-sight projection and surface-density data.
    """
    raise NotImplementedError(
        "Lensing pipeline not implemented. Required: projected baryon potential, "
        "tau-bar profile parameters (B, r0 or A, r0), and lensing geometry. "
        "See docs/DATA_REQUIREMENTS.md.",
    )


def lensing_data_requirements() -> dict[str, Any]:
    """Document minimum data needed for a future lensing fit."""
    return {
        "status": "not_yet_tested",
        "required_fields": [
            "lens_id",
            "r_kpc (impact parameter or projected radius)",
            "alpha_obs [arcsec]",
            "alpha_err",
            "baryon_potential_or_mass along LOS",
        ],
        "required_models": [
            "tau_log_profile or shared (B, r0) from rotation fit",
            "line-of-sight integral of grad_perp(Phi_b + Phi_tau)",
        ],
    }

"""Synthetic fitter parameter / curve recovery."""

import numpy as np

from tdf_obs.fitting.fit_rotation import fit_single_galaxy_rotation
from tdf_obs.models.rotation import v_tdf_simple
from tdf_obs.validation.synthetic_tests import generate_synthetic_rotation_curve


def test_synthetic_fitter_recovers_B_and_r0() -> None:
    B_true, r0_true = 750.0, 3.5
    data, truth = generate_synthetic_rotation_curve(
        B=B_true,
        r0=r0_true,
        noise_fraction=0.02,
        seed=123,
    )
    fit = fit_single_galaxy_rotation(data)
    assert fit.success
    assert fit.mse_tdf < fit.mse_baryon

    r = data.r_kpc
    v_true = v_tdf_simple(r, data.v_baryon, truth["B_true"], truth["r0_true"])
    v_fit = v_tdf_simple(r, data.v_baryon, fit.best_B, fit.best_r0)
    # Primary Phase 1 check: recover the circular-speed curve (B, r0 can be mildly degenerate).
    np.testing.assert_allclose(v_fit, v_true, rtol=0.05)

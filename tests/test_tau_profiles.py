"""Tests for tau profile derivatives."""

import numpy as np

from tdf_obs.models.tau_profiles import d_tau_log_dr, tau_log_profile


def test_log_tau_derivative_matches_finite_difference() -> None:
    A, r0 = 2.0, 5.0
    r = np.linspace(0.5, 20.0, 50)
    analytic = d_tau_log_dr(r, A, r0)
    dr = 1e-4
    numeric = (tau_log_profile(r + dr, A, r0) - tau_log_profile(r - dr, A, r0)) / (2 * dr)
    np.testing.assert_allclose(analytic, numeric, rtol=1e-5, atol=1e-8)

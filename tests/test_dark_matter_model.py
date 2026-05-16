"""NFW simple halo baseline model tests."""

import numpy as np

from tdf_obs.models.dark_matter import v2_nfw_simple, v_nfw_simple


def test_nfw_simple_finite_non_negative() -> None:
    r = np.linspace(0.5, 30.0, 40)
    v_b = 100.0 + 20.0 * np.sqrt(r / (r + 2.0))
    v2 = v2_nfw_simple(r, v_b, Vh2=3000.0, r_s=5.0)
    v = v_nfw_simple(r, v_b, Vh2=3000.0, r_s=5.0)
    assert np.all(np.isfinite(v2))
    assert np.all(np.isfinite(v))
    assert np.all(v >= 0)
    assert np.all(v2 >= 0)


def test_nfw_small_r_stable() -> None:
    """Halo term is finite and non-divergent at very small r (clamped factor)."""
    v_b = np.array([50.0])
    r = np.array([1e-9])
    v2 = v2_nfw_simple(r, v_b, Vh2=5000.0, r_s=10.0)
    assert np.isfinite(v2[0])
    np.testing.assert_allclose(v2[0], v_b[0] ** 2, rtol=0, atol=1e-6)

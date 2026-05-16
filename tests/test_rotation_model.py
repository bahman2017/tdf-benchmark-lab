"""Rotation model finite-value checks."""

import numpy as np

from tdf_obs.models.rotation import v2_tdf_simple, v_tdf_simple


def test_rotation_model_finite() -> None:
    r = np.linspace(0.5, 30.0, 40)
    v_b = 150.0 * np.ones_like(r)
    v2 = v2_tdf_simple(r, v_b, B=500.0, r0=3.0)
    v = v_tdf_simple(r, v_b, B=500.0, r0=3.0)
    assert np.all(np.isfinite(v2))
    assert np.all(np.isfinite(v))
    assert np.all(v >= 0)

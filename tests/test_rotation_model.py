"""Rotation model finite-value checks."""

import numpy as np
import pytest

from tdf_obs.models.rotation import (
    v2_tdf_kessence,
    v2_tdf_simple,
    v_tdf_kessence,
    v_tdf_simple,
)


def test_rotation_model_finite() -> None:
    r = np.linspace(0.5, 30.0, 40)
    v_b = 150.0 * np.ones_like(r)
    v2 = v2_tdf_simple(r, v_b, B=500.0, r0=3.0)
    v = v_tdf_simple(r, v_b, B=500.0, r0=3.0)
    assert np.all(np.isfinite(v2))
    assert np.all(np.isfinite(v))
    assert np.all(v >= 0)


def test_kessence_finite_positive() -> None:
    r = np.linspace(0.5, 35.0, 50)
    v_b = 180.0 * np.sqrt(r / (r + 2.0)) + 20.0
    v2 = v2_tdf_kessence(r, v_b, a0=800.0)
    v = v_tdf_kessence(r, v_b, a0=800.0)
    assert np.all(np.isfinite(v2))
    assert np.all(np.isfinite(v))
    assert np.all(v2 > 0)
    assert np.all(v > 0)


def test_kessence_reduces_to_baryon_when_a0_zero() -> None:
    r = np.linspace(1.0, 25.0, 20)
    v_b = 100.0 + 40.0 * np.sqrt(r / (r + 3.0))
    v2 = v2_tdf_kessence(r, v_b, a0=0.0)
    assert v2 == pytest.approx(v_b**2, rel=1e-12)
    assert v_tdf_kessence(r, v_b, a0=0.0) == pytest.approx(v_b, rel=1e-12)

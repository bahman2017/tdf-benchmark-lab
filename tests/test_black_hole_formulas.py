"""Black-hole phenomenology limits."""

import numpy as np

from tdf_obs.constants import SOLAR_MASS
from tdf_obs.models.black_hole import (
    hawking_temperature,
    non_return_radius_tdf,
    schwarzschild_radius,
    tdf_temperature,
)


def test_schwarzschild_positive() -> None:
    M = 10.0 * SOLAR_MASS
    assert schwarzschild_radius(M) > 0


def test_non_return_nan_when_rc_gt_rs() -> None:
    M = SOLAR_MASS
    rs = schwarzschild_radius(M)
    assert np.isnan(non_return_radius_tdf(M, rs * 1.1))


def test_tdf_temperature_gr_limit() -> None:
    M = 5.0 * SOLAR_MASS
    T_H = hawking_temperature(M)
    T_small_rc = tdf_temperature(M, rc=0.0)
    np.testing.assert_allclose(T_small_rc, T_H, rtol=1e-6)


def test_tdf_temperature_zero_at_rc_equals_rs() -> None:
    M = SOLAR_MASS
    rs = schwarzschild_radius(M)
    assert tdf_temperature(M, rs) == 0.0

"""Solar-system epsilon checks."""

from tdf_obs.models.solar_system import epsilon_tau, passes_gr_safe_limit


def test_epsilon_and_limit() -> None:
    eps = epsilon_tau(1e-12, 1.0)
    assert eps == 1e-12
    assert passes_gr_safe_limit(eps, 1e-9)
    assert not passes_gr_safe_limit(1e-3, 1e-9)

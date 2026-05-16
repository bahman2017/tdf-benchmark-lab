"""Metric unit tests."""

from tdf_obs.fitting.metrics import mse, percent_improvement, reduced_chi_square


def test_mse_and_improvement() -> None:
    y = [1.0, 2.0, 3.0]
    assert mse(y, y) == 0.0
    assert percent_improvement(100.0, 50.0) == 50.0


def test_reduced_chi_square() -> None:
    assert reduced_chi_square(10.0, 12, 2) == 1.0

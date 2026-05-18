"""Phase 7A — K-essence source viability benchmark tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tdf_obs.validation.kessence_source_viability import (
    BANNER_CALIBRATION,
    BANNER_KESSENCE_SOURCE,
    REQUIRED_SUMMARY_COLUMNS,
    baryon_density_profile,
    conformal_trace_source,
    disformal_static_dust_source,
    enclosed_source_integral,
    kessence_stability_proxy,
    mu_interpolation,
    outer_log_slope,
    run_conformal_canonical_newtonian,
    run_conformal_kessence_deep_mond,
    run_conformal_kessence_simple_interpolation,
    run_excessive_gradient_stability_fail,
    run_insufficient_coupling_expected_fail,
    run_kessence_source_viability_benchmark,
    run_pure_disformal_static_dust_expected_fail,
    solve_sigma_prime_from_integrated_equation,
)


def test_pure_disformal_source_near_zero() -> None:
    r = np.linspace(0.2, 10.0, 80)
    rho = baryon_density_profile(r, "exponential_spherical_proxy")
    src = disformal_static_dust_source(r, rho, omega=1.0)
    assert np.max(np.abs(src)) < 1e-12


def test_conformal_trace_source_nonzero() -> None:
    rho = np.array([0.1, 0.2, 0.3])
    s = conformal_trace_source(rho, beta_over_M=0.5)
    assert np.all(s > 0)


def test_canonical_outer_slope_near_minus_two() -> None:
    res = run_conformal_canonical_newtonian()
    assert res.overall_pass
    assert res.outer_gradient_slope == pytest.approx(-2.0, abs=0.5)


def test_deep_mond_outer_slope_near_minus_one() -> None:
    res = run_conformal_kessence_deep_mond()
    assert res.overall_pass
    assert res.outer_gradient_slope == pytest.approx(-1.0, abs=0.5)


def test_simple_interpolation_passes() -> None:
    assert run_conformal_kessence_simple_interpolation().overall_pass


def test_insufficient_coupling_fail_detected() -> None:
    res = run_insufficient_coupling_expected_fail()
    assert res.expected_status == "fail"
    assert res.failure_detected
    assert res.overall_pass
    assert res.expected_failure_matched
    assert "tau" in res.failure_reason


def test_excessive_gradient_fail_detected() -> None:
    res = run_excessive_gradient_stability_fail()
    assert res.expected_status == "fail"
    assert res.failure_detected
    assert res.max_abs_tau_acceleration > 25.0
    assert res.overall_pass
    assert res.expected_failure_matched


def test_stability_proxy_passes_viable_deep_mond() -> None:
    res = run_conformal_kessence_deep_mond()
    stab = kessence_stability_proxy(res.sigma_prime, 1.0, "deep_mond")
    assert stab["kx_positive"]
    assert stab["hyperbolicity_pass"]


def test_solve_sigma_prime_analytic_limits() -> None:
    r = np.array([1.0, 2.0, 5.0])
    i_r = r**3 * 0.01
    sp_can = solve_sigma_prime_from_integrated_equation(r, i_r, 1.0, "canonical")
    assert sp_can[1] == pytest.approx(i_r[1] / r[1] ** 2, rel=1e-6)
    sp_dm = solve_sigma_prime_from_integrated_equation(r, i_r, 1.0, "deep_mond")
    assert sp_dm[1] == pytest.approx(np.sqrt(i_r[1]) / r[1], rel=1e-6)


def test_mu_models_positive() -> None:
    y = np.linspace(0.01, 5.0, 20)
    for model in ("canonical", "deep_mond", "simple", "standard"):
        assert np.all(mu_interpolation(y, model) > 0)


def test_pure_disformal_expected_fail_counts_as_pass() -> None:
    res = run_pure_disformal_static_dust_expected_fail()
    assert res.expected_status == "fail"
    assert res.failure_detected
    assert res.overall_pass
    assert res.expected_failure_matched
    assert res.source_norm < 1e-12


def test_pipeline_outputs(tmp_path: Path) -> None:
    df, results = run_kessence_source_viability_benchmark(outputs_root=tmp_path / "out")
    assert len(results) == 7
    for col in REQUIRED_SUMMARY_COLUMNS:
        assert col in df.columns
    report = (tmp_path / "out" / "reports" / "kessence_source_viability_report.md").read_text(
        encoding="utf-8",
    )
    assert "NOT OBSERVATIONAL VALIDATION" in report
    assert BANNER_KESSENCE_SOURCE in report
    assert "pure disformal" in report.lower()
    assert "div J" in report
    assert "Expected outcomes matched" in report and "7/7" in report
    assert (tmp_path / "out" / "figures" / "kessence_source_terms.png").is_file()


def test_all_outcomes_match() -> None:
    """Each case should meet its pass criteria or detect the expected failure."""
    _, results = run_kessence_source_viability_benchmark(outputs_root=None)
    assert len(results) == 7
    assert all(r.expected_failure_matched for r in results)
    assert sum(1 for r in results if r.overall_pass) == 7

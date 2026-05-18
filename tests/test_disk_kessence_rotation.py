"""Phase 7B — Disk K-essence rotation benchmark tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tdf_obs.validation.disk_kessence_rotation import (
    BANNER_DISK_KESSENCE,
    REQUIRED_SUMMARY_COLUMNS,
    compact_disk_surface_density,
    disk_source_from_surface_density,
    exponential_disk_surface_density,
    integrated_disk_source,
    mu_interpolation,
    run_disk_kessence_rotation_benchmark,
    run_exponential_disk_deep_kessence,
    run_excessive_coupling_disk_expected_fail,
    run_pure_disformal_disk_expected_fail,
    run_simple_interpolation_disk,
    run_weak_coupling_disk_expected_fail,
    solve_disk_sigma_prime,
)


def test_exponential_disk_surface_density_decreases() -> None:
    r = np.linspace(0.5, 20.0, 50)
    sigma = exponential_disk_surface_density(r, sigma0=0.1, rd=3.0)
    assert sigma[0] > sigma[-1]
    assert np.all(sigma > 0)


def test_integrated_disk_source_monotonic() -> None:
    r = np.linspace(0.2, 15.0, 80)
    s = disk_source_from_surface_density(
        exponential_disk_surface_density(r, 0.08, 2.5),
        beta_over_m=1.0,
    )
    i_r = integrated_disk_source(r, s)
    assert np.all(np.diff(i_r) >= -1e-12)


def test_pure_disformal_disk_expected_fail_detected() -> None:
    res = run_pure_disformal_disk_expected_fail()
    assert res.expected_status == "fail"
    assert res.failure_detected
    assert res.overall_pass
    assert not res.source_nonzero


def test_conformal_disk_source_nonzero() -> None:
    r = np.linspace(0.5, 10.0, 40)
    sigma = exponential_disk_surface_density(r, 0.1, 2.0)
    s = disk_source_from_surface_density(sigma, beta_over_m=0.8)
    assert np.all(s > 0)


def test_deep_kessence_improves_outer_flatness() -> None:
    res = run_exponential_disk_deep_kessence()
    assert res.overall_pass
    assert res.flatness_improvement > 0.0
    assert res.total_outer_flatness_score >= res.baryon_outer_flatness_score


def test_simple_interpolation_disk_passes() -> None:
    assert run_simple_interpolation_disk().overall_pass


def test_weak_coupling_expected_fail_detected() -> None:
    res = run_weak_coupling_disk_expected_fail()
    assert res.failure_detected
    assert res.overall_pass


def test_excessive_coupling_expected_fail_detected() -> None:
    res = run_excessive_coupling_disk_expected_fail()
    assert res.failure_detected
    assert res.overall_pass


def test_solve_disk_sigma_prime_analytic() -> None:
    r = np.array([1.0, 2.0, 5.0])
    i_r = r * 0.05
    sp_can = solve_disk_sigma_prime(r, i_r, 1.0, "canonical")
    assert sp_can[1] == pytest.approx(i_r[1] / r[1], rel=1e-6)
    sp_dm = solve_disk_sigma_prime(r, i_r, 1.0, "deep_mond")
    assert sp_dm[1] == pytest.approx(np.sqrt(1.0 * i_r[1] / r[1]), rel=1e-6)


def test_mu_models_positive() -> None:
    y = np.linspace(0.01, 4.0, 15)
    for model in ("canonical", "deep_mond", "simple", "standard"):
        assert np.all(mu_interpolation(y, model) > 0)


def test_pipeline_outputs(tmp_path: Path) -> None:
    df, results = run_disk_kessence_rotation_benchmark(outputs_root=tmp_path / "out")
    assert len(results) == 8
    for col in REQUIRED_SUMMARY_COLUMNS:
        assert col in df.columns
    report = (tmp_path / "out" / "reports" / "disk_kessence_rotation_report.md").read_text(
        encoding="utf-8",
    )
    assert BANNER_DISK_KESSENCE in report
    assert "NOT REAL SPARC VALIDATION" in report
    assert "Expected outcomes matched" in report
    assert (tmp_path / "out" / "figures" / "disk_kessence_rotation_curves.png").is_file()


def test_all_outcomes_match() -> None:
    _, results = run_disk_kessence_rotation_benchmark(outputs_root=None)
    assert len(results) == 8
    assert all(r.overall_pass for r in results)

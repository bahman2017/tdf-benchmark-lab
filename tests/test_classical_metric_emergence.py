"""Phase 6E — Classical metric emergence benchmark tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tdf_obs.validation.classical_metric_emergence import (
    BANNER_CLASSICAL,
    REQUIRED_SUMMARY_COLUMNS,
    VAR_SUPPRESSION_MIN,
    apply_spatial_averaging,
    effective_metric_1p1,
    fluctuation_variance_before_after,
    gaussian_averaging_kernel,
    generate_tau_micro_grid,
    metric_lorentzian_check,
    run_classical_metric_emergence_benchmark,
    run_excessive_gradient_fail,
    run_insufficient_averaging_fail,
    run_noisy_micro_tau,
    run_smooth_control,
    run_two_branch_decohered_metric_merge,
)


def test_gaussian_kernel_normalized() -> None:
    x = np.linspace(-2, 2, 50)
    k = gaussian_averaging_kernel(x, 0.3)
    assert k.sum() == pytest.approx(1.0, rel=1e-10)


def test_averaging_reduces_high_frequency_variance() -> None:
    x = np.linspace(-3, 3, 200)
    t = np.linspace(0, 1, 20)
    tau = generate_tau_micro_grid(x, t, noise_strength=0.5, correlation_length=0.05, seed=0)
    tau_bar = apply_spatial_averaging(tau, x, 0.4)
    _, _, sup = fluctuation_variance_before_after(tau, tau_bar)
    assert sup > 1.5


def test_smooth_control_stable() -> None:
    assert run_smooth_control().overall_pass


def test_metric_lorentzian_small_gradients() -> None:
    x = np.linspace(-2, 2, 80)
    t = np.linspace(0, 1, 40)
    tau = generate_tau_micro_grid(
        x,
        t,
        smooth_profile=lambda X, T: 0.02 * X,
        noise_strength=0.01,
        seed=1,
    )
    g = effective_metric_1p1(tau, x, t, alpha_tau=1e-3)
    assert metric_lorentzian_check(g)


def test_excessive_gradient_can_fail_lorentzian() -> None:
    res = run_excessive_gradient_fail()
    assert res.expected_status == "fail"
    assert not res.overall_pass
    assert (
        not res.lorentzian_signature_pass
        or res.metric_smoothness_after < res.metric_smoothness_before
    )


def test_branch_metric_distance_decreases_after_averaging() -> None:
    res = run_two_branch_decohered_metric_merge()
    assert res.branch_metric_distance_after < res.branch_metric_distance_before


def test_insufficient_averaging_detected() -> None:
    res = run_insufficient_averaging_fail()
    assert res.expected_status == "fail"
    assert not res.overall_pass
    assert res.variance_suppression_factor < VAR_SUPPRESSION_MIN


def test_noisy_micro_passes() -> None:
    assert run_noisy_micro_tau().overall_pass


def test_pipeline_report_and_csv(tmp_path: Path) -> None:
    df, results = run_classical_metric_emergence_benchmark(outputs_root=tmp_path / "out")
    assert len(results) == 6
    for col in REQUIRED_SUMMARY_COLUMNS:
        assert col in df.columns
    report = (tmp_path / "out" / "reports" / "classical_metric_emergence_report.md").read_text(
        encoding="utf-8",
    )
    assert "NOT FULL OBJECTIVE-COLLAPSE SOLUTION" in report
    assert BANNER_CLASSICAL in report
    assert (tmp_path / "out" / "figures" / "classical_metric_tau_before_after.png").is_file()


def test_all_expected_outcomes() -> None:
    _, results = run_classical_metric_emergence_benchmark(outputs_root=None)
    for r in results:
        assert r.overall_pass == (r.expected_status == "pass")

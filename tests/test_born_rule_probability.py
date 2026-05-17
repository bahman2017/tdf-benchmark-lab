"""Phase 6F — Born-rule probability emergence benchmark tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tdf_obs.validation.born_rule_probability import (
    BANNER_BORN,
    FREQ_ERROR_TOL,
    REQUIRED_SUMMARY_COLUMNS,
    amplitudes_from_rho_tau,
    apply_decoherence_to_density_matrix,
    born_probability_from_state,
    chi_square_statistic,
    coarse_graining_additivity,
    compare_probability_rules,
    density_matrix_from_amplitudes,
    diagonal_probabilities,
    frequency_error,
    normalize_branch_weights,
    run_balanced_two_branch,
    run_born_rule_probability_benchmark,
    run_coarse_graining_additivity,
    run_decoherence_preserves_diagonal_weights,
    run_phase_invariance_check,
    run_unequal_two_branch,
    run_wrong_rules_fail_comparison,
    run_zero_weight_branch,
    simulate_measurement_counts,
    wrong_probability_rule,
)


def test_normalize_branch_weights_sums_to_one() -> None:
    p = normalize_branch_weights(np.array([2.0, 3.0, 5.0]))
    assert p.sum() == pytest.approx(1.0)


def test_amplitudes_recover_rho() -> None:
    rho = np.array([0.6, 0.4])
    c = amplitudes_from_rho_tau(rho, np.array([0.1, 0.9]))
    assert np.abs(c) ** 2 == pytest.approx(rho, rel=1e-12)


def test_decoherence_suppresses_off_diagonals_preserves_diagonals() -> None:
    rho = np.array([0.5, 0.5])
    c = amplitudes_from_rho_tau(rho, np.array([0.0, 1.2]))
    mat = density_matrix_from_amplitudes(c)
    before = diagonal_probabilities(mat)
    deco = apply_decoherence_to_density_matrix(mat, 5.0)
    after = diagonal_probabilities(deco)
    off_before = np.sum(np.abs(mat - np.diag(np.diag(mat))) ** 2)
    off_after = np.sum(np.abs(deco - np.diag(np.diag(deco))) ** 2)
    assert off_after < off_before
    assert after == pytest.approx(before, abs=1e-10)


def test_born_probabilities_phase_invariant() -> None:
    rho = np.array([0.5, 0.3, 0.2])
    p1 = born_probability_from_state(amplitudes_from_rho_tau(rho, np.array([0.0, 1.0, 2.0])))
    p2 = born_probability_from_state(
        amplitudes_from_rho_tau(rho, np.array([4.0, 0.5, 3.1])),
    )
    assert p1 == pytest.approx(p2, abs=1e-12)


def test_measurement_counts_converge() -> None:
    p = np.array([0.8, 0.2])
    counts = simulate_measurement_counts(p, 80_000, seed=0)
    assert frequency_error(counts, p) < FREQ_ERROR_TOL


def test_wrong_rules_larger_chi_square() -> None:
    rho = np.array([0.8, 0.2])
    counts = simulate_measurement_counts(rho, 50_000, seed=1)
    chi = compare_probability_rules(rho, counts)
    assert chi["born"] < chi["uniform"]
    assert chi["born"] < chi["amplitude_linear"]
    assert chi["born"] < chi["rho_squared"]


def test_coarse_graining_additivity_holds() -> None:
    probs = normalize_branch_weights(np.array([0.4, 0.35, 0.25]))
    err = coarse_graining_additivity(probs, [[0, 1], [2]])
    assert err < 1e-12


def test_zero_weight_branch_zero_probability() -> None:
    res = run_zero_weight_branch()
    assert res.overall_pass
    assert res.expected_probabilities[-1] < 1e-12


def test_balanced_and_unequal_cases_pass() -> None:
    assert run_balanced_two_branch().overall_pass
    assert run_unequal_two_branch().overall_pass
    assert run_decoherence_preserves_diagonal_weights().overall_pass
    assert run_wrong_rules_fail_comparison().overall_pass
    assert run_coarse_graining_additivity().overall_pass
    assert run_phase_invariance_check().overall_pass


def test_wrong_probability_rules_distinct_from_born() -> None:
    rho = np.array([0.7, 0.3])
    born = normalize_branch_weights(rho)
    assert not np.allclose(born, wrong_probability_rule(rho, "uniform"))
    assert not np.allclose(born, wrong_probability_rule(rho, "amplitude_linear"))


def test_pipeline_report_and_csv(tmp_path: Path) -> None:
    df, results = run_born_rule_probability_benchmark(outputs_root=tmp_path / "out")
    assert len(results) == 8
    for col in REQUIRED_SUMMARY_COLUMNS:
        assert col in df.columns
    report = (tmp_path / "out" / "reports" / "born_rule_probability_report.md").read_text(
        encoding="utf-8",
    )
    assert "NOT FULL BORN-RULE DERIVATION" in report
    assert BANNER_BORN in report
    assert (tmp_path / "out" / "figures" / "born_rule_expected_vs_observed.png").is_file()


def test_chi_square_statistic_finite() -> None:
    counts = np.array([800, 200])
    assert chi_square_statistic(counts, np.array([0.8, 0.2])) >= 0.0

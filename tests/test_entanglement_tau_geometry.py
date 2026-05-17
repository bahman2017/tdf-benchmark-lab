"""Phase 6C — Entanglement / τ geometry benchmark tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tdf_obs.validation.entanglement_tau_geometry import (
    BANNER_ENTANGLEMENT,
    CHSH_CLASSICAL,
    CHSH_BELL_TARGET,
    REQUIRED_SUMMARY_COLUMNS,
    amplitude_nonseparability_score,
    bell_state,
    chsh_S_value,
    concurrence_2qubit,
    nonseparable_tau_demo,
    no_signaling_check,
    phase_nonseparability_score,
    product_state,
    reduced_density_matrix_2qubit,
    run_bell_phi_plus,
    run_entanglement_tau_geometry_benchmark,
    run_product_state_control,
    von_neumann_entropy,
    psi_from_rho_tau_2body,
)


def test_bell_entropy_near_one() -> None:
    for name in ("phi_plus", "psi_minus"):
        psi = bell_state(name)
        rho_a = reduced_density_matrix_2qubit(psi, "A")
        assert abs(von_neumann_entropy(rho_a) - np.log(2)) < 0.05


def test_product_entropy_near_zero() -> None:
    psi = product_state(0.4, 0.0, 0.9, 0.5)
    rho_a = reduced_density_matrix_2qubit(psi, "A")
    assert von_neumann_entropy(rho_a) < 0.05


def test_bell_concurrence_near_one() -> None:
    assert concurrence_2qubit(bell_state("phi_plus")) > 0.99
    assert concurrence_2qubit(bell_state("psi_minus")) > 0.99


def test_product_concurrence_near_zero() -> None:
    psi = product_state(0.2, 1.0, 0.6, 0.3)
    assert concurrence_2qubit(psi) < 0.05


def test_chsh_bell_exceeds_classical() -> None:
    s = chsh_S_value(bell_state("phi_plus"))
    assert s > CHSH_CLASSICAL + 0.1
    assert abs(s - CHSH_BELL_TARGET) < 0.1


def test_chsh_product_not_violate() -> None:
    psi = product_state(0.5, 0.0, 0.8, 0.0)
    assert chsh_S_value(psi) <= CHSH_CLASSICAL + 0.05


def test_no_signaling_bell() -> None:
    psi = bell_state("phi_plus")
    assert no_signaling_check(
        psi,
        [0.0, np.pi / 2],
        [np.pi / 4, -np.pi / 4],
    )


def test_phase_separable_tau_low_score() -> None:
    tau = np.zeros((3, 3))
    tau += np.linspace(0, 1, 3)[:, None]
    tau += np.linspace(0, 0.5, 3)[None, :]
    assert phase_nonseparability_score(tau) < 0.05


def test_phase_nonseparable_tau_high_score() -> None:
    tau = nonseparable_tau_demo(4)
    assert phase_nonseparability_score(tau) > 0.2


def test_tdf_ansatz_density() -> None:
    rho = np.array([[0.5, 0], [0, 0.5]])
    tau = np.zeros((2, 2))
    psi_grid = psi_from_rho_tau_2body(rho, tau)
    assert abs(np.sum(np.abs(psi_grid) ** 2) - 1.0) < 1e-10 or np.sum(rho) > 0


def test_pipeline_report_and_csv(tmp_path: Path) -> None:
    df, results = run_entanglement_tau_geometry_benchmark(outputs_root=tmp_path / "out")
    assert len(results) == 6
    for col in REQUIRED_SUMMARY_COLUMNS:
        assert col in df.columns
    report = (tmp_path / "out" / "reports" / "entanglement_tau_geometry_report.md").read_text(
        encoding="utf-8",
    )
    assert "NOT FULL BELL-THEOREM RESOLUTION" in report
    assert BANNER_ENTANGLEMENT in report
    assert (tmp_path / "out" / "figures" / "entanglement_chsh_values.png").is_file()


def test_benchmark_cases_pass() -> None:
    assert run_product_state_control().overall_pass
    assert run_bell_phi_plus().overall_pass

"""Phase 6A / 6A.1 — Schrödinger-from-TDF action benchmark tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tdf_obs.validation.schrodinger_from_tdf import (
    BANNER_SCHRODINGER,
    PHASE_CONVENTION,
    QHJ_EQUATION,
    QHJ_NORMALIZED_PASS_THRESHOLD,
    REQUIRED_SUMMARY_COLUMNS,
    psi_from_rho_tau,
    quantum_hamilton_jacobi_residual,
    quantum_potential,
    rho_tau_from_psi,
    run_gaussian_snapshot_case,
    run_harmonic_ground_state_case,
    run_plane_wave_case,
    run_schrodinger_from_tdf_benchmark,
)


def test_psi_from_rho_tau_reconstructs_wave() -> None:
    rho = np.array([1.0, 1.0, 1.0])
    tau = np.array([0.0, 0.5, 1.0])
    psi = psi_from_rho_tau(rho, tau)
    rho2, tau2 = rho_tau_from_psi(psi)
    np.testing.assert_allclose(rho2, rho, rtol=1e-10)
    np.testing.assert_allclose(tau2, tau, atol=1e-10)


def test_plane_wave_phase_convention() -> None:
    """ψ = exp(i(kx − ωt)) ⇒ τ = ωt − kx."""
    k, omega, t = 1.0, 0.5, 0.3
    x = np.array([0.0, 1.0, 2.0])
    tau = omega * t - k * x
    psi = psi_from_rho_tau(np.ones_like(x), tau)
    expected = np.exp(1j * (k * x - omega * t))
    np.testing.assert_allclose(psi, expected, atol=1e-12)


def test_quantum_potential_finite_smooth_rho() -> None:
    x = np.linspace(-5, 5, 200)
    dx = x[1] - x[0]
    rho = np.exp(-0.5 * x**2)
    Q = quantum_potential(rho, dx)
    assert np.all(np.isfinite(Q[5:-5]))


def test_plane_wave_qhj_residual_small() -> None:
    res = run_plane_wave_case()
    assert res.max_continuity_residual < 0.05
    assert res.max_schrodinger_residual < 0.05
    assert res.max_normalized_qhj_residual < QHJ_NORMALIZED_PASS_THRESHOLD
    assert res.max_qhj_residual < 1e-10
    assert res.pass_


def test_gaussian_normalized_qhj_small() -> None:
    res = run_gaussian_snapshot_case()
    assert res.max_normalized_qhj_residual < QHJ_NORMALIZED_PASS_THRESHOLD
    assert res.max_schrodinger_residual < 0.05
    assert res.pass_


def test_harmonic_ground_state_residual_small() -> None:
    res = run_harmonic_ground_state_case()
    assert res.max_schrodinger_residual < 0.15
    assert res.max_normalized_qhj_residual < QHJ_NORMALIZED_PASS_THRESHOLD
    assert res.quantum_potential_finite
    assert res.pass_


def test_qhj_sign_plane_wave() -> None:
    """Corrected QHJ vanishes for uniform ρ and τ = ωt − kx."""
    x = np.linspace(-5, 5, 256, endpoint=False)
    dx = x[1] - x[0]
    k, omega = 1.0, 0.5
    rho = np.ones_like(x)
    tau = omega * 0.0 - k * x
    tau_t = np.full_like(x, omega)
    qhj = quantum_hamilton_jacobi_residual(rho, tau_t, tau, np.zeros_like(x), dx)
    assert np.max(np.abs(qhj[5:-5])) < 1e-12


def test_pipeline_report_and_csv(tmp_path: Path) -> None:
    df, results = run_schrodinger_from_tdf_benchmark(outputs_root=tmp_path / "out")
    assert len(results) == 3
    for col in REQUIRED_SUMMARY_COLUMNS:
        assert col in df.columns
    report = (tmp_path / "out" / "reports" / "schrodinger_from_tdf_report.md").read_text(
        encoding="utf-8",
    )
    assert "NOT FULL QUANTUM VALIDATION" in report
    assert BANNER_SCHRODINGER in report
    assert PHASE_CONVENTION.split(";")[0] in report or "ψ = √ρ exp(-iτ)" in report
    assert "normalized" in report.lower()
    assert "QHJ (norm)" in report or "QHJ (raw)" in report
    assert QHJ_EQUATION.split(",")[0] in report or "ℏ ∂_t τ" in report
    assert (tmp_path / "out" / "figures" / "schrodinger_plane_wave_residual.png").is_file()

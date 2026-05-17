"""Phase 6B — Dirac / spinor limit benchmark tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tdf_obs.validation.dirac_spinor_limit import (
    BANNER_DIRAC,
    REQUIRED_SUMMARY_COLUMNS,
    alpha_beta_from_gammas,
    check_clifford_algebra,
    compact_tau_mode_mass,
    dirac_energy_eigenvalues,
    dirac_hamiltonian_1d,
    dirac_residual_flat,
    effective_disformal_metric_flat,
    gamma_matrices_dirac,
    mass_from_tau_momentum,
    minkowski_eta,
    plane_wave_spinor,
    run_clifford_algebra_check,
    run_compact_tau_mass_ladder,
    run_dirac_spinor_limit_benchmark,
    run_disformal_metric_spinor_safety,
    run_flat_dirac_dispersion,
    run_positive_energy_plane_wave,
    run_tdf_spinor_phase_ansatz,
    spinor_norm,
    tdf_spinor_ansatz,
    tetrad_consistency_check,
)


def test_gamma_clifford_algebra() -> None:
    gammas = gamma_matrices_dirac()
    err = check_clifford_algebra(gammas)
    assert err < 1e-12


def test_alpha_beta_anticommutation() -> None:
    gammas = gamma_matrices_dirac()
    g0 = gammas[0]
    ax, ay, az, beta = alpha_beta_from_gammas(gammas)
    assert np.allclose(ax @ ax, np.eye(4))
    assert np.allclose(beta @ beta, np.eye(4))
    assert np.allclose(ax @ beta + beta @ ax, np.zeros((4, 4), dtype=complex))


def test_dirac_hamiltonian_dispersion() -> None:
    for k in [0.0, 0.8, 2.0]:
        H = dirac_hamiltonian_1d(k, m=1.0)
        evals = np.sort(np.linalg.eigvalsh(H))
        Ep, Em = dirac_energy_eigenvalues(k, m=1.0)
        assert np.allclose(evals, [Em, Em, Ep, Ep], atol=1e-10)


def test_plane_wave_dirac_residual_small() -> None:
    res = run_positive_energy_plane_wave()
    assert res.max_residual < 1e-10
    assert res.pass_


def test_tdf_spinor_ansatz_density() -> None:
    res = run_tdf_spinor_phase_ansatz()
    assert res.pass_
    chi = plane_wave_spinor(0.0, 1.0)
    psi = tdf_spinor_ansatz(1.0, 0.0, chi)
    assert abs(spinor_norm(psi) ** 2 - 1.0) < 1e-12


def test_disformal_metric_lorentzian_and_tetrad() -> None:
    res = run_disformal_metric_spinor_safety()
    assert res.lorentzian_signature_pass
    assert res.max_tetrad_reconstruction_error < 1e-3
    assert res.pass_
    g = effective_disformal_metric_flat(0.01, 0.02, alpha_tau=0.05)
    assert tetrad_consistency_check(g) < 2e-4


def test_compact_tau_mass_ladder_linear() -> None:
    res = run_compact_tau_mass_ladder()
    assert res.pass_
    m1 = compact_tau_mode_mass(1, 1.0)
    m3 = compact_tau_mode_mass(3, 1.0)
    assert abs(m3 - 3 * m1) < 1e-12
    assert abs(mass_from_tau_momentum(2.0) - 2.0) < 1e-12


def test_pipeline_report_and_csv(tmp_path: Path) -> None:
    df, results = run_dirac_spinor_limit_benchmark(outputs_root=tmp_path / "out")
    assert len(results) == 6
    for col in REQUIRED_SUMMARY_COLUMNS:
        assert col in df.columns
    report = (tmp_path / "out" / "reports" / "dirac_spinor_limit_report.md").read_text(
        encoding="utf-8",
    )
    assert "NOT FULL FERMION UNIFICATION" in report
    assert BANNER_DIRAC in report
    assert (tmp_path / "out" / "figures" / "dirac_dispersion_relation.png").is_file()


def test_all_benchmark_cases_pass() -> None:
    assert run_clifford_algebra_check().pass_
    assert run_flat_dirac_dispersion().pass_
    assert run_positive_energy_plane_wave().pass_
    assert run_tdf_spinor_phase_ansatz().pass_
    assert run_disformal_metric_spinor_safety().pass_
    assert run_compact_tau_mass_ladder().pass_

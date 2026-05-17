"""Phase 5D — Covariant action consistency checks."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tdf_obs.validation.covariant_action_checks import (
    BANNER_COVARIANT,
    NfwSummaryMissingError,
    REQUIRED_SUMMARY_COLUMNS,
    d_phi_tau_dr_analytic,
    d_phi_tau_dr_numeric,
    disformal_safety_proxy,
    effective_density_proxy,
    ensure_nfw_summary,
    k_essence_stability_proxy,
    load_nfw_fitted_parameters,
    run_covariant_action_checks_pipeline,
    run_covariant_checks_for_case,
    weak_field_rotation_identity_error,
)
from tdf_obs.validation.nfw_surrogate import run_nfw_surrogate_pipeline


def test_analytic_derivative_matches_numeric() -> None:
    B, r0 = 2000.0, 4.0
    r = np.linspace(0.5, 25.0, 60)
    ana = d_phi_tau_dr_analytic(r, B, r0)
    num = d_phi_tau_dr_numeric(r, B, r0)
    np.testing.assert_allclose(ana, num, rtol=1e-3, atol=1e-6)


def test_weak_field_identity_error_small() -> None:
    r = np.linspace(0.3, 30.0, 50)
    err, passed = weak_field_rotation_identity_error(r, 1500.0, 3.0)
    assert passed
    assert err < 1e-10


def test_disformal_proxy_finite() -> None:
    r = np.linspace(1.0, 20.0, 40)
    max_dg, passed = disformal_safety_proxy(r, 1800.0, 5.0)
    assert np.isfinite(max_dg)
    assert passed


def test_stability_proxy_passes_beta_zero() -> None:
    X = np.array([0.0, 1.0, 100.0, 1e4])
    _, _, passed = k_essence_stability_proxy(X, beta=0.0)
    assert bool(np.all(passed))


def test_stability_proxy_fails_bad_beta() -> None:
    X = np.array([10.0])
    _, _, passed = k_essence_stability_proxy(X, beta=-1.0, Lambda=1.0)
    assert not bool(passed)


def test_density_proxy_finite_valid_params() -> None:
    r = np.linspace(0.5, 25.0, 50)
    rho = effective_density_proxy(r, 1200.0, 2.5)
    assert np.all(np.isfinite(rho))
    assert np.nanmax(rho) <= 1.0 + 1e-9


def test_run_checks_for_known_case() -> None:
    res = run_covariant_checks_for_case("milky_way_like", 2000.0, 5.0)
    assert res.weak_field_identity_pass
    assert res.overall_action_consistency_pass


def test_pipeline_report_banner(tmp_path: Path) -> None:
    out = tmp_path / "out"
    run_nfw_surrogate_pipeline(outputs_root=out, case_names=["dwarf_low_mass"])
    df, results = run_covariant_action_checks_pipeline(
        outputs_root=out,
        case_names=["dwarf_low_mass"],
        run_nfw_if_missing=False,
    )
    report = (out / "reports" / "covariant_action_checks_report.md").read_text(encoding="utf-8")
    assert "NOT REAL OBSERVATIONAL DATA" in report
    assert BANNER_COVARIANT in report
    for col in REQUIRED_SUMMARY_COLUMNS:
        assert col in df.columns
    assert len(results) == 1


def test_missing_nfw_summary_no_run(tmp_path: Path) -> None:
    missing = tmp_path / "outputs" / "tables" / "nfw_surrogate_fit_summary.csv"
    with pytest.raises(NfwSummaryMissingError, match="run_nfw_surrogate"):
        load_nfw_fitted_parameters(missing)


def test_ensure_nfw_summary_message_when_no_run(tmp_path: Path) -> None:
    path = tmp_path / "tables" / "nfw_surrogate_fit_summary.csv"
    with pytest.raises(NfwSummaryMissingError, match="run_nfw_surrogate"):
        ensure_nfw_summary(path, outputs_root=tmp_path, run_if_missing=False)


def test_script_missing_summary_no_nfw(capsys) -> None:
    import subprocess
    import sys

    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "run_covariant_action_checks.py"
    env = {"PYTHONPATH": str(root / "src")}
    proc = subprocess.run(
        [sys.executable, str(script), "--no-run-nfw"],
        cwd=root,
        env={**dict(__import__("os").environ), **env},
        capture_output=True,
        text=True,
    )
    if (root / "outputs" / "tables" / "nfw_surrogate_fit_summary.csv").is_file():
        pytest.skip("NFW summary exists locally; cannot test missing path in CI")
    assert proc.returncode != 0
    assert "run_nfw_surrogate" in proc.stderr + proc.stdout

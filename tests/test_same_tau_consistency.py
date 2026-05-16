"""Phase 5C — Same-τ multi-observable consistency benchmark tests."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from tdf_obs.fitting.fit_rotation import _fit_tdf_params
from tdf_obs.validation.same_tau_consistency import (
    BANNER_SAME_TAU,
    BENCHMARK_CASE_REGISTRY,
    REQUIRED_SUMMARY_COLUMNS,
    PASS_TOLERANCE_PERCENT,
    generate_same_tau_teacher_case,
    fit_rotation_then_predict_lensing_redshift,
    get_benchmark_case,
    list_benchmark_cases,
    phi_tau_log,
    predict_lensing_proxy,
    predict_redshift_tau,
    run_same_tau_consistency_benchmark,
    tau_acceleration_term,
)


def test_phi_tau_log_derivative_matches_rotation_term() -> None:
    B, r0 = 1500.0, 3.0
    r = np.linspace(0.5, 20.0, 50)
    dr = 1e-5
    dphi_dr = (phi_tau_log(r + dr, B, r0) - phi_tau_log(r - dr, B, r0)) / (2 * dr)
    expected = float(B) / (r + r0)
    np.testing.assert_allclose(dphi_dr, expected, rtol=1e-4)
    np.testing.assert_allclose(r * dphi_dr, tau_acceleration_term(r, B, r0), rtol=1e-6)


def test_redshift_prediction_finite() -> None:
    z = predict_redshift_tau(np.array([2.0, 5.0]), np.array([10.0, 20.0]), 800.0, 2.5)
    assert np.all(np.isfinite(z))


def test_lensing_proxy_finite() -> None:
    R = np.linspace(1.0, 25.0, 30)
    alpha = predict_lensing_proxy(R, 1200.0, 4.0)
    assert np.all(np.isfinite(alpha))
    assert np.all(alpha > 0)


def test_rotation_fit_freezes_before_lensing_redshift() -> None:
    """Lensing/redshift predictions must use rotation-fitted (B, r0) only."""
    case = get_benchmark_case("same_tau_mid_mass")
    teacher = generate_same_tau_teacher_case(case)
    rot_df = teacher.rotation_df
    r = rot_df["r_kpc"].to_numpy()
    v_obs = rot_df["v_obs"].to_numpy()
    v_err = rot_df["v_err"].to_numpy()
    v_baryon = rot_df["v_baryon"].to_numpy()

    expected_B, expected_r0, _ = _fit_tdf_params(r, v_obs, v_err, v_baryon)

    with patch(
        "tdf_obs.validation.same_tau_consistency._fit_tdf_params",
        return_value=(expected_B, expected_r0, []),
    ) as mock_fit:
        with patch(
            "tdf_obs.validation.same_tau_consistency.predict_lensing_proxy",
            wraps=predict_lensing_proxy,
        ) as mock_lens:
            with patch(
                "tdf_obs.validation.same_tau_consistency.predict_redshift_tau",
                wraps=predict_redshift_tau,
            ) as mock_z:
                res = fit_rotation_then_predict_lensing_redshift(teacher)

    mock_fit.assert_called_once()
    mock_lens.assert_called()
    mock_z.assert_called()
    assert mock_lens.call_args[0][1] == expected_B
    assert mock_lens.call_args[0][2] == expected_r0
    assert mock_z.call_args[0][2] == expected_B
    assert mock_z.call_args[0][3] == expected_r0
    assert res.recovered_B == expected_B
    assert res.recovered_r0 == expected_r0


def test_registry_has_six_cases() -> None:
    assert len(BENCHMARK_CASE_REGISTRY) >= 6
    for name in (
        "same_tau_low_mass",
        "same_tau_mid_mass",
        "same_tau_high_mass",
        "same_tau_lsb",
        "same_tau_compact_baryon",
        "same_tau_core_like",
    ):
        assert name in list_benchmark_cases()


def test_one_case_end_to_end() -> None:
    case = get_benchmark_case("same_tau_low_mass")
    teacher = generate_same_tau_teacher_case(case)
    res = fit_rotation_then_predict_lensing_redshift(teacher)
    assert res.case_name == "same_tau_low_mass"
    assert np.isfinite(res.rotation_relative_error_percent)
    assert res.rotation_relative_error_percent < PASS_TOLERANCE_PERCENT


def test_pipeline_outputs(tmp_path: Path) -> None:
    df, results = run_same_tau_consistency_benchmark(
        outputs_root=tmp_path / "out",
        case_names=["same_tau_low_mass", "same_tau_mid_mass"],
    )
    assert len(results) == 2
    assert (tmp_path / "out" / "tables" / "same_tau_consistency_summary.csv").is_file()
    report = (tmp_path / "out" / "reports" / "same_tau_consistency_report.md").read_text(
        encoding="utf-8",
    )
    assert "NOT REAL OBSERVATIONAL DATA" in report
    assert BANNER_SAME_TAU in report
    for col in REQUIRED_SUMMARY_COLUMNS:
        assert col in df.columns
    assert (tmp_path / "out" / "figures" / "same_tau_same_tau_low_mass_rotation.png").is_file()


def test_summary_csv_columns(tmp_path: Path) -> None:
    df, _ = run_same_tau_consistency_benchmark(
        outputs_root=tmp_path / "out2",
        case_names=["same_tau_core_like"],
    )
    missing = [c for c in REQUIRED_SUMMARY_COLUMNS if c not in df.columns]
    assert not missing

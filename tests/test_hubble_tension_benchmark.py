"""Phase 5F — CMB-safe Hubble tension benchmark tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tdf_obs.validation.hubble_tension_benchmark import (
    BANNER_HUBBLE,
    REQUIRED_SUMMARY_COLUMNS,
    classify_hubble_case,
    compute_cmb_safety_metrics,
    compute_hubble_shift_metrics,
    get_benchmark_case,
    run_hubble_tension_benchmark,
    run_single_hubble_case,
)
from tdf_obs.validation.cmb_acoustic_benchmark import CosmologyParams


def test_zero_tau_zero_H0_shift_and_cmb_safe() -> None:
    res = run_single_hubble_case(get_benchmark_case("zero_tau"))
    assert res.cmb_safe_pass
    assert abs(res.shift.H0_shift_percent) < 0.01
    assert not res.hubble_shift_active
    assert not res.overall_success


def test_late_mild_shifts_H0_and_cmb_safe() -> None:
    res = run_single_hubble_case(get_benchmark_case("late_mild_3_percent"))
    assert res.cmb_safe_pass
    assert res.cmb.H_zstar_error_percent < 1.0
    assert 2.0 <= abs(res.shift.H0_shift_percent) <= 10.0


def test_recombination_leakage_fails_cmb_safe() -> None:
    res = run_single_hubble_case(get_benchmark_case("recombination_leakage_bad"))
    assert not res.cmb_safe_pass
    assert res.expected_status == "fail_cmb"


def test_early_leakage_fails_cmb_or_warns() -> None:
    res = run_single_hubble_case(get_benchmark_case("early_leakage_bad"))
    assert not res.cmb_safe_pass
    assert res.cmb.ell_A_error_percent > 1.0 or not res.cmb_safe_pass


def test_classification_logic() -> None:
    params = CosmologyParams()
    shift = compute_hubble_shift_metrics(params, "zero_tau", {})
    cmb = compute_cmb_safety_metrics(params, "zero_tau", {})
    safe, active, overall = classify_hubble_case(shift, cmb)
    assert safe
    assert not active
    assert not overall


def test_pipeline_csv_report_figures(tmp_path: Path) -> None:
    df, results = run_hubble_tension_benchmark(
        cases=["zero_tau", "late_moderate_6_percent", "recombination_leakage_bad"],
        outputs_root=tmp_path / "out",
    )
    assert len(results) == 3
    for col in REQUIRED_SUMMARY_COLUMNS:
        assert col in df.columns
    report = (tmp_path / "out" / "reports" / "hubble_tension_benchmark_report.md").read_text(
        encoding="utf-8",
    )
    assert "NOT REAL OBSERVATIONAL DATA" in report
    assert BANNER_HUBBLE in report
    assert (tmp_path / "out" / "figures" / "hubble_tension_epsilon_tau_cases.png").is_file()
    assert (tmp_path / "out" / "figures" / "hubble_tension_H_ratio_cases.png").is_file()


def test_late_moderate_overall_success() -> None:
    res = run_single_hubble_case(get_benchmark_case("late_moderate_6_percent"))
    assert res.overall_success
    assert res.expected_status == "success"


def test_lcdm_control_cmb_safe_only() -> None:
    res = run_single_hubble_case(get_benchmark_case("lcdm_control"))
    assert res.cmb_safe_pass
    assert not res.overall_success
    assert res.expected_status == "cmb_safe_only"

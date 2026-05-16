"""Phase 4D redshift/Doppler sanity benchmark tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from tdf_obs.validation.redshift_sanity_benchmark import (
    BANNER_REDSHIFT_SANITY,
    REQUIRED_SUMMARY_COLUMNS,
    classify_redshift_case,
    get_benchmark_case,
    run_redshift_sanity_benchmark,
    run_redshift_sanity_benchmark_pipeline,
    summarize_redshift_benchmark,
)


def test_negligible_case_passes() -> None:
    case = get_benchmark_case("negligible_tau_shift")
    assert classify_redshift_case(case.z_tau, case.max_allowed_abs_z_tau) == "pass"
    res = run_redshift_sanity_benchmark([case])[0]
    assert res.status == "pass"
    assert res.matches_expected


def test_borderline_case_classified_borderline() -> None:
    case = get_benchmark_case("borderline_shift")
    assert classify_redshift_case(case.z_tau, case.max_allowed_abs_z_tau) == "borderline"
    res = run_redshift_sanity_benchmark([case])[0]
    assert res.status == "borderline"
    assert res.warning


def test_too_large_case_fails() -> None:
    case = get_benchmark_case("too_large_shift")
    assert classify_redshift_case(case.z_tau, case.max_allowed_abs_z_tau) == "fail"
    res = run_redshift_sanity_benchmark([case])[0]
    assert res.status == "fail"
    assert res.expected_status == "fail"


def test_classify_boundaries() -> None:
    limit = 1e-8
    assert classify_redshift_case(0.79e-8, limit) == "pass"
    assert classify_redshift_case(0.8e-8, limit) == "borderline"
    assert classify_redshift_case(1.0e-8, limit) == "borderline"
    assert classify_redshift_case(1.1e-8, limit) == "fail"


def test_summarize_counts() -> None:
    results = run_redshift_sanity_benchmark()
    summary = summarize_redshift_benchmark(results)
    assert summary["n_cases"] == 7
    assert summary["n_pass"] + summary["n_borderline"] + summary["n_fail"] == 7
    assert summary["n_fail"] >= 1
    assert summary["n_borderline"] >= 1
    assert summary["n_failed_as_expected"] == 1


def test_pipeline_report_and_csv(tmp_path: Path) -> None:
    df, summary = run_redshift_sanity_benchmark_pipeline(outputs_root=tmp_path / "out")
    report = (tmp_path / "out" / "reports" / "redshift_sanity_benchmark_report.md").read_text(
        encoding="utf-8",
    )
    assert BANNER_REDSHIFT_SANITY in report
    assert "NOT REAL OBSERVATIONAL DATA" in report
    for col in REQUIRED_SUMMARY_COLUMNS:
        assert col in df.columns, f"missing column {col}"
    assert df["is_real_observational_data"].eq(False).all()
    assert summary["n_matches_expected"] == 7

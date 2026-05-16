"""Phase 4B expanded GR-safe local benchmark tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from tdf_obs.validation.gr_safe_benchmark import (
    BANNER_GR_SAFE,
    BENCHMARK_CASE_REGISTRY,
    REQUIRED_SUMMARY_COLUMNS,
    get_benchmark_case,
    list_benchmark_cases,
    run_gr_safe_benchmark_pipeline,
    run_single_gr_safe_case,
)


def test_registry_has_seven_cases() -> None:
    assert len(BENCHMARK_CASE_REGISTRY) == 7
    expected = {
        "mercury_perihelion",
        "earth_orbit",
        "gps_weak_field_clock",
        "light_bending_sun",
        "shapiro_delay",
        "lunar_laser_ranging",
        "binary_pulsar_weak_field",
    }
    assert set(BENCHMARK_CASE_REGISTRY) == expected


def test_each_case_has_required_fields() -> None:
    for name, case in BENCHMARK_CASE_REGISTRY.items():
        fields = case.required_fields()
        assert fields["case_name"] == name
        assert fields["phi_b_scale"] != 0
        assert fields["max_allowed_epsilon"] > 0
        assert (
            fields["assumed_phi_tau_scale"] is not None or fields["assumed_epsilon_tau"] is not None
        )


def test_binary_pulsar_uses_assumed_epsilon_tau() -> None:
    case = get_benchmark_case("binary_pulsar_weak_field")
    assert case.assumed_epsilon_tau is not None
    phi_tau = case.resolve_phi_tau()
    assert phi_tau == case.assumed_epsilon_tau * case.phi_b_scale


@pytest.mark.parametrize("case_name", list_benchmark_cases())
def test_all_registry_cases_pass_scaffold(case_name: str) -> None:
    res = run_single_gr_safe_case(get_benchmark_case(case_name))
    assert res.status == "pass"
    assert res.pass_


def test_pipeline_outputs_and_report(tmp_path: Path) -> None:
    df = run_gr_safe_benchmark_pipeline(outputs_root=tmp_path / "out")
    assert (tmp_path / "out" / "tables" / "gr_safe_benchmark_summary.csv").is_file()
    report = (tmp_path / "out" / "reports" / "gr_safe_benchmark_report.md").read_text(encoding="utf-8")
    assert BANNER_GR_SAFE in report
    assert "NOT REAL OBSERVATIONAL DATA" in report
    assert "compatibility scaffold" in report.lower()
    assert "real observational constraints are not yet fitted" in report.lower()
    for col in REQUIRED_SUMMARY_COLUMNS:
        assert col in df.columns, f"missing column {col}"
    assert df["is_real_observational_data"].eq(False).all()
    assert len(df) == 7


def test_single_case_cli_subset(tmp_path: Path) -> None:
    df = run_gr_safe_benchmark_pipeline(
        outputs_root=tmp_path / "out",
        case_names=["earth_orbit"],
    )
    assert len(df) == 1
    assert df["case_name"].iloc[0] == "earth_orbit"

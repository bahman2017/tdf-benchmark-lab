"""Phase 3C ΛCDM benchmark scaffold tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tdf_obs.validation.lcdm_benchmark import (
    BANNER_LCDM_BENCHMARK,
    run_black_hole_gr_limit_benchmark,
    run_lcdm_benchmark_pipeline,
    run_redshift_sanity_benchmark,
    run_solar_system_gr_benchmark,
)


def test_solar_system_benchmark_pass_fail() -> None:
    rows = run_solar_system_gr_benchmark()
    assert len(rows) == 4
    assert all(r.benchmark == "solar_system_gr_safe" for r in rows)
    assert any(r.status == "pass" for r in rows)


def test_black_hole_gr_limit_at_zero() -> None:
    rows = run_black_hole_gr_limit_benchmark(rc_over_rs_values=(0.0,))
    assert rows[0].status == "pass"
    assert np.isclose(rows[0].extra["r_nr_ratio"], 1.0)
    assert np.isclose(rows[0].extra["T_ratio"], 1.0)


def test_black_hole_small_rc_near_gr() -> None:
    rows = run_black_hole_gr_limit_benchmark(rc_over_rs_values=(1e-6,))
    assert rows[0].status == "pass"
    assert rows[0].extra["r_nr_ratio"] > 0.999


def test_redshift_sanity_expected_fail_case() -> None:
    rows = run_redshift_sanity_benchmark()
    too_large = [r for r in rows if r.case_name == "too_large"][0]
    assert too_large.status == "fail"


def test_pipeline_outputs_and_banner(tmp_path: Path) -> None:
    df = run_lcdm_benchmark_pipeline(outputs_root=tmp_path / "out", project_root=tmp_path)
    assert (tmp_path / "out" / "tables" / "lcdm_benchmark_summary.csv").is_file()
    report = (tmp_path / "out" / "reports" / "lcdm_benchmark_report.md").read_text(encoding="utf-8")
    assert BANNER_LCDM_BENCHMARK in report
    assert "not observational validation" in report.lower()
    assert df["is_real_observational_data"].eq(False).all()
    assert df["benchmark_mode"].eq("lcdm_benchmark_recovery").all()
    assert "solar_system_gr_safe" in df["benchmark"].values

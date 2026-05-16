"""Phase 4C black-hole exterior GR-limit benchmark tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tdf_obs.validation.black_hole_gr_benchmark import (
    BANNER_BH_GR_LIMIT,
    REQUIRED_SUMMARY_COLUMNS,
    classify_bh_gr_limit,
    expected_gr_ratio,
    run_black_hole_gr_benchmark_pipeline,
    run_black_hole_gr_limit_benchmark,
)


def test_q_zero_gives_ratios_one() -> None:
    rows = run_black_hole_gr_limit_benchmark([0.0])
    row = rows[0]
    assert row.q == 0.0
    np.testing.assert_allclose(row.r_nr_ratio, 1.0, rtol=0, atol=1e-12)
    np.testing.assert_allclose(row.temperature_ratio, 1.0, rtol=0, atol=1e-12)
    assert row.status == "GR-like"
    assert row.deviation_from_gr_percent < 0.1


def test_small_q_gr_like() -> None:
    rows = run_black_hole_gr_limit_benchmark([1e-8, 1e-6])
    assert all(r.status == "GR-like" for r in rows)


def test_q_half_strongly_modified() -> None:
    rows = run_black_hole_gr_limit_benchmark([0.5])
    row = rows[0]
    assert row.status == "strongly_modified"
    exp = expected_gr_ratio(0.5)
    np.testing.assert_allclose(row.temperature_ratio, exp, rtol=1e-6)
    assert row.deviation_from_gr_percent >= 5.0


def test_q_ge_one_no_horizon() -> None:
    rows = run_black_hole_gr_limit_benchmark([1.0, 1.1])
    assert rows[0].status == "no_horizon"
    assert not np.isfinite(rows[0].r_nr_ratio)
    assert rows[1].status == "no_horizon"


def test_classify_bh_gr_limit_boundaries() -> None:
    assert classify_bh_gr_limit(0.0, 1.0) == "GR-like"
    assert classify_bh_gr_limit(0.01, 0.9999) == "GR-like"
    assert classify_bh_gr_limit(0.5, 0.866) == "strongly_modified"
    assert classify_bh_gr_limit(1.0, float("nan")) == "no_horizon"


def test_pipeline_report_and_csv(tmp_path: Path) -> None:
    df = run_black_hole_gr_benchmark_pipeline(outputs_root=tmp_path / "out")
    report = (tmp_path / "out" / "reports" / "black_hole_gr_benchmark_report.md").read_text(
        encoding="utf-8",
    )
    assert BANNER_BH_GR_LIMIT in report
    assert "NOT REAL OBSERVATIONAL DATA" in report
    assert "phenomenological" in report.lower()
    for col in REQUIRED_SUMMARY_COLUMNS:
        assert col in df.columns, f"missing column {col}"
    assert df["is_real_observational_data"].eq(False).all()
    assert len(df) >= 12

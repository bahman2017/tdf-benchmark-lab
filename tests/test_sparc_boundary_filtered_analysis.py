"""SPARC Step 1 — boundary-filtered analysis tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tdf_obs.utils.run_outputs import make_versioned_output_dir
from tdf_obs.validation.sparc_boundary_filtered_analysis import (
    BANNER_BOUNDARY_FILTERED,
    FILTER_COMPARISON_COLUMNS,
    GALAXY_FLAG_COLUMNS,
    build_galaxy_boundary_flags,
    compute_filter_statistics,
    load_comparison_table,
    run_boundary_filtered_analysis,
    _apply_filter,
)

ROOT = Path(__file__).resolve().parents[1]
INPUT_RUN = ROOT / "outputs" / "runs" / "v0.20.2_corrected_mond_sparc_calibration"
COMPARISON_CSV = INPUT_RUN / "tables" / "sparc_model_comparison_by_galaxy.csv"
BOUNDARY_CSV = ROOT / "outputs" / "tables" / "sparc_parameter_boundary_flags.csv"


@pytest.fixture
def comparison_df() -> pd.DataFrame:
    if not COMPARISON_CSV.is_file():
        pytest.skip("corrected MOND comparison table missing")
    return load_comparison_table(COMPARISON_CSV)


def test_load_comparison_and_delta_bic(comparison_df: pd.DataFrame) -> None:
    assert "delta_bic_tdf_vs_nfw" in comparison_df.columns
    row = comparison_df.iloc[0]
    expected = float(row["bic_tdf_kessence"] - row["bic_nfw"])
    assert abs(float(row["delta_bic_tdf_vs_nfw"]) - expected) < 1e-6


def test_boundary_filter_reduces_or_equals_count(comparison_df: pd.DataFrame) -> None:
    if not BOUNDARY_CSV.is_file():
        pytest.skip("boundary flags missing")
    flags, available, _ = build_galaxy_boundary_flags(
        comparison_df, BOUNDARY_CSV, None,
    )
    assert available
    all_sub = _apply_filter(comparison_df, flags, "all_galaxies")
    filt_sub = _apply_filter(comparison_df, flags, "exclude_any_nfw_or_tdf_boundary_hit")
    assert len(filt_sub) <= len(all_sub)


def test_compute_filter_statistics_columns(comparison_df: pd.DataFrame) -> None:
    flags, _, _ = build_galaxy_boundary_flags(
        comparison_df,
        BOUNDARY_CSV if BOUNDARY_CSV.is_file() else None,
        INPUT_RUN / "tables" / "sparc_real_calibration_summary.csv",
    )
    sub = _apply_filter(comparison_df, flags, "all_galaxies")
    stats = compute_filter_statistics(sub, "all_galaxies", True)
    for col in FILTER_COMPARISON_COLUMNS:
        assert col in stats


def test_pipeline_writes_outputs(tmp_path: Path, comparison_df: pd.DataFrame) -> None:
    input_run = tmp_path / "input_run"
    (input_run / "tables").mkdir(parents=True)
    comparison_df.to_csv(
        input_run / "tables" / "sparc_model_comparison_by_galaxy.csv",
        index=False,
    )
    layout = make_versioned_output_dir(tmp_path, "test_boundary_filtered")
    result = run_boundary_filtered_analysis(
        input_run,
        layout.run_dir,
        boundary_flags_path=BOUNDARY_CSV if BOUNDARY_CSV.is_file() else None,
    )
    assert len(result.comparison_by_filter) == 3
    out_cmp = layout.tables / "boundary_filtered_model_comparison.csv"
    out_flags = layout.tables / "boundary_filter_galaxy_flags.csv"
    assert out_cmp.is_file()
    assert out_flags.is_file()
    cmp_df = pd.read_csv(out_cmp)
    for col in FILTER_COMPARISON_COLUMNS:
        assert col in cmp_df.columns
    flag_df = pd.read_csv(out_flags)
    for col in GALAXY_FLAG_COLUMNS:
        assert col in flag_df.columns
    report = (layout.reports / "boundary_filtered_analysis_report.md").read_text()
    assert "NOT FULL OBSERVATIONAL VALIDATION" in report
    assert BANNER_BOUNDARY_FILTERED in report


def test_production_run_not_written_by_test(tmp_path: Path, comparison_df: pd.DataFrame) -> None:
    prod = ROOT / "outputs" / "runs" / "sparc_step_1_boundary_filtered"
    report = prod / "reports" / "boundary_filtered_analysis_report.md"
    old_mtime = report.stat().st_mtime if report.is_file() else None
    input_run = tmp_path / "input_run"
    (input_run / "tables").mkdir(parents=True)
    comparison_df.head(5).to_csv(
        input_run / "tables" / "sparc_model_comparison_by_galaxy.csv",
        index=False,
    )
    layout = make_versioned_output_dir(tmp_path, "test_boundary_filtered")
    run_boundary_filtered_analysis(input_run, layout.run_dir)
    if old_mtime is not None:
        assert report.stat().st_mtime == old_mtime

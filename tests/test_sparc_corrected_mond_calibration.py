"""Phase 8A.2 — Corrected analytic MOND SPARC calibration tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tdf_obs.utils.run_outputs import (
    assert_production_output_safe,
    make_versioned_output_dir,
    resolve_versioned_run_dir,
)
from tdf_obs.validation.sparc_real_calibration import (
    A0_MOND_DEFAULT,
    BANNER_SPARC_CORRECTED_MOND,
    CORRECTED_COMPARISON_COLUMNS,
    SUMMARY_COLUMNS,
    check_mond_activity,
    mond_g_baryon_analytic,
    run_sparc_real_calibration,
    v_mond_analytic,
    v_mond_simple,
)

ROOT = Path(__file__).resolve().parents[1]
SPARC_CSV = ROOT / "data" / "processed" / "sparc_rotation.csv"
PRODUCTION_RUN_DIR = (
    ROOT / "outputs" / "runs" / "v0.20.2_corrected_mond_sparc_calibration"
)
PRODUCTION_REPORT = (
    PRODUCTION_RUN_DIR / "reports" / "sparc_real_calibration_report.md"
)


def test_analytic_mond_g_ge_gb() -> None:
    g_b = np.array([1.0, 100.0, 2000.0])
    g_m = mond_g_baryon_analytic(g_b, A0_MOND_DEFAULT)
    assert np.all(g_m >= g_b - 1e-9)


def test_corrected_mond_differs_from_baryon_low_acc() -> None:
    r = np.array([1.0, 3.0, 8.0, 15.0])
    v_gas = np.zeros(4)
    v_disk = np.array([25.0, 22.0, 18.0, 14.0])
    v_bulge = np.zeros(4)
    v_m, v_b, g_b = v_mond_analytic(
        r, v_gas, v_disk, v_bulge, 1.0, 0.0, return_components=True,
    )
    low = g_b < A0_MOND_DEFAULT
    assert np.any(low)
    assert np.any(v_m[low] > v_b[low] + 0.5)


def test_corrected_mond_finite_velocities() -> None:
    r = np.linspace(0.5, 20.0, 30)
    v = v_mond_simple(
        r,
        np.zeros_like(r),
        np.full_like(r, 30.0),
        np.zeros_like(r),
        0.8,
        0.0,
        A0_MOND_DEFAULT,
        analytic=True,
    )
    assert np.all(np.isfinite(v))
    assert np.all(v >= 0)


def test_mond_activity_flag_low_acc() -> None:
    r = np.linspace(1.0, 30.0, 25)
    act = check_mond_activity(
        r,
        np.zeros_like(r),
        np.linspace(40.0, 15.0, 25),
        np.zeros_like(r),
        1.0,
        0.0,
    )
    assert act["mond_active_flag"]


def test_production_guard_blocks_test_input_to_production_run() -> None:
    with pytest.raises(ValueError, match="Test-like input"):
        assert_production_output_safe(
            "/tmp/pytest-123/sparc.csv",
            ROOT / "outputs",
            "v0.20.2_corrected_mond_sparc_calibration",
            project_outputs_root=ROOT / "outputs",
        )


@pytest.fixture
def sparc_df() -> pd.DataFrame:
    if not SPARC_CSV.is_file():
        pytest.skip("sparc_rotation.csv missing")
    return pd.read_csv(SPARC_CSV)


def test_corrected_mond_pipeline_outputs(tmp_path: Path, sparc_df: pd.DataFrame) -> None:
    inp = tmp_path / "sparc.csv"
    sparc_df.to_csv(inp, index=False)
    layout = make_versioned_output_dir(tmp_path, "test_corrected_mond_run")
    summary, comparison, stats = run_sparc_real_calibration(
        inp,
        layout.run_dir,
        max_galaxies=3,
        corrected_mond=True,
        max_example_plots=2,
        file_suffix="",
        run_id="test_corrected_mond_run",
    )
    assert stats.corrected_mond
    assert "corrected_mond" in summary["model"].values
    for col in SUMMARY_COLUMNS:
        assert col in summary.columns
    for col in CORRECTED_COMPARISON_COLUMNS:
        assert col in comparison.columns
    report = (layout.reports / "sparc_real_calibration_report.md").read_text()
    assert "NOT FULL OBSERVATIONAL VALIDATION" in report
    assert "CORRECTED MOND BASELINE" in report
    assert BANNER_SPARC_CORRECTED_MOND in report
    assert (layout.tables / "sparc_real_calibration_summary.csv").is_file()
    assert not (tmp_path / "outputs" / "tables").exists()


def test_versioned_run_timestamp_when_exists(tmp_path: Path) -> None:
    layout1 = resolve_versioned_run_dir(tmp_path, "test_run_dup", overwrite_run=True)
    (layout1.tables / "marker.txt").write_text("x")
    layout2 = resolve_versioned_run_dir(tmp_path, "test_run_dup", overwrite_run=False)
    assert layout2.run_id != layout1.run_id
    assert layout2.run_id.startswith("test_run_dup__")


def test_production_report_not_touched_by_test_run(
    tmp_path: Path,
    sparc_df: pd.DataFrame,
) -> None:
    if not PRODUCTION_REPORT.is_file():
        pytest.skip("production corrected-MOND report not present")
    old_mtime = PRODUCTION_REPORT.stat().st_mtime
    inp = tmp_path / "sparc.csv"
    sparc_df.to_csv(inp, index=False)
    layout = make_versioned_output_dir(tmp_path, "test_corrected_mond_run")
    run_sparc_real_calibration(
        inp,
        layout.run_dir,
        max_galaxies=2,
        corrected_mond=True,
        file_suffix="",
    )
    assert PRODUCTION_REPORT.stat().st_mtime == old_mtime

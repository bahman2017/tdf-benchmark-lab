"""SPARC Step 6 — residual diagnostics tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tdf_obs.utils.run_outputs import make_versioned_output_dir
from tdf_obs.validation.sparc_residual_diagnostics import (
    BANNER_RESIDUAL,
    BY_POINT_COLUMNS,
    FAILURE_MODE_COLUMNS,
    build_failure_modes,
    build_tdf_nfw_comparison,
    compute_residuals_for_galaxy,
    galaxy_residual_metrics,
    radial_zone,
    run_residual_diagnostics,
    zone_mask,
)

ROOT = Path(__file__).resolve().parents[1]
INPUT_RUN = ROOT / "outputs" / "runs" / "v0.20.2_corrected_mond_sparc_calibration"
SPARC_CSV = ROOT / "data" / "processed" / "sparc_rotation.csv"


def test_residual_calculation() -> None:
    r = np.array([1.0, 5.0, 10.0])
    v_obs = np.array([20.0, 40.0, 50.0])
    v_err = np.array([2.0, 2.0, 2.0])
    v_model = v_obs + np.array([1.0, -1.0, 0.5])
    met = galaxy_residual_metrics(r, v_obs, v_err, v_model)
    assert np.allclose(met["residual"], v_model - v_obs)
    assert np.all(np.isfinite(met["weighted_residual"]))


def test_inner_mid_outer_masks_cover_points() -> None:
    r = np.linspace(1, 20, 20)
    zones = radial_zone(r, float(r.max()))
    assert zone_mask(zones, "inner").any()
    assert zone_mask(zones, "middle").any()
    assert zone_mask(zones, "outer").any()
    assert zone_mask(zones, "inner").sum() + zone_mask(zones, "middle").sum() + zone_mask(
        zones, "outer",
    ).sum() == len(r)


def test_weighted_residual_finite() -> None:
    r = np.array([1.0, 2.0, 3.0])
    met = galaxy_residual_metrics(
        r,
        np.array([10.0, 20.0, 30.0]),
        np.array([1.0, 1.0, 1.0]),
        np.array([11.0, 19.0, 31.0]),
    )
    assert np.all(np.isfinite(met["weighted_residual"]))


@pytest.fixture
def calibration_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not INPUT_RUN.is_dir():
        pytest.skip("v0.20.2 calibration run missing")
    summary = pd.read_csv(INPUT_RUN / "tables" / "sparc_real_calibration_summary.csv")
    comp = pd.read_csv(INPUT_RUN / "tables" / "sparc_model_comparison_by_galaxy.csv")
    return summary, comp


@pytest.fixture
def sparc_df() -> pd.DataFrame:
    if not SPARC_CSV.is_file():
        pytest.skip("sparc_rotation.csv missing")
    return pd.read_csv(SPARC_CSV)


def test_failure_mode_table_columns(
    calibration_tables: tuple[pd.DataFrame, pd.DataFrame],
    sparc_df: pd.DataFrame,
) -> None:
    _, comp = calibration_tables
    summary, _ = calibration_tables
    gid = summary["galaxy_id"].iloc[0]
    gdf = sparc_df[sparc_df["galaxy_id"] == gid]
    _, gals = compute_residuals_for_galaxy(
        str(gid), gdf, summary[summary["galaxy_id"] == gid], "dwarf", 50.0,
    )
    by_gal = pd.DataFrame(gals)
    tdf_nfw = build_tdf_nfw_comparison(by_gal)
    props = pd.DataFrame({"galaxy_id": [gid], "galaxy_class": ["dwarf"]})
    fm = build_failure_modes(comp, tdf_nfw, props)
    for col in FAILURE_MODE_COLUMNS:
        assert col in fm.columns


def test_pipeline_tmp_path_only(
    tmp_path: Path,
    calibration_tables: tuple[pd.DataFrame, pd.DataFrame],
    sparc_df: pd.DataFrame,
) -> None:
    summary, comp = calibration_tables
    inp_run = tmp_path / "input_run"
    (inp_run / "tables").mkdir(parents=True)
    sparc_sub = sparc_df.head(200)
    gids = set(sparc_sub["galaxy_id"].astype(str))
    summary = summary[summary["galaxy_id"].astype(str).isin(gids)]
    comp = comp[comp["galaxy_id"].astype(str).isin(gids)]
    summary.to_csv(inp_run / "tables" / "sparc_real_calibration_summary.csv", index=False)
    comp.to_csv(inp_run / "tables" / "sparc_model_comparison_by_galaxy.csv", index=False)
    sparc_path = tmp_path / "sparc.csv"
    sparc_sub.to_csv(sparc_path, index=False)

    layout = make_versioned_output_dir(tmp_path, "test_residuals")
    result = run_residual_diagnostics(inp_run, sparc_path, layout.run_dir, max_galaxies=3)
    assert len(result.by_point) > 0
    for col in BY_POINT_COLUMNS:
        assert col in result.by_point.columns
    report = (layout.reports / "residual_diagnostics_report.md").read_text()
    assert "NOT FULL OBSERVATIONAL VALIDATION" in report
    assert BANNER_RESIDUAL in report

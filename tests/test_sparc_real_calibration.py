"""Phase 8A — Real SPARC calibration tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tdf_obs.data.sparc_parser import REQUIRED_SPARC_COLUMNS
from tdf_obs.fitting.metrics import aic, bic, chi_square
from tdf_obs.validation.sparc_real_calibration import (
    BANNER_SPARC_CALIBRATION,
    COMPARISON_COLUMNS,
    SUMMARY_COLUMNS,
    SparcCalibrationSchemaError,
    compute_v_baryon,
    fit_galaxy_all_models,
    run_sparc_real_calibration,
    validate_sparc_input_schema,
    v_mond_simple,
    v_nfw_total,
    v_tdf_kessence_disk_proxy,
)

ROOT = Path(__file__).resolve().parents[1]
SPARC_CSV = ROOT / "data" / "processed" / "sparc_rotation.csv"


@pytest.fixture
def sparc_df() -> pd.DataFrame:
    if not SPARC_CSV.is_file():
        pytest.skip("sparc_rotation.csv not found; run parse_sparc_real_data.py first")
    return pd.read_csv(SPARC_CSV)


def test_input_schema_matches_parser_output(sparc_df: pd.DataFrame) -> None:
    for col in REQUIRED_SPARC_COLUMNS:
        assert col in sparc_df.columns
    validate_sparc_input_schema(sparc_df)


def test_schema_rejects_missing_columns() -> None:
    with pytest.raises(SparcCalibrationSchemaError):
        validate_sparc_input_schema(pd.DataFrame({"galaxy_id": ["A"]}))


def test_baryonic_velocity_calculation() -> None:
    v = compute_v_baryon(
        np.array([10.0, 20.0]),
        np.array([30.0, 40.0]),
        np.array([0.0, 0.0]),
        0.5,
        0.0,
    )
    assert np.all(np.isfinite(v))
    assert np.all(v >= 0)


def test_baryon_only_finite(sparc_df: pd.DataFrame) -> None:
    gid = sparc_df["galaxy_id"].iloc[0]
    gdf = sparc_df[sparc_df["galaxy_id"] == gid]
    fits, comp = fit_galaxy_all_models(gid, gdf, fit_ml=False)
    b = next(f for f in fits if f.model == "baryon_only")
    assert np.all(np.isfinite(b.v_pred))
    assert comp is not None


def test_nfw_finite(sparc_df: pd.DataFrame) -> None:
    gid = sparc_df["galaxy_id"].iloc[0]
    gdf = sparc_df[sparc_df["galaxy_id"] == gid]
    r = gdf["r_kpc"].to_numpy()
    v = v_nfw_total(
        r,
        gdf["v_gas"].to_numpy(),
        gdf["v_disk"].to_numpy(),
        gdf["v_bulge"].to_numpy(),
        0.5,
        0.0,
        80.0,
        2.0,
    )
    assert np.all(np.isfinite(v))


def test_mond_finite(sparc_df: pd.DataFrame) -> None:
    gid = sparc_df["galaxy_id"].iloc[0]
    gdf = sparc_df[sparc_df["galaxy_id"] == gid]
    r = gdf["r_kpc"].to_numpy()
    v = v_mond_simple(
        r,
        gdf["v_gas"].to_numpy(),
        gdf["v_disk"].to_numpy(),
        gdf["v_bulge"].to_numpy(),
        0.5,
        0.0,
        3700.0,
    )
    assert np.all(np.isfinite(v))


def test_tdf_finite(sparc_df: pd.DataFrame) -> None:
    gid = sparc_df["galaxy_id"].iloc[0]
    gdf = sparc_df[sparc_df["galaxy_id"] == gid]
    r = gdf["r_kpc"].to_numpy()
    v = v_tdf_kessence_disk_proxy(
        r,
        gdf["v_gas"].to_numpy(),
        gdf["v_disk"].to_numpy(),
        gdf["v_bulge"].to_numpy(),
        0.5,
        0.0,
        1.0,
        3700.0,
    )
    assert np.all(np.isfinite(v))


def test_aic_bic_penalize_extra_parameters() -> None:
    c2 = 10.0
    n = 20
    assert aic(c2, 2) > aic(c2, 1)
    assert bic(c2, n, 3) > bic(c2, n, 1)


def test_subset_pipeline(tmp_path: Path, sparc_df: pd.DataFrame) -> None:
    inp = tmp_path / "sparc.csv"
    sparc_df.to_csv(inp, index=False)
    summary, comparison, stats = run_sparc_real_calibration(
        inp,
        tmp_path / "out",
        max_galaxies=3,
        quality_min_points=5,
        fit_ml=True,
        max_example_plots=2,
    )
    assert stats.galaxies_fitted >= 1
    for col in SUMMARY_COLUMNS:
        assert col in summary.columns
    for col in COMPARISON_COLUMNS:
        assert col in comparison.columns
    report = (tmp_path / "out" / "reports" / "sparc_real_calibration_report.md").read_text(
        encoding="utf-8",
    )
    assert BANNER_SPARC_CALIBRATION in report
    assert "NOT FULL OBSERVATIONAL VALIDATION" in report
    assert (tmp_path / "out" / "tables" / "sparc_real_calibration_summary.csv").is_file()

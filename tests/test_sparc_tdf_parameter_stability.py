"""SPARC Step 5 — TDF parameter stability tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tdf_obs.utils.run_outputs import make_versioned_output_dir
from tdf_obs.validation.sparc_tdf_parameter_stability import (
    BANNER_TDF_STABILITY,
    SUMMARY_COLUMNS,
    build_stability_summary,
    classify_outliers,
    extract_tdf_parameters,
    merge_parameter_table,
    run_tdf_parameter_stability,
    tdf_boundary_flags,
)

ROOT = Path(__file__).resolve().parents[1]
INPUT_RUN = ROOT / "outputs" / "runs" / "v0.20.2_corrected_mond_sparc_calibration"
SPARC_CSV = ROOT / "data" / "processed" / "sparc_rotation.csv"


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


def test_extracts_tdf_parameters(calibration_tables: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    summary, _ = calibration_tables
    tdf = extract_tdf_parameters(summary)
    assert (tdf["model"] == "tdf_kessence").all()
    assert "beta_over_M" in tdf.columns
    assert tdf["beta_over_M"].notna().all()


def test_finite_summary_statistics(calibration_tables: tuple[pd.DataFrame, pd.DataFrame], sparc_df: pd.DataFrame) -> None:
    summary, comp = calibration_tables
    tdf = extract_tdf_parameters(summary)
    from tdf_obs.validation.sparc_tdf_parameter_stability import compute_galaxy_observables

    obs = compute_galaxy_observables(sparc_df, tdf)
    by_gal, _ = merge_parameter_table(
        tdf, obs, comp, sparc_df, run_global_beta_test=False,
    )
    stab = build_stability_summary(by_gal)
    assert "beta_over_M_median" in stab["metric"].values
    med = float(stab.loc[stab["metric"] == "beta_over_M_median", "value"].iloc[0])
    assert np.isfinite(med)
    assert med > 0


def test_detects_boundary_hits() -> None:
    row = pd.Series({"beta_over_M": 1e-4, "upsilon_disk": 3.0, "upsilon_bulge": np.nan})
    flags = tdf_boundary_flags(row, has_bulge=False)
    assert flags["beta_over_M_at_bound"]
    assert flags["upsilon_disk_at_bound"]
    assert flags["any_tdf_boundary_hit"]


def test_classifies_outliers() -> None:
    df = pd.DataFrame(
        {
            "galaxy_id": ["a", "b", "c", "d"],
            "beta_over_M": [0.2, 0.25, 0.3, 5.0],
            "reduced_chi2_tdf": [1.0, 1.1, 1.2, 50.0],
            "bic_tdf": [10.0, 11.0, 12.0, 13.0],
            "galaxy_class": ["dwarf"] * 4,
            "low_acceleration_fraction": [0.5] * 4,
            "any_tdf_boundary_hit": [False, False, False, True],
            "delta_bic_global_beta_approx": [0.0, 0.0, 0.0, 0.0],
        },
    )
    flagged, outliers = classify_outliers(df)
    assert flagged["is_parameter_outlier"].any()
    assert len(outliers) >= 1


def test_pipeline_tmp_path_only(tmp_path: Path, calibration_tables: tuple[pd.DataFrame, pd.DataFrame], sparc_df: pd.DataFrame) -> None:
    summary, comp = calibration_tables
    inp_run = tmp_path / "input_run"
    (inp_run / "tables").mkdir(parents=True)
    sparc_sub = sparc_df.head(300)
    sparc_path = tmp_path / "sparc.csv"
    sparc_sub.to_csv(sparc_path, index=False)
    gids = set(sparc_sub["galaxy_id"].astype(str))
    summary = summary[summary["galaxy_id"].astype(str).isin(gids)]
    comp = comp[comp["galaxy_id"].astype(str).isin(gids)]
    summary.to_csv(inp_run / "tables" / "sparc_real_calibration_summary.csv", index=False)
    comp.to_csv(inp_run / "tables" / "sparc_model_comparison_by_galaxy.csv", index=False)

    layout = make_versioned_output_dir(tmp_path, "test_tdf_stability")
    result = run_tdf_parameter_stability(
        inp_run,
        sparc_path,
        layout.run_dir,
        run_global_beta_test=True,
        global_beta_max_galaxies=3,
    )
    assert len(result.by_galaxy) >= 1
    for col in SUMMARY_COLUMNS:
        assert col in result.summary.columns
    report = (layout.reports / "tdf_parameter_stability_report.md").read_text()
    assert "NOT FULL OBSERVATIONAL VALIDATION" in report
    assert BANNER_TDF_STABILITY in report


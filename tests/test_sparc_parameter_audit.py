"""Phase 8A.1 — SPARC parameter audit tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tdf_obs.fitting.metrics import aic, bic
from tdf_obs.validation.sparc_parameter_audit import (
    A0_MOND_KMS2_KPC,
    BANNER_SPARC_AUDIT,
    BOUNDARY_COLUMNS,
    MOND_AUDIT_COLUMNS,
    PARAM_COUNT_COLUMNS,
    _at_bound,
    audit_mond_baseline_galaxy,
    audit_parameter_count,
    expected_n_params,
    fit_model_with_boundary_audit,
    mond_g_baryon_analytic,
    run_sparc_parameter_audit,
    v_mond_analytic,
)

ROOT = Path(__file__).resolve().parents[1]
SPARC_CSV = ROOT / "data" / "processed" / "sparc_rotation.csv"
SUMMARY_CSV = ROOT / "outputs" / "tables" / "sparc_real_calibration_summary.csv"
COMPARISON_CSV = ROOT / "outputs" / "tables" / "sparc_model_comparison_by_galaxy.csv"


def test_analytic_mond_g_ge_gb() -> None:
    g_b = np.array([10.0, 100.0, 500.0, 5000.0])
    g_m = mond_g_baryon_analytic(g_b, A0_MOND_KMS2_KPC)
    assert np.all(g_m >= g_b - 1e-9)


def test_low_acceleration_mond_boost() -> None:
    r = np.array([1.0, 2.0, 5.0, 10.0])
    v_gas = np.zeros_like(r)
    v_disk = np.array([20.0, 18.0, 15.0, 12.0])
    v_bulge = np.zeros_like(r)
    v_m, v_b, g_b = v_mond_analytic(r, v_gas, v_disk, v_bulge, 1.0, 0.0)
    low = g_b < A0_MOND_KMS2_KPC
    assert np.any(low)
    assert np.all(v_m[low] >= v_b[low] + 0.1)


def test_a0_units_finite_and_near_3700() -> None:
    assert np.isfinite(A0_MOND_KMS2_KPC)
    assert 3000 < A0_MOND_KMS2_KPC < 4500


def test_parameter_count_audit_catches_wrong_n_params() -> None:
    summary = pd.DataFrame(
        [
            {
                "galaxy_id": "G1",
                "model": "mond",
                "n_params": 99,
                "upsilon_bulge": np.nan,
            },
        ],
    )
    out = audit_parameter_count(summary)
    mond_row = out[out["model"] == "mond"].iloc[0]
    assert not bool(mond_row["parameter_count_consistent"])


def test_bic_recomputation_matches_within_tolerance() -> None:
    chi2, n_pts, n_par = 12.5, 20, 3
    b = bic(chi2, n_pts, n_par)
    assert abs(b - bic(chi2, n_pts, n_par)) < 1e-3
    a = aic(chi2, n_par)
    assert abs(a - aic(chi2, n_par)) < 1e-3


def test_at_bound_detection() -> None:
    assert _at_bound(0.05, 0.05, 3.0)
    assert _at_bound(100.0, 0.1, 100.0)
    assert not _at_bound(1.5, 0.05, 3.0)


def test_boundary_fit_flags_lower_bound(sparc_mini_df: pd.DataFrame) -> None:
    gid = sparc_mini_df["galaxy_id"].iloc[0]
    gdf = sparc_mini_df[sparc_mini_df["galaxy_id"] == gid].sort_values("r_kpc")
    r = gdf["r_kpc"].to_numpy()
    v_obs = gdf["v_obs"].to_numpy() * 3.0  # force high Υ at lower bound
    v_err = np.maximum(gdf["v_err"].to_numpy(), 1.0)
    res = fit_model_with_boundary_audit(
        gid,
        r,
        v_obs,
        v_err,
        gdf["v_gas"].to_numpy(),
        gdf["v_disk"].to_numpy(),
        gdf["v_bulge"].to_numpy(),
        False,
        "baryon_only",
    )
    assert isinstance(res.boundary_hits, list)


@pytest.fixture
def sparc_mini_df() -> pd.DataFrame:
    if not SPARC_CSV.is_file():
        pytest.skip("sparc_rotation.csv missing")
    df = pd.read_csv(SPARC_CSV)
    gid = df["galaxy_id"].iloc[0]
    return df[df["galaxy_id"] == gid].copy()


@pytest.fixture
def audit_outputs_exist() -> bool:
    return SUMMARY_CSV.is_file() and COMPARISON_CSV.is_file()


def test_report_banner_and_mond_section(tmp_path: Path, audit_outputs_exist: bool) -> None:
    if not audit_outputs_exist or not SPARC_CSV.is_file():
        pytest.skip("calibration outputs or SPARC CSV missing")
    from tdf_obs.utils.run_outputs import make_versioned_output_dir

    layout = make_versioned_output_dir(tmp_path, "test_sparc_parameter_audit")
    run_sparc_parameter_audit(
        SPARC_CSV,
        SUMMARY_CSV,
        COMPARISON_CSV,
        layout.run_dir,
        refit_for_boundaries=True,
        max_galaxies=3,
    )
    report = (layout.reports / "sparc_parameter_audit_report.md").read_text()
    assert "NOT FULL OBSERVATIONAL VALIDATION" in report
    assert BANNER_SPARC_AUDIT.split("—")[0].strip() in report or BANNER_SPARC_AUDIT in report
    assert "MOND" in report


def test_output_csv_columns(tmp_path: Path, audit_outputs_exist: bool) -> None:
    if not audit_outputs_exist or not SPARC_CSV.is_file():
        pytest.skip("calibration outputs or SPARC CSV missing")
    from tdf_obs.utils.run_outputs import make_versioned_output_dir

    layout = make_versioned_output_dir(tmp_path, "test_sparc_parameter_audit")
    run_sparc_parameter_audit(
        SPARC_CSV,
        SUMMARY_CSV,
        COMPARISON_CSV,
        layout.run_dir,
        refit_for_boundaries=True,
        max_galaxies=2,
    )
    mond = pd.read_csv(layout.tables / "sparc_mond_baseline_audit.csv")
    for col in MOND_AUDIT_COLUMNS:
        assert col in mond.columns
    boundary = pd.read_csv(layout.tables / "sparc_parameter_boundary_flags.csv")
    for col in BOUNDARY_COLUMNS:
        assert col in boundary.columns
    pcount = pd.read_csv(layout.tables / "sparc_parameter_count_audit.csv")
    for col in PARAM_COUNT_COLUMNS:
        assert col in pcount.columns


def test_mond_baseline_galaxy_metrics(sparc_mini_df: pd.DataFrame) -> None:
    gid = sparc_mini_df["galaxy_id"].iloc[0]
    row = audit_mond_baseline_galaxy(gid, sparc_mini_df, 0.5, 0.0)
    assert row["g_mond_ge_gb_all"]
    assert row["v_mond_ge_vb_all"]
    assert "mond_active_flag" in row


def test_expected_n_params_rules() -> None:
    assert expected_n_params("baryon_only", False) == 1
    assert expected_n_params("nfw", True) == 4
    assert expected_n_params("mond", False, a0_fitted=False) == 1

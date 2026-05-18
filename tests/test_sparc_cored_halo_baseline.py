"""SPARC Step 4 — cored halo baseline tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tdf_obs.fitting.metrics import bic
from tdf_obs.models.dark_matter import v2_burkert_halo_only, v2_pseudo_isothermal_halo_only
from tdf_obs.utils.run_outputs import make_versioned_output_dir
from tdf_obs.validation.sparc_cored_halo_baseline import (
    BANNER_CORED_HALO,
    COMPARISON_COLUMNS,
    fit_galaxy_cored_baseline,
    fit_model_cored_baseline,
    run_cored_halo_baseline,
    v_burkert_total,
    v_pseudo_isothermal_total,
)

ROOT = Path(__file__).resolve().parents[1]
SPARC_CSV = ROOT / "data" / "processed" / "sparc_rotation.csv"


@pytest.fixture
def sparc_df() -> pd.DataFrame:
    if not SPARC_CSV.is_file():
        pytest.skip("sparc_rotation.csv missing")
    return pd.read_csv(SPARC_CSV)


def test_burkert_velocity_finite_nonnegative() -> None:
    r = np.linspace(0.01, 30.0, 50)
    v2 = v2_burkert_halo_only(r, v_core=80.0, r_core=2.0)
    assert np.all(np.isfinite(v2))
    assert np.all(v2 >= 0.0)
    v = np.sqrt(v2)
    assert np.all(np.isfinite(v))


def test_pseudo_isothermal_velocity_finite() -> None:
    r = np.linspace(0.001, 40.0, 60)
    v2 = v2_pseudo_isothermal_halo_only(r, v_inf=120.0, r_core=3.0)
    assert np.all(np.isfinite(v2))
    assert np.all(v2 >= 0.0)
    assert v2[0] < v2[-1]


def test_model_comparison_includes_burkert(sparc_df: pd.DataFrame) -> None:
    gid = sparc_df["galaxy_id"].iloc[0]
    gdf = sparc_df[sparc_df["galaxy_id"] == gid]
    fits, comp = fit_galaxy_cored_baseline(gid, gdf, min_points=5)
    models = {f.model for f in fits}
    assert "burkert" in models
    assert "nfw" in models
    assert "tdf_kessence" in models
    assert comp is not None
    assert "bic_burkert" in comp
    assert "delta_bic_tdf_vs_burkert" in comp


def test_bic_with_extra_halo_parameters() -> None:
    n = 20
    chi2 = 15.0
    assert bic(chi2, 3, n) > bic(chi2, 2, n)
    assert bic(chi2, 4, n) > bic(chi2, 3, n)


def test_burkert_total_matches_components() -> None:
    r = np.array([1.0, 5.0, 10.0])
    v_gas = np.array([10.0, 12.0, 11.0])
    v_disk = np.array([20.0, 25.0, 22.0])
    v_bulge = np.zeros(3)
    v = v_burkert_total(r, v_gas, v_disk, v_bulge, 0.5, 0.0, 60.0, 2.0)
    assert np.all(np.isfinite(v))
    assert np.all(v > 0)


def test_pseudo_isothermal_total_finite() -> None:
    r = np.array([0.05, 2.0, 15.0])
    v = v_pseudo_isothermal_total(
        r,
        np.full(3, 10.0),
        np.full(3, 20.0),
        np.zeros(3),
        0.5,
        0.0,
        100.0,
        2.0,
    )
    assert np.all(np.isfinite(v))


def test_fit_burkert_n_params(sparc_df: pd.DataFrame) -> None:
    gid = sparc_df["galaxy_id"].iloc[0]
    gdf = sparc_df[sparc_df["galaxy_id"] == gid]
    r = gdf["r_kpc"].to_numpy()
    fr = fit_model_cored_baseline(
        r,
        gdf["v_obs"].to_numpy(),
        gdf["v_err"].to_numpy(),
        gdf["v_gas"].to_numpy(),
        gdf["v_disk"].to_numpy(),
        gdf["v_bulge"].to_numpy(),
        False,
        "burkert",
    )
    assert fr.n_params == 3
    assert fr.bic > 0


def test_pipeline_tmp_path_only(tmp_path: Path, sparc_df: pd.DataFrame) -> None:
    inp = tmp_path / "sparc.csv"
    sparc_df.head(200).to_csv(inp, index=False)
    layout = make_versioned_output_dir(tmp_path, "test_cored_halo")
    result = run_cored_halo_baseline(
        inp,
        layout.run_dir,
        max_galaxies=2,
        quality_min_points=5,
    )
    assert len(result.by_galaxy) >= 1
    assert "burkert" in result.model_summary["model"].values
    for col in COMPARISON_COLUMNS:
        assert col in result.by_galaxy.columns
    report = (layout.reports / "cored_halo_baseline_report.md").read_text()
    assert "NOT FULL OBSERVATIONAL VALIDATION" in report
    assert BANNER_CORED_HALO in report

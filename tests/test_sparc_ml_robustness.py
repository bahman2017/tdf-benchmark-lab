"""SPARC Step 3 — M/L robustness tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tdf_obs.utils.run_outputs import make_versioned_output_dir
from tdf_obs.validation.sparc_ml_robustness import (
    BANNER_ML_ROBUSTNESS,
    ML_REGIMES,
    SUMMARY_COLUMNS,
    fit_model_ml_regime,
    fit_galaxy_ml_regime,
    run_ml_robustness_analysis,
)
from tdf_obs.validation.sparc_real_calibration import v_mond_analytic

ROOT = Path(__file__).resolve().parents[1]
SPARC_CSV = ROOT / "data" / "processed" / "sparc_rotation.csv"


@pytest.fixture
def sparc_df() -> pd.DataFrame:
    if not SPARC_CSV.is_file():
        pytest.skip("sparc_rotation.csv missing")
    return pd.read_csv(SPARC_CSV)


def test_fixed_regime_zero_ml_params(sparc_df: pd.DataFrame) -> None:
    regime = ML_REGIMES["A_fixed_canonical"]
    gid = sparc_df["galaxy_id"].iloc[0]
    gdf = sparc_df[sparc_df["galaxy_id"] == gid]
    r = gdf["r_kpc"].to_numpy()
    fr = fit_model_ml_regime(
        r,
        gdf["v_obs"].to_numpy(),
        gdf["v_err"].to_numpy(),
        gdf["v_gas"].to_numpy(),
        gdf["v_disk"].to_numpy(),
        gdf["v_bulge"].to_numpy(),
        False,
        "baryon_only",
        regime,
    )
    assert fr.n_params == 0
    assert abs(fr.params["upsilon_disk"] - 0.5) < 1e-9


def test_narrow_prior_bounds_enforced(sparc_df: pd.DataFrame) -> None:
    regime = ML_REGIMES["B_narrow_prior"]
    gid = sparc_df["galaxy_id"].iloc[0]
    gdf = sparc_df[sparc_df["galaxy_id"] == gid]
    fits, _ = fit_galaxy_ml_regime(gid, gdf, regime, min_points=5)
    b = next(f for f in fits if f.model == "baryon_only")
    assert 0.3 - 0.01 <= b.params["upsilon_disk"] <= 0.8 + 0.01


def test_corrected_mond_active(sparc_df: pd.DataFrame) -> None:
    gid = sparc_df["galaxy_id"].iloc[0]
    gdf = sparc_df[sparc_df["galaxy_id"] == gid]
    r = gdf["r_kpc"].to_numpy()
    v_m = v_mond_analytic(
        r,
        gdf["v_gas"].to_numpy(),
        gdf["v_disk"].to_numpy(),
        gdf["v_bulge"].to_numpy(),
        0.5,
        0.0,
    )
    v_b = np.sqrt(
        np.maximum(
            gdf["v_gas"].to_numpy() ** 2
            + 0.5 * gdf["v_disk"].to_numpy() ** 2,
            0,
        ),
    )
    assert np.any(v_m > v_b + 0.1)


def test_pipeline_tmp_path_only(tmp_path: Path, sparc_df: pd.DataFrame) -> None:
    inp = tmp_path / "sparc.csv"
    sparc_df.to_csv(inp, index=False)
    layout = make_versioned_output_dir(tmp_path, "test_ml_robustness")
    result = run_ml_robustness_analysis(
        inp,
        layout.run_dir,
        max_galaxies=3,
        quality_min_points=5,
        regimes=("A_fixed_canonical", "B_narrow_prior"),
    )
    assert len(result.summary) == 2
    for col in SUMMARY_COLUMNS:
        assert col in result.summary.columns
    report = (layout.reports / "ml_robustness_report.md").read_text()
    assert "NOT FULL OBSERVATIONAL VALIDATION" in report
    assert BANNER_ML_ROBUSTNESS in report

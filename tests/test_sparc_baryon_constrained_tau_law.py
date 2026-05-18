"""SPARC Step 6C — baryon-constrained τ law tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tdf_obs.fitting.metrics import bic
from tdf_obs.utils.run_outputs import make_versioned_output_dir
from tdf_obs.validation.sparc_baryon_constrained_tau_law import (
    BANNER_TAU_LAW,
    R_core,
    SUMMARY_COLUMNS,
    beta_eff_law_A,
    beta_eff_law_B,
    law_global_dim,
    n_params_tau_law,
    run_baryon_constrained_tau_law,
    v_tau_law,
)
from tdf_obs.validation.sparc_inverse_tau_response import R_core as R_core_imported


def test_R_core_finite_bounded() -> None:
    r = np.linspace(0.01, 20.0, 40)
    rc = R_core(r, 1.5)
    assert np.all(np.isfinite(rc))
    assert np.all(rc >= 0)
    assert np.allclose(R_core_imported(r, 1.5), rc)


def test_beta_laws_nonnegative_finite() -> None:
    a_b = np.array([50.0, 500.0, 5000.0])
    assert np.all(beta_eff_law_A(a_b, 0.3, 1.0, 3700.0) >= 0)
    assert np.all(np.isfinite(beta_eff_law_B(a_b, 0.05, 0.8, 1.0, 3700.0)))


def test_v_tau_law_finite() -> None:
    r = np.linspace(0.5, 12.0, 15)
    v = v_tau_law(
        r,
        np.full(15, 5.0),
        np.full(15, 18.0),
        np.zeros(15),
        0.5,
        0.0,
        "A",
        np.array([0.3, 1.0]),
        1.5,
    )
    assert np.all(np.isfinite(v))
    assert np.all(v >= 0)


def test_bic_parameter_count_increases() -> None:
    n_a = n_params_tau_law("A", False, "global_r_c")
    n_b = n_params_tau_law("B", False, "global_r_c")
    n_d = n_params_tau_law("D", True, "bounded_galaxy_r_c")
    assert n_b > n_a
    assert n_d > n_b
    assert law_global_dim("D") == 4


@pytest.fixture
def minimal_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    sparc = tmp_path / "sparc.csv"
    rows = []
    for gid in ("g1", "g2"):
        for rk in [0.5, 1.0, 2.0, 5.0, 10.0, 14.0]:
            rows.append(
                {
                    "galaxy_id": gid,
                    "r_kpc": rk,
                    "v_obs": 22.0 + rk,
                    "v_err": 2.0,
                    "v_gas": 5.0,
                    "v_disk": 14.0,
                    "v_bulge": 0.0,
                    "source": "t",
                    "dataset_mode": "real_sparc",
                    "real_observational_data": True,
                },
            )
    pd.DataFrame(rows).to_csv(sparc, index=False)

    d6a = tmp_path / "6a"
    (d6a / "tables").mkdir(parents=True)
    pd.DataFrame(
        {
            "galaxy_id": ["g1", "g2"],
            "galaxy_class": ["dwarf", "massive"],
            "median_beta_eff": [0.3, 0.25],
            "core_r_c_soft": [1.0, 2.0],
        },
    ).to_csv(d6a / "tables" / "tau_effective_halo_summary.csv", index=False)

    d6b = tmp_path / "6b"
    (d6b / "tables").mkdir(parents=True)
    pd.DataFrame(
        {
            "galaxy_id": ["g1", "g2"],
            "bic_inverse_tdf_baryon_feature_beta": [20.0, 25.0],
        },
    ).to_csv(d6b / "tables" / "inverse_tau_comparison_by_galaxy.csv", index=False)

    return sparc, d6a, d6b


def test_pipeline_tmp_path(minimal_inputs: tuple[Path, Path, Path], tmp_path: Path) -> None:
    sparc, d6a, d6b = minimal_inputs
    layout = make_versioned_output_dir(tmp_path, "law6c")
    result = run_baryon_constrained_tau_law(
        sparc,
        layout.run_dir,
        inverse_design_run=d6a,
        inverse_response_run=d6b,
        max_galaxies=2,
    )
    summary = pd.read_csv(layout.tables / "tau_law_model_summary.csv")
    assert "model" in summary.columns
    assert "bic_win_count" in summary.columns
    assert "median_bic" in summary.columns
    assert any(summary["model"].str.startswith("tau_law_"))
    report = (layout.reports / "tau_law_report.md").read_text()
    assert "NOT FULL OBSERVATIONAL VALIDATION" in report
    assert BANNER_TAU_LAW in report
    assert len(result.comparison_by_galaxy) == 2

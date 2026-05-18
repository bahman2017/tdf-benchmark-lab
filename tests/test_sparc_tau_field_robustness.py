"""SPARC Step 6E — τ field robustness tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tdf_obs.utils.run_outputs import make_versioned_output_dir
from tdf_obs.validation.sparc_tau_field_robustness import (
    BANNER_TAU_FIELD_ROBUSTNESS,
    BOUNDARY_FILTER_COLUMNS,
    GLOBAL_LAMBDA_COLUMNS,
    PARAMETER_STABILITY_COLUMNS,
    boundary_filtered_summary,
    run_global_lambda_test,
    run_tau_field_robustness,
)
from tdf_obs.validation.sparc_tau_field_solver import (
    GalaxyFrame,
    run_tau_field_solver,
)

ROOT = Path(__file__).resolve().parents[1]


def _minimal_sparc(tmp_path: Path) -> Path:
    sparc = tmp_path / "sparc.csv"
    rows = []
    for gid in ("g1", "g2"):
        for rk in [0.5, 1.0, 2.0, 5.0, 10.0, 14.0]:
            rows.append(
                {
                    "galaxy_id": gid,
                    "r_kpc": rk,
                    "v_obs": 24.0 + rk,
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
    return sparc


def test_global_lambda_delta_bic_finite() -> None:
    gf = GalaxyFrame(
        galaxy_id="t",
        galaxy_class="dwarf",
        r=np.linspace(0.5, 12.0, 8),
        v_obs=np.linspace(20.0, 35.0, 8),
        v_err=np.full(8, 2.0),
        v_gas=np.full(8, 5.0),
        v_disk=np.full(8, 14.0),
        v_bulge=np.zeros(8),
        has_bulge=False,
        rmax=12.0,
    )
    out = run_global_lambda_test([gf])
    assert np.all(np.isfinite(out["delta_bic_global_vs_free"]))
    assert list(out.columns) == list(GLOBAL_LAMBDA_COLUMNS)


def test_boundary_filtered_summary_finite(tmp_path: Path) -> None:
    comp = pd.DataFrame(
        {
            "galaxy_id": ["g1", "g2"],
            "galaxy_class": ["dwarf", "dwarf"],
            "bic_tau_field_A": [10.0, 20.0],
            "bic_tau_field_E": [12.0, 18.0],
            "bic_old_tdf_baseline": [15.0, 15.0],
            "bic_pseudo_isothermal": [8.0, 25.0],
            "bic_nfw": [14.0, 16.0],
        },
    )
    bf = pd.DataFrame(
        {
            "galaxy_id": ["g1", "g1", "g2", "g2"],
            "model": ["tau_field_A", "nfw", "tau_field_A", "nfw"],
            "any_boundary_hit": [False, True, False, False],
        },
    )
    summ = boundary_filtered_summary(comp, bf)
    for col in BOUNDARY_FILTER_COLUMNS:
        assert col in summ.columns
    assert np.all(np.isfinite(summ["median_delta_bic_tau_field_A_vs_old_tdf"]))


@pytest.fixture
def mini_field_run(tmp_path: Path) -> Path:
    sparc = _minimal_sparc(tmp_path)
    layout = make_versioned_output_dir(tmp_path, "field6d")
    run_tau_field_solver(sparc, layout.run_dir, max_galaxies=2)
    return layout.run_dir


def test_pipeline_tmp_path(mini_field_run: Path, tmp_path: Path) -> None:
    sparc = _minimal_sparc(tmp_path)
    layout = make_versioned_output_dir(tmp_path, "robust6e")
    result = run_tau_field_robustness(
        mini_field_run,
        sparc,
        layout.run_dir,
        max_galaxies=2,
    )
    stab = pd.read_csv(layout.tables / "tau_field_parameter_stability.csv")
    for col in PARAMETER_STABILITY_COLUMNS:
        assert col in stab.columns
    report = (layout.reports / "tau_field_robustness_report.md").read_text()
    assert "NOT FULL OBSERVATIONAL VALIDATION" in report
    assert BANNER_TAU_FIELD_ROBUSTNESS in report
    assert len(result.global_parameter_test) == 2

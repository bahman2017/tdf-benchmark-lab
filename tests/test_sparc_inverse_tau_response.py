"""SPARC Step 6B — inverse τ response benchmark tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tdf_obs.fitting.metrics import bic
from tdf_obs.utils.run_outputs import make_versioned_output_dir
from tdf_obs.validation.sparc_inverse_tau_response import (
    BANNER_INVERSE_TAU,
    R_core,
    a_tau_inverse,
    beta_eff_saturation,
    n_params_for_model,
    run_inverse_tau_benchmark,
    v_inverse_tau,
)
ROOT = Path(__file__).resolve().parents[1]


def test_R_core_finite_bounded() -> None:
    r = np.linspace(0.01, 20.0, 50)
    rc = R_core(r, 1.5)
    assert np.all(np.isfinite(rc))
    assert np.all(rc >= 0)
    assert rc[0] < r[0]


def test_a_tau_nonnegative_finite() -> None:
    r = np.linspace(0.5, 15.0, 20)
    v2 = 100.0 + 50.0 * r
    a = a_tau_inverse(r, v2, 0.3, 3700.0, 1.0)
    assert np.all(np.isfinite(a))
    assert np.all(a >= 0)


def test_beta_eff_variants_finite() -> None:
    a_b = np.array([100.0, 500.0, 5000.0])
    be = beta_eff_saturation(a_b, 0.1, 0.8, 1.0, 3700.0)
    assert np.all(np.isfinite(be))


def test_bic_parameter_count_increases() -> None:
    b1, _ = n_params_for_model("baryon_only", False)
    b2, _ = n_params_for_model("old_tdf_baseline", False)
    b3, c3 = n_params_for_model("inverse_tdf_baryon_feature_beta", False)
    assert b2 > b1
    assert b3 > b2
    assert c3 == 0
    _, c_class = n_params_for_model("inverse_tdf_class_beta_core", False)
    assert c_class == 3


def test_inverse_tdf_prediction_finite() -> None:
    r = np.linspace(0.5, 10.0, 15)
    v = v_inverse_tau(
        r,
        np.full(15, 5.0),
        np.full(15, 20.0),
        np.zeros(15),
        0.5,
        0.0,
        0.3,
        3700.0,
        1.0,
    )
    assert np.all(np.isfinite(v))
    assert np.all(v >= 0)


def test_bic_formula_matches_metrics() -> None:
    chi2 = 10.0
    n, k = 20, 3
    assert bic(chi2, n, k) == pytest.approx(chi2 + k * np.log(n))


@pytest.fixture
def minimal_6a(tmp_path: Path) -> Path:
    run = tmp_path / "step6a"
    (run / "tables").mkdir(parents=True)
    pd.DataFrame(
        {
            "galaxy_id": ["g1", "g2"],
            "galaxy_class": ["dwarf", "massive"],
            "median_beta_eff": [0.3, 0.2],
            "core_r_c_soft": [1.0, 2.0],
        },
    ).to_csv(run / "tables" / "tau_effective_halo_summary.csv", index=False)
    pd.DataFrame({"galaxy_id": ["g1", "g2"]}).to_csv(
        run / "tables" / "tau_design_constraints.csv",
        index=False,
    )
    return run


@pytest.fixture
def minimal_sparc(tmp_path: Path) -> Path:
    path = tmp_path / "sparc.csv"
    rows = []
    for gid in ("g1", "g2"):
        for rk in [0.5, 1.0, 2.0, 4.0, 8.0, 12.0]:
            rows.append(
                {
                    "galaxy_id": gid,
                    "r_kpc": rk,
                    "v_obs": 25.0 + rk * 2,
                    "v_err": 2.0,
                    "v_gas": 5.0,
                    "v_disk": 15.0,
                    "v_bulge": 0.0,
                    "source": "test",
                    "dataset_mode": "real_sparc",
                    "real_observational_data": True,
                },
            )
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_pipeline_tmp_path(
    minimal_6a: Path,
    minimal_sparc: Path,
    tmp_path: Path,
) -> None:
    layout = make_versioned_output_dir(tmp_path, "inv_tau")
    result = run_inverse_tau_benchmark(
        minimal_sparc,
        layout.run_dir,
        inverse_design_run=minimal_6a,
        max_galaxies=2,
    )
    assert len(result.comparison_by_galaxy) == 2
    report = (layout.reports / "inverse_tau_response_report.md").read_text()
    assert "NOT FULL OBSERVATIONAL VALIDATION" in report
    assert BANNER_INVERSE_TAU in report
    assert "dark matter solved" not in report.lower()

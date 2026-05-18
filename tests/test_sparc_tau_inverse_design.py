"""SPARC Step 6A — τ inverse-design tests (tmp_path only for pipeline)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tdf_obs.utils.run_outputs import make_versioned_output_dir
from tdf_obs.validation.sparc_tau_inverse_design import (
    BANNER_TAU_INVERSE,
    CONSTRAINT_COLUMNS,
    HALO_SUMMARY_COLUMNS,
    PROFILE_COLUMNS,
    beta_eff_profile,
    compute_a_tau_required,
    fit_core_regularization,
    fit_outer_slope,
    run_tau_inverse_design,
    v2_baryon_user,
)

FORBIDDEN = (
    "dark matter solved",
    "observational validation",
    "replaces dark matter",
)


def test_a_tau_required_nonnegative_finite() -> None:
    r = np.array([0.5, 1.0, 2.0, 5.0, 10.0])
    v_obs = np.array([30.0, 40.0, 50.0, 55.0, 58.0])
    v2_b = np.array([100.0, 200.0, 400.0, 800.0, 1200.0])
    _, _, a_tau = compute_a_tau_required(r, v_obs, v2_b)
    assert np.all(np.isfinite(a_tau))
    assert np.all(a_tau >= 0)


def test_outer_slope_toy_mond_like() -> None:
    r = np.linspace(1.0, 20.0, 40)
    rmax = float(r.max())
    a_tau = 800.0 / np.maximum(r, 0.1)
    s, status = fit_outer_slope(r, a_tau, rmax)
    assert status == "ok"
    assert np.isfinite(s)
    assert -1.4 < s < -0.6


def test_beta_eff_finite() -> None:
    r = np.linspace(1.0, 10.0, 10)
    a_b = 500.0 / r
    a_tau = 200.0 / r
    beta = beta_eff_profile(a_tau, a_b, 3700.0)
    assert np.all(np.isfinite(beta))
    assert np.all(beta >= 0)


def test_core_fit_positive_rc() -> None:
    r = np.linspace(0.5, 15.0, 30)
    a_tau = 1200.0 / np.sqrt(r**2 + 1.5**2)
    _, rc, rmse = fit_core_regularization(r, a_tau, "soft")
    assert np.isfinite(rc) and rc > 0
    assert np.isfinite(rmse)


@pytest.fixture
def minimal_calibration(tmp_path: Path) -> Path:
    run = tmp_path / "cal"
    (run / "tables").mkdir(parents=True)
    rows = []
    for gid in ("g1", "g2"):
        rows.append(
            {
                "galaxy_id": gid,
                "model": "tdf_kessence",
                "n_points": 6,
                "success": True,
                "upsilon_disk": 0.5,
                "upsilon_bulge": 0.0,
                "a0": 3700.0,
                "beta_over_M": 0.3,
            },
        )
    pd.DataFrame(rows).to_csv(run / "tables" / "sparc_real_calibration_summary.csv", index=False)
    pd.DataFrame({"galaxy_id": ["g1", "g2"]}).to_csv(
        run / "tables" / "sparc_model_comparison_by_galaxy.csv",
        index=False,
    )
    return run


@pytest.fixture
def minimal_sparc(tmp_path: Path) -> Path:
    path = tmp_path / "sparc_rotation.csv"
    rows = []
    for gid in ("g1", "g2"):
        for i, rk in enumerate([0.5, 1.0, 2.0, 4.0, 8.0, 12.0], start=1):
            rows.append(
                {
                    "galaxy_id": gid,
                    "r_kpc": rk,
                    "v_obs": 20.0 + 3.0 * i,
                    "v_err": 2.0,
                    "v_gas": 5.0,
                    "v_disk": 15.0 + i,
                    "v_bulge": 0.0,
                    "source": "test",
                    "dataset_mode": "real_sparc",
                    "real_observational_data": True,
                },
            )
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_pipeline_tmp_path(
    minimal_calibration: Path,
    minimal_sparc: Path,
    tmp_path: Path,
) -> None:
    layout = make_versioned_output_dir(tmp_path, "tau_6a")
    result = run_tau_inverse_design(
        minimal_calibration,
        minimal_sparc,
        layout.run_dir,
    )
    assert len(result.halo_summary) == 2
    for col in HALO_SUMMARY_COLUMNS:
        assert col in result.halo_summary.columns
    for col in PROFILE_COLUMNS:
        assert col in result.profiles.columns
    for col in CONSTRAINT_COLUMNS:
        assert col in result.design_constraints.columns
    report = (layout.reports / "tau_inverse_design_report.md").read_text()
    assert "NOT FULL OBSERVATIONAL VALIDATION" in report
    assert BANNER_TAU_INVERSE in report
    lower = report.lower()
    assert "dark matter solved" not in lower
    assert "not" in lower and "replaces dark matter" in lower
    assert "not full observational validation" in lower
    assert "not constitute observational validation" in lower


def test_v2_baryon_user_gas_sign() -> None:
    v_gas = np.array([-10.0, 5.0])
    v2 = v2_baryon_user(v_gas, np.zeros(2), np.zeros(2), 0.5, 0.0)
    assert v2[0] == -100.0
    assert v2[1] == 25.0

"""SPARC Step 6D — τ field-equation solver tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tdf_obs.fitting.metrics import bic
from tdf_obs.utils.run_outputs import make_versioned_output_dir
from tdf_obs.validation.sparc_tau_field_solver import (
    BANNER_TAU_FIELD,
    PROFILE_COLUMNS,
    mu_tau,
    n_params_tau_field,
    predict_tau_field,
    solve_sigma_prime_at_radius,
    solve_sigma_prime_profile,
    run_tau_field_solver,
)

ROOT = Path(__file__).resolve().parents[1]


def test_mu_tau_positive_finite() -> None:
    y = np.linspace(0.01, 10.0, 20)
    for kind in ("A", "B", "C", "D", "E"):
        vals = [mu_tau(float(yi), kind, r_kpc=5.0, p=1.0, r_c=1.0) for yi in y]  # type: ignore[arg-type]
        assert np.all(np.isfinite(vals))
        assert np.all(np.asarray(vals) >= 0)


def test_sigma_prime_root_finite() -> None:
    for kind in ("A", "B", "C"):
        s = solve_sigma_prime_at_radius(100.0, kind, 3700.0, 5.0)  # type: ignore[arg-type]
        assert np.isfinite(s)
        assert s >= 0


def test_sigma_prime_profile_finite() -> None:
    r = np.linspace(0.5, 15.0, 12)
    sig = 500.0 / np.maximum(r, 0.1)
    sp, _ = solve_sigma_prime_profile(r, sig, 0.3, "B", 3700.0)
    assert np.all(np.isfinite(sp))


def test_v_total_prediction_finite() -> None:
    r = np.linspace(0.5, 12.0, 10)
    v, _, _, _, diag = predict_tau_field(
        r,
        np.full(10, 5.0),
        np.full(10, 16.0),
        np.zeros(10),
        0.5,
        0.0,
        "B",
        0.3,
        1.0,
        3700.0,
    )
    assert np.all(np.isfinite(v))
    assert np.all(v >= 0)
    assert isinstance(diag.stable, bool)


def test_bic_parameter_count() -> None:
    assert n_params_tau_field("A", False) == 2
    assert n_params_tau_field("E", True) == 5
    n, k = 20, n_params_tau_field("D", False)
    assert bic(10.0, n, k) == pytest.approx(10.0 + k * np.log(n))


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
    d6b = tmp_path / "6b"
    (d6b / "tables").mkdir(parents=True)
    pd.DataFrame(
        {
            "galaxy_id": ["g1", "g2"],
            "bic_inverse_tdf_baryon_feature_beta": [20.0, 22.0],
        },
    ).to_csv(d6b / "tables" / "inverse_tau_comparison_by_galaxy.csv", index=False)
    d6c = tmp_path / "6c"
    (d6c / "tables").mkdir(parents=True)
    pd.DataFrame(
        {
            "galaxy_id": ["g1", "g2"],
            "bic_tau_law_A": [30.0, 32.0],
            "bic_tau_law_B": [28.0, 31.0],
            "bic_tau_law_C": [35.0, 36.0],
            "bic_tau_law_D": [40.0, 41.0],
        },
    ).to_csv(d6c / "tables" / "tau_law_comparison_by_galaxy.csv", index=False)
    return sparc, d6b, d6c


def test_pipeline_tmp_path(minimal_inputs: tuple[Path, Path, Path], tmp_path: Path) -> None:
    sparc, d6b, d6c = minimal_inputs
    layout = make_versioned_output_dir(tmp_path, "field6d")
    result = run_tau_field_solver(
        sparc,
        layout.run_dir,
        inverse_response_run=d6b,
        tau_law_run=d6c,
        max_galaxies=2,
    )
    prof = pd.read_csv(layout.tables / "tau_field_profiles.csv")
    for col in PROFILE_COLUMNS:
        assert col in prof.columns
    assert len(result.comparison_by_galaxy) == 2
    report = (layout.reports / "tau_field_solver_report.md").read_text()
    assert "NOT FULL OBSERVATIONAL VALIDATION" in report
    assert BANNER_TAU_FIELD in report

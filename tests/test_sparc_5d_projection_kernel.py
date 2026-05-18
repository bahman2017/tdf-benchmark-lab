"""SPARC Step 6F — 5D projection kernel tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tdf_obs.fitting.metrics import bic
from tdf_obs.utils.run_outputs import make_versioned_output_dir
from tdf_obs.validation.sparc_5d_projection_kernel import (
    BANNER_PROJECTION,
    PROFILE_COLUMNS,
    _trapz_weights,
    build_normalized_kernel_matrix,
    n_params_projection,
    predict_projection,
    projected_source,
    run_5d_projection_kernel,
    solve_sigma_prime_from_source,
    KernelHyper,
)
from tdf_obs.validation.sparc_tau_field_solver import GalaxyFrame, run_tau_field_solver

ROOT = Path(__file__).resolve().parents[1]


def test_kernel_normalized_finite() -> None:
    r = np.linspace(0.5, 12.0, 15)
    w = _trapz_weights(r)
    for kind in ("A", "B", "C", "D", "E"):
        k = build_normalized_kernel_matrix(r, kind, 2.0, np.ones_like(r) * 10.0)  # type: ignore[arg-type]
        assert np.all(np.isfinite(k))
        assert np.all(k >= 0)
        for i in range(len(r)):
            row_int = float(np.sum(k[i, :] * r * w))
            assert row_int == pytest.approx(1.0, rel=0.08)


def test_projected_source_finite() -> None:
    r = np.linspace(0.5, 10.0, 12)
    sig = 100.0 / r
    k = build_normalized_kernel_matrix(r, "C", 1.5)
    s = projected_source(r, sig, k, 0.3)
    assert np.all(np.isfinite(s))
    assert np.all(s >= 0)


def test_field_and_velocity_finite() -> None:
    r = np.linspace(0.5, 12.0, 10)
    sig = 80.0 / r
    k = build_normalized_kernel_matrix(r, "B", 2.0)
    s = projected_source(r, sig, k, 0.25)
    sp, _ = solve_sigma_prime_from_source(r, s)
    assert np.all(np.isfinite(sp))
    v, _, _, _, _, _ = predict_projection(
        r, np.full(10, 5.0), np.full(10, 14.0), np.zeros(10), 0.5, 0.0, "C", KernelHyper(0.3, 2.0),
    )
    assert np.all(np.isfinite(v))
    assert np.all(v >= 0)


def test_bic_parameter_count() -> None:
    assert n_params_projection("A", False) == 2
    assert n_params_projection("E", True) == 6
    n, k = 20, n_params_projection("C", False)
    assert bic(10.0, n, k) == pytest.approx(10.0 + k * np.log(n))


@pytest.fixture
def mini_runs(tmp_path: Path) -> tuple[Path, Path, Path]:
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
    layout6d = make_versioned_output_dir(tmp_path, "field6d")
    run_tau_field_solver(sparc, layout6d.run_dir, max_galaxies=2)
    layout6e = make_versioned_output_dir(tmp_path, "robust6e")
    (layout6e.tables).mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "galaxy_id": ["g1", "g2"],
            "acceptable_delta_lt_6": [True, False],
        },
    ).to_csv(layout6e.tables / "tau_field_global_parameter_test.csv", index=False)
    return sparc, layout6d.run_dir, layout6e.run_dir


def test_pipeline_tmp_path(mini_runs: tuple[Path, Path, Path], tmp_path: Path) -> None:
    sparc, field_run, robust_run = mini_runs
    layout = make_versioned_output_dir(tmp_path, "proj6f")
    result = run_5d_projection_kernel(
        sparc,
        layout.run_dir,
        field_run=field_run,
        robustness_run=robust_run,
        max_galaxies=2,
    )
    prof = pd.read_csv(layout.tables / "projection_kernel_profiles.csv")
    for col in PROFILE_COLUMNS:
        assert col in prof.columns
    assert len(result.comparison_by_galaxy) == 2
    report = (layout.reports / "projection_kernel_report.md").read_text()
    assert "NOT FULL OBSERVATIONAL VALIDATION" in report
    assert BANNER_PROJECTION in report

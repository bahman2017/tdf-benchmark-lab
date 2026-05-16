"""Phase 5A core–cusp stress test."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tdf_obs.fitting.fit_rotation import N_PARAMS_BARYON, N_PARAMS_NFW, N_PARAMS_TDF
from tdf_obs.validation.core_cusp_stress import (
    BANNER_CORE_CUSP,
    BENCHMARK_CASE_REGISTRY,
    N_PARAMS_TDF_CORE,
    REQUIRED_SUMMARY_COLUMNS,
    cored_teacher_velocity,
    fit_core_cusp_case,
    generate_cored_teacher_case,
    generate_cuspy_teacher_case,
    get_benchmark_case,
    run_core_cusp_stress_pipeline,
    v2_core_proxy,
    v_core_proxy,
)


def test_core_proxy_finite_positive() -> None:
    r = np.linspace(0.2, 20.0, 40)
    v2 = v2_core_proxy(r, 3000.0, 1.5)
    v = v_core_proxy(r, 3000.0, 1.5)
    assert np.all(np.isfinite(v2))
    assert np.all(np.isfinite(v))
    assert np.all(v >= 0)


def test_cored_teacher_linear_near_center() -> None:
    r = np.array([0.01, 0.02, 0.05])
    v_b = np.zeros_like(r)
    Vc2, rc = 4000.0, 2.0
    v = cored_teacher_velocity(r, v_b, Vc2, rc)
    v_expected_slope = v_core_proxy(r, Vc2, rc)
    np.testing.assert_allclose(v, v_expected_slope, rtol=0.05)
    assert v[1] / r[1] > 0


def test_registry_has_at_least_eight_cases() -> None:
    assert len(BENCHMARK_CASE_REGISTRY) >= 8
    assert sum(1 for c in BENCHMARK_CASE_REGISTRY.values() if c.teacher_type == "cuspy") >= 4
    assert sum(1 for c in BENCHMARK_CASE_REGISTRY.values() if c.teacher_type == "cored") >= 4


def test_one_case_end_to_end() -> None:
    case = get_benchmark_case("core_small_rc")
    df = generate_cored_teacher_case(case)
    res = fit_core_cusp_case(case, df)
    assert res.case_name == "core_small_rc"
    assert res.teacher_type == "cored"
    assert np.isfinite(res.mse_tdf_core)


def test_bic_parameter_counts() -> None:
    assert N_PARAMS_BARYON == 0
    assert N_PARAMS_TDF == N_PARAMS_TDF_CORE == N_PARAMS_NFW == 2
    case = get_benchmark_case("nfw_mild_cusp")
    res = fit_core_cusp_case(case, generate_cuspy_teacher_case(case))
    assert np.isfinite(res.bic_baryon)
    assert np.isfinite(res.bic_tdf_simple)
    assert np.isfinite(res.bic_tdf_core)
    assert np.isfinite(res.bic_nfw)


def test_pipeline_outputs(tmp_path: Path) -> None:
    df, results = run_core_cusp_stress_pipeline(
        outputs_root=tmp_path / "out",
        case_names=["core_small_rc", "nfw_mild_cusp"],
    )
    assert len(results) == 2
    assert (tmp_path / "out" / "tables" / "core_cusp_stress_summary.csv").is_file()
    report = (tmp_path / "out" / "reports" / "core_cusp_stress_report.md").read_text(encoding="utf-8")
    assert BANNER_CORE_CUSP in report
    assert "NOT REAL OBSERVATIONAL DATA" in report
    for col in REQUIRED_SUMMARY_COLUMNS:
        assert col in df.columns, f"missing {col}"
    assert list((tmp_path / "out" / "figures").glob("core_cusp_*.png"))

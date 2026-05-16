"""Phase 5B rotation-curve diversity stress tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from tdf_obs.fitting.fit_rotation import N_PARAMS_BARYON, N_PARAMS_NFW, N_PARAMS_TDF
from tdf_obs.validation.core_cusp_stress import N_PARAMS_TDF_CORE
from tdf_obs.validation.rotation_diversity_stress import (
    BANNER_ROTATION_DIVERSITY,
    BENCHMARK_CASE_REGISTRY,
    REQUIRED_SUMMARY_COLUMNS,
    diversity_teacher_velocity,
    fit_diversity_case,
    generate_diversity_case,
    get_benchmark_case,
    run_rotation_diversity_stress,
)


def test_registry_has_at_least_ten_cases() -> None:
    assert len(BENCHMARK_CASE_REGISTRY) >= 10


def test_all_teachers_finite_positive() -> None:
    r = np.linspace(0.3, 20.0, 40)
    v_b = 30.0 * np.sqrt(r / (r + 2.0))
    for cfg in BENCHMARK_CASE_REGISTRY.values():
        v = diversity_teacher_velocity(r, v_b, cfg.teacher_shape_type, cfg.teacher_params)
        assert np.all(np.isfinite(v)), cfg.case_name
        assert np.all(v >= 0), cfg.case_name


def test_one_case_end_to_end() -> None:
    name = "fast_rising_flat"
    df = generate_diversity_case(name)
    res = fit_diversity_case(name, df)
    assert res.case_name == name
    assert np.isfinite(res.mse_tdf_simple)
    assert "galaxy_id" in df.columns
    assert df["dataset_mode"].iloc[0] != "real_data_calibration"


def test_pipeline_outputs(tmp_path: Path) -> None:
    df, results = run_rotation_diversity_stress(
        outputs_root=tmp_path / "out",
        case_names=["slow_rising_lsb", "declining_outer_curve"],
    )
    assert len(results) == 2
    report = (tmp_path / "out" / "reports" / "rotation_diversity_stress_report.md").read_text(
        encoding="utf-8",
    )
    assert BANNER_ROTATION_DIVERSITY in report
    assert "NOT REAL OBSERVATIONAL DATA" in report
    for col in REQUIRED_SUMMARY_COLUMNS:
        assert col in df.columns, col
    assert list((tmp_path / "out" / "figures").glob("rotation_diversity_*.png"))


def test_bic_four_model_families() -> None:
    assert N_PARAMS_BARYON == 0
    assert N_PARAMS_TDF == N_PARAMS_TDF_CORE == N_PARAMS_NFW == 2
    res = fit_diversity_case("rising_outer_curve", generate_diversity_case("rising_outer_curve"))
    for attr in ("bic_baryon", "bic_tdf_simple", "bic_tdf_core", "bic_nfw"):
        assert np.isfinite(getattr(res, attr))
    assert res.best_model_by_bic in ("baryon_only", "tdf_simple", "tdf_core", "nfw_simple")


def test_at_least_one_case_can_favor_nfw() -> None:
    """Full registry should include shapes where NFW wins by BIC (not a failure)."""
    from tdf_obs.validation.rotation_diversity_stress import list_benchmark_cases

    nfw_wins = sum(
        1
        for name in list_benchmark_cases()
        if fit_diversity_case(name, generate_diversity_case(name)).nfw_best_flag
    )
    assert nfw_wins >= 1, "expected at least one synthetic case where NFW is best by BIC"

"""Phase 5G — BAO/SNe distance consistency benchmark tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tdf_obs.validation.bao_sne_distance_benchmark import (
    BANNER_BAO_SNE,
    REQUIRED_SUMMARY_COLUMNS,
    angular_diameter_distance,
    bao_DV_proxy,
    classify_distance_case,
    comoving_distance,
    compute_distance_metrics,
    get_benchmark_case,
    luminosity_distance,
    run_bao_sne_distance_benchmark,
    run_single_distance_case,
)
from tdf_obs.validation.cmb_acoustic_benchmark import CosmologyParams, H_lcdm


def test_zero_tau_zero_distance_errors() -> None:
    res = run_single_distance_case(get_benchmark_case("zero_tau"))
    assert res.distance_safe_pass
    assert res.metrics.max_rel_error_D_L_percent < 0.01
    assert res.metrics.max_rel_error_D_M_percent < 0.01
    assert not res.hubble_shift_active


def test_late_mild_shifts_H0_and_distance_safe() -> None:
    res = run_single_distance_case(get_benchmark_case("late_mild_3_percent"))
    assert res.distance_safe_pass
    assert res.hubble_shift_active
    assert res.overall_success
    assert 2.0 <= abs(res.metrics.H0_shift_percent) <= 3.5


def test_distance_distorting_bad_fails() -> None:
    res = run_single_distance_case(get_benchmark_case("distance_distorting_bad"))
    assert not res.distance_safe_pass
    assert res.expected_status == "fail_distance"


def test_DL_equals_one_plus_z_DM() -> None:
    params = CosmologyParams()
    Hf = lambda z: H_lcdm(z, params)
    z = 0.5
    dm = comoving_distance(z, Hf)
    dl = luminosity_distance(z, Hf)
    np.testing.assert_allclose(dl, (1.0 + z) * dm, rtol=1e-6)


def test_DA_equals_DM_over_one_plus_z() -> None:
    params = CosmologyParams()
    Hf = lambda z: H_lcdm(z, params)
    z = 0.8
    dm = comoving_distance(z, Hf)
    da = angular_diameter_distance(z, Hf)
    np.testing.assert_allclose(da, dm / (1.0 + z), rtol=1e-6)


def test_DV_finite_positive() -> None:
    params = CosmologyParams()
    Hf = lambda z: H_lcdm(z, params)
    for z in (0.35, 1.0, 2.0):
        dv = bao_DV_proxy(z, Hf)
        assert np.isfinite(dv)
        assert dv > 0


def test_classification_logic() -> None:
    params = CosmologyParams()
    m = compute_distance_metrics(params, "zero_tau", {})
    safe, active, overall = classify_distance_case(m)
    assert safe
    assert not active
    assert not overall


def test_pipeline_outputs(tmp_path: Path) -> None:
    df, results = run_bao_sne_distance_benchmark(
        cases=["zero_tau", "late_moderate_6_percent", "distance_distorting_bad"],
        outputs_root=tmp_path / "out",
    )
    assert len(results) == 3
    for col in REQUIRED_SUMMARY_COLUMNS:
        assert col in df.columns
    report = (tmp_path / "out" / "reports" / "bao_sne_distance_benchmark_report.md").read_text(
        encoding="utf-8",
    )
    assert "NOT REAL OBSERVATIONAL DATA" in report
    assert BANNER_BAO_SNE in report
    fig_dir = tmp_path / "out" / "figures"
    assert (fig_dir / "bao_sne_distance_H_ratio_cases.png").is_file()
    assert (fig_dir / "bao_sne_distance_DL_ratio_cases.png").is_file()
    assert (fig_dir / "bao_sne_distance_DV_ratio_cases.png").is_file()


def test_late_moderate_hubble_active_only() -> None:
    res = run_single_distance_case(get_benchmark_case("late_moderate_6_percent"))
    assert res.hubble_shift_active
    assert not res.distance_safe_pass
    assert not res.overall_success
    assert res.expected_status == "hubble_active_only"

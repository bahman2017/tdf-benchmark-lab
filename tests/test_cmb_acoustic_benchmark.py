"""Phase 5E — CMB acoustic-scale compatibility benchmark tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tdf_obs.validation.cmb_acoustic_benchmark import (
    BANNER_CMB,
    CosmologyParams,
    REQUIRED_SUMMARY_COLUMNS,
    H_lcdm,
    H_tdf,
    acoustic_scale,
    comoving_distance,
    get_benchmark_case,
    run_cmb_acoustic_benchmark,
    run_single_cmb_case,
    sound_horizon_proxy,
    sound_speed_photon_baryon,
)


def test_H_lcdm_finite_positive() -> None:
    params = CosmologyParams()
    z = np.array([0.0, 1.0, 100.0, params.z_star])
    H = H_lcdm(z, params)
    assert np.all(np.isfinite(H))
    assert np.all(H > 0)


def test_H_tdf_equals_lcdm_for_zero_tau() -> None:
    params = CosmologyParams()
    z = np.linspace(0.0, 2000.0, 50)
    h_l = H_lcdm(z, params)
    h_t = H_tdf(z, params, "zero_tau", {})
    np.testing.assert_allclose(h_t, h_l, rtol=0.0, atol=0.0)


def test_sound_horizon_finite_positive() -> None:
    params = CosmologyParams()
    Hf = lambda z: H_lcdm(z, params)
    rs = sound_horizon_proxy(params.z_star, params.z_max_horizon, Hf, params, n_steps=800)
    assert np.isfinite(rs)
    assert rs > 0


def test_acoustic_scale_finite() -> None:
    ell = acoustic_scale(1000.0, 150.0)
    assert np.isfinite(ell)
    assert ell > 0


def test_zero_tau_case_passes_near_zero_error() -> None:
    res = run_single_cmb_case(get_benchmark_case("zero_tau"))
    assert res.cmb_compatibility_pass
    assert res.rel_error_ell_A_percent < 0.01
    assert res.rel_error_H_zstar_percent < 0.01


def test_too_large_tau_case_fails() -> None:
    res = run_single_cmb_case(get_benchmark_case("recombination_bump_too_large"))
    assert not res.cmb_compatibility_pass
    assert not res.expected_pass


def test_early_tau_too_large_fails() -> None:
    res = run_single_cmb_case(get_benchmark_case("early_tau_fraction_too_large"))
    assert not res.cmb_compatibility_pass


def test_pipeline_report_and_csv(tmp_path: Path) -> None:
    df, results = run_cmb_acoustic_benchmark(
        cases=["lcdm_control", "zero_tau", "recombination_bump_too_large"],
        outputs_root=tmp_path / "out",
    )
    assert len(results) == 3
    report = (tmp_path / "out" / "reports" / "cmb_acoustic_benchmark_report.md").read_text(
        encoding="utf-8",
    )
    assert "NOT REAL OBSERVATIONAL DATA" in report
    assert BANNER_CMB in report
    for col in REQUIRED_SUMMARY_COLUMNS:
        assert col in df.columns
    assert (tmp_path / "out" / "figures" / "cmb_acoustic_epsilon_tau_cases.png").is_file()


def test_sound_speed_positive() -> None:
    params = CosmologyParams()
    cs = sound_speed_photon_baryon(np.array([0.0, params.z_star]), params)
    assert np.all(cs > 0)


def test_comoving_distance_finite() -> None:
    params = CosmologyParams()
    D = comoving_distance(params.z_star, lambda z: H_lcdm(z, params), n_steps=800)
    assert np.isfinite(D)
    assert D > 0

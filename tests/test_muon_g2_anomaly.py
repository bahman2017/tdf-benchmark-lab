"""Phase 6H — Muon g-2 anomaly benchmark tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tdf_obs.validation.muon_g2_anomaly import (
    BANNER_CALIBRATION,
    BANNER_MUON_G2,
    DELTA_A_MU_EXP_DEFAULT,
    EPSILON_TAU_REFERENCE,
    INTERPRETATION_DISCLAIMER,
    MuonG2ExperimentalReference,
    REQUIRED_SUMMARY_COLUMNS,
    calculate_delta_a_mu,
    estimate_epsilon_tau,
    muon_compton_wavelength_m,
    run_muon_g2_anomaly_benchmark,
    run_reference_consistency_case,
)


def test_calculate_delta_a_mu_matches_experiment_at_reference_epsilon() -> None:
    delta = calculate_delta_a_mu(EPSILON_TAU_REFERENCE)
    assert delta == pytest.approx(DELTA_A_MU_EXP_DEFAULT, rel=0.02)


def test_estimate_epsilon_tau_positive_inputs() -> None:
    eps = estimate_epsilon_tau(1.7e-18, 1.17e-15)
    assert eps > 0.0
    assert eps == pytest.approx((1.7e-18 / 1.17e-15) ** 2, rel=1e-6)


def test_estimate_epsilon_tau_rejects_non_positive() -> None:
    with pytest.raises(ValueError):
        estimate_epsilon_tau(0.0, 1.0)
    with pytest.raises(ValueError):
        estimate_epsilon_tau(1.0, -1.0)


def test_calculate_delta_a_mu_rejects_negative_epsilon() -> None:
    with pytest.raises(ValueError):
        calculate_delta_a_mu(-1e-8)


def test_reference_consistency_case_passes() -> None:
    res = run_reference_consistency_case()
    assert res.overall_pass


def test_compton_wavelength_positive() -> None:
    assert muon_compton_wavelength_m() > 0.0


def test_experimental_reference_dataclass() -> None:
    ref = MuonG2ExperimentalReference(
        delta_a_mu=2.51e-10,
        uncertainty=0.48e-10,
    )
    lo, hi = ref.band()
    assert lo < ref.delta_a_mu < hi


def test_pipeline_outputs_and_banners(tmp_path: Path) -> None:
    df, results = run_muon_g2_anomaly_benchmark(outputs_root=tmp_path / "out")
    assert len(results) == 2
    for col in REQUIRED_SUMMARY_COLUMNS:
        assert col in df.columns
    report = (tmp_path / "out" / "reports" / "muon_g2_anomaly_report.md").read_text(
        encoding="utf-8",
    )
    assert BANNER_CALIBRATION in report
    assert BANNER_MUON_G2 in report
    assert INTERPRETATION_DISCLAIMER in report
    assert (tmp_path / "out" / "figures" / "muon_g2_epsilon_sweep.png").is_file()

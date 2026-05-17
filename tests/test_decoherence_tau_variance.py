"""Phase 6D — Decoherence from τ-variance benchmark tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tdf_obs.validation.decoherence_tau_variance import (
    BANNER_DECOHERENCE,
    REQUIRED_SUMMARY_COLUMNS,
    coherence_from_variance,
    decoherence_rate_from_variance,
    delta_tau,
    evolve_coherence_from_rate,
    generate_correlated_noise_tau_case,
    generate_environment_strength_sweep,
    generate_gaussian_noise_tau_case,
    generate_linear_decoherence_case,
    generate_mass_dependent_decoherence_case,
    run_coherent_control,
    run_decoherence_tau_variance_benchmark,
    run_linear_decoherence,
    variance_delta_tau,
)


def test_coherence_from_variance_zero() -> None:
    assert coherence_from_variance(0.0) == pytest.approx(1.0)


def test_coherence_decreases_with_variance() -> None:
    assert coherence_from_variance(0.1) > coherence_from_variance(1.0)
    assert coherence_from_variance(1.0) > coherence_from_variance(4.0)


def test_linear_decoherence_exponential() -> None:
    gamma = 0.4
    sim = generate_linear_decoherence_case(gamma=gamma, n_steps=300, t_max=3.0)
    expected = np.exp(-gamma * sim.time)
    np.testing.assert_allclose(sim.coherence, expected, rtol=0.05, atol=0.05)


def test_decoherence_rate_recovers_gamma() -> None:
    gamma = 0.6
    sim = generate_linear_decoherence_case(gamma=gamma)
    mean_gamma = float(np.mean(sim.gamma[10:-10]))
    assert abs(mean_gamma - gamma) < 0.1


def test_correlated_noise_lower_variance_than_uncorrelated() -> None:
    sigma = 0.4
    uncorr = generate_gaussian_noise_tau_case(sigma_tau=sigma, seed=5)
    corr = generate_correlated_noise_tau_case(sigma_tau=sigma, seed=5)
    assert np.max(corr.var_delta_tau) <= np.max(uncorr.var_delta_tau) + 0.1


def test_environment_strength_sweep_monotonic() -> None:
    sims = generate_environment_strength_sweep()
    finals = [float(s.coherence[-1]) for s in sims]
    assert all(finals[i] >= finals[i + 1] - 1e-6 for i in range(len(finals) - 1))


def test_mass_proxy_sweep_monotonic() -> None:
    masses = [0.5, 1.0, 2.0, 3.0]
    finals = [
        float(generate_mass_dependent_decoherence_case(m, gamma_base=0.1).coherence[-1])
        for m in masses
    ]
    assert all(finals[i] >= finals[i + 1] - 1e-6 for i in range(len(finals) - 1))


def test_evolve_coherence_from_rate() -> None:
    t = np.linspace(0, 2, 100)
    g = np.full_like(t, 0.5)
    c = evolve_coherence_from_rate(t, g, C0=1.0)
    assert c[0] == pytest.approx(1.0)
    assert c[-1] < c[0]


def test_delta_tau_and_variance() -> None:
    a = np.array([0.0, 1.0, 2.0])
    b = np.array([0.0, 0.5, 1.0])
    d = delta_tau(a, b)
    assert variance_delta_tau(d) == pytest.approx(1.0 / 6.0, rel=1e-6)


def test_pipeline_report_and_csv(tmp_path: Path) -> None:
    df, results = run_decoherence_tau_variance_benchmark(outputs_root=tmp_path / "out")
    assert len(results) == 6
    for col in REQUIRED_SUMMARY_COLUMNS:
        assert col in df.columns
    report = (tmp_path / "out" / "reports" / "decoherence_tau_variance_report.md").read_text(
        encoding="utf-8",
    )
    assert "NOT FULL MEASUREMENT-PROBLEM SOLUTION" in report
    assert BANNER_DECOHERENCE in report
    assert (tmp_path / "out" / "figures" / "decoherence_coherence_curves.png").is_file()


def test_benchmark_cases_pass() -> None:
    assert run_coherent_control().overall_pass
    assert run_linear_decoherence(0.5).overall_pass

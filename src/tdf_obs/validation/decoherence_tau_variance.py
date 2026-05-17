"""Phase 6D — Decoherence from τ-variance benchmark (not full measurement solution)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

BENCHMARK_MODE = "decoherence_tau_variance_benchmark"
BANNER_DECOHERENCE = (
    "DECOHERENCE FROM TAU VARIANCE BENCHMARK — NOT FULL MEASUREMENT-PROBLEM SOLUTION"
)

COHERENCE_MIN_FINAL = 0.95
COHERENCE_MAX_INITIAL = 1.01
GAMMA_FIT_TOL_PERCENT = 15.0
MONOTONIC_TOL = 1e-6

REQUIRED_SUMMARY_COLUMNS: tuple[str, ...] = (
    "case_name",
    "initial_coherence",
    "final_coherence",
    "max_var_delta_tau",
    "fitted_gamma",
    "expected_gamma",
    "gamma_relative_error_percent",
    "monotonic_decoherence_pass",
    "coherence_bounds_pass",
    "overall_pass",
    "warnings",
)


@dataclass
class TauBranchSimulation:
    """Time series for two τ branches and derived decoherence quantities."""

    time: np.ndarray
    tau_A: np.ndarray
    tau_B: np.ndarray
    delta_tau: np.ndarray
    var_delta_tau: np.ndarray
    coherence: np.ndarray
    gamma: np.ndarray
    expected_gamma: float = float("nan")
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DecoherenceBenchmarkResult:
    case_name: str
    initial_coherence: float
    final_coherence: float
    max_var_delta_tau: float
    fitted_gamma: float
    expected_gamma: float
    gamma_relative_error_percent: float
    monotonic_decoherence_pass: bool
    coherence_bounds_pass: bool
    overall_pass: bool
    warnings: str = ""
    simulation: TauBranchSimulation | None = None


def delta_tau(tau_A: np.ndarray, tau_B: np.ndarray) -> np.ndarray:
    """Δτ_AB = τ_A − τ_B."""
    return np.asarray(tau_A, dtype=float) - np.asarray(tau_B, dtype=float)


def variance_delta_tau(
    delta_tau_samples: np.ndarray,
    axis: int | None = None,
) -> float | np.ndarray:
    """Var(Δτ) over sample axis (or global scalar if 1D)."""
    d = np.asarray(delta_tau_samples, dtype=float)
    if d.ndim == 1:
        return float(np.var(d))
    return np.var(d, axis=axis)


def coherence_from_variance(var_delta_tau: float | np.ndarray) -> float | np.ndarray:
    """C = exp(−½ Var(Δτ))."""
    v = np.asarray(var_delta_tau, dtype=float)
    return np.exp(-0.5 * v)


def decoherence_rate_from_variance(
    time: np.ndarray,
    var_delta_tau: np.ndarray,
) -> np.ndarray:
    """Γ(t) = ½ d Var(Δτ) / dt (finite differences)."""
    t = np.asarray(time, dtype=float)
    v = np.asarray(var_delta_tau, dtype=float)
    rate = np.gradient(v, t)
    return 0.5 * rate


def evolve_coherence_from_rate(
    time: np.ndarray,
    gamma: np.ndarray,
    C0: float = 1.0,
) -> np.ndarray:
    """Integrate dC/dt = −Γ(t) C with trapezoidal cumulative decay."""
    t = np.asarray(time, dtype=float)
    g = np.asarray(gamma, dtype=float)
    if len(t) < 2:
        return np.array([C0], dtype=float)
    integral = np.zeros_like(t)
    integral[1:] = np.cumsum(0.5 * (g[1:] + g[:-1]) * np.diff(t))
    return C0 * np.exp(-integral)


def _fit_gamma_from_coherence(time: np.ndarray, coherence: np.ndarray) -> float:
    """Fit Γ from C(t) ≈ exp(−Γ t) for t > 0."""
    t = np.asarray(time, dtype=float)
    c = np.clip(np.asarray(coherence, dtype=float), 1e-12, 1.0)
    mask = (t > 1e-12) & (c < 0.999)
    if np.sum(mask) < 2:
        return float("nan")
    slope, _ = np.polyfit(t[mask], -np.log(c[mask]), 1)
    return float(max(slope, 0.0))


def _coherence_monotonic_decreasing(coherence: np.ndarray) -> bool:
    c = np.asarray(coherence, dtype=float)
    return bool(np.all(np.diff(c) <= MONOTONIC_TOL))


def _cumulative_variance_series(samples: np.ndarray) -> np.ndarray:
    """Var(Δτ) up to each time index (branch-path variance growth)."""
    x = np.asarray(samples, dtype=float)
    out = np.zeros_like(x)
    for i in range(1, len(x)):
        out[i] = float(np.var(x[: i + 1]))
    return out


def _build_simulation(
    time: np.ndarray,
    tau_A: np.ndarray,
    tau_B: np.ndarray,
    *,
    var_delta_tau: np.ndarray | None = None,
    expected_gamma: float = float("nan"),
    metadata: dict[str, Any] | None = None,
) -> TauBranchSimulation:
    dt = delta_tau(tau_A, tau_B)
    var_series = (
        np.asarray(var_delta_tau, dtype=float)
        if var_delta_tau is not None
        else _cumulative_variance_series(dt)
    )
    coh = coherence_from_variance(var_series)
    gamma = decoherence_rate_from_variance(time, var_series)
    return TauBranchSimulation(
        time=time,
        tau_A=tau_A,
        tau_B=tau_B,
        delta_tau=dt,
        var_delta_tau=var_series,
        coherence=coh,
        gamma=gamma,
        expected_gamma=expected_gamma,
        metadata=metadata or {},
    )


def generate_static_coherent_case(
    n_steps: int = 200,
    t_max: float = 2.0,
) -> TauBranchSimulation:
    """Δτ variance ≈ 0 ⇒ C ≈ 1."""
    time = np.linspace(0.0, t_max, n_steps)
    tau_A = np.zeros_like(time)
    tau_B = np.zeros_like(time)
    return _build_simulation(time, tau_A, tau_B, expected_gamma=0.0)


def generate_linear_decoherence_case(
    gamma: float = 0.5,
    n_steps: int = 200,
    t_max: float = 2.0,
) -> TauBranchSimulation:
    """Var(Δτ) = 2 Γ t ⇒ C(t) = exp(−Γ t)."""
    time = np.linspace(0.0, t_max, n_steps)
    var_t = 2.0 * gamma * time
    # Construct τ_A, τ_B with specified variance of difference: τ_A = sqrt(var/2), τ_B = -sqrt(var/2)
    half = np.sqrt(np.maximum(var_t, 0.0) / 2.0)
    tau_A = half
    tau_B = -half
    return _build_simulation(
        time,
        tau_A,
        tau_B,
        var_delta_tau=var_t,
        expected_gamma=gamma,
    )


def generate_gaussian_noise_tau_case(
    sigma_tau: float = 0.4,
    n_steps: int = 300,
    t_max: float = 3.0,
    seed: int = 0,
) -> TauBranchSimulation:
    """Independent Gaussian phase noise on each branch."""
    rng = np.random.default_rng(seed)
    time = np.linspace(0.0, t_max, n_steps)
    noise_A = np.cumsum(rng.normal(0.0, sigma_tau, size=n_steps)) / np.sqrt(n_steps)
    noise_B = np.cumsum(rng.normal(0.0, sigma_tau, size=n_steps)) / np.sqrt(n_steps)
    return _build_simulation(
        time,
        noise_A,
        noise_B,
        metadata={"sigma_tau": sigma_tau, "noise_type": "uncorrelated"},
    )


def generate_correlated_noise_tau_case(
    sigma_tau: float = 0.4,
    correlation: float = 0.85,
    n_steps: int = 300,
    t_max: float = 3.0,
    seed: int = 1,
) -> TauBranchSimulation:
    """Correlated noise: shared component reduces Var(Δτ)."""
    rng = np.random.default_rng(seed)
    time = np.linspace(0.0, t_max, n_steps)
    common = np.cumsum(rng.normal(0.0, sigma_tau, size=n_steps)) / np.sqrt(n_steps)
    indep_A = np.cumsum(rng.normal(0.0, sigma_tau * np.sqrt(1 - correlation**2), size=n_steps))
    indep_B = np.cumsum(rng.normal(0.0, sigma_tau * np.sqrt(1 - correlation**2), size=n_steps))
    tau_A = correlation * common + indep_A / np.sqrt(n_steps)
    tau_B = correlation * common + indep_B / np.sqrt(n_steps)
    return _build_simulation(
        time,
        tau_A,
        tau_B,
        metadata={"sigma_tau": sigma_tau, "correlation": correlation},
    )


def generate_environment_strength_sweep(
    sigmas: list[float] | None = None,
    n_steps: int = 250,
    t_max: float = 2.5,
    seed: int = 10,
) -> list[TauBranchSimulation]:
    """
    Sweep environmental τ-noise strength σ_τ.

    Uses deterministic Var(Δτ) ≈ (σ t)² for monotonic coherence vs σ; stochastic case
    remains in ``generate_gaussian_noise_tau_case``.
    """
    if sigmas is None:
        sigmas = [0.05, 0.15, 0.3, 0.5, 0.8]
    sims: list[TauBranchSimulation] = []
    time = np.linspace(0.0, t_max, n_steps)
    for i, s in enumerate(sigmas):
        var_t = (s * time) ** 2
        half = np.sqrt(np.maximum(var_t, 0.0) / 2.0)
        sims.append(
            _build_simulation(
                time,
                half,
                -half,
                var_delta_tau=var_t,
                metadata={"sigma_tau": s, "model": "deterministic_env_sweep"},
            ),
        )
    return sims


def generate_mass_dependent_decoherence_case(
    mass_proxy: float = 2.0,
    gamma_base: float = 0.2,
    n_steps: int = 200,
    t_max: float = 2.0,
) -> TauBranchSimulation:
    """
    TDF-inspired proxy: Γ_eff = γ₀ × (M_proxy)².

    **Proxy only** — not derived from first principles in this benchmark.
    """
    gamma_eff = gamma_base * mass_proxy**2
    return generate_linear_decoherence_case(gamma=gamma_eff, n_steps=n_steps, t_max=t_max)


def _evaluate_simulation(
    case_name: str,
    sim: TauBranchSimulation,
    *,
    expect_gamma: float | None = None,
    require_monotonic: bool = True,
    final_coherence_max: float | None = None,
    final_coherence_min: float | None = None,
    warnings: str = "",
) -> DecoherenceBenchmarkResult:
    c0 = float(sim.coherence[0])
    cf = float(sim.coherence[-1])
    max_var = float(np.max(sim.var_delta_tau))
    fitted_g = _fit_gamma_from_coherence(sim.time, sim.coherence)
    expected_g = expect_gamma if expect_gamma is not None else sim.expected_gamma
    if np.isnan(expected_g) or expected_g < 1e-12:
        gamma_err_pct = 0.0 if fitted_g < 0.05 else float("nan")
    else:
        gamma_err_pct = abs(fitted_g - expected_g) / expected_g * 100.0

    mono = _coherence_monotonic_decreasing(sim.coherence) if require_monotonic else True
    bounds = (
        c0 <= COHERENCE_MAX_INITIAL
        and cf >= 0.0
        and cf <= 1.0 + 1e-9
    )
    if final_coherence_max is not None:
        bounds = bounds and cf <= final_coherence_max
    if final_coherence_min is not None:
        bounds = bounds and cf >= final_coherence_min

    gamma_ok = True
    if expect_gamma is not None and expect_gamma > 1e-6:
        gamma_ok = gamma_err_pct <= GAMMA_FIT_TOL_PERCENT
    elif expect_gamma is not None and expect_gamma <= 1e-6:
        gamma_ok = fitted_g < 0.1

    overall = mono and bounds and gamma_ok
    if case_name == "coherent_control":
        overall = cf >= COHERENCE_MIN_FINAL and max_var < 0.01 and bounds

    return DecoherenceBenchmarkResult(
        case_name=case_name,
        initial_coherence=c0,
        final_coherence=cf,
        max_var_delta_tau=max_var,
        fitted_gamma=fitted_g,
        expected_gamma=expected_g if not np.isnan(expected_g) else float("nan"),
        gamma_relative_error_percent=gamma_err_pct,
        monotonic_decoherence_pass=mono,
        coherence_bounds_pass=bounds,
        overall_pass=overall,
        warnings=warnings,
        simulation=sim,
    )


def run_coherent_control() -> DecoherenceBenchmarkResult:
    sim = generate_static_coherent_case()
    return _evaluate_simulation(
        "coherent_control",
        sim,
        expect_gamma=0.0,
        require_monotonic=False,
        final_coherence_min=COHERENCE_MIN_FINAL,
    )


def run_linear_decoherence(gamma: float = 0.5) -> DecoherenceBenchmarkResult:
    sim = generate_linear_decoherence_case(gamma=gamma)
    return _evaluate_simulation(
        "linear_decoherence",
        sim,
        expect_gamma=gamma,
    )


def run_gaussian_noise_decoherence() -> DecoherenceBenchmarkResult:
    sim = generate_gaussian_noise_tau_case(sigma_tau=1.2, n_steps=500, t_max=5.0, seed=0)
    res = _evaluate_simulation(
        "gaussian_noise_decoherence",
        sim,
        require_monotonic=False,
        warnings="Stochastic τ noise; single-trajectory proxy.",
    )
    decohered = (res.initial_coherence - res.final_coherence) > 0.03
    res.overall_pass = res.overall_pass and decohered and res.final_coherence < 0.98
    return res


def run_correlated_noise_protection(
    sigma: float = 0.9,
) -> DecoherenceBenchmarkResult:
    uncorr = generate_gaussian_noise_tau_case(sigma_tau=sigma, n_steps=400, t_max=4.0, seed=2)
    corr = generate_correlated_noise_tau_case(
        sigma_tau=sigma,
        n_steps=400,
        t_max=4.0,
        seed=2,
    )
    pass_prot = float(corr.coherence[-1]) >= float(uncorr.coherence[-1]) - 0.02
    var_lower = np.max(corr.var_delta_tau) <= np.max(uncorr.var_delta_tau) + 0.05
    res = _evaluate_simulation(
        "correlated_noise_protection",
        corr,
        require_monotonic=False,
        warnings=f"Uncorr final C={uncorr.coherence[-1]:.3f}, corr final C={corr.coherence[-1]:.3f}",
    )
    res.overall_pass = pass_prot and var_lower
    if not pass_prot:
        res.warnings += "; correlated coherence not higher than uncorrelated."
    return res


def run_environment_strength_sweep() -> DecoherenceBenchmarkResult:
    sims = generate_environment_strength_sweep()
    finals = [float(s.coherence[-1]) for s in sims]
    sigmas = [s.metadata.get("sigma_tau", 0.0) for s in sims]
    monotonic = all(finals[i] >= finals[i + 1] - MONOTONIC_TOL for i in range(len(finals) - 1))
    # Representative row: strongest decoherence case
    worst = sims[-1]
    res = _evaluate_simulation(
        "environment_strength_sweep",
        worst,
        warnings=f"σ sweep {sigmas}; finals={[round(f,3) for f in finals]}",
    )
    res.monotonic_decoherence_pass = monotonic
    res.overall_pass = monotonic and res.coherence_bounds_pass
    return res


def run_mass_dependent_proxy() -> DecoherenceBenchmarkResult:
    masses = [0.5, 1.0, 1.5, 2.0, 3.0]
    gamma_base = 0.15
    finals: list[float] = []
    sims: list[TauBranchSimulation] = []
    for m in masses:
        sim = generate_mass_dependent_decoherence_case(mass_proxy=m, gamma_base=gamma_base)
        sims.append(sim)
        finals.append(float(sim.coherence[-1]))
    # Higher mass ⇒ faster decay ⇒ lower final coherence
    monotonic = all(finals[i] >= finals[i + 1] - MONOTONIC_TOL for i in range(len(finals) - 1))
    heaviest = sims[-1]
    expected_g = gamma_base * masses[-1] ** 2
    res = _evaluate_simulation(
        "mass_dependent_proxy",
        heaviest,
        expect_gamma=expected_g,
        warnings="Γ_eff = γ₀ M² proxy only; not proven TDF physics.",
    )
    res.monotonic_decoherence_pass = monotonic
    res.overall_pass = monotonic and res.coherence_bounds_pass and res.overall_pass
    return res


def _result_to_row(res: DecoherenceBenchmarkResult) -> dict[str, Any]:
    return {
        "case_name": res.case_name,
        "initial_coherence": res.initial_coherence,
        "final_coherence": res.final_coherence,
        "max_var_delta_tau": res.max_var_delta_tau,
        "fitted_gamma": res.fitted_gamma,
        "expected_gamma": res.expected_gamma,
        "gamma_relative_error_percent": res.gamma_relative_error_percent,
        "monotonic_decoherence_pass": res.monotonic_decoherence_pass,
        "coherence_bounds_pass": res.coherence_bounds_pass,
        "overall_pass": res.overall_pass,
        "warnings": res.warnings,
        "dataset_mode": BENCHMARK_MODE,
        "is_real_observational_data": False,
    }


def _plot_coherence_curves(results: list[DecoherenceBenchmarkResult], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))
    for res in results:
        if res.simulation is None:
            continue
        s = res.simulation
        ax.plot(s.time, s.coherence, label=res.case_name)
    ax.set_xlabel("t")
    ax.set_ylabel("C_AB(t)")
    ax.set_title("Coherence vs time (Phase 6D)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_variance_curves(results: list[DecoherenceBenchmarkResult], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))
    for res in results:
        if res.simulation is None:
            continue
        s = res.simulation
        ax.plot(s.time, s.var_delta_tau, label=res.case_name)
    ax.set_xlabel("t")
    ax.set_ylabel("Var(Δτ_AB)")
    ax.set_title("τ-difference variance (Phase 6D)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_environment_sweep(path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sims = generate_environment_strength_sweep()
    sigmas = [s.metadata["sigma_tau"] for s in sims]
    finals = [float(s.coherence[-1]) for s in sims]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(sigmas, finals, "o-")
    ax.set_xlabel("σ_τ (noise strength)")
    ax.set_ylabel("final coherence")
    ax.set_title("Environment strength sweep")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_mass_proxy_sweep(path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    masses = np.linspace(0.5, 3.0, 6)
    gamma_base = 0.15
    finals = [
        float(generate_mass_dependent_decoherence_case(m, gamma_base).coherence[-1])
        for m in masses
    ]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(masses, finals, "s-", color="C2")
    ax.set_xlabel("mass proxy M")
    ax.set_ylabel("final coherence")
    ax.set_title("Mass-proxy decoherence (Γ ∝ M², proxy only)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _build_report(results: list[DecoherenceBenchmarkResult]) -> str:
    n_pass = sum(1 for r in results if r.overall_pass)
    lines = [
        "# Decoherence from τ-variance report (Phase 6D)",
        "",
        f"## ⚠️ {BANNER_DECOHERENCE}",
        "",
        "## Purpose",
        "",
        "Test whether **growth in Var(Δτ_AB)** between quantum branches suppresses "
        "accessible phase coherence in controlled two-branch benchmarks.",
        "",
        "> **NOT FULL MEASUREMENT-PROBLEM SOLUTION.** No wavefunction-collapse claim.",
        "",
        "## Equations",
        "",
        "```text",
        "Δτ_AB(t) = τ_A(t) − τ_B(t)",
        "C_AB(t) = exp[−½ Var(Δτ_AB)]",
        "Γ_AB(t) = ½ d/dt Var(Δτ_AB)",
        "dC_AB/dt = −Γ_AB(t) C_AB(t)",
        "```",
        "",
        "## Summary",
        "",
        f"- **Cases:** {len(results)}",
        f"- **Pass:** {n_pass} / {len(results)}",
        "",
        "## Results table",
        "",
        "| Case | C₀ | C_final | max Var(Δτ) | Γ_fit | Γ_exp | pass |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in results:
        lines.append(
            f"| {r.case_name} | {r.initial_coherence:.3f} | {r.final_coherence:.3f} | "
            f"{r.max_var_delta_tau:.3f} | {r.fitted_gamma:.3f} | "
            f"{r.expected_gamma:.3f} | {'✓' if r.overall_pass else '✗'} |",
        )
    lines.extend(
        [
            "",
            "## Scientific interpretation",
            "",
            "- **Passing** means TDF can model decoherence as τ-phase variance growth "
            "between branches in these toy benchmarks.",
            "- Does **not** prove full wavefunction collapse or derive the Born rule.",
            "- Does **not** solve the full measurement problem.",
            "",
            "## Failure modes",
            "",
            "- Two-branch toy model; environment = stochastic τ noise only.",
            "- No many-body detector, relativistic QFT, or experimental decoherence data.",
            "- Mass scaling is an explicit **proxy** (Γ ∝ M²), not proven TDF physics.",
            "",
            "## Disclaimer",
            "",
            "- Numerical consistency benchmark only.",
            "",
        ],
    )
    return "\n".join(lines)


def run_decoherence_tau_variance_benchmark(
    outputs_root: Path | None = None,
) -> tuple[pd.DataFrame, list[DecoherenceBenchmarkResult]]:
    """Run Phase 6D benchmarks; write CSV, report, figures."""
    root = Path(__file__).resolve().parents[3]
    outputs = outputs_root or (root / "outputs")
    tables_dir = outputs / "tables"
    reports_dir = outputs / "reports"
    figures_dir = outputs / "figures"
    for d in (tables_dir, reports_dir, figures_dir):
        d.mkdir(parents=True, exist_ok=True)

    results = [
        run_coherent_control(),
        run_linear_decoherence(),
        run_gaussian_noise_decoherence(),
        run_correlated_noise_protection(),
        run_environment_strength_sweep(),
        run_mass_dependent_proxy(),
    ]

    out_df = pd.DataFrame([_result_to_row(r) for r in results])
    out_df.to_csv(tables_dir / "decoherence_tau_variance_summary.csv", index=False)
    (reports_dir / "decoherence_tau_variance_report.md").write_text(
        _build_report(results),
        encoding="utf-8",
    )

    plot_cases = [r for r in results if r.simulation is not None]
    _plot_coherence_curves(plot_cases, figures_dir / "decoherence_coherence_curves.png")
    _plot_variance_curves(plot_cases, figures_dir / "decoherence_variance_curves.png")
    _plot_environment_sweep(figures_dir / "decoherence_environment_sweep.png")
    _plot_mass_proxy_sweep(figures_dir / "decoherence_mass_proxy_sweep.png")

    return out_df, results

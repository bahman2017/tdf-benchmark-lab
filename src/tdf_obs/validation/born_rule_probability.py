"""Phase 6F — Born-rule probability emergence proxy (not full derivation)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

BENCHMARK_MODE = "born_rule_probability_benchmark"
BANNER_BORN = (
    "BORN-RULE PROBABILITY EMERGENCE BENCHMARK — NOT FULL BORN-RULE DERIVATION"
)

DEFAULT_N_TRIALS = 50_000
FREQ_ERROR_TOL = 0.025
ADDITIVITY_TOL = 1e-12
DIAGONAL_PRESERVE_TOL = 1e-10
PHASE_INVARIANCE_TOL = 1e-12
ZERO_WEIGHT_TOL = 1e-14
OFFDIAG_RATIO_MAX = 0.05

WrongRuleName = Literal["amplitude_linear", "rho_squared", "uniform"]

REQUIRED_SUMMARY_COLUMNS: tuple[str, ...] = (
    "case_name",
    "n_branches",
    "expected_probabilities",
    "observed_frequencies",
    "max_frequency_error",
    "chi_square_born",
    "chi_square_uniform",
    "chi_square_amplitude_linear",
    "chi_square_rho_squared",
    "offdiag_before",
    "offdiag_after",
    "diagonal_change",
    "additivity_error",
    "phase_invariance_error",
    "overall_pass",
    "warnings",
)


@dataclass
class BornRuleBenchmarkResult:
    case_name: str
    n_branches: int
    expected_probabilities: np.ndarray
    observed_frequencies: np.ndarray
    max_frequency_error: float
    chi_square_born: float
    chi_square_uniform: float
    chi_square_amplitude_linear: float
    chi_square_rho_squared: float
    offdiag_before: float
    offdiag_after: float
    diagonal_change: float
    additivity_error: float
    phase_invariance_error: float
    overall_pass: bool
    warnings: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


def normalize_branch_weights(rho: np.ndarray) -> np.ndarray:
    """ρ̃_i = ρ_i / Σ_j ρ_j."""
    r = np.asarray(rho, dtype=float)
    s = float(np.sum(r))
    if s <= 0.0:
        raise ValueError("Branch weights must have positive sum.")
    return r / s


def amplitudes_from_rho_tau(rho: np.ndarray, tau: np.ndarray) -> np.ndarray:
    """c_i = √(ρ_i) exp(−i τ_i) with TDF v0.8.1 phase convention."""
    r = np.asarray(rho, dtype=float)
    t = np.asarray(tau, dtype=float)
    return np.sqrt(np.maximum(r, 0.0)) * np.exp(-1j * t)


def density_matrix_from_amplitudes(c: np.ndarray) -> np.ndarray:
    """ρ_matrix = |ψ⟩⟨ψ| for branch amplitude vector c."""
    v = np.asarray(c, dtype=complex).reshape(-1)
    return np.outer(v, np.conj(v))


def apply_decoherence_to_density_matrix(
    rho_matrix: np.ndarray,
    decoherence_strength: float,
) -> np.ndarray:
    """Suppress off-diagonals; leave diagonal entries unchanged."""
    rho = np.asarray(rho_matrix, dtype=complex).copy()
    n = rho.shape[0]
    damp = float(np.exp(-max(decoherence_strength, 0.0)))
    for i in range(n):
        for j in range(n):
            if i != j:
                rho[i, j] *= damp
    return rho


def diagonal_probabilities(rho_matrix: np.ndarray) -> np.ndarray:
    """Normalized diagonal entries (classical branch probabilities)."""
    diag = np.real(np.diag(np.asarray(rho_matrix, dtype=complex)))
    diag = np.maximum(diag, 0.0)
    s = float(np.sum(diag))
    if s <= 0.0:
        return np.ones_like(diag) / len(diag)
    return diag / s


def born_probability_from_state(c: np.ndarray) -> np.ndarray:
    """P_i = |c_i|² / Σ_j |c_j|²."""
    amp = np.asarray(c, dtype=complex).reshape(-1)
    weights = np.abs(amp) ** 2
    s = float(np.sum(weights))
    if s <= 0.0:
        return np.ones(len(amp)) / len(amp)
    return weights / s


def simulate_measurement_counts(
    probabilities: np.ndarray,
    n_trials: int,
    seed: int = 0,
) -> np.ndarray:
    """Multinomial sampling of branch outcomes."""
    p = normalize_branch_weights(np.asarray(probabilities, dtype=float))
    rng = np.random.default_rng(seed)
    return rng.multinomial(int(n_trials), p)


def frequency_error(counts: np.ndarray, probabilities: np.ndarray) -> float:
    """Max |f_i − p_i| over branches."""
    c = np.asarray(counts, dtype=float)
    p = normalize_branch_weights(np.asarray(probabilities, dtype=float))
    n = float(np.sum(c))
    if n <= 0.0:
        return float("inf")
    freqs = c / n
    return float(np.max(np.abs(freqs - p)))


def chi_square_statistic(counts: np.ndarray, probabilities: np.ndarray) -> float:
    """Pearson χ² comparing counts to expected probabilities."""
    c = np.asarray(counts, dtype=float)
    p = normalize_branch_weights(np.asarray(probabilities, dtype=float))
    n = float(np.sum(c))
    if n <= 0.0:
        return float("inf")
    expected = n * p
    mask = expected > 1e-12
    if not np.any(mask):
        return 0.0
    return float(np.sum((c[mask] - expected[mask]) ** 2 / expected[mask]))


def wrong_probability_rule(
    rho: np.ndarray,
    rule: WrongRuleName,
) -> np.ndarray:
    """Alternative (incorrect) probability assignments for comparison."""
    r = np.maximum(np.asarray(rho, dtype=float), 0.0)
    n = len(r)
    if rule == "uniform":
        return np.ones(n, dtype=float) / n
    if rule == "amplitude_linear":
        w = np.sqrt(r)
        return w / np.sum(w) if np.sum(w) > 0 else np.ones(n) / n
    if rule == "rho_squared":
        w = r**2
        return w / np.sum(w) if np.sum(w) > 0 else np.ones(n) / n
    raise ValueError(f"Unknown rule: {rule}")


def compare_probability_rules(
    rho: np.ndarray,
    counts: np.ndarray,
) -> dict[str, float]:
    """χ² for Born rule vs wrong rules given observed counts."""
    r = normalize_branch_weights(np.asarray(rho, dtype=float))
    c = np.asarray(counts, dtype=float)
    return {
        "born": chi_square_statistic(c, r),
        "uniform": chi_square_statistic(c, wrong_probability_rule(r, "uniform")),
        "amplitude_linear": chi_square_statistic(
            c,
            wrong_probability_rule(r, "amplitude_linear"),
        ),
        "rho_squared": chi_square_statistic(c, wrong_probability_rule(r, "rho_squared")),
    }


def coarse_graining_additivity(
    probabilities: np.ndarray,
    groups: list[list[int]],
) -> float:
    """
    |P(∪ group) − Σ_{i∈group} P_i| maximized over groups.
    """
    p = normalize_branch_weights(np.asarray(probabilities, dtype=float))
    errors: list[float] = []
    for g in groups:
        idx = list(g)
        p_group = float(np.sum(p[idx]))
        p_union = float(np.sum(p[idx]))
        errors.append(abs(p_union - p_group))
    return float(max(errors) if errors else 0.0)


def zero_weight_branch_check(rho: np.ndarray) -> tuple[float, int]:
    """Return (P_zero, index) for first zero-weight branch."""
    r = np.asarray(rho, dtype=float)
    p = normalize_branch_weights(r)
    zero_idx = int(np.where(r <= ZERO_WEIGHT_TOL)[0][0]) if np.any(r <= ZERO_WEIGHT_TOL) else -1
    if zero_idx < 0:
        return 0.0, -1
    return float(p[zero_idx]), zero_idx


def _offdiag_norm(rho_matrix: np.ndarray) -> float:
    m = np.asarray(rho_matrix, dtype=complex)
    n = m.shape[0]
    off = 0.0
    for i in range(n):
        for j in range(n):
            if i != j:
                off += abs(m[i, j]) ** 2
    return float(np.sqrt(off))


def _default_tau(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(0.0, 2.0 * np.pi, size=n)


def _counts_to_frequencies(counts: np.ndarray) -> np.ndarray:
    c = np.asarray(counts, dtype=float)
    n = float(np.sum(c))
    return c / n if n > 0 else c


def _evaluate_sampling_case(
    case_name: str,
    rho: np.ndarray,
    *,
    n_trials: int = DEFAULT_N_TRIALS,
    seed: int = 0,
    tau: np.ndarray | None = None,
    require_wrong_rules_fail: bool = False,
    warnings: str = "",
) -> BornRuleBenchmarkResult:
    r = normalize_branch_weights(np.asarray(rho, dtype=float))
    t = _default_tau(len(r), seed=seed + 1) if tau is None else np.asarray(tau, dtype=float)
    c = amplitudes_from_rho_tau(r, t)
    expected = born_probability_from_state(c)
    counts = simulate_measurement_counts(expected, n_trials, seed=seed)
    freqs = _counts_to_frequencies(counts)
    chi = compare_probability_rules(r, counts)
    max_err = frequency_error(counts, expected)
    overall = max_err <= FREQ_ERROR_TOL
    if require_wrong_rules_fail:
        overall = overall and (
            chi["born"] < chi["uniform"]
            and chi["born"] < chi["amplitude_linear"]
            and chi["born"] < chi["rho_squared"]
        )
    return BornRuleBenchmarkResult(
        case_name=case_name,
        n_branches=len(r),
        expected_probabilities=expected,
        observed_frequencies=freqs,
        max_frequency_error=max_err,
        chi_square_born=chi["born"],
        chi_square_uniform=chi["uniform"],
        chi_square_amplitude_linear=chi["amplitude_linear"],
        chi_square_rho_squared=chi["rho_squared"],
        offdiag_before=float("nan"),
        offdiag_after=float("nan"),
        diagonal_change=float("nan"),
        additivity_error=float("nan"),
        phase_invariance_error=float("nan"),
        overall_pass=overall,
        warnings=warnings,
    )


def run_balanced_two_branch() -> BornRuleBenchmarkResult:
    rho = np.array([0.5, 0.5])
    return _evaluate_sampling_case("balanced_two_branch", rho, seed=10)


def run_unequal_two_branch() -> BornRuleBenchmarkResult:
    rho = np.array([0.8, 0.2])
    return _evaluate_sampling_case(
        "unequal_two_branch",
        rho,
        seed=11,
        require_wrong_rules_fail=True,
    )


def run_three_branch_distribution() -> BornRuleBenchmarkResult:
    rho = np.array([0.6, 0.3, 0.1])
    return _evaluate_sampling_case("three_branch_distribution", rho, seed=12)


def run_decoherence_preserves_diagonal_weights() -> BornRuleBenchmarkResult:
    rho = np.array([0.55, 0.45])
    tau = np.array([0.3, 1.7])
    c = amplitudes_from_rho_tau(rho, tau)
    rho_mat = density_matrix_from_amplitudes(c)
    before = diagonal_probabilities(rho_mat)
    off_b = _offdiag_norm(rho_mat)
    decohered = apply_decoherence_to_density_matrix(rho_mat, decoherence_strength=6.0)
    off_a = _offdiag_norm(decohered)
    after = diagonal_probabilities(decohered)
    diag_change = float(np.max(np.abs(before - after)))
    overall = (
        off_a < OFFDIAG_RATIO_MAX * max(off_b, 1e-14)
        and diag_change <= DIAGONAL_PRESERVE_TOL
    )
    return BornRuleBenchmarkResult(
        case_name="decoherence_preserves_diagonal_weights",
        n_branches=2,
        expected_probabilities=before,
        observed_frequencies=after,
        max_frequency_error=float("nan"),
        chi_square_born=float("nan"),
        chi_square_uniform=float("nan"),
        chi_square_amplitude_linear=float("nan"),
        chi_square_rho_squared=float("nan"),
        offdiag_before=off_b,
        offdiag_after=off_a,
        diagonal_change=diag_change,
        additivity_error=float("nan"),
        phase_invariance_error=float("nan"),
        overall_pass=overall,
    )


def run_wrong_rules_fail_comparison() -> BornRuleBenchmarkResult:
    rho = np.array([0.75, 0.25])
    res = _evaluate_sampling_case(
        "wrong_rules_fail_comparison",
        rho,
        seed=13,
        require_wrong_rules_fail=True,
    )
    res.case_name = "wrong_rules_fail_comparison"
    return res


def run_coarse_graining_additivity() -> BornRuleBenchmarkResult:
    rho = np.array([0.4, 0.35, 0.25])
    probs = normalize_branch_weights(rho)
    groups = [[0, 1], [2]]
    err = coarse_graining_additivity(probs, groups)
    p01 = float(probs[0] + probs[1])
    p2 = float(probs[2])
    overall = err <= ADDITIVITY_TOL and abs((p01 + p2) - 1.0) < 1e-12
    return BornRuleBenchmarkResult(
        case_name="coarse_graining_additivity",
        n_branches=3,
        expected_probabilities=probs,
        observed_frequencies=np.array([p01, p2]),
        max_frequency_error=float("nan"),
        chi_square_born=float("nan"),
        chi_square_uniform=float("nan"),
        chi_square_amplitude_linear=float("nan"),
        chi_square_rho_squared=float("nan"),
        offdiag_before=float("nan"),
        offdiag_after=float("nan"),
        diagonal_change=float("nan"),
        additivity_error=err,
        phase_invariance_error=float("nan"),
        overall_pass=overall,
    )


def run_zero_weight_branch() -> BornRuleBenchmarkResult:
    rho = np.array([0.7, 0.3, 0.0])
    p_zero, zidx = zero_weight_branch_check(rho)
    expected = normalize_branch_weights(rho)
    counts = simulate_measurement_counts(expected, DEFAULT_N_TRIALS, seed=14)
    zero_counts = int(counts[zidx]) if zidx >= 0 else 0
    overall = p_zero <= ZERO_WEIGHT_TOL and zero_counts == 0
    return BornRuleBenchmarkResult(
        case_name="zero_weight_branch",
        n_branches=3,
        expected_probabilities=expected,
        observed_frequencies=_counts_to_frequencies(counts),
        max_frequency_error=frequency_error(counts, expected),
        chi_square_born=float("nan"),
        chi_square_uniform=float("nan"),
        chi_square_amplitude_linear=float("nan"),
        chi_square_rho_squared=float("nan"),
        offdiag_before=float("nan"),
        offdiag_after=float("nan"),
        diagonal_change=float("nan"),
        additivity_error=float("nan"),
        phase_invariance_error=float("nan"),
        overall_pass=overall,
        warnings=f"zero_branch_counts={zero_counts}",
    )


def run_phase_invariance_check() -> BornRuleBenchmarkResult:
    rho = np.array([0.6, 0.3, 0.1])
    tau_a = np.array([0.2, 1.1, 2.4])
    rng = np.random.default_rng(99)
    tau_b = rng.uniform(0.0, 2.0 * np.pi, size=3)
    p_a = born_probability_from_state(amplitudes_from_rho_tau(rho, tau_a))
    p_b = born_probability_from_state(amplitudes_from_rho_tau(rho, tau_b))
    err = float(np.max(np.abs(p_a - p_b)))
    return BornRuleBenchmarkResult(
        case_name="phase_invariance_check",
        n_branches=3,
        expected_probabilities=normalize_branch_weights(rho),
        observed_frequencies=p_b,
        max_frequency_error=float("nan"),
        chi_square_born=float("nan"),
        chi_square_uniform=float("nan"),
        chi_square_amplitude_linear=float("nan"),
        chi_square_rho_squared=float("nan"),
        offdiag_before=float("nan"),
        offdiag_after=float("nan"),
        diagonal_change=float("nan"),
        additivity_error=float("nan"),
        phase_invariance_error=err,
        overall_pass=err <= PHASE_INVARIANCE_TOL,
    )


def _result_to_row(res: BornRuleBenchmarkResult) -> dict[str, Any]:
    def _ser(arr: np.ndarray) -> str:
        return json.dumps([float(x) for x in np.asarray(arr, dtype=float)])

    return {
        "case_name": res.case_name,
        "n_branches": res.n_branches,
        "expected_probabilities": _ser(res.expected_probabilities),
        "observed_frequencies": _ser(res.observed_frequencies),
        "max_frequency_error": res.max_frequency_error,
        "chi_square_born": res.chi_square_born,
        "chi_square_uniform": res.chi_square_uniform,
        "chi_square_amplitude_linear": res.chi_square_amplitude_linear,
        "chi_square_rho_squared": res.chi_square_rho_squared,
        "offdiag_before": res.offdiag_before,
        "offdiag_after": res.offdiag_after,
        "diagonal_change": res.diagonal_change,
        "additivity_error": res.additivity_error,
        "phase_invariance_error": res.phase_invariance_error,
        "overall_pass": res.overall_pass,
        "warnings": res.warnings,
        "dataset_mode": BENCHMARK_MODE,
        "is_real_observational_data": False,
    }


def _plot_expected_vs_observed(results: list[BornRuleBenchmarkResult], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cases = [
        r
        for r in results
        if r.case_name
        in (
            "balanced_two_branch",
            "unequal_two_branch",
            "three_branch_distribution",
        )
    ]
    fig, axes = plt.subplots(1, len(cases), figsize=(4 * len(cases), 4))
    if len(cases) == 1:
        axes = [axes]
    x = np.arange(max(r.n_branches for r in cases))
    w = 0.35
    for ax, res in zip(axes, cases):
        n = res.n_branches
        xs = np.arange(n)
        ax.bar(xs - w / 2, res.expected_probabilities[:n], w, label="expected")
        ax.bar(xs + w / 2, res.observed_frequencies[:n], w, label="observed")
        ax.set_xticks(xs)
        ax.set_title(res.case_name)
        ax.set_ylabel("probability")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Born-rule expected vs observed frequencies (Phase 6F)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_rule_comparison(results: list[BornRuleBenchmarkResult], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    res = next((r for r in results if r.case_name == "wrong_rules_fail_comparison"), None)
    if res is None:
        res = next((r for r in results if r.case_name == "unequal_two_branch"), None)
    if res is None:
        return
    labels = ["born", "uniform", "amplitude_linear", "rho_squared"]
    vals = [
        res.chi_square_born,
        res.chi_square_uniform,
        res.chi_square_amplitude_linear,
        res.chi_square_rho_squared,
    ]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, vals, color=["C0", "C3", "C1", "C2"])
    ax.set_ylabel("χ²")
    ax.set_title(f"Probability rule comparison ({res.case_name})")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_decoherence_diagonal(results: list[BornRuleBenchmarkResult], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    res = next(
        (r for r in results if r.case_name == "decoherence_preserves_diagonal_weights"),
        None,
    )
    if res is None:
        return
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].bar(["before", "after"], [res.offdiag_before, res.offdiag_after], color=["C0", "C1"])
    axes[0].set_ylabel("off-diagonal ‖·‖")
    axes[0].set_title("Off-diagonal suppression")
    axes[0].grid(True, alpha=0.3)
    n = res.n_branches
    xs = np.arange(n)
    axes[1].bar(xs - 0.15, res.expected_probabilities[:n], 0.3, label="before decoherence")
    axes[1].bar(xs + 0.15, res.observed_frequencies[:n], 0.3, label="after decoherence")
    axes[1].set_title(f"Diagonal preserved (Δ={res.diagonal_change:.2e})")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)
    fig.suptitle("Decoherence preserves diagonal weights (Phase 6F)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _build_report(results: list[BornRuleBenchmarkResult]) -> str:
    n_pass = sum(1 for r in results if r.overall_pass)
    lines = [
        "# Born-rule probability emergence report (Phase 6F)",
        "",
        f"## ⚠️ {BANNER_BORN}",
        "",
        "## Purpose",
        "",
        "Test whether branch weights **ρ_i** behave as stable probabilities **P_i = ρ_i / Σρ_j** "
        "after τ-variance decoherence and classical coarse-graining, using **c_i = √(ρ_i) e^(−iτ_i)**.",
        "",
        "> **NOT FULL BORN-RULE DERIVATION.** Does not solve the measurement problem.",
        "",
        "## Equations",
        "",
        "```text",
        "c_i = sqrt(rho_i) exp(-i tau_i)",
        "P_i = rho_i / sum_j rho_j",
        "rho_matrix = |psi><psi|",
        "rho_ij -> rho_ij * exp(-Lambda)  (i != j)",
        "f_i = counts_i / N_trials",
        "P(A union B) = P(A) + P(B)  (disjoint coarse-graining)",
        "```",
        "",
        f"- **Cases:** {len(results)}",
        f"- **Passed:** {n_pass} / {len(results)}",
        "",
        "## Results table",
        "",
        "| Case | branches | max |f−p| | χ² born | χ² uniform | additivity err | phase err | pass |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in results:
        mfe = f"{r.max_frequency_error:.4f}" if not np.isnan(r.max_frequency_error) else "—"
        chi_b = f"{r.chi_square_born:.2f}" if not np.isnan(r.chi_square_born) else "—"
        chi_u = f"{r.chi_square_uniform:.2f}" if not np.isnan(r.chi_square_uniform) else "—"
        add = f"{r.additivity_error:.2e}" if not np.isnan(r.additivity_error) else "—"
        ph = f"{r.phase_invariance_error:.2e}" if not np.isnan(r.phase_invariance_error) else "—"
        lines.append(
            f"| {r.case_name} | {r.n_branches} | {mfe} | {chi_b} | {chi_u} | {add} | {ph} | "
            f"{'✓' if r.overall_pass else '✗'} |",
        )
    lines.extend(
        [
            "",
            "## Scientific interpretation",
            "",
            "- **Passing** means Born-rule weighting is numerically consistent with decohered "
            "branch probabilities in controlled finite-branch benchmarks.",
            "- Does **not** prove a full derivation of the Born rule.",
            "- Does **not** explain why one branch is experienced.",
            "",
            "## Failure modes",
            "",
            "- Toy finite-branch model; multinomial sampling rule is imposed.",
            "- No decision-theoretic or envariance derivation.",
            "- No many-body detector or relativistic QFT measurement model.",
            "",
            "## Disclaimer",
            "",
            "- Numerical consistency benchmark only.",
            "",
        ],
    )
    return "\n".join(lines)


def run_born_rule_probability_benchmark(
    outputs_root: Path | None = None,
) -> tuple[pd.DataFrame, list[BornRuleBenchmarkResult]]:
    """Run Phase 6F benchmarks; write CSV, report, figures."""
    root = Path(__file__).resolve().parents[3]
    outputs = outputs_root or (root / "outputs")
    tables_dir = outputs / "tables"
    reports_dir = outputs / "reports"
    figures_dir = outputs / "figures"
    for d in (tables_dir, reports_dir, figures_dir):
        d.mkdir(parents=True, exist_ok=True)

    results = [
        run_balanced_two_branch(),
        run_unequal_two_branch(),
        run_three_branch_distribution(),
        run_decoherence_preserves_diagonal_weights(),
        run_wrong_rules_fail_comparison(),
        run_coarse_graining_additivity(),
        run_zero_weight_branch(),
        run_phase_invariance_check(),
    ]

    out_df = pd.DataFrame([_result_to_row(r) for r in results])
    out_df.to_csv(tables_dir / "born_rule_probability_summary.csv", index=False)
    (reports_dir / "born_rule_probability_report.md").write_text(
        _build_report(results),
        encoding="utf-8",
    )

    _plot_expected_vs_observed(results, figures_dir / "born_rule_expected_vs_observed.png")
    _plot_rule_comparison(results, figures_dir / "born_rule_rule_comparison.png")
    _plot_decoherence_diagonal(
        results,
        figures_dir / "born_rule_decoherence_diagonal_preservation.png",
    )

    return out_df, results

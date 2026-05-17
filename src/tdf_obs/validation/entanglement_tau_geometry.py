"""Phase 6C — Entanglement from configuration-space τ geometry (not Bell resolution)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np
import pandas as pd

BENCHMARK_MODE = "entanglement_tau_geometry_benchmark"
BANNER_ENTANGLEMENT = (
    "ENTANGLEMENT / NONLOCAL CORRELATION BENCHMARK — NOT FULL BELL-THEOREM RESOLUTION"
)

ENTROPY_MAX_PRODUCT = 0.05
ENTROPY_MIN_BELL = 0.65  # log(2) nats for maximally entangled reduced state
CONCURRENCE_MAX_PRODUCT = 0.12
CONCURRENCE_MIN_BELL = 0.9
CHSH_CLASSICAL = 2.0
CHSH_BELL_TARGET = 2.0 * np.sqrt(2.0)
CHSH_TOL = 0.05
PHASE_SEP_MAX_PRODUCT = 0.1
PHASE_NONSEP_MIN = 0.2
NO_SIGNALING_TOL = 1e-10

DEFAULT_CHSH_ANGLES: dict[str, float] = {
    "a": 0.0,
    "a_prime": np.pi / 2.0,
    "b": np.pi / 4.0,
    "b_prime": 3.0 * np.pi / 4.0,
}

REQUIRED_SUMMARY_COLUMNS: tuple[str, ...] = (
    "case_name",
    "entropy_A",
    "concurrence",
    "chsh_S",
    "bell_violation",
    "no_signaling_pass",
    "phase_nonseparability_score",
    "amplitude_nonseparability_score",
    "overall_pass",
    "warning",
)


@dataclass
class EntanglementBenchmarkResult:
    case_name: str
    entropy_A: float
    concurrence: float
    chsh_S: float
    bell_violation: bool
    no_signaling_pass: bool
    phase_nonseparability_score: float
    amplitude_nonseparability_score: float
    overall_pass: bool
    warning: str = ""


def pauli_matrices() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """σ_x, σ_y, σ_z."""
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    return sx, sy, sz


def _pauli_axis_xz(angle: float) -> np.ndarray:
    """σ·n with n in the x–z plane: n = (sin θ, 0, cos θ)."""
    _, _, sz = pauli_matrices()
    sx, _, _ = pauli_matrices()
    return np.cos(angle) * sz + np.sin(angle) * sx


def psi_from_rho_tau_2body(rho: np.ndarray, tau: np.ndarray) -> np.ndarray:
    """Ψ = √ρ exp(−iτ) (elementwise on configuration grid)."""
    rho = np.asarray(rho, dtype=float)
    tau = np.asarray(tau, dtype=float)
    return np.sqrt(np.maximum(rho, 0.0)) * np.exp(-1j * tau)


def _state_to_density_matrix(state_vector: np.ndarray) -> np.ndarray:
    psi = np.asarray(state_vector, dtype=complex).reshape(-1)
    psi = psi / np.linalg.norm(psi)
    return np.outer(psi, psi.conj())


def reduced_density_matrix_2qubit(
    state_vector: np.ndarray,
    subsystem: Literal["A", "B"],
) -> np.ndarray:
    """Partial trace over the other qubit (|00>,|01>,|10>,|11> ordering)."""
    psi = np.asarray(state_vector, dtype=complex).reshape(4)
    rho = _state_to_density_matrix(psi).reshape(2, 2, 2, 2)
    if subsystem == "A":
        return np.einsum("ijik->jk", rho)
    return np.einsum("ijkj->ik", rho)


def von_neumann_entropy(rho_matrix: np.ndarray) -> float:
    """S = −Tr(ρ log ρ) in natural log (nats)."""
    rho = np.asarray(rho_matrix, dtype=complex)
    rho = 0.5 * (rho + rho.conj().T)
    evals = np.linalg.eigvalsh(rho)
    evals = np.clip(evals.real, 0.0, 1.0)
    nz = evals[evals > 1e-14]
    if nz.size == 0:
        return 0.0
    return float(-np.sum(nz * np.log(nz)))


def concurrence_2qubit(state_vector: np.ndarray) -> float:
    """Concurrence for a pure two-qubit state."""
    psi = np.asarray(state_vector, dtype=complex).reshape(4)
    psi = psi / np.linalg.norm(psi)
    _, sy, _ = pauli_matrices()
    yy = np.kron(sy, sy)
    return float(np.abs(psi.conj() @ yy @ psi))


def bell_state(name: str) -> np.ndarray:
    """Normalized Bell state in |00>,|01>,|10>,|11> ordering."""
    key = name.lower().replace(" ", "_").replace("-", "_")
    states = {
        "phi_plus": np.array([1, 0, 0, 1], dtype=complex),
        "phi_minus": np.array([1, 0, 0, -1], dtype=complex),
        "psi_plus": np.array([0, 1, 1, 0], dtype=complex),
        "psi_minus": np.array([0, 1, -1, 0], dtype=complex),
    }
    if key not in states:
        raise ValueError(f"Unknown Bell state: {name}")
    psi = states[key]
    return psi / np.linalg.norm(psi)


def product_state(
    theta1: float,
    phi1: float,
    theta2: float,
    phi2: float,
) -> np.ndarray:
    """|ψ₁(θ₁,φ₁)⟩ ⊗ |ψ₂(θ₂,φ₂)⟩ on the Bloch sphere."""
    def single(theta: float, phi: float) -> np.ndarray:
        c = np.cos(theta / 2.0)
        s = np.sin(theta / 2.0)
        return np.array([c, s * np.exp(1j * phi)], dtype=complex)

    return np.kron(single(theta1, phi1), single(theta2, phi2))


def partially_entangled_state(theta: float) -> np.ndarray:
    """cos θ|00⟩ + sin θ|11⟩ (real, normalized)."""
    psi = np.array([np.cos(theta), 0, 0, np.sin(theta)], dtype=complex)
    return psi / np.linalg.norm(psi)


def state_to_tau_rho_grids(state_vector: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Build 2×2 configuration grids ρ_ij, τ_ij from |ψ⟩.

    ψ reshaped as ψ_{i_A, i_B}; τ_ij = −arg(ψ_ij), ρ_ij = |ψ_ij|².
    """
    psi = np.asarray(state_vector, dtype=complex).reshape(2, 2)
    rho = np.abs(psi) ** 2
    tau = -np.angle(psi)
    return rho, tau


def phase_nonseparability_score(tau_matrix: np.ndarray) -> float:
    """
    Normalized residual for τ_ij ≈ a_i + b_j (least squares, b_0 = 0 gauge).
    """
    tau = np.asarray(tau_matrix, dtype=float)
    n, m = tau.shape
    if n != m:
        raise ValueError("tau_matrix must be square")
    rows: list[np.ndarray] = []
    targets: list[float] = []
    n_unknown = 2 * n - 1
    for i in range(n):
        for j in range(n):
            row = np.zeros(n_unknown)
            row[i] = 1.0
            if j > 0:
                row[n + j - 1] = 1.0
            rows.append(row)
            targets.append(tau[i, j])
    design = np.stack(rows)
    sol, _, _, _ = np.linalg.lstsq(design, np.array(targets), rcond=None)
    fitted = design @ sol
    residual = np.array(targets) - fitted
    scale = max(1.0, float(np.max(np.abs(tau))))
    return float(np.max(np.abs(residual)) / scale)


def amplitude_nonseparability_score(rho_matrix: np.ndarray) -> float:
    """
    Normalized residual for ρ_ij ≈ ρ_i ρ_j (least squares, product form on grid).
    """
    rho = np.asarray(rho_matrix, dtype=float)
    n, m = rho.shape
    if n != m:
        raise ValueError("rho_matrix must be square")
    rows: list[np.ndarray] = []
    targets: list[float] = []
    n_unknown = 2 * n - 1
    for i in range(n):
        for j in range(n):
            row = np.zeros(n_unknown)
            row[i] = 1.0
            if j > 0:
                row[n + j - 1] = 1.0
            rows.append(row)
            targets.append(rho[i, j])
    design = np.stack(rows)
    # Product form: use log-linear fit when rho > 0, else linear fallback
    if np.all(rho > 1e-14):
        targets_log = np.log(np.array(targets))
        sol, _, _, _ = np.linalg.lstsq(design, targets_log, rcond=None)
        fitted = np.exp(design @ sol)
    else:
        sol, _, _, _ = np.linalg.lstsq(design, np.array(targets), rcond=None)
        fitted = design @ sol
    residual = np.array(targets) - fitted
    scale = max(1.0, float(np.max(np.abs(rho))))
    return float(np.max(np.abs(residual)) / scale)


def _measurement_projectors(angle: float) -> tuple[np.ndarray, np.ndarray]:
    """Projectors onto ± eigenstates of σ·n (x–z plane)."""
    op = _pauli_axis_xz(angle)
    evals, evecs = np.linalg.eigh(op)
    idx_plus = int(np.argmax(evals.real))
    idx_minus = 1 - idx_plus
    p_plus = np.outer(evecs[:, idx_plus], evecs[:, idx_plus].conj())
    p_minus = np.outer(evecs[:, idx_minus], evecs[:, idx_minus].conj())
    return p_plus, p_minus


def chsh_correlation_for_state(
    state_vector: np.ndarray,
    angles: dict[str, float] | None = None,
) -> dict[str, float]:
    """Correlations E(a,b), E(a,b′), E(a′,b), E(a′,b′)."""
    if angles is None:
        angles = DEFAULT_CHSH_ANGLES
    psi = np.asarray(state_vector, dtype=complex).reshape(4)
    rho = _state_to_density_matrix(psi)

    def correlation(theta_a: float, theta_b: float) -> float:
        pa_p, pa_m = _measurement_projectors(theta_a)
        pb_p, pb_m = _measurement_projectors(theta_b)
        ops = [
            (np.kron(pa_p, pb_p), +1.0),
            (np.kron(pa_p, pb_m), -1.0),
            (np.kron(pa_m, pb_p), -1.0),
            (np.kron(pa_m, pb_m), +1.0),
        ]
        return float(sum(w * np.real(np.trace(op @ rho)) for op, w in ops))

    return {
        "E_ab": correlation(angles["a"], angles["b"]),
        "E_ab_prime": correlation(angles["a"], angles["b_prime"]),
        "E_a_prime_b": correlation(angles["a_prime"], angles["b"]),
        "E_a_prime_b_prime": correlation(angles["a_prime"], angles["b_prime"]),
    }


def chsh_S_value(
    state_vector: np.ndarray,
    angles: dict[str, float] | None = None,
) -> float:
    """S = E(a,b) − E(a,b′) + E(a′,b) + E(a′,b′)."""
    c = chsh_correlation_for_state(state_vector, angles)
    return (
        c["E_ab"]
        - c["E_ab_prime"]
        + c["E_a_prime_b"]
        + c["E_a_prime_b_prime"]
    )


def no_signaling_check(
    state_vector: np.ndarray,
    measurement_basis_A: Sequence[float],
    measurement_basis_B: Sequence[float],
    *,
    tol: float = NO_SIGNALING_TOL,
) -> bool:
    """
    Marginal outcome probabilities for A must not depend on B's measurement choice.
    """
    psi = np.asarray(state_vector, dtype=complex).reshape(4)
    rho = _state_to_density_matrix(psi)

    def marginal_a(theta_b: float) -> np.ndarray:
        pb_p, pb_m = _measurement_projectors(theta_b)
        probs = []
        for theta_a in measurement_basis_A:
            pa_p, pa_m = _measurement_projectors(theta_a)
            for pa in (pa_p, pa_m):
                for pb in (pb_p, pb_m):
                    op = np.kron(pa, pb)
                    probs.append(float(np.real(np.trace(op @ rho))))
        # Return Alice marginal over her ± outcomes for first setting in list
        pa_p, pa_m = _measurement_projectors(measurement_basis_A[0])
        p_plus = float(np.real(np.trace(np.kron(pa_p, np.eye(2)) @ rho)))
        p_minus = float(np.real(np.trace(np.kron(pa_m, np.eye(2)) @ rho)))
        return np.array([p_plus, p_minus])

    if len(measurement_basis_B) < 2:
        return True
    m0 = marginal_a(measurement_basis_B[0])
    for theta_b in measurement_basis_B[1:]:
        if np.max(np.abs(marginal_a(theta_b) - m0)) > tol:
            return False
    return True


def nonseparable_tau_demo(size: int = 4) -> np.ndarray:
    """τ_ij matrix that cannot be written as a_i + b_j (non-additive phase grid)."""
    i = np.arange(size, dtype=float)[:, None]
    j = np.arange(size, dtype=float)[None, :]
    return (i * j + (i - j) ** 2) * (np.pi / (2.0 * size))


def _evaluate_case(
    case_name: str,
    state: np.ndarray | None,
    *,
    tau_demo: np.ndarray | None = None,
    expect_entangled: bool,
    expect_chsh_above_classical: bool | None = None,
    entropy_range: tuple[float, float] | None = None,
    concurrence_range: tuple[float, float] | None = None,
    phase_sep_max: float | None = None,
    phase_nonsep_min: float | None = None,
    warning: str = "",
) -> EntanglementBenchmarkResult:
    if state is not None:
        rho_a = reduced_density_matrix_2qubit(state, "A")
        entropy = von_neumann_entropy(rho_a)
        conc = concurrence_2qubit(state)
        chsh = chsh_S_value(state)
        bell_viol = abs(chsh) > CHSH_CLASSICAL + 1e-9
        no_sig = no_signaling_check(
            state,
            [DEFAULT_CHSH_ANGLES["a"], DEFAULT_CHSH_ANGLES["a_prime"]],
            [DEFAULT_CHSH_ANGLES["b"], DEFAULT_CHSH_ANGLES["b_prime"]],
        )
        rho_grid, tau_grid = state_to_tau_rho_grids(state)
        phase_score = phase_nonseparability_score(tau_grid)
        amp_score = amplitude_nonseparability_score(rho_grid)
    else:
        entropy = float("nan")
        conc = float("nan")
        chsh = float("nan")
        bell_viol = False
        no_sig = True
        rho_grid = np.ones((4, 4)) / 16.0
        phase_score = phase_nonseparability_score(tau_demo)  # type: ignore[arg-type]
        amp_score = float("nan")

    checks: list[bool] = [no_sig]
    if state is not None:
        if expect_entangled:
            if concurrence_range is None:
                checks.append(conc >= CONCURRENCE_MIN_BELL)
            if entropy_range is None:
                checks.append(entropy >= ENTROPY_MIN_BELL)
            checks.append(amp_score >= PHASE_NONSEP_MIN or conc >= 0.5)
        else:
            checks.append(entropy <= ENTROPY_MAX_PRODUCT)
            checks.append(conc <= CONCURRENCE_MAX_PRODUCT)
            checks.append(amp_score <= PHASE_SEP_MAX_PRODUCT)
            if phase_sep_max is not None:
                checks.append(phase_score <= phase_sep_max)
        if expect_chsh_above_classical is True:
            checks.append(abs(chsh) > CHSH_CLASSICAL + 0.01)
        elif expect_chsh_above_classical is False:
            checks.append(abs(chsh) <= CHSH_CLASSICAL + CHSH_TOL)
        if entropy_range is not None:
            checks.append(entropy_range[0] <= entropy <= entropy_range[1])
        if concurrence_range is not None:
            checks.append(concurrence_range[0] <= conc <= concurrence_range[1])
        if "bell" in case_name:
            checks.append(abs(abs(chsh) - CHSH_BELL_TARGET) < 0.15)
    else:
        checks.append(phase_score >= PHASE_NONSEP_MIN)

    overall = all(checks)

    return EntanglementBenchmarkResult(
        case_name=case_name,
        entropy_A=entropy,
        concurrence=conc,
        chsh_S=chsh,
        bell_violation=bell_viol,
        no_signaling_pass=no_sig,
        phase_nonseparability_score=phase_score,
        amplitude_nonseparability_score=amp_score,
        overall_pass=overall,
        warning=warning,
    )


def run_product_state_control() -> EntanglementBenchmarkResult:
    state = product_state(0.3, 0.2, 0.7, 1.1)
    return _evaluate_case(
        "product_state_control",
        state,
        expect_entangled=False,
        expect_chsh_above_classical=False,
        phase_sep_max=PHASE_SEP_MAX_PRODUCT,
    )


def run_bell_phi_plus() -> EntanglementBenchmarkResult:
    return _evaluate_case(
        "bell_phi_plus",
        bell_state("phi_plus"),
        expect_entangled=True,
        expect_chsh_above_classical=True,
    )


def run_bell_psi_minus() -> EntanglementBenchmarkResult:
    return _evaluate_case(
        "bell_psi_minus",
        bell_state("psi_minus"),
        expect_entangled=True,
        expect_chsh_above_classical=True,
    )


def run_partially_entangled_state(theta: float = np.pi / 6.0) -> EntanglementBenchmarkResult:
    state = partially_entangled_state(theta)
    return _evaluate_case(
        "partially_entangled_state",
        state,
        expect_entangled=True,
        expect_chsh_above_classical=True,
        entropy_range=(0.05, 0.95),
        concurrence_range=(0.05, 0.99),
        warning=f"θ={theta:.3f}; partial entanglement.",
    )


def run_random_product_state(seed: int = 42) -> EntanglementBenchmarkResult:
    rng = np.random.default_rng(seed)
    state = product_state(
        float(rng.uniform(0, np.pi)),
        float(rng.uniform(0, 2 * np.pi)),
        float(rng.uniform(0, np.pi)),
        float(rng.uniform(0, 2 * np.pi)),
    )
    return _evaluate_case(
        "random_product_state",
        state,
        expect_entangled=False,
        expect_chsh_above_classical=False,
        phase_sep_max=PHASE_SEP_MAX_PRODUCT,
    )


def run_mixed_phase_nonseparable_tau_demo() -> EntanglementBenchmarkResult:
    tau = nonseparable_tau_demo(4)
    return _evaluate_case(
        "mixed_phase_nonseparable_tau_demo",
        None,
        tau_demo=tau,
        expect_entangled=True,
        warning="Geometric τ coupling demo only; not a physical mixed state.",
    )


def _result_to_row(res: EntanglementBenchmarkResult) -> dict[str, Any]:
    return {
        "case_name": res.case_name,
        "entropy_A": res.entropy_A,
        "concurrence": res.concurrence,
        "chsh_S": res.chsh_S,
        "bell_violation": res.bell_violation,
        "no_signaling_pass": res.no_signaling_pass,
        "phase_nonseparability_score": res.phase_nonseparability_score,
        "amplitude_nonseparability_score": res.amplitude_nonseparability_score,
        "overall_pass": res.overall_pass,
        "warning": res.warning,
        "dataset_mode": BENCHMARK_MODE,
        "is_real_observational_data": False,
    }


def _plot_chsh(results: list[EntanglementBenchmarkResult], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names, values = [], []
    for r in results:
        if not np.isnan(r.chsh_S):
            names.append(r.case_name)
            values.append(r.chsh_S)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(names, values, color="C0")
    ax.axhline(CHSH_CLASSICAL, color="k", ls="--", label="classical bound S=2")
    ax.axhline(CHSH_BELL_TARGET, color="C3", ls=":", label="Tsirelson S=2√2")
    ax.set_ylabel("CHSH S")
    ax.set_title("CHSH values (Phase 6C)")
    plt.xticks(rotation=25, ha="right")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_entropy_concurrence(results: list[EntanglementBenchmarkResult], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names, ent, conc = [], [], []
    for r in results:
        if not np.isnan(r.entropy_A):
            names.append(r.case_name)
            ent.append(r.entropy_A)
            conc.append(r.concurrence)
    x = np.arange(len(names))
    w = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - w / 2, ent, w, label="S_A (nats)")
    ax.bar(x + w / 2, conc, w, label="concurrence")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=25, ha="right")
    ax.set_title("Entropy and concurrence (Phase 6C)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_tau_scores(results: list[EntanglementBenchmarkResult], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [r.case_name for r in results]
    phase = [r.phase_nonseparability_score for r in results]
    amp = [r.amplitude_nonseparability_score for r in results]
    x = np.arange(len(names))
    w = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - w / 2, phase, w, label="phase nonseparability")
    ax.bar(x + w / 2, [a if not np.isnan(a) else 0 for a in amp], w, label="amplitude nonseparability")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=25, ha="right")
    ax.set_title("τ / ρ nonseparability scores (Phase 6C)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _build_report(results: list[EntanglementBenchmarkResult]) -> str:
    n_pass = sum(1 for r in results if r.overall_pass)
    lines = [
        "# Entanglement / τ geometry report (Phase 6C)",
        "",
        f"## ⚠️ {BANNER_ENTANGLEMENT}",
        "",
        "## Purpose",
        "",
        "Test whether **nonseparable configuration-space τ geometry** can represent "
        "entangled two-qubit correlations while preserving **no-signaling**.",
        "",
        "> **NOT FULL BELL-THEOREM RESOLUTION.** No claim of superluminal signaling.",
        "",
        "## Equations",
        "",
        "```text",
        "Ψ(x₁,x₂) = √ρ(x₁,x₂) exp(−i τ(x₁,x₂))",
        "Separable: τ(x₁,x₂) = τ₁(x₁) + τ₂(x₂),  ρ(x₁,x₂) = ρ₁(x₁)ρ₂(x₂)",
        "ρ_A = Tr_B |Ψ⟩⟨Ψ|",
        "S_A = −Tr(ρ_A log ρ_A)",
        "Concurrence C(|ψ⟩)",
        "S_CHSH = E(a,b) − E(a,b′) + E(a′,b) + E(a′,b′)",
        "No-signaling: P(a | x, y) = P(a | x)  (marginals independent of remote setting y)",
        "```",
        "",
        "## Summary",
        "",
        f"- **Cases:** {len(results)}",
        f"- **Pass:** {n_pass} / {len(results)}",
        "",
        "## Results table",
        "",
        "| Case | S_A | C | CHSH S | Bell viol. | no-signal | phase score | amp score | pass |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in results:
        lines.append(
            f"| {r.case_name} | {r.entropy_A:.3f} | {r.concurrence:.3f} | {r.chsh_S:.3f} | "
            f"{'yes' if r.bell_violation else 'no'} | "
            f"{'yes' if r.no_signaling_pass else 'no'} | "
            f"{r.phase_nonseparability_score:.3f} | {r.amplitude_nonseparability_score:.3f} | "
            f"{'✓' if r.overall_pass else '✗'} |",
        )
    lines.extend(
        [
            "",
            "## Scientific interpretation",
            "",
            "- **Passing** means TDF can encode entangled states as nonseparable τ geometry "
            "in controlled two-qubit benchmarks with quantum no-signaling preserved.",
            "- Does **not** prove a hidden-variable resolution of Bell's theorem.",
            "- Does **not** allow faster-than-light signaling.",
            "",
            "## Failure modes",
            "",
            "- Two-qubit toy system only; configuration-space τ, not local τ(x).",
            "- No relativistic QFT, measurement collapse, or experimental Bell data.",
            "",
            "## Disclaimer",
            "",
            "- Numerical consistency benchmark only.",
            "",
        ],
    )
    return "\n".join(lines)


def run_entanglement_tau_geometry_benchmark(
    outputs_root: Path | None = None,
) -> tuple[pd.DataFrame, list[EntanglementBenchmarkResult]]:
    """Run Phase 6C benchmarks; write CSV, report, figures."""
    root = Path(__file__).resolve().parents[3]
    outputs = outputs_root or (root / "outputs")
    tables_dir = outputs / "tables"
    reports_dir = outputs / "reports"
    figures_dir = outputs / "figures"
    for d in (tables_dir, reports_dir, figures_dir):
        d.mkdir(parents=True, exist_ok=True)

    results = [
        run_product_state_control(),
        run_bell_phi_plus(),
        run_bell_psi_minus(),
        run_partially_entangled_state(),
        run_random_product_state(),
        run_mixed_phase_nonseparable_tau_demo(),
    ]

    out_df = pd.DataFrame([_result_to_row(r) for r in results])
    out_df.to_csv(tables_dir / "entanglement_tau_geometry_summary.csv", index=False)
    (reports_dir / "entanglement_tau_geometry_report.md").write_text(
        _build_report(results),
        encoding="utf-8",
    )
    _plot_chsh(results, figures_dir / "entanglement_chsh_values.png")
    _plot_entropy_concurrence(results, figures_dir / "entanglement_entropy_concurrence.png")
    _plot_tau_scores(results, figures_dir / "tau_nonseparability_scores.png")
    return out_df, results

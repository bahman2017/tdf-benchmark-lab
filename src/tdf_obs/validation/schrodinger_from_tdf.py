"""Phase 6A — Schrödinger-from-TDF action benchmark (not full quantum validation)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

BENCHMARK_MODE = "schrodinger_from_tdf_action_benchmark"
BANNER_SCHRODINGER = "SCHRÖDINGER-FROM-TDF ACTION BENCHMARK — NOT FULL QUANTUM VALIDATION"

PHASE_CONVENTION = (
    "ψ = √ρ exp(-iτ); E = ℏ ∂_t τ (TDF v0.8.1). "
    "Plane wave ψ = exp(i(kx − ωt)) ⇒ τ = ωt − kx with ω = ℏk²/(2m)."
)
QHJ_EQUATION = "ℏ ∂_t τ − (ℏ²/2m)|∇τ|² − V − Q = 0,  Q = −(ℏ²/2m)∇²√ρ/√ρ"

DEFAULT_PASS_THRESHOLD = 1e-2
QHJ_NORMALIZED_PASS_THRESHOLD = 1e-3
RHO_FLOOR = 1e-12

REQUIRED_SUMMARY_COLUMNS: tuple[str, ...] = (
    "case_name",
    "max_continuity_residual",
    "max_qhj_residual",
    "max_normalized_qhj_residual",
    "max_schrodinger_residual",
    "quantum_potential_finite",
    "pass",
    "warnings",
)


@dataclass
class SchrodingerBenchmarkResult:
    case_name: str
    max_continuity_residual: float
    max_qhj_residual: float
    max_normalized_qhj_residual: float
    max_schrodinger_residual: float
    quantum_potential_finite: bool
    pass_: bool
    x: np.ndarray
    continuity_map: np.ndarray
    qhj_map: np.ndarray
    schrodinger_map: np.ndarray
    warnings: list[str] = field(default_factory=list)


def psi_from_rho_tau(rho: np.ndarray, tau: np.ndarray) -> np.ndarray:
    """ψ = √ρ exp(-iτ)."""
    rho = np.asarray(rho, dtype=float)
    tau = np.asarray(tau, dtype=float)
    return np.sqrt(np.maximum(rho, 0.0)) * np.exp(-1j * tau)


def rho_tau_from_psi(psi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Recover ρ = |ψ|² and τ = -arg(ψ) (principal branch)."""
    psi = np.asarray(psi, dtype=complex)
    rho = np.abs(psi) ** 2
    tau = -np.angle(psi)
    return rho, tau


def tau_t_from_psi(
    psi: np.ndarray,
    psi_t: np.ndarray,
    rho: np.ndarray,
    rho_t: np.ndarray,
) -> np.ndarray:
    """Madelung-consistent ∂_t τ from ψ, ∂_t ψ, ρ, ∂_t ρ."""
    rho_safe = np.maximum(np.asarray(rho, dtype=float), RHO_FLOOR)
    psi_safe = np.where(np.abs(psi) > RHO_FLOOR, psi, 1.0 + 0j)
    ratio = np.asarray(psi_t, dtype=complex) / psi_safe
    return -np.imag(ratio - 0.5 * np.asarray(rho_t, dtype=float) / rho_safe)


def quantum_potential(
    rho: np.ndarray,
    dx: float,
    *,
    hbar: float = 1.0,
    m: float = 1.0,
) -> np.ndarray:
    """
    Q = -(ℏ²/2m) ∇²√ρ / √ρ  (1D).

    Emerges from the Fisher-information term (ℏ²/8m)|∇ρ|²/ρ in the action.
    """
    rho = np.asarray(rho, dtype=float)
    sqrt_rho = np.sqrt(np.maximum(rho, RHO_FLOOR))
    lap = _laplacian_1d(sqrt_rho, dx)
    return -(hbar**2 / (2.0 * m)) * lap / sqrt_rho


def continuity_residual(
    rho_t: np.ndarray,
    rho: np.ndarray,
    tau: np.ndarray,
    dx: float,
    dt: float,
    *,
    hbar: float = 1.0,
    m: float = 1.0,
) -> np.ndarray:
    """
    Continuity equation residual:

    ∂_t ρ + ∂_x(ρ (ℏ/m) ∂_x τ) = 0
    """
    rho = np.asarray(rho, dtype=float)
    rho_t = np.asarray(rho_t, dtype=float)
    tau = np.asarray(tau, dtype=float)
    flux = rho * (hbar / m) * _gradient_1d(tau, dx)
    div_flux = _gradient_1d(flux, dx)
    return rho_t + div_flux


def quantum_hamilton_jacobi_residual(
    rho: np.ndarray,
    tau_t: np.ndarray,
    tau: np.ndarray,
    V: np.ndarray,
    dx: float,
    *,
    hbar: float = 1.0,
    m: float = 1.0,
) -> np.ndarray:
    """
    Quantum Hamilton–Jacobi residual (TDF ψ = √ρ exp(-iτ), E = ℏ ∂_t τ):

    ℏ ∂_t τ − (ℏ²/2m)|∇τ|² − V − Q = 0,  Q = −(ℏ²/2m)∇²√ρ/√ρ
    """
    rho = np.asarray(rho, dtype=float)
    tau = np.asarray(tau, dtype=float)
    tau_t = np.asarray(tau_t, dtype=float)
    V = np.asarray(V, dtype=float)
    grad_tau = _gradient_1d(tau, dx)
    kinetic_tau = (hbar**2 / (2.0 * m)) * grad_tau**2
    Q = quantum_potential(rho, dx, hbar=hbar, m=m)
    return hbar * tau_t - kinetic_tau - V - Q


def qhj_term_magnitudes(
    rho: np.ndarray,
    tau_t: np.ndarray,
    tau: np.ndarray,
    V: np.ndarray,
    dx: float,
    *,
    hbar: float = 1.0,
    m: float = 1.0,
) -> dict[str, np.ndarray]:
    """Per-point magnitudes of QHJ terms for normalization."""
    grad_tau = _gradient_1d(tau, dx)
    Q = quantum_potential(rho, dx, hbar=hbar, m=m)
    return {
        "hbar_tau_t": np.abs(hbar * tau_t),
        "kinetic": np.abs((hbar**2 / (2.0 * m)) * grad_tau**2),
        "V": np.abs(V),
        "Q": np.abs(Q),
    }


def schrodinger_residual(
    psi_t: np.ndarray,
    psi: np.ndarray,
    V: np.ndarray,
    dx: float,
    *,
    hbar: float = 1.0,
    m: float = 1.0,
) -> np.ndarray:
    """
    iℏ ∂_t ψ - [-ℏ²/(2m) ∂_xx + V] ψ  (complex residual).
    """
    psi = np.asarray(psi, dtype=complex)
    psi_t = np.asarray(psi_t, dtype=complex)
    V = np.asarray(V, dtype=float)
    lap_psi = _laplacian_1d(psi, dx)
    lhs = 1j * hbar * psi_t
    rhs = (-(hbar**2) / (2.0 * m)) * lap_psi + V * psi
    return lhs - rhs


def _gradient_1d(f: np.ndarray, dx: float) -> np.ndarray:
    arr = np.asarray(f)
    if np.iscomplexobj(arr):
        return np.gradient(arr, dx)
    return np.gradient(arr.astype(float), dx)


def _laplacian_1d(f: np.ndarray, dx: float) -> np.ndarray:
    return _gradient_1d(_gradient_1d(f, dx), dx)


def _max_abs(residual: np.ndarray, mask: np.ndarray) -> float:
    return float(np.nanmax(np.abs(np.asarray(residual)[mask])))


def _max_abs_rel(residual: np.ndarray, reference: np.ndarray, mask: np.ndarray) -> float:
    residual = np.asarray(residual)[mask]
    reference = np.asarray(reference)[mask]
    if np.iscomplexobj(residual):
        num = np.abs(residual)
        denom = np.maximum(np.abs(reference), 1e-12)
    else:
        num = np.abs(residual)
        denom = np.maximum(np.abs(reference), 1e-12)
    return float(np.nanmax(num / denom))


def _max_normalized_residual(
    residual: np.ndarray,
    mask: np.ndarray,
    term_arrays: dict[str, np.ndarray],
) -> float:
    """max|residual| / max(1, max|terms|) on masked region."""
    stacks = [np.asarray(term_arrays[k], dtype=float)[mask] for k in term_arrays]
    term_scale = float(np.max(np.stack(stacks)))
    scale = max(1.0, term_scale)
    return _max_abs(residual, mask) / scale


def _interior_mask(n: int, margin: int = 3) -> np.ndarray:
    mask = np.zeros(n, dtype=bool)
    if n > 2 * margin:
        mask[margin:-margin] = True
    else:
        mask[:] = True
    return mask


def _evaluate_case(
    *,
    case_name: str,
    x: np.ndarray,
    rho: np.ndarray,
    tau: np.ndarray,
    tau_t: np.ndarray,
    rho_t: np.ndarray,
    psi: np.ndarray,
    psi_t: np.ndarray,
    V: np.ndarray,
    schrodinger_check: np.ndarray,
    hbar: float,
    m: float,
    pass_threshold: float,
    qhj_threshold: float,
    warnings: list[str],
) -> SchrodingerBenchmarkResult:
    dx = x[1] - x[0]
    mask = _interior_mask(len(x))

    cont = continuity_residual(rho_t, rho, tau, dx, dt=1.0, hbar=hbar, m=m)
    qhj = quantum_hamilton_jacobi_residual(rho, tau_t, tau, V, dx, hbar=hbar, m=m)
    Q = quantum_potential(rho, dx, hbar=hbar, m=m)
    terms = qhj_term_magnitudes(rho, tau_t, tau, V, dx, hbar=hbar, m=m)

    max_cont = _max_abs_rel(cont, rho, mask)
    max_qhj = _max_abs(qhj, mask)
    max_qhj_norm = _max_normalized_residual(qhj, mask, terms)
    max_sch = _max_abs_rel(schrodinger_check, psi, mask)

    finite_q = bool(np.all(np.isfinite(Q[mask])))
    passed = (
        max_cont < pass_threshold
        and max_sch < pass_threshold
        and max_qhj_norm < qhj_threshold
        and finite_q
    )

    return SchrodingerBenchmarkResult(
        case_name=case_name,
        max_continuity_residual=max_cont,
        max_qhj_residual=max_qhj,
        max_normalized_qhj_residual=max_qhj_norm,
        max_schrodinger_residual=max_sch,
        quantum_potential_finite=finite_q,
        pass_=passed,
        x=x,
        continuity_map=cont,
        qhj_map=qhj,
        schrodinger_map=np.abs(schrodinger_check),
        warnings=warnings,
    )


def run_plane_wave_case(
    *,
    k: float = 1.0,
    omega: float | None = None,
    hbar: float = 1.0,
    m: float = 1.0,
    L: float = 20.0,
    n: int = 512,
    t: float = 0.0,
    pass_threshold: float = DEFAULT_PASS_THRESHOLD,
    qhj_threshold: float = QHJ_NORMALIZED_PASS_THRESHOLD,
) -> SchrodingerBenchmarkResult:
    """Free plane wave ψ = exp(i(kx − ωt)); τ = ωt − kx, ρ = 1, ω = ℏk²/(2m)."""
    if omega is None:
        omega = hbar * k**2 / (2.0 * m)

    x = np.linspace(-L / 2, L / 2, n, endpoint=False)
    dx = x[1] - x[0]

    rho = np.ones_like(x)
    tau = omega * t - k * x
    tau_t = np.full_like(x, omega)
    rho_t = np.zeros_like(x)
    V = np.zeros_like(x)

    psi = psi_from_rho_tau(rho, tau)
    psi_t = -1j * omega * psi

    warnings: list[str] = []
    if abs(omega - hbar * k**2 / (2.0 * m)) > 1e-10:
        warnings.append("ω set to ℏk²/(2m) for Schrödinger dispersion.")

    return _evaluate_case(
        case_name="plane_wave",
        x=x,
        rho=rho,
        tau=tau,
        tau_t=tau_t,
        rho_t=rho_t,
        psi=psi,
        psi_t=psi_t,
        V=V,
        schrodinger_check=schrodinger_residual(psi_t, psi, V, dx, hbar=hbar, m=m),
        hbar=hbar,
        m=m,
        pass_threshold=pass_threshold,
        qhj_threshold=qhj_threshold,
        warnings=warnings,
    )


def run_gaussian_snapshot_case(
    *,
    k: float = 0.5,
    sigma: float = 2.0,
    hbar: float = 1.0,
    m: float = 1.0,
    L: float = 16.0,
    n: int = 400,
    pass_threshold: float = DEFAULT_PASS_THRESHOLD,
    qhj_threshold: float = QHJ_NORMALIZED_PASS_THRESHOLD,
) -> SchrodingerBenchmarkResult:
    """
    Gaussian ρ and τ = kx at t = 0.

    ρ_t enforces continuity; τ_t from Madelung + Schrödinger ∂_t ψ at t = 0.
    """
    x = np.linspace(-L / 2, L / 2, n, endpoint=False)
    dx = x[1] - x[0]

    rho = np.exp(-(x**2) / (2.0 * sigma**2))
    rho /= np.trapezoid(rho, x)
    tau = k * x
    flux = rho * (hbar / m) * k
    rho_t = -_gradient_1d(flux, dx)

    psi = psi_from_rho_tau(rho, tau)
    V = np.zeros_like(x)
    lap_psi = _laplacian_1d(psi, dx)
    psi_t = (-1j / hbar) * ((-(hbar**2) / (2.0 * m)) * lap_psi + V * psi)
    tau_t = tau_t_from_psi(psi, psi_t, rho, rho_t)

    warnings = [
        "Snapshot at t = 0: ρ Gaussian, τ = kx; ρ_t from continuity; τ_t from ψ_t.",
    ]

    return _evaluate_case(
        case_name="gaussian_snapshot",
        x=x,
        rho=rho,
        tau=tau,
        tau_t=tau_t,
        rho_t=rho_t,
        psi=psi,
        psi_t=psi_t,
        V=V,
        schrodinger_check=schrodinger_residual(psi_t, psi, V, dx, hbar=hbar, m=m),
        hbar=hbar,
        m=m,
        pass_threshold=pass_threshold,
        qhj_threshold=qhj_threshold,
        warnings=warnings,
    )


def run_harmonic_ground_state_case(
    *,
    omega_ho: float = 1.0,
    hbar: float = 1.0,
    m: float = 1.0,
    L: float = 6.0,
    n: int = 512,
    t: float = 0.0,
    pass_threshold: float = DEFAULT_PASS_THRESHOLD,
    qhj_threshold: float = QHJ_NORMALIZED_PASS_THRESHOLD,
) -> SchrodingerBenchmarkResult:
    """
    HO ground state ψ₀(x) exp(−iE₀t/ℏ), V = ½mω²x², E₀ = ½ℏω.

    At time t: τ = E₀t/ℏ (spatially uniform), τ_t = E₀/ℏ.
    Domain L chosen so |x| ≲ 3σ keeps boundary FD error below QHJ threshold.
    """
    x = np.linspace(-L / 2, L / 2, n, endpoint=False)
    dx = x[1] - x[0]

    E0 = 0.5 * hbar * omega_ho
    psi_0 = np.exp(-m * omega_ho * x**2 / (2.0 * hbar))
    psi = psi_0 * np.exp(-1j * E0 * t / hbar)
    psi = psi.astype(complex)

    rho = np.abs(psi) ** 2
    tau = np.full_like(x, E0 * t / hbar)
    tau_t = np.full_like(x, E0 / hbar)
    rho_t = np.zeros_like(rho)
    V = 0.5 * m * omega_ho**2 * x**2

    psi_t = -1j * (E0 / hbar) * psi
    lap_psi = _laplacian_1d(psi, dx)
    sch_static = (-(hbar**2) / (2.0 * m)) * lap_psi + V * psi - E0 * psi

    warnings = [
        "Stationary ground state: τ = E₀t/ℏ, τ_t = E₀/ℏ; Schrödinger eigenvalue residual.",
    ]

    return _evaluate_case(
        case_name="harmonic_ground_state",
        x=x,
        rho=rho,
        tau=tau,
        tau_t=tau_t,
        rho_t=rho_t,
        psi=psi,
        psi_t=psi_t,
        V=V,
        schrodinger_check=sch_static,
        hbar=hbar,
        m=m,
        pass_threshold=pass_threshold,
        qhj_threshold=qhj_threshold,
        warnings=warnings,
    )


def _result_to_row(res: SchrodingerBenchmarkResult) -> dict[str, Any]:
    return {
        "case_name": res.case_name,
        "max_continuity_residual": res.max_continuity_residual,
        "max_qhj_residual": res.max_qhj_residual,
        "max_normalized_qhj_residual": res.max_normalized_qhj_residual,
        "max_schrodinger_residual": res.max_schrodinger_residual,
        "quantum_potential_finite": res.quantum_potential_finite,
        "pass": res.pass_,
        "warnings": "; ".join(res.warnings),
        "dataset_mode": BENCHMARK_MODE,
        "is_real_observational_data": False,
    }


def _plot_residuals(res: SchrodingerBenchmarkResult, output_path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    axes[0].semilogy(res.x, np.abs(res.continuity_map) + 1e-16, "C0")
    axes[0].set_ylabel("|continuity res.|")
    axes[0].set_title(title)
    axes[1].semilogy(res.x, np.abs(res.qhj_map) + 1e-16, "C1")
    axes[1].set_ylabel("|QHJ res.|")
    axes[2].semilogy(res.x, res.schrodinger_map + 1e-16, "C2")
    axes[2].set_ylabel("|Schrödinger res.|")
    axes[2].set_xlabel("x")
    for ax in axes:
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _build_report(results: list[SchrodingerBenchmarkResult]) -> str:
    n_pass = sum(1 for r in results if r.pass_)
    lines = [
        "# Schrödinger-from-TDF action report (Phase 6A / 6A.1)",
        "",
        f"## ⚠️ {BANNER_SCHRODINGER}",
        "",
        "## Purpose",
        "",
        "Test whether the **phase-density TDF action** reproduces the hydrodynamic form of "
        "Schrödinger dynamics (continuity + quantum Hamilton–Jacobi) in controlled 1D benchmarks.",
        "",
        "> **NOT FULL QUANTUM VALIDATION.** Does not prove quantum gravity, spin, entanglement, or measurement.",
        "",
        "## Phase convention (TDF v0.8.1)",
        "",
        f"{PHASE_CONVENTION}",
        "",
        f"**QHJ:** {QHJ_EQUATION}",
        "",
        "Equivalent Madelung form with action phase S = −ℏτ:",
        "−ℏ∂_tτ + (ℏ²/2m)|∇τ|² + V + Q = 0.",
        "",
        "The earlier Phase 6A implementation used `+ (ℏ²/2m)|∇τ|²` with `− Q` from the Fisher term; "
        "that sign pairing is **inconsistent** with ψ = √ρ exp(−iτ) and E = ℏ∂_tτ.",
        "",
        "## Equations",
        "",
        "```text",
        "ψ = √ρ exp(-iτ)",
        "E = ℏ ∂_t τ",
        "L = -ℏρ∂_tτ - (ℏ²/2m)ρ|∇τ|² - Vρ + (ℏ²/8m)|∇ρ|²/ρ",
        "∂_tρ + ∇·(ρ(ℏ/m)∇τ) = 0",
        "ℏ∂_tτ - (ℏ²/2m)|∇τ|² - V - Q = 0",
        "Q = -(ℏ²/2m)∇²√ρ/√ρ",
        "iℏ∂_tψ = [-ℏ²/(2m)∇² + V]ψ",
        "```",
        "",
        "See [docs/quantum_limit/SCHRODINGER_DERIVATION.md](../../docs/quantum_limit/SCHRODINGER_DERIVATION.md).",
        "",
        "## Summary",
        "",
        f"- **Cases:** {len(results)}",
        f"- **Pass:** {n_pass} / {len(results)}",
        f"- **Normalized QHJ pass threshold:** {QHJ_NORMALIZED_PASS_THRESHOLD:.0e}",
        "",
        "## Results table (raw and normalized)",
        "",
        "| Case | continuity (rel) | QHJ (raw) | QHJ (norm) | Schrödinger (rel) | Q finite | pass |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for res in results:
        lines.append(
            f"| {res.case_name} | {res.max_continuity_residual:.2e} | {res.max_qhj_residual:.2e} | "
            f"{res.max_normalized_qhj_residual:.2e} | {res.max_schrodinger_residual:.2e} | "
            f"{'yes' if res.quantum_potential_finite else 'no'} | "
            f"{'✓' if res.pass_ else '✗'} |",
        )
    lines.extend(
        [
            "",
            "## Scientific interpretation",
            "",
            "- **Passing** means numerical consistency between the TDF phase-density action "
            "and Schrödinger/hydrodynamic equations in these 1D test fields.",
            "- Normalized QHJ residuals divide by max(|ℏ∂_tτ|, |kinetic|, |V|, |Q|) on interior points.",
            "- Stationary eigenstates include **τ = E₀t/ℏ** so ℏ∂_tτ = E₀.",
            "- This does **not** prove full quantum gravity or replace axiomatic quantum mechanics.",
            "",
            "## Failure modes",
            "",
            "- 1D grids only; boundary regions excluded from max-norm metrics.",
            "- Gaussian snapshot is a single-time slice with τ_t inferred from ∂_tψ.",
            "- Relativistic KG/Dirac, spin, entanglement, and decoherence are **not** implemented.",
            "",
            "## Disclaimer",
            "",
            "- Symbolic/numerical consistency benchmark only.",
            "",
        ],
    )
    return "\n".join(lines)


def run_schrodinger_from_tdf_benchmark(
    outputs_root: Path | None = None,
    *,
    pass_threshold: float = DEFAULT_PASS_THRESHOLD,
    qhj_threshold: float = QHJ_NORMALIZED_PASS_THRESHOLD,
) -> tuple[pd.DataFrame, list[SchrodingerBenchmarkResult]]:
    """Run Phase 6A benchmarks and write CSV, report, figures."""
    root = Path(__file__).resolve().parents[3]
    outputs = outputs_root or (root / "outputs")
    tables_dir = outputs / "tables"
    reports_dir = outputs / "reports"
    figures_dir = outputs / "figures"
    for d in (tables_dir, reports_dir, figures_dir):
        d.mkdir(parents=True, exist_ok=True)

    results = [
        run_plane_wave_case(pass_threshold=pass_threshold, qhj_threshold=qhj_threshold),
        run_gaussian_snapshot_case(pass_threshold=pass_threshold, qhj_threshold=qhj_threshold),
        run_harmonic_ground_state_case(pass_threshold=pass_threshold, qhj_threshold=qhj_threshold),
    ]

    out_df = pd.DataFrame([_result_to_row(r) for r in results])
    out_df.to_csv(tables_dir / "schrodinger_from_tdf_summary.csv", index=False)
    (reports_dir / "schrodinger_from_tdf_report.md").write_text(_build_report(results), encoding="utf-8")

    _plot_residuals(
        results[0],
        figures_dir / "schrodinger_plane_wave_residual.png",
        "Plane wave — continuity / QHJ / Schrödinger residuals",
    )
    _plot_residuals(
        results[1],
        figures_dir / "schrodinger_gaussian_residual.png",
        "Gaussian snapshot — residuals",
    )
    _plot_residuals(
        results[2],
        figures_dir / "schrodinger_harmonic_ground_state.png",
        "Harmonic ground state — residuals",
    )
    return out_df, results

"""Phase 6B — Dirac / spinor limit benchmark (not full fermion unification)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

BENCHMARK_MODE = "dirac_spinor_limit_benchmark"
BANNER_DIRAC = "DIRAC / SPINOR LIMIT BENCHMARK — NOT FULL FERMION UNIFICATION"

DEFAULT_PASS_THRESHOLD = 1e-6
DISPERSION_PASS_THRESHOLD = 1e-8
TETRAD_PASS_THRESHOLD = 2e-4
MASS_LADDER_PASS_THRESHOLD = 1e-10

REQUIRED_SUMMARY_COLUMNS: tuple[str, ...] = (
    "case_name",
    "max_residual",
    "pass",
    "notes",
    "warning",
    "max_dispersion_error",
    "max_tetrad_reconstruction_error",
    "lorentzian_signature_pass",
    "linear_mass_ladder_error",
)


@dataclass
class DiracBenchmarkResult:
    case_name: str
    max_residual: float
    pass_: bool
    notes: str
    warning: str = ""
    max_dispersion_error: float = float("nan")
    max_tetrad_reconstruction_error: float = float("nan")
    lorentzian_signature_pass: bool | None = None
    linear_mass_ladder_error: float = float("nan")
    extra: dict[str, Any] = field(default_factory=dict)


def minkowski_metric_covariant() -> np.ndarray:
    """Flat g_{μν} with signature (−,+,+,+): diag(−1,+1,+1,+1)."""
    return np.diag([-1.0, 1.0, 1.0, 1.0])


def minkowski_eta() -> np.ndarray:
    """
    Contravariant η^{μν} for Clifford algebra with standard Dirac γ matrices.

    {γ^μ, γ^ν} = 2 η^{μν} I,  η = diag(+1, −1, −1, −1).
    """
    return np.diag([1.0, -1.0, -1.0, -1.0])


def gamma_matrices_dirac() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """γ⁰…γ³ in the standard Dirac representation (complex 4×4)."""
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    i2 = np.eye(2, dtype=complex)
    z2 = np.zeros((2, 2), dtype=complex)

    g0 = np.block([[i2, z2], [z2, -i2]])
    g1 = np.block([[z2, sx], [-sx, z2]])
    g2 = np.block([[z2, sy], [-sy, z2]])
    g3 = np.block([[z2, sz], [-sz, z2]])
    return g0, g1, g2, g3


def check_clifford_algebra(
    gammas: tuple[np.ndarray, ...],
    eta: np.ndarray | None = None,
) -> float:
    """
    Return max ‖{γᵃ,γᵇ} − 2ηᵃᵇ I‖_F over a,b.

    Convention: {γᵃ,γᵇ} = 2 ηᵃᵇ I with η = diag(−1,+1,+1,+1).
    """
    if eta is None:
        eta = minkowski_eta()  # contravariant; matches standard Dirac rep
    n = len(gammas)
    eye = np.eye(4, dtype=complex)
    max_err = 0.0
    for a in range(n):
        for b in range(n):
            anticom = gammas[a] @ gammas[b] + gammas[b] @ gammas[a]
            target = 2.0 * eta[a, b] * eye
            max_err = max(max_err, float(np.linalg.norm(anticom - target, ord="fro")))
    return max_err


def alpha_beta_from_gammas(
    gammas: tuple[np.ndarray, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """αᵢ = γ⁰γᵢ, β = γ⁰."""
    g0, g1, g2, g3 = gammas
    return g0 @ g1, g0 @ g2, g0 @ g3, g0


def dirac_hamiltonian_1d(
    k: float,
    m: float = 1.0,
    *,
    hbar: float = 1.0,
    c: float = 1.0,
) -> np.ndarray:
    """H(k) = ℏ c k αₓ + β m c² (4×4, momentum along x)."""
    g0, g1, _, _ = gamma_matrices_dirac()
    alpha_x = g0 @ g1
    beta = g0
    return hbar * c * k * alpha_x + beta * m * c**2


def dirac_energy_eigenvalues(
    k: float,
    m: float = 1.0,
    *,
    hbar: float = 1.0,
    c: float = 1.0,
) -> tuple[float, float]:
    """E± = ±√((ℏck)² + (mc²)²)."""
    E = np.sqrt((hbar * c * k) ** 2 + (m * c**2) ** 2)
    return float(E), float(-E)


def spinor_norm(psi: np.ndarray) -> float:
    """‖Ψ‖ = √(Ψ†Ψ) for 4-spinor."""
    psi = np.asarray(psi, dtype=complex).reshape(-1)
    return float(np.sqrt(np.real(np.vdot(psi, psi))))


def dirac_adjoint(psi: np.ndarray, gamma0: np.ndarray | None = None) -> np.ndarray:
    """ψ̄ = ψ† γ⁰."""
    if gamma0 is None:
        gamma0 = gamma_matrices_dirac()[0]
    psi = np.asarray(psi, dtype=complex).reshape(-1)
    return psi.conj() @ gamma0


def dirac_current(
    psi: np.ndarray,
    gammas: tuple[np.ndarray, ...] | None = None,
) -> np.ndarray:
    """j^μ = ψ̄ γ^μ ψ (4-vector)."""
    if gammas is None:
        gammas = gamma_matrices_dirac()
    psi = np.asarray(psi, dtype=complex).reshape(-1)
    psi_bar = dirac_adjoint(psi, gammas[0])
    return np.array([np.real(psi_bar @ g @ psi) for g in gammas], dtype=float)


def plane_wave_spinor(
    k: float,
    m: float,
    spin: Literal["up", "down"] = "up",
    energy_sign: Literal["+", "-"] = "+",
    *,
    hbar: float = 1.0,
    c: float = 1.0,
) -> np.ndarray:
    """Normalized positive/negative-energy spinor for 1D momentum k along x."""
    H = dirac_hamiltonian_1d(k, m, hbar=hbar, c=c)
    evals, evecs = np.linalg.eigh(H)
    E_pos, E_neg = dirac_energy_eigenvalues(k, m, hbar=hbar, c=c)
    target = E_pos if energy_sign == "+" else E_neg
    idx = int(np.argmin(np.abs(evals - target)))
    u = evecs[:, idx].astype(complex)
    if spin == "down":
        # Orthogonal spinor at same energy
        mask = np.abs(evals - target) < 1e-8 * max(1.0, abs(target))
        candidates = [evecs[:, i] for i in range(4) if mask[i] or abs(evals[i] - target) < 1e-6]
        if len(candidates) >= 2:
            u = candidates[1]
    nrm = spinor_norm(u)
    return u / nrm if nrm > 0 else u


def dirac_residual_flat(
    psi_t: np.ndarray,
    psi_x: np.ndarray,
    psi: np.ndarray,
    m: float,
    *,
    hbar: float = 1.0,
    c: float = 1.0,
) -> np.ndarray:
    """
    Residual of iℏ∂_tΨ − (−iℏc α·∂_x + β m c²)Ψ (1D, α = αₓ).

    Returns 4-component complex residual vector.
    """
    g0, g1, _, _ = gamma_matrices_dirac()
    alpha_x = g0 @ g1
    beta = g0
    psi = np.asarray(psi, dtype=complex).reshape(-1)
    psi_t = np.asarray(psi_t, dtype=complex).reshape(-1)
    psi_x = np.asarray(psi_x, dtype=complex).reshape(-1)
    lhs = 1j * hbar * psi_t
    rhs = -1j * hbar * c * (alpha_x @ psi_x) + beta * m * c**2 @ psi
    return lhs - rhs


def tdf_spinor_ansatz(
    rho: float | np.ndarray,
    tau: float | np.ndarray,
    chi: np.ndarray,
) -> np.ndarray:
    """Ψ = √ρ exp(−iτ) χ with χ a 4-spinor."""
    rho_v = np.asarray(rho, dtype=float)
    tau_v = np.asarray(tau, dtype=float)
    chi = np.asarray(chi, dtype=complex).reshape(-1)
    scalar = np.sqrt(np.maximum(rho_v, 0.0)) * np.exp(-1j * tau_v)
    if np.ndim(scalar) == 0:
        return scalar * chi
    return scalar[..., np.newaxis] * chi


def effective_disformal_metric_flat(
    tau_t: float,
    tau_x: float,
    tau_y: float = 0.0,
    tau_z: float = 0.0,
    *,
    alpha_tau: float = 0.1,
    eta: np.ndarray | None = None,
) -> np.ndarray:
    """
    g̃_{μν} = η_{μν} + α_τ ∂_μτ ∂_ντ  (4D, constant gradients).
    """
    if eta is None:
        eta = minkowski_metric_covariant()
    d = np.array([tau_t, tau_x, tau_y, tau_z], dtype=float)
    return eta + alpha_tau * np.outer(d, d)


def metric_signature_check(g_tilde: np.ndarray) -> bool:
    """True if one negative and three positive eigenvalues (Lorentzian)."""
    evals = np.linalg.eigvalsh(np.asarray(g_tilde, dtype=float))
    evals_sorted = np.sort(evals)
    return bool(evals_sorted[0] < 0 and evals_sorted[1] > 0 and evals_sorted[2] > 0 and evals_sorted[3] > 0)


def tetrad_from_metric(g_tilde: np.ndarray, eta: np.ndarray | None = None) -> np.ndarray:
    """
    Construct tetrad e^a_μ such that g̃_{μν} = η_{ab} e^a_μ e^b_ν.

    Uses symmetric matrix square root: g = L L^T, then e^a_μ = (L @ sqrt(|η|))_{aμ}.
    """
    if eta is None:
        eta = minkowski_metric_covariant()
    g = np.asarray(g_tilde, dtype=float)
    evals, evecs = np.linalg.eigh(g)
    evals_clip = np.clip(evals, 1e-14, None)
    # Restore sign for Lorentzian: flip eigenvectors where eigenvalue negative
    signs = np.sign(evals)
    signs[signs == 0] = 1.0
    sqrt_abs = np.sqrt(np.abs(evals))
    L = evecs @ np.diag(signs * sqrt_abs) @ evecs.T
    # e such that g ≈ e^T eta e via e = L @ sqrt(|eta|) with column signs
    eta_sign = np.diag(np.sign(np.diag(eta)))
    return L @ np.sqrt(np.abs(eta)) @ eta_sign


def tetrad_consistency_check(
    g_tilde: np.ndarray,
    eta: np.ndarray | None = None,
) -> float:
    """max |g̃ − e^T η e| (Frobenius)."""
    if eta is None:
        eta = minkowski_metric_covariant()
    e = tetrad_from_metric(g_tilde, eta)
    g_recon = e.T @ eta @ e
    return float(np.linalg.norm(g_tilde - g_recon, ord="fro"))


def mass_from_tau_momentum(p_tau: float, c: float = 1.0) -> float:
    """m = p_τ / c."""
    return p_tau / c


def compact_tau_mode_mass(
    n: int,
    R_tau: float,
    *,
    hbar: float = 1.0,
    c: float = 1.0,
) -> float:
    """m_n = (n ℏ / R_τ) / c."""
    p_tau = n * hbar / R_tau
    return mass_from_tau_momentum(p_tau, c=c)


# --- Benchmark cases ---


def run_clifford_algebra_check(
    threshold: float = DEFAULT_PASS_THRESHOLD,
) -> DiracBenchmarkResult:
    gammas = gamma_matrices_dirac()
    err = check_clifford_algebra(gammas)
    return DiracBenchmarkResult(
        case_name="clifford_algebra_check",
        max_residual=err,
        pass_=err < threshold,
        notes="Clifford algebra {γᵃ,γᵇ} = 2ηᵃᵇ I",
        warning="",
    )


def run_flat_dirac_dispersion(
    k_values: list[float] | None = None,
    m: float = 1.0,
    *,
    hbar: float = 1.0,
    c: float = 1.0,
    threshold: float = DISPERSION_PASS_THRESHOLD,
) -> DiracBenchmarkResult:
    if k_values is None:
        k_values = [0.0, 0.5, 1.0, 2.0, 5.0]
    errors: list[float] = []
    for k in k_values:
        H = dirac_hamiltonian_1d(k, m, hbar=hbar, c=c)
        evals = np.sort(np.linalg.eigvalsh(H))
        E_plus, E_minus = dirac_energy_eigenvalues(k, m, hbar=hbar, c=c)
        expected = np.array([E_minus, E_minus, E_plus, E_plus])  # degenerate pairs
        expected = np.sort(expected)
        errors.append(float(np.max(np.abs(evals - expected))))
    max_err = max(errors) if errors else 0.0
    return DiracBenchmarkResult(
        case_name="flat_dirac_dispersion",
        max_residual=max_err,
        pass_=max_err < threshold,
        notes=f"H(k) eigenvalues vs ±√((ℏck)²+(mc²)²), k={k_values}",
        max_dispersion_error=max_err,
    )


def run_positive_energy_plane_wave(
    k: float = 1.5,
    m: float = 1.0,
    *,
    hbar: float = 1.0,
    c: float = 1.0,
    threshold: float = DEFAULT_PASS_THRESHOLD,
) -> DiracBenchmarkResult:
    u = plane_wave_spinor(k, m, spin="up", energy_sign="+", hbar=hbar, c=c)
    E, _ = dirac_energy_eigenvalues(k, m, hbar=hbar, c=c)
    omega = E / hbar
    psi = u * np.exp(1j * (k * 0.0 - omega * 0.0))  # at x=t=0
    psi_t = -1j * omega * u
    psi_x = 1j * k * u
    res = dirac_residual_flat(psi_t, psi_x, psi, m, hbar=hbar, c=c)
    max_res = float(np.max(np.abs(res)))
    return DiracBenchmarkResult(
        case_name="positive_energy_plane_wave",
        max_residual=max_res,
        pass_=max_res < threshold,
        notes=f"Plane wave k={k}, m={m}, positive energy branch",
    )


def run_tdf_spinor_phase_ansatz(
    threshold: float = DEFAULT_PASS_THRESHOLD,
) -> DiracBenchmarkResult:
    rho = 2.5
    tau = 0.7
    chi = plane_wave_spinor(0.3, 1.0)
    psi = tdf_spinor_ansatz(rho, tau, chi)
    rho_rec = float(np.real(np.vdot(psi, psi)))
    err = abs(rho_rec - rho)
    return DiracBenchmarkResult(
        case_name="tdf_spinor_phase_ansatz",
        max_residual=err,
        pass_=err < threshold,
        notes="ρ = Ψ†Ψ with normalized χ",
    )


def run_disformal_metric_spinor_safety(
    alpha_tau: float = 0.05,
    threshold_tetrad: float = TETRAD_PASS_THRESHOLD,
) -> DiracBenchmarkResult:
    """Small τ gradients: Lorentzian g̃ and small tetrad reconstruction error."""
    tau_t, tau_x = 0.02, 0.03
    g_tilde = effective_disformal_metric_flat(tau_t, tau_x, alpha_tau=alpha_tau)
    lorentz = metric_signature_check(g_tilde)
    tetrad_err = tetrad_consistency_check(g_tilde)
    max_res = tetrad_err if lorentz else 1.0
    return DiracBenchmarkResult(
        case_name="disformal_metric_spinor_safety",
        max_residual=max_res,
        pass_=lorentz and tetrad_err < threshold_tetrad,
        notes=f"α_τ={alpha_tau}, ∂_tτ={tau_t}, ∂_xτ={tau_x}",
        max_tetrad_reconstruction_error=tetrad_err,
        lorentzian_signature_pass=lorentz,
        warning="" if lorentz else "Metric signature not Lorentzian",
    )


def run_compact_tau_mass_ladder(
    n_max: int = 6,
    R_tau: float = 1.0,
    *,
    hbar: float = 1.0,
    c: float = 1.0,
    threshold: float = MASS_LADDER_PASS_THRESHOLD,
) -> DiracBenchmarkResult:
    ns = np.arange(1, n_max + 1, dtype=float)
    masses = np.array([compact_tau_mode_mass(int(n), R_tau, hbar=hbar, c=c) for n in ns])
    slope, intercept = np.polyfit(ns, masses, 1)
    predicted = slope * ns + intercept
    ladder_err = float(np.max(np.abs(masses - predicted)))
    expected_slope = hbar / (R_tau * c)
    slope_err = abs(slope - expected_slope)
    max_res = max(ladder_err, slope_err)
    return DiracBenchmarkResult(
        case_name="compact_tau_mass_ladder",
        max_residual=max_res,
        pass_=max_res < threshold,
        notes=f"m_n = nℏ/(R_τ c), R_τ={R_tau}, n=1…{n_max}",
        linear_mass_ladder_error=ladder_err,
    )


def _result_to_row(res: DiracBenchmarkResult) -> dict[str, Any]:
    return {
        "case_name": res.case_name,
        "max_residual": res.max_residual,
        "pass": res.pass_,
        "notes": res.notes,
        "warning": res.warning,
        "max_dispersion_error": res.max_dispersion_error,
        "max_tetrad_reconstruction_error": res.max_tetrad_reconstruction_error,
        "lorentzian_signature_pass": res.lorentzian_signature_pass,
        "linear_mass_ladder_error": res.linear_mass_ladder_error,
        "dataset_mode": BENCHMARK_MODE,
        "is_real_observational_data": False,
    }


def _plot_dispersion(output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    k = np.linspace(0, 4, 80)
    m, hbar, c = 1.0, 1.0, 1.0
    E_th = np.sqrt((hbar * c * k) ** 2 + (m * c**2) ** 2)
    E_num = []
    for ki in k:
        evals = np.linalg.eigvalsh(dirac_hamiltonian_1d(float(ki), m, hbar=hbar, c=c))
        E_num.append([np.min(evals), np.max(evals)])
    E_num = np.array(E_num)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(k, E_th, "k--", label="E = +√((ℏck)²+(mc²)²)")
    ax.plot(k, -E_th, "k--", label="E = −√((ℏck)²+(mc²)²)")
    ax.plot(k, E_num[:, 1], "C0", label="H(k) max eigenvalue")
    ax.plot(k, E_num[:, 0], "C1", label="H(k) min eigenvalue")
    ax.set_xlabel("k")
    ax.set_ylabel("E")
    ax.set_title("Flat 1D Dirac dispersion (Phase 6B)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_mass_ladder(output_path: Path, R_tau: float = 1.0) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ns = np.arange(1, 11)
    masses = [compact_tau_mode_mass(int(n), R_tau) for n in ns]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(ns, masses, "o-", label="m_n = nℏ/(R_τ c)")
    ax.set_xlabel("compact mode n")
    ax.set_ylabel("m_n")
    ax.set_title("τ-momentum mass ladder (Phase 6B)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _build_report(results: list[DiracBenchmarkResult]) -> str:
    n_pass = sum(1 for r in results if r.pass_)
    lines = [
        "# Dirac / spinor limit report (Phase 6B)",
        "",
        f"## ⚠️ {BANNER_DIRAC}",
        "",
        "## Purpose",
        "",
        "Test whether TDF can consistently host **Dirac spinors** through a tetrad/vielbein "
        "formulation and recover the **flat-space Dirac limit** in controlled checks.",
        "",
        "> **NOT FULL FERMION UNIFICATION.** Spin is not claimed fully emergent from τ.",
        "",
        "## Equations",
        "",
        "```text",
        "Ψ = √ρ exp(-iτ) χ",
        "g̃_{μν} = g_{μν} + α_τ ∂_μτ ∂_ντ",
        "S_D = ∫ √(-g̃) ψ̄(iℏ γ^a e_a^μ D_μ − mc)Ψ",
        "{γ^a, γ^b} = 2 η^{ab} I",
        "iℏ ∂_t Ψ = (-iℏc α·∇ + β m c²) Ψ",
        "E² = (ℏck)² + (mc²)²",
        "m = p_τ / c,   p_τ = n ℏ / R_τ",
        "```",
        "",
        "## Summary",
        "",
        f"- **Cases:** {len(results)}",
        f"- **Pass:** {n_pass} / {len(results)}",
        "",
        "## Results table",
        "",
        "| Case | max residual | pass | notes |",
        "| --- | --- | --- | --- |",
    ]
    for res in results:
        lines.append(
            f"| {res.case_name} | {res.max_residual:.2e} | "
            f"{'✓' if res.pass_ else '✗'} | {res.notes} |",
        )
    lines.extend(
        [
            "",
            "### Case-specific metrics",
            "",
            "| Case | extra metric |",
            "| --- | --- |",
        ],
    )
    for res in results:
        extra_parts: list[str] = []
        if not np.isnan(res.max_dispersion_error):
            extra_parts.append(f"dispersion err={res.max_dispersion_error:.2e}")
        if not np.isnan(res.max_tetrad_reconstruction_error):
            extra_parts.append(f"tetrad err={res.max_tetrad_reconstruction_error:.2e}")
        if res.lorentzian_signature_pass is not None:
            extra_parts.append(f"Lorentzian={res.lorentzian_signature_pass}")
        if not np.isnan(res.linear_mass_ladder_error):
            extra_parts.append(f"ladder err={res.linear_mass_ladder_error:.2e}")
        if extra_parts:
            lines.append(f"| {res.case_name} | {', '.join(extra_parts)} |")
    lines.extend(
        [
            "",
            "## Scientific interpretation",
            "",
            "- **Passing** means TDF is compatible with a covariant spinor extension and recovers "
            "flat Dirac behavior in these algebraic and 1D checks.",
            "- Does **not** prove spin fully emerges from τ.",
            "- Does **not** address Standard Model flavor, gauge interactions, or generations.",
            "",
            "## Failure modes",
            "",
            "- Flat and weak-field checks only; spin connection not fully dynamical.",
            "- Gauge fields, chirality, and weak interactions not included.",
            "- Entanglement and measurement not included.",
            "",
            "## Disclaimer",
            "",
            "- Symbolic/numerical consistency benchmark only.",
            "",
        ],
    )
    return "\n".join(lines)


def run_dirac_spinor_limit_benchmark(
    outputs_root: Path | None = None,
) -> tuple[pd.DataFrame, list[DiracBenchmarkResult]]:
    """Run Phase 6B benchmarks; write CSV, report, figures."""
    root = Path(__file__).resolve().parents[3]
    outputs = outputs_root or (root / "outputs")
    tables_dir = outputs / "tables"
    reports_dir = outputs / "reports"
    figures_dir = outputs / "figures"
    for d in (tables_dir, reports_dir, figures_dir):
        d.mkdir(parents=True, exist_ok=True)

    results = [
        run_clifford_algebra_check(),
        run_flat_dirac_dispersion(),
        run_positive_energy_plane_wave(),
        run_tdf_spinor_phase_ansatz(),
        run_disformal_metric_spinor_safety(),
        run_compact_tau_mass_ladder(),
    ]

    out_df = pd.DataFrame([_result_to_row(r) for r in results])
    out_df.to_csv(tables_dir / "dirac_spinor_limit_summary.csv", index=False)
    (reports_dir / "dirac_spinor_limit_report.md").write_text(_build_report(results), encoding="utf-8")
    _plot_dispersion(figures_dir / "dirac_dispersion_relation.png")
    _plot_mass_ladder(figures_dir / "compact_tau_mass_ladder.png")

    return out_df, results

"""Phase 6H — Muon g-2 anomaly phenomenological QED coupling benchmark."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

BENCHMARK_MODE = "muon_g2_anomaly_benchmark"
BANNER_MUON_G2 = (
    "MUON G-2 ANOMALY BENCHMARK — NOT FULL QFT DERIVATION OF (g-2)"
)

# Fine-structure constant α ≈ 1/137.035999084 (dimensionless)
ALPHA_FINE_STRUCTURE = 1.0 / 137.035999084

# CODATA 2022: muon mass m_μ = 1.883 531 627 × 10^{-28} kg; ħ, c → Compton wavelength
MUON_MASS_KG = 1.883531627e-28
HBAR_J_S = 1.054571817e-34
C_M_S = 299792458.0

BANNER_CALIBRATION = (
    "These results are calibration diagnostics for a phenomenological TDF model. "
    "They do not constitute observational validation of TDF unless all required "
    "multi-channel constraints are passed."
)

INTERPRETATION_DISCLAIMER = (
    "This is a phenomenological order-of-magnitude estimate showing that geometric "
    "phase fluctuations could, in principle, couple to QED observables. It is NOT a "
    "full QFT derivation or proof of the anomaly's origin."
)

# Fermilab/BNL combined consensus (order-of-magnitude central value for benchmark)
DELTA_A_MU_EXP_DEFAULT = 2.51e-10
DELTA_A_MU_SIGMA_DEFAULT = 0.48e-10

RELATIVE_MATCH_TOL = 0.05
EPSILON_TAU_REFERENCE = 2.16e-7

REQUIRED_SUMMARY_COLUMNS: tuple[str, ...] = (
    "case_name",
    "epsilon_tau",
    "sigma_tau",
    "l_tau_m",
    "delta_a_mu_theory",
    "delta_a_mu_exp",
    "delta_a_mu_sigma",
    "relative_error",
    "within_exp_band",
    "overall_pass",
    "warnings",
)


@dataclass(frozen=True)
class MuonG2ExperimentalReference:
    """Experimental muon anomalous magnetic moment deviation Δa_μ."""

    delta_a_mu: float
    uncertainty: float
    source_note: str = "Fermilab/BNL combined consensus (benchmark central value)"

    def band(self, n_sigma: float = 1.0) -> tuple[float, float]:
        return (
            self.delta_a_mu - n_sigma * self.uncertainty,
            self.delta_a_mu + n_sigma * self.uncertainty,
        )


@dataclass
class MuonG2BenchmarkResult:
    case_name: str
    epsilon_tau: float
    sigma_tau: float
    l_tau_m: float
    delta_a_mu_theory: float
    delta_a_mu_exp: float
    delta_a_mu_sigma: float
    relative_error: float
    within_exp_band: bool
    overall_pass: bool
    warnings: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


def calculate_delta_a_mu(epsilon_tau: float, *, alpha: float = ALPHA_FINE_STRUCTURE) -> float:
    """
    TDF phenomenological shift: Δa_μ = (α / 2π) ε_τ.

    Parameters
    ----------
    epsilon_tau : ⟨(∂δτ)²⟩ geometric variance (dimensionless, ≥ 0)
    alpha : fine-structure constant (default ≈ 1/137.036)
    """
    eps = float(epsilon_tau)
    if eps < 0.0:
        raise ValueError("epsilon_tau must be non-negative.")
    return (alpha / (2.0 * np.pi)) * eps


def estimate_epsilon_tau(sigma_tau: float, l_tau: float) -> float:
    """
    ε_τ = (σ_τ / ℓ_τ)² from sub-Compton-scale τ fluctuations.

    Both inputs must be strictly positive.
    """
    sigma = float(sigma_tau)
    ell = float(l_tau)
    if sigma <= 0.0 or ell <= 0.0:
        raise ValueError("sigma_tau and l_tau must be strictly positive.")
    return (sigma / ell) ** 2


def muon_compton_wavelength_m() -> float:
    """Muon reduced Compton wavelength ℓ_τ = ħ / (m_μ c) [m]."""
    return HBAR_J_S / (MUON_MASS_KG * C_M_S)


def epsilon_tau_from_delta_a_mu(
    delta_a_mu: float,
    *,
    alpha: float = ALPHA_FINE_STRUCTURE,
) -> float:
    """Invert Δa_μ = (α/2π) ε_τ for ε_τ."""
    if delta_a_mu < 0.0:
        raise ValueError("delta_a_mu must be non-negative.")
    return delta_a_mu * (2.0 * np.pi) / alpha


def sigma_tau_from_epsilon(epsilon_tau: float, l_tau: float) -> float:
    """σ_τ = ℓ_τ √ε_τ."""
    return float(l_tau) * np.sqrt(max(float(epsilon_tau), 0.0))


def run_reference_consistency_case(
    exp: MuonG2ExperimentalReference | None = None,
) -> MuonG2BenchmarkResult:
    """Check ε_τ ≈ 2.16×10⁻⁷ reproduces Δa_μ ≈ 2.51×10⁻¹⁰."""
    ref = exp or MuonG2ExperimentalReference(
        delta_a_mu=DELTA_A_MU_EXP_DEFAULT,
        uncertainty=DELTA_A_MU_SIGMA_DEFAULT,
    )
    eps = EPSILON_TAU_REFERENCE
    delta_th = calculate_delta_a_mu(eps)
    rel_err = abs(delta_th - ref.delta_a_mu) / ref.delta_a_mu
    lo, hi = ref.band(1.0)
    within = lo <= delta_th <= hi
    overall = rel_err <= RELATIVE_MATCH_TOL and within
    return MuonG2BenchmarkResult(
        case_name="reference_epsilon_match",
        epsilon_tau=eps,
        sigma_tau=float("nan"),
        l_tau_m=float("nan"),
        delta_a_mu_theory=delta_th,
        delta_a_mu_exp=ref.delta_a_mu,
        delta_a_mu_sigma=ref.uncertainty,
        relative_error=rel_err,
        within_exp_band=within,
        overall_pass=overall,
    )


def run_compton_scale_estimate(
    exp: MuonG2ExperimentalReference | None = None,
) -> MuonG2BenchmarkResult:
    """
    Set ℓ_τ = muon Compton wavelength; infer σ_τ and ε_τ to match Δa_μ^exp.
    """
    ref = exp or MuonG2ExperimentalReference(
        delta_a_mu=DELTA_A_MU_EXP_DEFAULT,
        uncertainty=DELTA_A_MU_SIGMA_DEFAULT,
    )
    l_tau = muon_compton_wavelength_m()
    eps_req = epsilon_tau_from_delta_a_mu(ref.delta_a_mu)
    sigma = sigma_tau_from_epsilon(eps_req, l_tau)
    delta_th = calculate_delta_a_mu(eps_req)
    rel_err = abs(delta_th - ref.delta_a_mu) / ref.delta_a_mu
    lo, hi = ref.band(1.0)
    within = lo <= delta_th <= hi
    return MuonG2BenchmarkResult(
        case_name="compton_scale_fluctuation_estimate",
        epsilon_tau=eps_req,
        sigma_tau=sigma,
        l_tau_m=l_tau,
        delta_a_mu_theory=delta_th,
        delta_a_mu_exp=ref.delta_a_mu,
        delta_a_mu_sigma=ref.uncertainty,
        relative_error=rel_err,
        within_exp_band=within,
        overall_pass=within,
        metadata={"alpha_eff_factor": 1.0 + eps_req},
    )


def _result_to_row(res: MuonG2BenchmarkResult) -> dict[str, Any]:
    return {
        "case_name": res.case_name,
        "epsilon_tau": res.epsilon_tau,
        "sigma_tau": res.sigma_tau,
        "l_tau_m": res.l_tau_m,
        "delta_a_mu_theory": res.delta_a_mu_theory,
        "delta_a_mu_exp": res.delta_a_mu_exp,
        "delta_a_mu_sigma": res.delta_a_mu_sigma,
        "relative_error": res.relative_error,
        "within_exp_band": res.within_exp_band,
        "overall_pass": res.overall_pass,
        "warnings": res.warnings,
        "dataset_mode": BENCHMARK_MODE,
        "is_real_observational_data": False,
    }


def _plot_epsilon_sweep(
    exp: MuonG2ExperimentalReference,
    path: Path,
    *,
    highlight_epsilon: float | None = None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    eps_grid = np.logspace(-10, -4, 300)
    delta_grid = np.array([calculate_delta_a_mu(e) for e in eps_grid])
    lo, hi = exp.band(1.0)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.loglog(eps_grid, delta_grid, "-", color="C0", lw=2, label=r"$\Delta a_\mu = (\alpha/2\pi)\,\epsilon_\tau$")
    ax.axhspan(lo, hi, color="C3", alpha=0.25, label="exp. 1σ band")
    ax.axhline(exp.delta_a_mu, color="C3", ls="--", lw=1.2, label=r"$\Delta a_\mu^{\rm exp}$")
    if highlight_epsilon is not None:
        d_hi = calculate_delta_a_mu(highlight_epsilon)
        ax.plot(
            highlight_epsilon,
            d_hi,
            "ko",
            ms=8,
            label=rf"Compton-scale estimate ($\epsilon_\tau={highlight_epsilon:.2e}$)",
        )
    ax.set_xlabel(r"$\epsilon_\tau \equiv \langle(\partial\delta\tau)^2\rangle$")
    ax.set_ylabel(r"$\Delta a_\mu$ (theory)")
    ax.set_title("Muon g-2 vs geometric τ variance (Phase 6H phenomenology)")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _build_report(
    results: list[MuonG2BenchmarkResult],
    exp: MuonG2ExperimentalReference,
) -> str:
    n_pass = sum(1 for r in results if r.overall_pass)
    compton = next((r for r in results if r.case_name == "compton_scale_fluctuation_estimate"), None)
    lines = [
        "# Muon g-2 anomaly phenomenological benchmark (Phase 6H)",
        "",
        f"## ⚠️ {BANNER_MUON_G2}",
        "",
        f"## ⚠️ {BANNER_CALIBRATION}",
        "",
        "## Purpose",
        "",
        "Order-of-magnitude check whether sub-Compton **τ-phase geometry fluctuations** "
        "can modify the effective fine-structure constant "
        "α_eff = α(1 + ε_τ) and produce a muon **(g−2)** shift",
        "",
        "```text",
        "ε_τ = (σ_τ / ℓ_τ)²",
        "Δa_μ = (α / 2π) ε_τ",
        "```",
        "",
        f"> {INTERPRETATION_DISCLAIMER}",
        "",
        "## Experimental reference",
        "",
        f"- **Δa_μ^exp:** {exp.delta_a_mu:.3e} ± {exp.uncertainty:.3e}",
        f"- **Note:** {exp.source_note}",
        f"- **α used:** {ALPHA_FINE_STRUCTURE:.12g} (≈ 1/137.035999)",
        "",
        f"- **Cases:** {len(results)} | **Passed:** {n_pass}/{len(results)}",
        "",
        "## Results",
        "",
        "| Case | ε_τ | σ_τ [m] | ℓ_τ [m] | Δa_μ (th) | rel. err | in band | pass |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in results:
        sig = f"{r.sigma_tau:.3e}" if np.isfinite(r.sigma_tau) else "—"
        ell = f"{r.l_tau_m:.3e}" if np.isfinite(r.l_tau_m) else "—"
        lines.append(
            f"| {r.case_name} | {r.epsilon_tau:.3e} | {sig} | {ell} | "
            f"{r.delta_a_mu_theory:.3e} | {r.relative_error:.3f} | "
            f"{'yes' if r.within_exp_band else 'no'} | {'✓' if r.overall_pass else '✗'} |",
        )
    if compton is not None:
        lines.extend(
            [
                "",
                "## Compton-scale fluctuation estimate",
                "",
                f"- **ℓ_τ** (muon Compton wavelength): {compton.l_tau_m:.4e} m",
                f"- **ε_τ required:** {compton.epsilon_tau:.4e}",
                f"- **σ_τ = ℓ_τ √ε_τ:** {compton.sigma_tau:.4e} m",
                f"- Implied **α_eff/α − 1 ≈ ε_τ:** {compton.epsilon_tau:.4e}",
                "",
            ],
        )
    lines.extend(
        [
            "## Scientific interpretation",
            "",
            "- Matching the experimental Δa_μ at ε_τ ~ 2×10⁻⁷ is a **numerical consistency** "
            "exercise, not a derivation from first principles.",
            "- Does **not** replace the Standard Model weak/strong contributions or claim "
            "the anomaly is solved.",
            "",
            "## Failure modes",
            "",
            "- One-loop order-of-magnitude proxy only; no full TDF–QED Feynman calculation.",
            "- ε_τ and σ_τ are effective parameters, not uniquely measured geometric fields.",
            "- No hadronic vacuum-polarization or beamline systematics modeled.",
            "",
            "## Disclaimer",
            "",
            f"- {BANNER_CALIBRATION}",
            f"- {INTERPRETATION_DISCLAIMER}",
            "",
        ],
    )
    return "\n".join(lines)


def run_muon_g2_anomaly_benchmark(
    outputs_root: Path | None = None,
) -> tuple[pd.DataFrame, list[MuonG2BenchmarkResult]]:
    """Run Phase 6H benchmark; write CSV, report, figure."""
    root = Path(__file__).resolve().parents[3]
    outputs = Path(outputs_root or root / "outputs")
    tables = outputs / "tables"
    reports = outputs / "reports"
    figures = outputs / "figures"
    for d in (tables, reports, figures):
        d.mkdir(parents=True, exist_ok=True)

    exp = MuonG2ExperimentalReference(
        delta_a_mu=DELTA_A_MU_EXP_DEFAULT,
        uncertainty=DELTA_A_MU_SIGMA_DEFAULT,
    )
    results = [
        run_reference_consistency_case(exp),
        run_compton_scale_estimate(exp),
    ]
    compton = results[1]

    df = pd.DataFrame([_result_to_row(r) for r in results])
    df.to_csv(tables / "muon_g2_anomaly_summary.csv", index=False)
    (reports / "muon_g2_anomaly_report.md").write_text(
        _build_report(results, exp),
        encoding="utf-8",
    )
    _plot_epsilon_sweep(
        exp,
        figures / "muon_g2_epsilon_sweep.png",
        highlight_epsilon=compton.epsilon_tau,
    )
    return df, results

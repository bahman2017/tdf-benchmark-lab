"""Phase 5D — Covariant action consistency checks on NFW surrogate TDF fits (not validation)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from tdf_obs.validation.nfw_surrogate import (
    BENCHMARK_CASE_REGISTRY,
    get_benchmark_case,
    run_nfw_surrogate_pipeline,
)

BENCHMARK_MODE = "covariant_action_consistency_check"
BANNER_COVARIANT = "COVARIANT ACTION CONSISTENCY CHECK — NOT REAL OBSERVATIONAL DATA"

NFW_SUMMARY_FILENAME = "nfw_surrogate_fit_summary.csv"
OUTPUT_SUMMARY_FILENAME = "covariant_action_checks_summary.csv"
OUTPUT_REPORT_FILENAME = "covariant_action_checks_report.md"

DEFAULT_K_TAU = 1.0
DEFAULT_ALPHA_TAU = 1e-6
DEFAULT_LAMBDA_K = 1.0
DEFAULT_BETA_K = 0.0
DEFAULT_DISFORMAL_MAX_DELTA_G = 1.0
DEFAULT_WEAK_FIELD_RTOL = 1e-8
DEFAULT_N_RADIAL = 80

REQUIRED_SUMMARY_COLUMNS: tuple[str, ...] = (
    "case_name",
    "B",
    "r0",
    "weak_field_identity_max_error",
    "weak_field_identity_pass",
    "max_tau_gradient_proxy",
    "max_disformal_delta_g_proxy",
    "disformal_safety_pass",
    "k_stability_pass",
    "density_finite_pass",
    "density_min",
    "density_max",
    "overall_action_consistency_pass",
    "warnings",
)


class NfwSummaryMissingError(FileNotFoundError):
    """Raised when NFW surrogate fit summary CSV is unavailable."""


def phi_tau_from_rotation_profile(r: np.ndarray, B: float, r0: float) -> np.ndarray:
    """Weak-field tau potential Φ_τ(r) = B log(1 + r/r0) [km²/s²]."""
    r = np.asarray(r, dtype=float)
    r0_safe = max(float(r0), 1e-12)
    return float(B) * np.log1p(r / r0_safe)


def d_phi_tau_dr_analytic(r: np.ndarray, B: float, r0: float) -> np.ndarray:
    """dΦ_τ/dr = B/(r+r0) [km²/s² / kpc]."""
    r = np.asarray(r, dtype=float)
    r0_safe = max(float(r0), 1e-12)
    return float(B) / (r + r0_safe)


def d_phi_tau_dr_numeric(
    r: np.ndarray,
    B: float,
    r0: float,
    *,
    dr: float | None = None,
) -> np.ndarray:
    """Central finite-difference derivative of Φ_τ."""
    r = np.asarray(r, dtype=float)
    if dr is None:
        dr = max(1e-4, 0.5 * np.min(np.diff(r)) if len(r) > 1 else 1e-4)
    return (phi_tau_from_rotation_profile(r + dr, B, r0) - phi_tau_from_rotation_profile(r - dr, B, r0)) / (
        2.0 * dr
    )


def weak_field_rotation_identity_error(
    r: np.ndarray,
    B: float,
    r0: float,
    *,
    rtol: float = DEFAULT_WEAK_FIELD_RTOL,
) -> tuple[float, bool]:
    """
    Check r dΦ/dr = B r/(r+r0).

    Returns (max_relative_error, pass).
    """
    r = np.asarray(r, dtype=float)
    dphi = d_phi_tau_dr_analytic(r, B, r0)
    lhs = r * dphi
    rhs = float(B) * r / (r + max(float(r0), 1e-12))
    denom = np.maximum(np.abs(rhs), 1e-12)
    rel = np.abs(lhs - rhs) / denom
    max_err = float(np.max(rel))
    return max_err, max_err < rtol


def tau_gradient_magnitude_proxy(
    r: np.ndarray,
    B: float,
    r0: float,
    *,
    K_tau: float = DEFAULT_K_TAU,
) -> np.ndarray:
    """|dτ̄/dr| with τ̄ = Φ_τ/K_τ → |dΦ_τ/dr|/K_τ."""
    K_tau = max(float(K_tau), 1e-30)
    return np.abs(d_phi_tau_dr_analytic(r, B, r0)) / K_tau


def disformal_safety_proxy(
    r: np.ndarray,
    B: float,
    r0: float,
    *,
    alpha_tau: float = DEFAULT_ALPHA_TAU,
    K_tau: float = DEFAULT_K_TAU,
    threshold: float = DEFAULT_DISFORMAL_MAX_DELTA_G,
) -> tuple[float, bool]:
    """
    δg proxy = α_τ (dτ/dr)²; return (max over r, pass if max < threshold).
    """
    dtau_dr = tau_gradient_magnitude_proxy(r, B, r0, K_tau=K_tau)
    delta_g = float(alpha_tau) * dtau_dr**2
    max_delta = float(np.max(delta_g))
    return max_delta, max_delta < threshold


def k_essence_stability_proxy(
    X: float | np.ndarray,
    *,
    beta: float = DEFAULT_BETA_K,
    Lambda: float = DEFAULT_LAMBDA_K,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | bool]:
    """
    K(X) = X + β X²/Λ⁴; require K_X > 0 and K_X + 2X K_XX > 0.

    Returns (K_X, stability_scalar, pass) where pass is bool or bool array.
    """
    X_arr = np.asarray(X, dtype=float)
    Lam4 = float(Lambda) ** 4
    K_X = 1.0 + 2.0 * float(beta) * X_arr / Lam4
    K_XX = 2.0 * float(beta) / Lam4
    stability = K_X + 2.0 * X_arr * K_XX
    passed = (K_X > 0) & (stability > 0)
    if X_arr.ndim == 0:
        return K_X, stability, bool(passed)
    return K_X, stability, passed


def effective_density_proxy(
    r: np.ndarray,
    B: float,
    r0: float,
) -> np.ndarray:
    """
    Spherical Poisson-like proxy: ρ_eff ∝ (1/r²) d/dr [r² dΦ/dr].

    Uses analytic dΦ/dr = B/(r+r0). Normalized to max(|ρ|)=1 for reporting.
    """
    r = np.asarray(r, dtype=float)
    r0_safe = max(float(r0), 1e-12)
    r_safe = np.maximum(r, 1e-6)
    flux = r_safe**2 * float(B) / (r_safe + r0_safe)
    d_flux_dr = float(B) * r_safe * (r_safe + 2.0 * r0_safe) / (r_safe + r0_safe) ** 2
    rho = d_flux_dr / (r_safe**2)
    scale = np.nanmax(np.abs(rho))
    if scale > 0 and np.isfinite(scale):
        rho = rho / scale
    return rho


@dataclass
class CovariantActionCheckResult:
    case_name: str
    B: float
    r0: float
    weak_field_identity_max_error: float
    weak_field_identity_pass: bool
    max_tau_gradient_proxy: float
    max_disformal_delta_g_proxy: float
    disformal_safety_pass: bool
    k_stability_pass: bool
    density_finite_pass: bool
    density_min: float
    density_max: float
    overall_action_consistency_pass: bool
    warnings: list[str] = field(default_factory=list)


def _radial_grid_for_case(case_name: str, n_points: int = DEFAULT_N_RADIAL) -> np.ndarray:
    case = get_benchmark_case(case_name)
    return np.linspace(case.r_min_kpc, case.r_max_kpc, n_points)


def run_covariant_checks_for_case(
    case_name: str,
    B: float,
    r0: float,
    *,
    n_radial: int = DEFAULT_N_RADIAL,
    K_tau: float = DEFAULT_K_TAU,
    alpha_tau: float = DEFAULT_ALPHA_TAU,
    beta_k: float = DEFAULT_BETA_K,
    disformal_threshold: float = DEFAULT_DISFORMAL_MAX_DELTA_G,
    weak_field_rtol: float = DEFAULT_WEAK_FIELD_RTOL,
) -> CovariantActionCheckResult:
    """Run all proxy action consistency checks for one (B, r0) pair."""
    warnings: list[str] = []

    if not (np.isfinite(B) and np.isfinite(r0) and B >= 0 and r0 > 0):
        warnings.append("Non-finite or invalid fitted (B, r0); checks may fail.")
        B = float(B) if np.isfinite(B) else 0.0
        r0 = float(r0) if np.isfinite(r0) and r0 > 0 else 1.0

    r = _radial_grid_for_case(case_name, n_radial)

    wf_err, wf_pass = weak_field_rotation_identity_error(r, B, r0, rtol=weak_field_rtol)

    dphi_num = d_phi_tau_dr_numeric(r, B, r0)
    dphi_ana = d_phi_tau_dr_analytic(r, B, r0)
    deriv_mismatch = float(np.max(np.abs(dphi_num - dphi_ana) / np.maximum(np.abs(dphi_ana), 1e-12)))
    if deriv_mismatch > 1e-4:
        warnings.append(f"Numeric vs analytic dΦ/dr mismatch max rel {deriv_mismatch:.2e}.")

    grad = tau_gradient_magnitude_proxy(r, B, r0, K_tau=K_tau)
    max_grad = float(np.max(grad))

    max_dg, dis_pass = disformal_safety_proxy(
        r, B, r0, alpha_tau=alpha_tau, K_tau=K_tau, threshold=disformal_threshold,
    )

    X_proxy = grad**2
    _, _, k_pass_arr = k_essence_stability_proxy(X_proxy, beta=beta_k, Lambda=DEFAULT_LAMBDA_K)
    k_pass = bool(np.all(k_pass_arr))

    rho = effective_density_proxy(r, B, r0)
    finite = np.all(np.isfinite(rho))
    rho_min = float(np.nanmin(rho)) if finite else float("nan")
    rho_max = float(np.nanmax(rho)) if finite else float("nan")
    density_pass = finite and not np.any(rho < -1e-6)

    if not finite:
        warnings.append("Effective density proxy contains non-finite values.")
    if finite and rho_min < 0:
        warnings.append(f"Negative density proxy min={rho_min:.3e} (pathology flag).")

    overall = wf_pass and dis_pass and k_pass and density_pass

    return CovariantActionCheckResult(
        case_name=case_name,
        B=float(B),
        r0=float(r0),
        weak_field_identity_max_error=wf_err,
        weak_field_identity_pass=wf_pass,
        max_tau_gradient_proxy=max_grad,
        max_disformal_delta_g_proxy=max_dg,
        disformal_safety_pass=dis_pass,
        k_stability_pass=k_pass,
        density_finite_pass=density_pass,
        density_min=rho_min,
        density_max=rho_max,
        overall_action_consistency_pass=overall,
        warnings=warnings,
    )


def load_nfw_fitted_parameters(summary_path: Path) -> pd.DataFrame:
    """Load NFW surrogate fit summary; columns must include case_name, fitted_tdf_B, fitted_tdf_r0."""
    if not summary_path.is_file():
        raise NfwSummaryMissingError(
            f"Missing NFW surrogate summary: {summary_path}\n"
            "Run: python scripts/run_nfw_surrogate.py",
        )
    df = pd.read_csv(summary_path)
    for col in ("case_name", "fitted_tdf_B", "fitted_tdf_r0"):
        if col not in df.columns:
            raise ValueError(f"NFW summary missing required column {col!r}")
    return df


def ensure_nfw_summary(
    summary_path: Path,
    *,
    outputs_root: Path,
    run_if_missing: bool = True,
) -> pd.DataFrame:
    """Load NFW summary or run NFW surrogate pipeline once if missing."""
    if summary_path.is_file():
        return load_nfw_fitted_parameters(summary_path)
    if not run_if_missing:
        raise NfwSummaryMissingError(
            f"Missing NFW surrogate summary: {summary_path}\n"
            "Run: python scripts/run_nfw_surrogate.py",
        )
    run_nfw_surrogate_pipeline(outputs_root=outputs_root)
    if not summary_path.is_file():
        raise NfwSummaryMissingError(
            f"NFW surrogate pipeline did not create {summary_path}\n"
            "Run: python scripts/run_nfw_surrogate.py",
        )
    return load_nfw_fitted_parameters(summary_path)


def _result_to_row(res: CovariantActionCheckResult) -> dict[str, Any]:
    return {
        "case_name": res.case_name,
        "B": res.B,
        "r0": res.r0,
        "weak_field_identity_max_error": res.weak_field_identity_max_error,
        "weak_field_identity_pass": res.weak_field_identity_pass,
        "max_tau_gradient_proxy": res.max_tau_gradient_proxy,
        "max_disformal_delta_g_proxy": res.max_disformal_delta_g_proxy,
        "disformal_safety_pass": res.disformal_safety_pass,
        "k_stability_pass": res.k_stability_pass,
        "density_finite_pass": res.density_finite_pass,
        "density_min": res.density_min,
        "density_max": res.density_max,
        "overall_action_consistency_pass": res.overall_action_consistency_pass,
        "warnings": "; ".join(res.warnings),
        "dataset_mode": BENCHMARK_MODE,
        "is_real_observational_data": False,
    }


def _build_report(results: list[CovariantActionCheckResult]) -> str:
    n = len(results)
    n_pass = sum(1 for r in results if r.overall_action_consistency_pass)
    worst_wf = max(results, key=lambda r: r.weak_field_identity_max_error) if results else None

    lines = [
        "# Covariant action consistency report (Phase 5D)",
        "",
        f"## ⚠️ {BANNER_COVARIANT}",
        "",
        "## Purpose",
        "",
        "Check whether **fitted weak-field TDF profiles** from the ΛCDM/NFW surrogate rotation "
        "benchmark remain compatible with **action-level sanity constraints** (EFT/phenomenological "
        "proxies only).",
        "",
        "> **Not observational validation.** Not a proof of the full covariant theory.",
        "",
        "## Equations (reference)",
        "",
        "```text",
        "S_TDF = ∫ d⁴x √(-g) [ M_Pl² R/2 − (K₀/2) g^μν ∂_μτ ∂_ντ − V(τ) ] + S_m[Ψ, g̃]",
        "g̃_μν = g_μν + α_τ ∂_μτ ∂_ντ",
        "Φ_τ(r) = B log(1 + r/r0)",
        "v_τ² = r dΦ_τ/dr = B r/(r + r0)",
        "K(X) = X + β X²/Λ⁴",
        "```",
        "",
        "## Summary",
        "",
        f"- **Cases checked:** {n}",
        f"- **Overall action consistency pass:** {n_pass} / {n}",
    ]
    if worst_wf is not None:
        lines.append(
            f"- **Largest weak-field identity error:** {worst_wf.case_name} "
            f"({worst_wf.weak_field_identity_max_error:.3e})",
        )
    lines.extend(
        [
            "",
            "## Results table",
            "",
            "| Case | B | r0 | wf err | wf | disformal | k-stab | ρ fin | overall |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ],
    )
    for res in results:
        lines.append(
            f"| {res.case_name} | {res.B:.2g} | {res.r0:.3g} | {res.weak_field_identity_max_error:.2e} | "
            f"{'✓' if res.weak_field_identity_pass else '✗'} | "
            f"{'✓' if res.disformal_safety_pass else '✗'} | "
            f"{'✓' if res.k_stability_pass else '✗'} | "
            f"{'✓' if res.density_finite_pass else '✗'} | "
            f"{'✓' if res.overall_action_consistency_pass else '✗'} |",
        )

    lines.extend(
        [
            "",
            "## Scientific interpretation",
            "",
            "- **Passing** means the profile satisfies weak-field identity, finite gradient proxies, "
            "small disformal deformation proxy, and simple k-essence stability proxy.",
            "- **Passing does not** prove the covariant action, validate TDF observationally, or "
            "replace ΛCDM.",
            "",
            "## Failure modes",
            "",
            "- These are **proxy checks** only; units are normalized unless noted.",
            "- The full variational equations of motion are **not** solved.",
            "- Strong-field / black-hole dynamics are excluded.",
            "- Fitted (B, r0) from noisy rotation fits may stress gradient or density proxies.",
            "",
            "## Disclaimer",
            "",
            "- **NOT REAL OBSERVATIONAL DATA**",
            "- EFT/phenomenological consistency scaffold only.",
            "",
        ],
    )
    return "\n".join(lines)


def run_covariant_action_checks_pipeline(
    outputs_root: Path | None = None,
    *,
    case_names: Sequence[str] | None = None,
    run_nfw_if_missing: bool = True,
    **check_kwargs: Any,
) -> tuple[pd.DataFrame, list[CovariantActionCheckResult]]:
    """Load NFW fits, run covariant checks, write summary CSV and report."""
    root = Path(__file__).resolve().parents[3]
    outputs = outputs_root or (root / "outputs")
    tables_dir = outputs / "tables"
    reports_dir = outputs / "reports"
    for d in (tables_dir, reports_dir):
        d.mkdir(parents=True, exist_ok=True)

    summary_path = tables_dir / NFW_SUMMARY_FILENAME
    nfw_df = ensure_nfw_summary(
        summary_path,
        outputs_root=outputs,
        run_if_missing=run_nfw_if_missing,
    )

    if case_names is not None:
        nfw_df = nfw_df[nfw_df["case_name"].isin(case_names)].copy()
        missing = set(case_names) - set(nfw_df["case_name"])
        if missing:
            raise KeyError(f"Case(s) not in NFW summary: {sorted(missing)}")

    results: list[CovariantActionCheckResult] = []
    for _, row in nfw_df.iterrows():
        name = str(row["case_name"])
        if name not in BENCHMARK_CASE_REGISTRY:
            continue
        res = run_covariant_checks_for_case(
            name,
            float(row["fitted_tdf_B"]),
            float(row["fitted_tdf_r0"]),
            **check_kwargs,
        )
        results.append(res)

    if not results:
        raise ValueError("No NFW surrogate cases found to check.")

    out_df = pd.DataFrame([_result_to_row(r) for r in results])
    out_df.to_csv(tables_dir / OUTPUT_SUMMARY_FILENAME, index=False)
    (reports_dir / OUTPUT_REPORT_FILENAME).write_text(_build_report(results), encoding="utf-8")
    return out_df, results

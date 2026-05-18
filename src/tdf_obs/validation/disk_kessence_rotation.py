"""Phase 7B — Axisymmetric disk K-essence rotation benchmark (thin-disk proxy)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy.optimize import brentq

BENCHMARK_MODE = "disk_kessence_rotation_benchmark"
BANNER_DISK_KESSENCE = (
    "DISK K-ESSENCE ROTATION BENCHMARK — NOT REAL SPARC VALIDATION"
)
BANNER_CALIBRATION = (
    "These results are calibration diagnostics for a phenomenological TDF model. "
    "They do not constitute observational validation of TDF unless all required "
    "multi-channel constraints are passed."
)

MuModel = Literal["canonical", "deep_mond", "simple", "standard"]
DiskModel = Literal[
    "exponential_disk",
    "lsb_extended_disk",
    "compact_high_surface_brightness",
]
CouplingType = Literal["pure_disformal", "conformal_trace"]
ExpectedFailureMode = Literal["none", "pure_disformal", "weak_coupling", "excessive_coupling"]

OUTER_FLATNESS_MIN = 0.80
FLATNESS_IMPROVEMENT_MIN = 0.04
SOURCE_ZERO_TOL = 1e-10
GRADIENT_EXCESS_THRESHOLD = 25.0
TAU_V2_FRACTION_MIN = 0.10
TAU_ACCEL_STRENGTH_MIN = 1e-4
BTF_REL_ERROR_MAX = 5.0  # loose log-scale proxy only; not a pass gate

REQUIRED_SUMMARY_COLUMNS: tuple[str, ...] = (
    "case_name",
    "disk_model",
    "mu_model",
    "coupling_type",
    "source_nonzero",
    "source_norm",
    "tau_gradient_outer_slope",
    "baryon_outer_flatness_score",
    "total_outer_flatness_score",
    "flatness_improvement",
    "v_flat_proxy",
    "M_b_proxy",
    "btf_proxy_error",
    "gradient_safety_pass",
    "expected_status",
    "failure_detected",
    "failure_reason",
    "overall_pass",
    "warnings",
)


@dataclass
class DiskKessenceCaseResult:
    case_name: str
    disk_model: str
    mu_model: str
    coupling_type: str
    source_nonzero: bool
    source_norm: float
    tau_gradient_outer_slope: float
    baryon_outer_flatness_score: float
    total_outer_flatness_score: float
    flatness_improvement: float
    v_flat_proxy: float
    m_b_proxy: float
    btf_proxy_error: float
    gradient_safety_pass: bool
    expected_status: Literal["pass", "fail"]
    failure_detected: bool = False
    failure_reason: str = ""
    overall_pass: bool = False
    warnings: str = ""
    r: np.ndarray | None = None
    sigma_b: np.ndarray | None = None
    sigma_prime: np.ndarray | None = None
    v2_baryon: np.ndarray | None = None
    v2_total: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def exponential_disk_surface_density(
    r: np.ndarray,
    sigma0: float,
    rd: float,
) -> np.ndarray:
    """Σ_b(R) = Σ₀ exp(−R/R_d) [mass/length²]."""
    r = np.asarray(r, dtype=float)
    rd = max(float(rd), 1e-6)
    return float(sigma0) * np.exp(-r / rd)


def compact_disk_surface_density(
    r: np.ndarray,
    sigma0: float,
    rd: float,
    rcore: float,
) -> np.ndarray:
    """Compact core + exponential tail (thin-disk proxy)."""
    r = np.asarray(r, dtype=float)
    rd = max(float(rd), 1e-6)
    rc = max(float(rcore), 1e-6)
    core = 1.0 / (1.0 + (r / rc) ** 2) ** 2
    tail = np.exp(-r / rd)
    return float(sigma0) * core * (0.35 + 0.65 * tail)


def lsb_disk_surface_density(
    r: np.ndarray,
    sigma0: float,
    rd: float,
) -> np.ndarray:
    """Low surface-brightness extended disk: lower Σ₀, larger R_d scale."""
    r = np.asarray(r, dtype=float)
    rd = max(float(rd), 1e-6)
    return float(sigma0) * np.exp(-r / (1.8 * rd))


def disk_source_from_surface_density(
    sigma_b: np.ndarray,
    beta_over_m: float,
) -> np.ndarray:
    """
    Thin-disk conformal trace source proxy: S_disk(R) = (β/M) Σ_b(R).

    Sign convention matches Phase 7A: positive β/M and Σ_b > 0 give
    outward-growing I(R) in the cylindrical integrated equation.
    """
    return float(beta_over_m) * np.asarray(sigma_b, dtype=float)


def integrated_disk_source(r: np.ndarray, s_disk: np.ndarray) -> np.ndarray:
    """I(R) = ∫₀^R S_disk(R') R' dR'."""
    r = np.asarray(r, dtype=float)
    s = np.asarray(s_disk, dtype=float)
    integrand = s * r
    return np.array([np.trapz(integrand[: i + 1], r[: i + 1]) for i in range(len(r))])


def mu_interpolation(y: np.ndarray | float, model: MuModel) -> np.ndarray:
    """μ(y) for cylindrical K-essence disk equation."""
    y_arr = np.maximum(np.asarray(y, dtype=float), 0.0)
    if model == "canonical":
        return np.ones_like(y_arr)
    if model == "deep_mond":
        return y_arr
    if model == "simple":
        return y_arr / (1.0 + y_arr)
    if model == "standard":
        return y_arr / np.sqrt(1.0 + y_arr**2)
    raise ValueError(f"Unknown mu model: {model}")


def _disk_lhs(r: float, sigma_prime: float, a0: float, mu_model: MuModel) -> float:
    y = abs(sigma_prime) / max(a0, 1e-30)
    mu = float(mu_interpolation(y, mu_model))
    return r * mu * sigma_prime


def solve_disk_sigma_prime(
    r: np.ndarray,
    i_r: np.ndarray,
    a0: float,
    mu_model: MuModel,
) -> np.ndarray:
    """
    Solve R μ(|σ'|/a₀) σ' = I(R) pointwise.

    Analytic limits:
    - canonical (μ=1): σ' = I/R
    - deep_mond (μ=y, σ'>0): σ' = √(a₀ I/R)
    """
    r = np.asarray(r, dtype=float)
    i_r = np.asarray(i_r, dtype=float)
    a0 = max(float(a0), 1e-30)
    out = np.zeros_like(r)

    for k, (rk, ik) in enumerate(zip(r, i_r)):
        if rk <= 0.0 or ik <= 0.0:
            out[k] = 0.0
            continue
        if mu_model == "canonical":
            out[k] = ik / rk
            continue
        if mu_model == "deep_mond":
            out[k] = np.sqrt(max(ik * a0 / rk, 0.0))
            continue

        def func(sp: float) -> float:
            return _disk_lhs(rk, sp, a0, mu_model) - ik

        hi = max(np.sqrt(ik * a0 / rk), ik / rk, 1e-12) * 100.0
        try:
            out[k] = brentq(func, 1e-14, hi, maxiter=200)
        except ValueError:
            out[k] = np.sqrt(max(ik * a0 / rk, 0.0))
    return out


def freeman_disk_baryon_rotation_proxy(
    r: np.ndarray,
    sigma_b: np.ndarray,
    *,
    g_newton: float = 1.0,
) -> np.ndarray:
    """
    Disk baryon rotation proxy (not full Freeman Bessel inversion).

    v_b²(R) = G M_b(<R) / max(R, ε) with M_b(<R) = 2π ∫₀^R Σ(R') R' dR'.
    """
    r = np.asarray(r, dtype=float)
    sigma = np.asarray(sigma_b, dtype=float)
    m_enc = np.array(
        [2.0 * np.pi * np.trapz(sigma[: i + 1] * r[: i + 1], r[: i + 1]) for i in range(len(r))],
    )
    return g_newton * m_enc / np.maximum(r, 1e-12)


def tau_acceleration_proxy(
    sigma_prime: np.ndarray,
    coupling: float,
) -> np.ndarray:
    """a_τ = coupling · σ'(R)."""
    return float(coupling) * np.asarray(sigma_prime, dtype=float)


def total_rotation_curve(
    r: np.ndarray,
    v_b2: np.ndarray,
    a_tau: np.ndarray,
) -> np.ndarray:
    """v_total² = v_b² + R · a_τ (cylindrical disk proxy)."""
    r = np.asarray(r, dtype=float)
    return np.maximum(np.asarray(v_b2, dtype=float) + r * np.asarray(a_tau, dtype=float), 0.0)


def rotation_flatness_score(
    r: np.ndarray,
    v2_total: np.ndarray,
    outer_fraction: float = 0.5,
) -> float:
    """1 − fractional std of v² in outer radial band (higher = flatter)."""
    r = np.asarray(r, dtype=float)
    v2 = np.asarray(v2_total, dtype=float)
    mask = r >= outer_fraction * r[-1]
    outer = v2[mask]
    if len(outer) < 2:
        return 0.0
    mean = float(np.mean(outer))
    if mean <= 0.0:
        return 0.0
    cv = float(np.std(outer) / mean)
    return float(np.clip(1.0 - cv, 0.0, 1.0))


def outer_log_slope(
    r: np.ndarray,
    y: np.ndarray,
    outer_fraction: float = 0.5,
) -> float:
    """Fit d log|y| / d log R in outer disk."""
    r = np.asarray(r, dtype=float)
    y = np.maximum(np.abs(np.asarray(y, dtype=float)), 1e-30)
    mask = r >= outer_fraction * r[-1]
    if np.sum(mask) < 3:
        mask = np.ones_like(r, dtype=bool)
    slope, _ = np.polyfit(np.log(r[mask]), np.log(y[mask]), 1)
    return float(slope)


def baryonic_tully_fisher_proxy(
    m_b: float,
    v_flat: float,
) -> float:
    """
    Relative error of v_flat⁴ ∝ M_b scaling (deep-MOND-like check).

    Returns |log₁₀(v⁴/M) − median_reference| using a fixed reference scale.
    """
    m_b = max(float(m_b), 1e-12)
    v_flat = max(float(v_flat), 1e-12)
    ratio = (v_flat**4) / m_b
    ref = 1.0e4  # reference scale in benchmark units
    return float(abs(np.log10(ratio / ref)))


def compare_against_nfw_surrogate_shape(
    r: np.ndarray,
    v_total: np.ndarray,
    *,
    v_scale: float | None = None,
    r_s: float | None = None,
) -> float:
    """
    Optional shape comparison: Pearson correlation of v(R) with a simplified
    NFW-like rising-then-flat surrogate. No observational claim.
    """
    r = np.asarray(r, dtype=float)
    v = np.maximum(np.asarray(v_total, dtype=float), 0.0)
    if len(r) < 4:
        return float("nan")
    v0 = float(np.max(v)) if v_scale is None else float(v_scale)
    rs = float(0.25 * r[-1]) if r_s is None else float(r_s)
    v_nfw = v0 * np.sqrt(np.log(1.0 + r / rs) - (r / rs) / (1.0 + r / rs))
    if np.std(v) < 1e-12 or np.std(v_nfw) < 1e-12:
        return float("nan")
    return float(np.corrcoef(v, v_nfw)[0, 1])


def _default_r_grid(n: int = 200) -> np.ndarray:
    return np.linspace(0.2, 40.0, n)


def _disk_surface_density(
    r: np.ndarray,
    disk_model: DiskModel,
    params: dict[str, float] | None,
) -> np.ndarray:
    p = params or {}
    sigma0 = float(p.get("Sigma0", 0.08))
    rd = float(p.get("Rd", 3.0))
    if disk_model == "exponential_disk":
        return exponential_disk_surface_density(r, sigma0, rd)
    if disk_model == "lsb_extended_disk":
        return lsb_disk_surface_density(r, sigma0, rd)
    if disk_model == "compact_high_surface_brightness":
        return compact_disk_surface_density(
            r, sigma0, rd, float(p.get("Rcore", 0.5)),
        )
    raise ValueError(f"Unknown disk model: {disk_model}")


def _outer_tau_metrics(
    r: np.ndarray,
    a_tau: np.ndarray,
    v2_tot: np.ndarray,
    outer_fraction: float = 0.5,
) -> tuple[float, float]:
    mask = r >= outer_fraction * r[-1]
    tau_frac = float(
        np.median((r[mask] * np.abs(a_tau[mask])) / np.maximum(v2_tot[mask], 1e-30)),
    )
    tau_strength = float(np.median(np.abs(a_tau[mask])))
    return tau_frac, tau_strength


def _detect_expected_failure(
    mode: ExpectedFailureMode,
    *,
    source_norm: float,
    source_tolerance: float,
    max_abs_sigma_prime: float,
    max_abs_tau_acceleration: float,
    tau_v2_fraction: float,
    tau_acceleration_strength: float,
    flatness_improvement: float,
    gradient_safety_pass: bool,
) -> tuple[bool, str]:
    if mode == "pure_disformal":
        if source_norm < source_tolerance and max_abs_sigma_prime < 1e-10:
            return True, "pure_disformal_no_disk_tau_source"
        return False, "unexpected_conformal_source"
    if mode == "weak_coupling":
        reasons: list[str] = []
        if tau_acceleration_strength < TAU_ACCEL_STRENGTH_MIN:
            reasons.append("tau_acceleration_too_weak")
        if tau_v2_fraction < TAU_V2_FRACTION_MIN:
            reasons.append("tau_v2_boost_too_small")
        if flatness_improvement < FLATNESS_IMPROVEMENT_MIN:
            reasons.append("flatness_improvement_insufficient")
        if reasons:
            return True, ";".join(reasons)
        return False, "coupling_still_sufficient"
    if mode == "excessive_coupling":
        reasons = []
        if max_abs_sigma_prime > GRADIENT_EXCESS_THRESHOLD:
            reasons.append("sigma_prime_excess")
        if max_abs_tau_acceleration > GRADIENT_EXCESS_THRESHOLD:
            reasons.append("tau_acceleration_excess")
        if not gradient_safety_pass:
            reasons.append("gradient_safety_fail")
        if reasons:
            return True, ";".join(reasons)
        return False, "coupling_within_safety_bounds"
    return False, ""


def _finalize_case(
    res: DiskKessenceCaseResult,
    *,
    pass_criteria_met: bool,
    failure_mode: ExpectedFailureMode,
    source_tolerance: float,
    tau_v2_fraction: float,
    max_abs_sigma_prime: float,
    max_abs_tau_acceleration: float,
    tau_acceleration_strength: float,
) -> DiskKessenceCaseResult:
    if failure_mode != "none":
        detected, reason = _detect_expected_failure(
            failure_mode,
            source_norm=res.source_norm,
            source_tolerance=source_tolerance,
            max_abs_sigma_prime=max_abs_sigma_prime,
            max_abs_tau_acceleration=max_abs_tau_acceleration,
            tau_v2_fraction=tau_v2_fraction,
            tau_acceleration_strength=tau_acceleration_strength,
            flatness_improvement=res.flatness_improvement,
            gradient_safety_pass=res.gradient_safety_pass,
        )
        res.failure_detected = detected
        res.failure_reason = reason
        res.overall_pass = detected
    else:
        res.failure_detected = False
        res.failure_reason = ""
        res.overall_pass = pass_criteria_met
    return res


def _evaluate_disk_case(
    case_name: str,
    *,
    disk_model: DiskModel,
    disk_params: dict[str, float] | None,
    mu_model: MuModel,
    coupling_type: CouplingType,
    beta_over_m: float,
    a0: float,
    expected_status: Literal["pass", "fail"] = "pass",
    expected_failure_mode: ExpectedFailureMode = "none",
    require_flatness_improvement: bool = False,
    require_canonical_sanity: bool = False,
    require_simple_transition: bool = False,
    warnings: str = "",
) -> DiskKessenceCaseResult:
    r = _default_r_grid()
    sigma_b = _disk_surface_density(r, disk_model, disk_params)

    if coupling_type == "pure_disformal":
        s_disk = np.zeros_like(sigma_b)
        sigma_p = np.zeros_like(r)
        coupling = 0.0
    else:
        s_disk = disk_source_from_surface_density(sigma_b, beta_over_m)
        i_r = integrated_disk_source(r, s_disk)
        sigma_p = solve_disk_sigma_prime(r, i_r, a0, mu_model)
        coupling = beta_over_m

    source_norm = float(np.max(np.abs(s_disk)))
    ref_scale = max(
        float(np.max(np.abs(disk_source_from_surface_density(sigma_b, max(beta_over_m, 1e-12))))),
        1e-12,
    )
    source_nz = source_norm > SOURCE_ZERO_TOL * ref_scale
    source_tolerance = SOURCE_ZERO_TOL * ref_scale

    a_tau = tau_acceleration_proxy(sigma_p, coupling)
    v_b2 = freeman_disk_baryon_rotation_proxy(r, sigma_b)
    v2_tot = total_rotation_curve(r, v_b2, a_tau)

    flat_b = rotation_flatness_score(r, v_b2)
    flat_t = rotation_flatness_score(r, v2_tot)
    flat_improve = flat_t - flat_b
    tau_slope = outer_log_slope(r, sigma_p)

    outer_mask = r >= 0.5 * r[-1]
    v_flat = float(np.sqrt(np.median(v2_tot[outer_mask])))
    m_b = float(2.0 * np.pi * np.trapz(sigma_b * r, r))
    btf_err = baryonic_tully_fisher_proxy(m_b, v_flat)

    max_sigma = float(np.max(np.abs(sigma_p)))
    max_tau = float(np.max(np.abs(a_tau)))
    tau_frac, tau_strength = _outer_tau_metrics(r, a_tau, v2_tot)
    grad_safe = (
        max_sigma <= GRADIENT_EXCESS_THRESHOLD
        and max_tau <= GRADIENT_EXCESS_THRESHOLD
    )

    checks: list[bool] = []
    if require_canonical_sanity:
        checks.append(source_nz)
    if require_flatness_improvement:
        checks.append(source_nz)
        checks.append(flat_t >= OUTER_FLATNESS_MIN)
        checks.append(flat_improve >= FLATNESS_IMPROVEMENT_MIN)
        checks.append(grad_safe)
    if require_simple_transition:
        checks.append(source_nz)
        inner = np.median(np.abs(sigma_p[r < 0.35 * r[-1]]))
        outer = np.median(np.abs(sigma_p[r > 0.65 * r[-1]]))
        y_in = inner / max(a0, 1e-30)
        y_out = outer / max(a0, 1e-30)
        mu_in = float(mu_interpolation(y_in, "simple"))
        mu_out = float(mu_interpolation(y_out, "simple"))
        checks.append(mu_in > mu_out and mu_out < 0.9)

    pass_criteria_met = all(checks) if checks else True

    res = DiskKessenceCaseResult(
        case_name=case_name,
        disk_model=disk_model,
        mu_model=mu_model,
        coupling_type=coupling_type,
        source_nonzero=source_nz,
        source_norm=source_norm,
        tau_gradient_outer_slope=tau_slope,
        baryon_outer_flatness_score=flat_b,
        total_outer_flatness_score=flat_t,
        flatness_improvement=flat_improve,
        v_flat_proxy=v_flat,
        m_b_proxy=m_b,
        btf_proxy_error=btf_err,
        gradient_safety_pass=grad_safe,
        expected_status=expected_status,
        warnings=warnings,
        r=r,
        sigma_b=sigma_b,
        sigma_prime=sigma_p,
        v2_baryon=v_b2,
        v2_total=v2_tot,
        metadata={
            "nfw_shape_correlation": compare_against_nfw_surrogate_shape(
                r, np.sqrt(v2_tot),
            ),
        },
    )
    return _finalize_case(
        res,
        pass_criteria_met=pass_criteria_met,
        failure_mode=expected_failure_mode,
        source_tolerance=source_tolerance,
        tau_v2_fraction=tau_frac,
        max_abs_sigma_prime=max_sigma,
        max_abs_tau_acceleration=max_tau,
        tau_acceleration_strength=tau_strength,
    )


def run_exponential_disk_canonical() -> DiskKessenceCaseResult:
    return _evaluate_disk_case(
        "exponential_disk_canonical",
        disk_model="exponential_disk",
        disk_params={"Sigma0": 0.1, "Rd": 2.5},
        mu_model="canonical",
        coupling_type="conformal_trace",
        beta_over_m=0.9,
        a0=1.0,
        expected_status="pass",
        require_canonical_sanity=True,
        warnings="Newtonian disk sanity; flatness not required.",
    )


def run_exponential_disk_deep_kessence() -> DiskKessenceCaseResult:
    return _evaluate_disk_case(
        "exponential_disk_deep_kessence",
        disk_model="exponential_disk",
        disk_params={"Sigma0": 0.1, "Rd": 2.8},
        mu_model="deep_mond",
        coupling_type="conformal_trace",
        beta_over_m=1.1,
        a0=1.0,
        expected_status="pass",
        require_flatness_improvement=True,
    )


def run_lsb_extended_disk_deep_kessence() -> DiskKessenceCaseResult:
    return _evaluate_disk_case(
        "lsb_extended_disk_deep_kessence",
        disk_model="lsb_extended_disk",
        disk_params={"Sigma0": 0.08, "Rd": 3.2},
        mu_model="deep_mond",
        coupling_type="conformal_trace",
        beta_over_m=2.0,
        a0=1.0,
        expected_status="pass",
        require_flatness_improvement=True,
    )


def run_compact_high_surface_brightness_disk() -> DiskKessenceCaseResult:
    return _evaluate_disk_case(
        "compact_high_surface_brightness_disk",
        disk_model="compact_high_surface_brightness",
        disk_params={"Sigma0": 0.35, "Rd": 1.8, "Rcore": 0.4},
        mu_model="deep_mond",
        coupling_type="conformal_trace",
        beta_over_m=1.0,
        a0=1.0,
        expected_status="pass",
        require_flatness_improvement=True,
    )


def run_simple_interpolation_disk() -> DiskKessenceCaseResult:
    return _evaluate_disk_case(
        "simple_interpolation_disk",
        disk_model="exponential_disk",
        disk_params={"Sigma0": 0.09, "Rd": 3.0},
        mu_model="simple",
        coupling_type="conformal_trace",
        beta_over_m=1.0,
        a0=1.0,
        expected_status="pass",
        require_simple_transition=True,
        require_flatness_improvement=True,
    )


def run_pure_disformal_disk_expected_fail() -> DiskKessenceCaseResult:
    return _evaluate_disk_case(
        "pure_disformal_disk_expected_fail",
        disk_model="exponential_disk",
        disk_params={"Sigma0": 0.1, "Rd": 2.5},
        mu_model="deep_mond",
        coupling_type="pure_disformal",
        beta_over_m=0.0,
        a0=1.0,
        expected_status="fail",
        expected_failure_mode="pure_disformal",
        warnings="Intentional fail: no conformal disk source.",
    )


def run_weak_coupling_disk_expected_fail() -> DiskKessenceCaseResult:
    return _evaluate_disk_case(
        "weak_coupling_disk_expected_fail",
        disk_model="exponential_disk",
        disk_params={"Sigma0": 0.1, "Rd": 2.8},
        mu_model="deep_mond",
        coupling_type="conformal_trace",
        beta_over_m=1e-5,
        a0=1.0,
        expected_status="fail",
        expected_failure_mode="weak_coupling",
        warnings="Intentional fail: τ boost too weak.",
    )


def run_excessive_coupling_disk_expected_fail() -> DiskKessenceCaseResult:
    return _evaluate_disk_case(
        "excessive_coupling_disk_expected_fail",
        disk_model="compact_high_surface_brightness",
        disk_params={"Sigma0": 12.0, "Rd": 0.5, "Rcore": 0.2},
        mu_model="deep_mond",
        coupling_type="conformal_trace",
        beta_over_m=150.0,
        a0=0.005,
        expected_status="fail",
        expected_failure_mode="excessive_coupling",
        warnings="Intentional fail: excessive τ gradient / acceleration.",
    )


def _result_to_row(res: DiskKessenceCaseResult) -> dict[str, Any]:
    return {
        "case_name": res.case_name,
        "disk_model": res.disk_model,
        "mu_model": res.mu_model,
        "coupling_type": res.coupling_type,
        "source_nonzero": res.source_nonzero,
        "source_norm": res.source_norm,
        "tau_gradient_outer_slope": res.tau_gradient_outer_slope,
        "baryon_outer_flatness_score": res.baryon_outer_flatness_score,
        "total_outer_flatness_score": res.total_outer_flatness_score,
        "flatness_improvement": res.flatness_improvement,
        "v_flat_proxy": res.v_flat_proxy,
        "M_b_proxy": res.m_b_proxy,
        "btf_proxy_error": res.btf_proxy_error,
        "gradient_safety_pass": res.gradient_safety_pass,
        "expected_status": res.expected_status,
        "failure_detected": res.failure_detected,
        "failure_reason": res.failure_reason,
        "overall_pass": res.overall_pass,
        "warnings": res.warnings,
        "dataset_mode": BENCHMARK_MODE,
        "is_real_observational_data": False,
    }


def _plot_surface_density(results: list[DiskKessenceCaseResult], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    for res in results[:4]:
        if res.r is None or res.sigma_b is None:
            continue
        ax.semilogy(res.r, res.sigma_b, label=res.case_name, lw=1.5)
    ax.set_xlabel("R")
    ax.set_ylabel("Σ_b(R)")
    ax.set_title("Disk surface density profiles (Phase 7B)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_tau_gradient(results: list[DiskKessenceCaseResult], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    for res in results:
        if res.r is None or res.sigma_prime is None:
            continue
        ax.loglog(res.r, np.maximum(np.abs(res.sigma_prime), 1e-30), label=res.case_name, lw=1.2)
    ax.set_xlabel("R")
    ax.set_ylabel("|σ'(R)|")
    ax.set_title("Disk τ gradient σ'(R)")
    ax.legend(fontsize=6)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_rotation_curves(results: list[DiskKessenceCaseResult], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    for res in results:
        if res.r is None or res.v2_baryon is None or res.v2_total is None:
            continue
        ax.plot(res.r, np.sqrt(res.v2_baryon), "--", alpha=0.35, lw=1.0)
        ax.plot(res.r, np.sqrt(res.v2_total), label=res.case_name, lw=1.5)
    ax.set_xlabel("R")
    ax.set_ylabel("v(R)")
    ax.set_title("Disk rotation curves (dashed = baryon proxy)")
    ax.legend(fontsize=6)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_flatness_scores(results: list[DiskKessenceCaseResult], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [r.case_name for r in results]
    bary = [r.baryon_outer_flatness_score for r in results]
    total = [r.total_outer_flatness_score for r in results]
    x = np.arange(len(names))
    w = 0.35
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.bar(x - w / 2, bary, w, label="baryon-only flatness")
    ax.bar(x + w / 2, total, w, label="total flatness")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=28, ha="right")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("Outer disk rotation flatness scores")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_btf_proxy(results: list[DiskKessenceCaseResult], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    deep = [r for r in results if r.mu_model == "deep_mond" and r.expected_status == "pass"]
    m = [r.m_b_proxy for r in deep]
    v4 = [r.v_flat_proxy**4 for r in deep]
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.loglog(m, v4, "o", label="deep-MOND pass cases")
    if len(m) >= 2:
        coef = np.polyfit(np.log10(m), np.log10(v4), 1)
        m_line = np.logspace(np.log10(min(m)), np.log10(max(m)), 50)
        v4_line = 10 ** (coef[0] * np.log10(m_line) + coef[1])
        ax.loglog(m_line, v4_line, "--", alpha=0.6, label="fit slope")
    ax.set_xlabel("M_b proxy")
    ax.set_ylabel("v_flat⁴ proxy")
    ax.set_title("Baryonic Tully–Fisher proxy (not SPARC)")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _build_report(results: list[DiskKessenceCaseResult]) -> str:
    n_ok = sum(1 for r in results if r.overall_pass)
    lines = [
        "# Disk K-essence rotation report (Phase 7B)",
        "",
        f"## ⚠️ {BANNER_DISK_KESSENCE}",
        "",
        f"## ⚠️ {BANNER_CALIBRATION}",
        "",
        "## Purpose",
        "",
        "Test whether the **corrected conformal-disformal K-essence source** remains "
        "viable in **axisymmetric disk-like** baryonic geometries and can improve "
        "outer rotation-curve flatness in controlled synthetic disk benchmarks.",
        "",
        "## Equations (cylindrical thin-disk proxy)",
        "",
        "```text",
        "Σ_b(R)                          surface density",
        "S_disk(R) = (β/M) Σ_b(R)        conformal trace source",
        "(1/R) d/dR [ R μ(|σ'|/a₀) σ' ] = S_disk(R)",
        "R μ(|σ'|/a₀) σ' = ∫₀^R S_disk(R') R' dR'",
        "v_b²(R) = G M_b(<R) / R         Freeman disk proxy",
        "v_total² = v_b² + R a_τ,   a_τ = (β/M) σ'",
        "```",
        "",
        "**Deep regime (μ=y):** outer disk may give σ' ~ R^(−1/2); benchmark targets "
        "**flat or slowly varying v_total²**, not a fixed σ' slope.",
        "",
        f"- **Cases:** {len(results)} | **Expected outcomes matched:** {n_ok}/{len(results)}",
        "",
        "## Results",
        "",
        "| Case | disk | μ | source | baryon flat | total flat | Δ flat | physical | expect | matched |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in results:
        if r.expected_status == "pass":
            physical = "pass" if r.overall_pass else "fail"
        else:
            physical = "fail (detected)" if r.failure_detected else "fail (missed)"
        lines.append(
            f"| {r.case_name} | {r.disk_model} | {r.mu_model} | "
            f"{'yes' if r.source_nonzero else 'no'} | {r.baryon_outer_flatness_score:.2f} | "
            f"{r.total_outer_flatness_score:.2f} | {r.flatness_improvement:+.2f} | "
            f"{physical} | {r.expected_status} | {'✓' if r.overall_pass else '✗'} |",
        )
    lines.extend(
        [
            "",
            "### Expected-fail detection",
            "",
            "| Case | failure_detected | failure_reason |",
            "| --- | --- | --- |",
        ],
    )
    for r in results:
        if r.expected_status == "fail":
            lines.append(
                f"| {r.case_name} | {r.failure_detected} | {r.failure_reason or '—'} |",
            )
    lines.extend(
        [
            "",
            "## Scientific interpretation",
            "",
            "- **Passing** means the corrected source can improve disk-like rotation-curve "
            "flatness in synthetic axisymmetric benchmarks.",
            "- Does **not** validate against SPARC or real galaxies.",
            "- Does **not** prove dark-matter replacement or lensing consistency.",
            "",
            "## Failure modes",
            "",
            "- Thin-disk cylindrical approximation only; no full 3D Poisson solver.",
            "- No real SPARC fitting; no gas/stars M/L uncertainty.",
            "- No lensing or cosmological consistency; no full relativistic perturbations.",
            "",
            "## Disclaimer",
            "",
            f"- {BANNER_CALIBRATION}",
            "",
        ],
    )
    return "\n".join(lines)


def run_disk_kessence_rotation_benchmark(
    outputs_root: Path | None = None,
) -> tuple[pd.DataFrame, list[DiskKessenceCaseResult]]:
    """Run Phase 7B; write CSV, report, and figures."""
    root = Path(__file__).resolve().parents[3]
    outputs = Path(outputs_root or root / "outputs")
    tables = outputs / "tables"
    reports = outputs / "reports"
    figures = outputs / "figures"
    for d in (tables, reports, figures):
        d.mkdir(parents=True, exist_ok=True)

    results = [
        run_exponential_disk_canonical(),
        run_exponential_disk_deep_kessence(),
        run_lsb_extended_disk_deep_kessence(),
        run_compact_high_surface_brightness_disk(),
        run_simple_interpolation_disk(),
        run_pure_disformal_disk_expected_fail(),
        run_weak_coupling_disk_expected_fail(),
        run_excessive_coupling_disk_expected_fail(),
    ]

    df = pd.DataFrame([_result_to_row(r) for r in results])
    df.to_csv(tables / "disk_kessence_rotation_summary.csv", index=False)
    (reports / "disk_kessence_rotation_report.md").write_text(
        _build_report(results),
        encoding="utf-8",
    )

    _plot_surface_density(results, figures / "disk_kessence_surface_density.png")
    _plot_tau_gradient(results, figures / "disk_kessence_tau_gradient.png")
    _plot_rotation_curves(results, figures / "disk_kessence_rotation_curves.png")
    _plot_flatness_scores(results, figures / "disk_kessence_flatness_scores.png")
    _plot_btf_proxy(results, figures / "disk_kessence_btf_proxy.png")

    return df, results

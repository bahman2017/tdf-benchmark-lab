"""Phase 7A — K-essence source viability from static baryonic matter (spherical proxy)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy.optimize import brentq

BENCHMARK_MODE = "kessence_source_viability_benchmark"
BANNER_KESSENCE_SOURCE = (
    "K-ESSENCE SOURCE VIABILITY BENCHMARK — NOT OBSERVATIONAL VALIDATION"
)

BANNER_CALIBRATION = (
    "These results are calibration diagnostics for a phenomenological TDF model. "
    "They do not constitute observational validation of TDF unless all required "
    "multi-channel constraints are passed."
)

MuModel = Literal["canonical", "deep_mond", "simple", "standard"]
BaryonProfile = Literal["plummer_sphere", "exponential_spherical_proxy", "compact_bulge_proxy"]
CouplingType = Literal["pure_disformal", "conformal_trace"]

SLOPE_TOL = 0.45
FLATNESS_MIN = 0.82
SOURCE_ZERO_TOL = 1e-10 * 1.0  # relative to typical conformal source scale
RA_TAU_SLOPE_TOL = 0.12  # outer d log(a_tau) / d log(r) ≈ -1 when r a_tau is flat
GRADIENT_EXCESS_THRESHOLD = 25.0
TAU_V2_FRACTION_MIN = 0.15  # outer median (r a_tau) / v_total^2 for MOND-like boost
TAU_ACCEL_STRENGTH_MIN = 1e-4  # outer median |a_tau| for insufficient-coupling detection
SIGMA_PRIME_ZERO_TOL = 1e-10  # no dynamically sourced spatial gradient

ExpectedFailureMode = Literal[
    "none",
    "pure_disformal",
    "insufficient_coupling",
    "excessive_gradient",
]

REQUIRED_SUMMARY_COLUMNS: tuple[str, ...] = (
    "case_name",
    "coupling_type",
    "mu_model",
    "baryon_profile",
    "source_norm",
    "source_nonzero",
    "outer_gradient_slope",
    "expected_outer_slope",
    "slope_error",
    "tau_acceleration_slope",
    "rotation_flatness_score",
    "max_abs_sigma_prime",
    "max_abs_tau_acceleration",
    "tau_acceleration_strength",
    "kx_positive",
    "hyperbolicity_pass",
    "failure_detected",
    "failure_reason",
    "pass_criteria_met",
    "expected_status",
    "expected_failure_matched",
    "overall_pass",
    "warnings",
)


@dataclass
class KessenceSourceCaseResult:
    case_name: str
    coupling_type: str
    mu_model: str
    baryon_profile: str
    source_norm: float
    source_nonzero: bool
    outer_gradient_slope: float
    expected_outer_slope: float
    slope_error: float
    tau_acceleration_slope: float
    rotation_flatness_score: float
    kx_positive: bool
    hyperbolicity_pass: bool
    expected_status: Literal["pass", "fail"]
    max_abs_sigma_prime: float = 0.0
    max_abs_tau_acceleration: float = 0.0
    tau_acceleration_strength: float = 0.0
    failure_detected: bool = False
    failure_reason: str = ""
    pass_criteria_met: bool = False
    expected_failure_matched: bool = False
    overall_pass: bool = False
    warnings: str = ""
    r: np.ndarray | None = None
    rho_b: np.ndarray | None = None
    sigma_prime: np.ndarray | None = None
    v2_total: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def baryon_density_profile(
    r: np.ndarray,
    model: BaryonProfile,
    params: dict[str, float] | None = None,
) -> np.ndarray:
    """
    Spherical baryon density ρ_b(r) [mass/length³ in code units].

    Models are phenomenological proxies for viability tests only.
    """
    r = np.asarray(r, dtype=float)
    p = params or {}
    if model == "plummer_sphere":
        m = float(p.get("M", 1.0))
        a = max(float(p.get("a", 1.0)), 1e-6)
        return (3.0 * m) / (4.0 * np.pi * a**3) * (1.0 + (r / a) ** 2) ** (-2.5)
    if model == "exponential_spherical_proxy":
        rho0 = float(p.get("rho0", 0.08))
        rd = max(float(p.get("Rd", 2.0)), 1e-6)
        return rho0 * np.exp(-r / rd)
    if model == "compact_bulge_proxy":
        rho0 = float(p.get("rho0", 2.0))
        rb = max(float(p.get("Rb", 0.3)), 1e-6)
        return rho0 / (r / rb + 1.0) ** 4
    raise ValueError(f"Unknown baryon profile: {model}")


def static_dust_disformal_current(
    rho_b: np.ndarray,
    omega: float,
) -> tuple[np.ndarray, np.ndarray]:
    """J^0 = ρ_b ω, J^r = 0 for static dust with τ = ω t + σ(r)."""
    rho = np.asarray(rho_b, dtype=float)
    j0 = rho * float(omega)
    jr = np.zeros_like(rho)
    return j0, jr


def divergence_static_spherical_current(r: np.ndarray, jr: np.ndarray) -> np.ndarray:
    """div J = (1/r²) d/dr (r² J^r) in spherical symmetry."""
    r = np.asarray(r, dtype=float)
    jr = np.asarray(jr, dtype=float)
    if len(r) < 2:
        return np.zeros_like(r)
    flux = r**2 * jr
    div = np.gradient(flux, r) / np.maximum(r**2, 1e-30)
    return div


def disformal_static_dust_source(
    r: np.ndarray,
    rho_b: np.ndarray,
    omega: float,
) -> np.ndarray:
    """
    Pure-disformal static-dust source proxy: div of T^{μν} ∂_ν τ.

    With J^r = 0, the spherical divergence vanishes identically.
    """
    _j0, jr = static_dust_disformal_current(rho_b, omega)
    return divergence_static_spherical_current(r, jr)


def conformal_trace_source(
    rho_b: np.ndarray,
    beta_over_M: float,
) -> np.ndarray:
    """
    Conformal trace source S_b ∝ β/M · ρ_b with T ≈ −ρ_b.

    Sign convention: positive S_b for positive β/M and ρ_b > 0 drives
    outward-growing integrated source I(r) in the benchmark radial equation.
    """
    rho = np.asarray(rho_b, dtype=float)
    return float(beta_over_M) * rho


def enclosed_source_integral(r: np.ndarray, s_b: np.ndarray) -> np.ndarray:
    """I(r) = ∫_0^r S_b(r') r'^2 dr'."""
    r = np.asarray(r, dtype=float)
    s = np.asarray(s_b, dtype=float)
    integrand = s * r**2
    return np.array([np.trapz(integrand[: i + 1], r[: i + 1]) for i in range(len(r))])


def mu_interpolation(y: np.ndarray | float, model: MuModel) -> np.ndarray:
    """μ(y) for K-essence radial equation."""
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


def _lhs_integrated(r: float, sigma_prime: float, a0: float, mu_model: MuModel) -> float:
    y = abs(sigma_prime) / max(a0, 1e-30)
    mu = float(mu_interpolation(y, mu_model))
    return r**2 * mu * sigma_prime


def solve_sigma_prime_from_integrated_equation(
    r: np.ndarray,
    i_r: np.ndarray,
    a0: float,
    mu_model: MuModel,
) -> np.ndarray:
    """
    Solve r² μ(|σ'|/a0) σ' = I(r) pointwise.

    Analytic limits:
    - canonical (μ=1): σ' = I / r²
    - deep_mond (μ=y, σ'>0): σ' = √(I a0) / r
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
            out[k] = ik / rk**2
            continue
        if mu_model == "deep_mond":
            out[k] = np.sqrt(max(ik * a0, 0.0)) / rk
            continue

        def func(sp: float) -> float:
            return _lhs_integrated(rk, sp, a0, mu_model) - ik

        hi = max(np.sqrt(ik * a0) / rk, ik / rk**2, 1e-12) * 100.0
        try:
            out[k] = brentq(func, 1e-14, hi, maxiter=200)
        except ValueError:
            out[k] = np.sqrt(max(ik * a0, 0.0)) / rk
    return out


def tau_acceleration_proxy(
    sigma_prime: np.ndarray,
    coupling: float,
) -> np.ndarray:
    """a_τ = coupling · σ'(r) [acceleration-scale proxy]."""
    return float(coupling) * np.asarray(sigma_prime, dtype=float)


def baryon_rotation_proxy(
    r: np.ndarray,
    rho_b: np.ndarray,
    *,
    g_newton: float = 1.0,
) -> np.ndarray:
    """v_b² = G M_enc(r) / r with M_enc = 4π ∫ ρ r'² dr'."""
    r = np.asarray(r, dtype=float)
    rho = np.asarray(rho_b, dtype=float)
    m_enc = np.array(
        [4.0 * np.pi * np.trapz(rho[: i + 1] * r[: i + 1] ** 2, r[: i + 1]) for i in range(len(r))],
    )
    return g_newton * m_enc / np.maximum(r, 1e-12)


def total_rotation_proxy(
    r: np.ndarray,
    v_b2: np.ndarray,
    a_tau: np.ndarray,
) -> np.ndarray:
    """v_total² = v_b² + r · a_τ (spherical proxy)."""
    r = np.asarray(r, dtype=float)
    return np.maximum(np.asarray(v_b2, dtype=float) + r * np.asarray(a_tau, dtype=float), 0.0)


def outer_log_slope(
    r: np.ndarray,
    y: np.ndarray,
    r_min_fraction: float = 0.6,
) -> float:
    """Fit d log(y) / d log(r) in outer radial band."""
    r = np.asarray(r, dtype=float)
    y = np.maximum(np.asarray(y, dtype=float), 1e-30)
    mask = r >= r_min_fraction * r[-1]
    if np.sum(mask) < 3:
        mask = np.ones_like(r, dtype=bool)
    lr = np.log(r[mask])
    ly = np.log(y[mask])
    slope, _ = np.polyfit(lr, ly, 1)
    return float(slope)


def rotation_flatness_score(
    r: np.ndarray,
    v2_total: np.ndarray,
    r_min_fraction: float = 0.6,
) -> float:
    """1 − fractional std of v² in outer region (higher = flatter)."""
    r = np.asarray(r, dtype=float)
    v2 = np.asarray(v2_total, dtype=float)
    mask = r >= r_min_fraction * r[-1]
    outer = v2[mask]
    if len(outer) < 2:
        return 0.0
    mean = float(np.mean(outer))
    if mean <= 0.0:
        return 0.0
    cv = float(np.std(outer) / mean)
    return float(np.clip(1.0 - cv, 0.0, 1.0))


def kessence_stability_proxy(
    gradient: np.ndarray,
    a0: float,
    mu_model: MuModel,
) -> dict[str, Any]:
    """
    Proxy stability: μ > 0 and d/dy[y μ(y)] > 0 at median y = |σ'|/a0.
    """
    g = np.asarray(gradient, dtype=float)
    a0 = max(float(a0), 1e-30)
    y = np.median(np.maximum(np.abs(g) / a0, 1e-12))
    mu_y = float(mu_interpolation(y, mu_model))
    eps = 1e-8
    y_mu_plus = (y + eps) * float(mu_interpolation(y + eps, mu_model))
    y_mu_minus = max(y - eps, 0.0) * float(mu_interpolation(max(y - eps, 0.0), mu_model))
    d_dy = (y_mu_plus - y_mu_minus) / (2.0 * eps)
    warnings: list[str] = []
    if mu_y <= 0.0:
        warnings.append("μ(y) ≤ 0 at median y.")
    if d_dy <= 0.0:
        warnings.append("d/dy[y μ(y)] ≤ 0 (hyperbolicity proxy fail).")
    return {
        "kx_positive": mu_y > 0.0,
        "hyperbolicity_pass": d_dy > 0.0,
        "warnings": "; ".join(warnings),
    }


def _default_r_grid(n: int = 200) -> np.ndarray:
    return np.linspace(0.05, 30.0, n)


def _outer_tau_metrics(
    r: np.ndarray,
    a_tau: np.ndarray,
    v2_tot: np.ndarray,
    r_min_fraction: float = 0.6,
) -> tuple[float, float]:
    """Outer median τ boost fraction and |a_τ| strength."""
    r = np.asarray(r, dtype=float)
    mask = r >= r_min_fraction * r[-1]
    if not np.any(mask):
        mask = np.ones_like(r, dtype=bool)
    tau_frac = float(
        np.median(
            (r[mask] * np.abs(a_tau[mask])) / np.maximum(v2_tot[mask], 1e-30),
        ),
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
    tau_acceleration_strength: float,
    tau_v2_fraction: float,
    rotation_flatness_score: float,
    kx_positive: bool,
    hyperbolicity_pass: bool,
) -> tuple[bool, str]:
    """Return (failure_detected, failure_reason) for intentional fail cases."""
    if mode == "pure_disformal":
        if source_norm < source_tolerance:
            return True, "pure_disformal_source_zero"
        return False, "source_not_zero"
    if mode == "insufficient_coupling":
        reasons: list[str] = []
        if tau_acceleration_strength < TAU_ACCEL_STRENGTH_MIN:
            reasons.append("tau_acceleration_too_weak")
        if tau_v2_fraction < TAU_V2_FRACTION_MIN:
            reasons.append("tau_v2_boost_too_small")
        if rotation_flatness_score < FLATNESS_MIN:
            reasons.append("rotation_flatness_below_threshold")
        if reasons:
            return True, ";".join(reasons)
        return False, "coupling_still_sufficient"
    if mode == "excessive_gradient":
        reasons = []
        gradient_excess = (
            max_abs_sigma_prime > GRADIENT_EXCESS_THRESHOLD
            or max_abs_tau_acceleration > GRADIENT_EXCESS_THRESHOLD
        )
        if gradient_excess:
            reasons.append("gradient_exceeds_threshold")
        if not kx_positive:
            reasons.append("kx_not_positive")
        if not hyperbolicity_pass:
            reasons.append("hyperbolicity_failed")
        if reasons:
            return True, ";".join(reasons)
        return False, "gradient_and_stability_within_bounds"
    return False, ""


def _finalize_benchmark_outcome(
    res: KessenceSourceCaseResult,
    *,
    pass_criteria_met: bool,
    failure_mode: ExpectedFailureMode,
    source_tolerance: float,
    tau_v2_fraction: float,
) -> KessenceSourceCaseResult:
    """Set failure_detected, overall_pass, and expected_failure_matched."""
    if failure_mode != "none":
        detected, reason = _detect_expected_failure(
            failure_mode,
            source_norm=res.source_norm,
            source_tolerance=source_tolerance,
            max_abs_sigma_prime=res.max_abs_sigma_prime,
            max_abs_tau_acceleration=res.max_abs_tau_acceleration,
            tau_acceleration_strength=res.tau_acceleration_strength,
            tau_v2_fraction=tau_v2_fraction,
            rotation_flatness_score=res.rotation_flatness_score,
            kx_positive=res.kx_positive,
            hyperbolicity_pass=res.hyperbolicity_pass,
        )
        res.failure_detected = detected
        res.failure_reason = reason
    else:
        res.failure_detected = False
        res.failure_reason = ""

    res.pass_criteria_met = pass_criteria_met
    if res.expected_status == "pass":
        res.overall_pass = pass_criteria_met
    else:
        res.overall_pass = res.failure_detected
    res.expected_failure_matched = res.overall_pass
    return res


def _evaluate_case(
    case_name: str,
    *,
    coupling_type: CouplingType,
    mu_model: MuModel,
    baryon_profile: BaryonProfile,
    profile_params: dict[str, float] | None,
    beta_over_M: float,
    a0: float,
    omega: float = 1.0,
    coupling_scale: float | None = None,
    expected_status: Literal["pass", "fail"] = "pass",
    expected_failure_mode: ExpectedFailureMode = "none",
    expected_outer_slope: float = float("nan"),
    require_flat_rotation: bool = False,
    require_source_nonzero: bool = True,
    stability_must_pass: bool = True,
    warnings: str = "",
) -> KessenceSourceCaseResult:
    r = _default_r_grid()
    rho = baryon_density_profile(r, baryon_profile, profile_params)

    if coupling_type == "pure_disformal":
        source = disformal_static_dust_source(r, rho, omega)
        sigma_p = np.zeros_like(r)
        coupling = 0.0
    else:
        source = conformal_trace_source(rho, beta_over_M)
        i_r = enclosed_source_integral(r, source)
        sigma_p = solve_sigma_prime_from_integrated_equation(r, i_r, a0, mu_model)
        coupling = float(coupling_scale if coupling_scale is not None else beta_over_M)

    source_norm = float(np.max(np.abs(source)))
    ref_scale = max(float(np.max(np.abs(conformal_trace_source(rho, max(beta_over_M, 1e-12))))), 1e-12)
    source_near_zero = source_norm < SOURCE_ZERO_TOL * ref_scale
    source_nz = source_norm > SOURCE_ZERO_TOL * ref_scale

    a_tau = tau_acceleration_proxy(sigma_p, coupling)
    v_b2 = baryon_rotation_proxy(r, rho)
    v2_tot = total_rotation_proxy(r, v_b2, a_tau)

    grad_slope = outer_log_slope(r, sigma_p)
    slope_err = abs(grad_slope - expected_outer_slope) if np.isfinite(expected_outer_slope) else 0.0
    a_slope = outer_log_slope(r, np.maximum(np.abs(a_tau), 1e-30))
    flat = rotation_flatness_score(r, v2_tot)
    stab = kessence_stability_proxy(sigma_p, a0, mu_model)
    max_sigma = float(np.max(np.abs(sigma_p)))
    max_tau_accel = float(np.max(np.abs(a_tau)))
    tau_frac, tau_strength = _outer_tau_metrics(r, a_tau, v2_tot)
    source_tolerance = SOURCE_ZERO_TOL * ref_scale

    checks: list[bool] = []
    if require_source_nonzero:
        checks.append(source_nz)
    if np.isfinite(expected_outer_slope):
        checks.append(slope_err <= SLOPE_TOL)
    if require_flat_rotation:
        checks.append(flat >= FLATNESS_MIN)
        checks.append(abs(a_slope + 1.0) < RA_TAU_SLOPE_TOL)
        checks.append(tau_frac >= TAU_V2_FRACTION_MIN)
        checks.append(tau_strength >= TAU_ACCEL_STRENGTH_MIN)
    if stability_must_pass:
        checks.append(stab["kx_positive"] and stab["hyperbolicity_pass"])
        checks.append(max_sigma <= GRADIENT_EXCESS_THRESHOLD)
        checks.append(max_tau_accel <= GRADIENT_EXCESS_THRESHOLD)

    pass_criteria_met = all(checks) if checks else True
    all_warn = "; ".join(filter(None, [warnings, stab.get("warnings", "")]))

    res = KessenceSourceCaseResult(
        case_name=case_name,
        coupling_type=coupling_type,
        mu_model=mu_model,
        baryon_profile=baryon_profile,
        source_norm=source_norm,
        source_nonzero=source_nz,
        outer_gradient_slope=grad_slope,
        expected_outer_slope=expected_outer_slope,
        slope_error=slope_err,
        tau_acceleration_slope=a_slope,
        rotation_flatness_score=flat,
        kx_positive=bool(stab["kx_positive"]),
        hyperbolicity_pass=bool(stab["hyperbolicity_pass"]),
        max_abs_sigma_prime=max_sigma,
        max_abs_tau_acceleration=max_tau_accel,
        tau_acceleration_strength=tau_strength,
        expected_status=expected_status,
        warnings=all_warn,
        r=r,
        rho_b=rho,
        sigma_prime=sigma_p,
        v2_total=v2_tot,
    )
    return _finalize_benchmark_outcome(
        res,
        pass_criteria_met=pass_criteria_met,
        failure_mode=expected_failure_mode,
        source_tolerance=source_tolerance,
        tau_v2_fraction=tau_frac,
    )


def run_pure_disformal_static_dust_expected_fail() -> KessenceSourceCaseResult:
    return _evaluate_case(
        "pure_disformal_static_dust_expected_fail",
        coupling_type="pure_disformal",
        mu_model="deep_mond",
        baryon_profile="exponential_spherical_proxy",
        profile_params={"rho0": 0.1, "Rd": 3.0},
        beta_over_M=0.0,
        a0=1.0,
        expected_status="fail",
        expected_failure_mode="pure_disformal",
        require_source_nonzero=False,
        stability_must_pass=False,
        warnings="Intentional fail: pure disformal static dust has div J = 0.",
    )


def run_conformal_canonical_newtonian() -> KessenceSourceCaseResult:
    return _evaluate_case(
        "conformal_canonical_newtonian",
        coupling_type="conformal_trace",
        mu_model="canonical",
        baryon_profile="exponential_spherical_proxy",
        profile_params={"rho0": 0.12, "Rd": 2.5},
        beta_over_M=0.8,
        a0=1.0,
        expected_status="pass",
        expected_outer_slope=-2.0,
        require_source_nonzero=True,
        warnings="Newtonian sanity: μ=1 gives σ' ~ I/r², not flat rotation.",
    )


def run_conformal_kessence_deep_mond() -> KessenceSourceCaseResult:
    return _evaluate_case(
        "conformal_kessence_deep_mond",
        coupling_type="conformal_trace",
        mu_model="deep_mond",
        baryon_profile="exponential_spherical_proxy",
        profile_params={"rho0": 0.1, "Rd": 3.0},
        beta_over_M=1.2,
        a0=1.0,
        expected_status="pass",
        expected_outer_slope=-1.0,
        require_flat_rotation=True,
        require_source_nonzero=True,
    )


def run_conformal_kessence_simple_interpolation() -> KessenceSourceCaseResult:
    res = _evaluate_case(
        "conformal_kessence_simple_interpolation",
        coupling_type="conformal_trace",
        mu_model="simple",
        baryon_profile="plummer_sphere",
        profile_params={"M": 1.0, "a": 2.0},
        beta_over_M=1.0,
        a0=1.0,
        expected_status="pass",
        expected_outer_slope=-1.1,
        require_source_nonzero=True,
    )
    r = res.r
    if r is not None and res.sigma_prime is not None:
        inner = np.median(np.abs(res.sigma_prime[r < 0.35 * r[-1]]))
        outer = np.median(np.abs(res.sigma_prime[r > 0.65 * r[-1]]))
        y_inner = inner / 1.0
        y_outer = outer / 1.0
        mu_in = float(mu_interpolation(y_inner, "simple"))
        mu_out = float(mu_interpolation(y_outer, "simple"))
        transition_ok = mu_in > mu_out and mu_out < 0.85
        if not transition_ok:
            res.warnings += "; simple μ transition check failed."
        tau_v2_frac = 0.0
        if res.r is not None and res.sigma_prime is not None and res.v2_total is not None:
            a_tau = tau_acceleration_proxy(res.sigma_prime, 1.0)
            tau_v2_frac, _ = _outer_tau_metrics(res.r, a_tau, res.v2_total)
        return _finalize_benchmark_outcome(
            res,
            pass_criteria_met=res.pass_criteria_met and transition_ok,
            failure_mode="none",
            source_tolerance=SOURCE_ZERO_TOL,
            tau_v2_fraction=tau_v2_frac,
        )
    return res


def run_insufficient_coupling_expected_fail() -> KessenceSourceCaseResult:
    return _evaluate_case(
        "insufficient_coupling_expected_fail",
        coupling_type="conformal_trace",
        mu_model="deep_mond",
        baryon_profile="exponential_spherical_proxy",
        profile_params={"rho0": 0.1, "Rd": 3.0},
        beta_over_M=1e-5,
        a0=1.0,
        expected_status="fail",
        expected_failure_mode="insufficient_coupling",
        require_source_nonzero=True,
        warnings="Intentional fail: coupling too weak for flat rotation proxy.",
    )


def run_excessive_gradient_stability_fail() -> KessenceSourceCaseResult:
    return _evaluate_case(
        "excessive_gradient_stability_fail",
        coupling_type="conformal_trace",
        mu_model="deep_mond",
        baryon_profile="compact_bulge_proxy",
        profile_params={"rho0": 500.0, "Rb": 0.2},
        beta_over_M=80.0,
        a0=0.01,
        expected_status="fail",
        expected_failure_mode="excessive_gradient",
        require_source_nonzero=True,
        stability_must_pass=False,
        warnings="Intentional fail: excessive coupling / gradient.",
    )


def run_compact_bulge_deep_mond_profile() -> KessenceSourceCaseResult:
    return _evaluate_case(
        "compact_bulge_deep_mond_profile",
        coupling_type="conformal_trace",
        mu_model="deep_mond",
        baryon_profile="compact_bulge_proxy",
        profile_params={"rho0": 1.5, "Rb": 0.4},
        beta_over_M=1.5,
        a0=1.0,
        expected_status="pass",
        expected_outer_slope=-1.0,
        require_flat_rotation=True,
        require_source_nonzero=True,
    )


def _result_to_row(res: KessenceSourceCaseResult) -> dict[str, Any]:
    return {
        "case_name": res.case_name,
        "coupling_type": res.coupling_type,
        "mu_model": res.mu_model,
        "baryon_profile": res.baryon_profile,
        "source_norm": res.source_norm,
        "source_nonzero": res.source_nonzero,
        "outer_gradient_slope": res.outer_gradient_slope,
        "expected_outer_slope": res.expected_outer_slope,
        "slope_error": res.slope_error,
        "tau_acceleration_slope": res.tau_acceleration_slope,
        "rotation_flatness_score": res.rotation_flatness_score,
        "max_abs_sigma_prime": res.max_abs_sigma_prime,
        "max_abs_tau_acceleration": res.max_abs_tau_acceleration,
        "tau_acceleration_strength": res.tau_acceleration_strength,
        "kx_positive": res.kx_positive,
        "hyperbolicity_pass": res.hyperbolicity_pass,
        "failure_detected": res.failure_detected,
        "failure_reason": res.failure_reason,
        "pass_criteria_met": res.pass_criteria_met,
        "expected_status": res.expected_status,
        "expected_failure_matched": res.expected_failure_matched,
        "overall_pass": res.overall_pass,
        "warnings": res.warnings,
        "dataset_mode": BENCHMARK_MODE,
        "is_real_observational_data": False,
    }


def _plot_source_terms(results: list[KessenceSourceCaseResult], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    pure = next(r for r in results if r.case_name == "pure_disformal_static_dust_expected_fail")
    conf = next(r for r in results if r.case_name == "conformal_kessence_deep_mond")
    if pure.r is not None:
        src_p = disformal_static_dust_source(pure.r, pure.rho_b, 1.0)
        axes[0].semilogy(pure.r, np.abs(src_p) + 1e-30, label="pure disformal div J")
    if conf.r is not None and conf.rho_b is not None:
        src_c = conformal_trace_source(conf.rho_b, 1.2)
        axes[1].semilogy(conf.r, src_c, label="conformal S_b")
    for ax, title in zip(axes, ["Pure disformal (≈0)", "Conformal trace"]):
        ax.set_xlabel("r")
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Source terms (Phase 7A)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_profiles(
    results: list[KessenceSourceCaseResult],
    path: Path,
    field: str,
    ylabel: str,
    title: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    for res in results:
        if res.r is None:
            continue
        y = getattr(res, field)
        if y is None:
            continue
        ax.loglog(res.r, np.maximum(np.abs(y), 1e-30), label=res.case_name, lw=1.5)
    ax.set_xlabel("r")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=7)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_outer_slope(results: list[KessenceSourceCaseResult], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [r.case_name for r in results]
    slopes = [r.outer_gradient_slope for r in results]
    expected = [r.expected_outer_slope for r in results]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(x - 0.2, slopes, 0.4, label="measured σ' slope")
    exp_mask = [np.isfinite(e) for e in expected]
    exp_x = [i for i, e in zip(x, expected) if np.isfinite(e)]
    exp_y = [e for e in expected if np.isfinite(e)]
    if exp_x:
        ax.scatter(exp_x, exp_y, color="C3", zorder=5, label="expected")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel("d log σ' / d log r")
    ax.set_title("Outer gradient slopes")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _stability_viable(res: KessenceSourceCaseResult) -> bool:
    """Combined stability proxy including gradient safety bound."""
    return (
        res.kx_positive
        and res.hyperbolicity_pass
        and res.max_abs_sigma_prime <= GRADIENT_EXCESS_THRESHOLD
        and res.max_abs_tau_acceleration <= GRADIENT_EXCESS_THRESHOLD
    )


def _plot_stability(results: list[KessenceSourceCaseResult], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [r.case_name for r in results]
    kx = [1.0 if r.kx_positive else 0.0 for r in results]
    hyp = [1.0 if r.hyperbolicity_pass else 0.0 for r in results]
    grad_ok = [
        1.0
        if r.max_abs_sigma_prime <= GRADIENT_EXCESS_THRESHOLD
        and r.max_abs_tau_acceleration <= GRADIENT_EXCESS_THRESHOLD
        else 0.0
        for r in results
    ]
    x = np.arange(len(names))
    w = 0.25
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(x - w, kx, w, label="μ>0 proxy")
    ax.bar(x, hyp, w, label="hyperbolicity proxy")
    ax.bar(x + w, grad_ok, w, label="|σ'| within bound")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylim(-0.1, 1.2)
    ax.set_title("K-essence stability proxies")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _build_report(results: list[KessenceSourceCaseResult]) -> str:
    n_ok = sum(1 for r in results if r.expected_failure_matched)
    lines = [
        "# K-essence source viability report (Phase 7A)",
        "",
        f"## ⚠️ {BANNER_KESSENCE_SOURCE}",
        "",
        f"## ⚠️ {BANNER_CALIBRATION}",
        "",
        "## Purpose",
        "",
        "Test whether **corrected TDF source structure** (conformal factor "
        "A(τ)² g + disformal τ terms) can **dynamically generate spatial τ gradients** "
        "from **static baryonic** matter in spherical proxy geometry.",
        "",
        "## Fatal flaw diagnosis (pure disformal + static dust)",
        "",
        "Old coupling: g̃ = g + α_τ ∂τ ∂τ with τ = ωt + σ(r) and static dust gives",
        "",
        "```text",
        "J^0 = ρ_b ω,   J^r = 0",
        "div J = (1/r²) d/dr (r² J^r) = 0",
        "```",
        "",
        "So **pure disformal coupling cannot source** a spatial τ halo for a static galaxy.",
        "",
        "## Corrected conformal source",
        "",
        "```text",
        "g̃ = A(τ)² g + α_τ ∂τ ∂τ",
        "A(τ) ≈ exp(β τ / M)",
        "T ≈ −ρ_b",
        "S_b ∝ (β/M) ρ_b",
        "```",
        "",
        "## K-essence radial equation",
        "",
        "```text",
        "(1/r²) d/dr [ r² μ(|σ'|/a₀) σ' ] = S_b(r)",
        "r² μ(|σ'|/a₀) σ' = ∫₀ʳ S_b(r') r'² dr'",
        "```",
        "",
        "**Deep-MOND limit:** μ(y)=y ⇒ σ' ∝ 1/r ⇒ r·a_τ ≈ const (flat rotation proxy).",
        "",
        f"- **Cases:** {len(results)} | **Expected outcomes matched:** {n_ok}/{len(results)}",
        "",
        "## Results",
        "",
        "| Case | coupling | μ | source | σ' slope | flatness | stable | physical | expect | matched |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in results:
        if r.expected_status == "pass":
            physical = "pass" if r.pass_criteria_met else "fail"
        else:
            physical = "fail (detected)" if r.failure_detected else "fail (missed)"
        stable = "yes" if _stability_viable(r) else "no"
        lines.append(
            f"| {r.case_name} | {r.coupling_type} | {r.mu_model} | "
            f"{'yes' if r.source_nonzero else 'no'} | {r.outer_gradient_slope:.2f} | "
            f"{r.rotation_flatness_score:.2f} | {stable} | {physical} | "
            f"{r.expected_status} | {'✓' if r.expected_failure_matched else '✗'} |",
        )
    lines.extend(
        [
            "",
            "### Failure detection (expected-fail cases)",
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
            "- **Passing** means the corrected source can produce a MOND-like τ gradient "
            "in controlled spherical benchmarks.",
            "- Does **not** observationally validate TDF.",
            "- Does **not** replace SPARC fitting or prove the full 5D action.",
            "- Conformal source is a **viability correction** pending derivation from geometry.",
            "",
            "## Failure modes",
            "",
            "- Spherical approximation only; disk geometry not modeled.",
            "- No real SPARC data; no lensing consistency; no relativistic perturbation theory.",
            "",
            "## Disclaimer",
            "",
            f"- {BANNER_CALIBRATION}",
            "",
        ],
    )
    return "\n".join(lines)


def run_kessence_source_viability_benchmark(
    outputs_root: Path | None = None,
) -> tuple[pd.DataFrame, list[KessenceSourceCaseResult]]:
    """Run Phase 7A; write CSV, report, figures."""
    root = Path(__file__).resolve().parents[3]
    outputs = Path(outputs_root or root / "outputs")
    tables = outputs / "tables"
    reports = outputs / "reports"
    figures = outputs / "figures"
    for d in (tables, reports, figures):
        d.mkdir(parents=True, exist_ok=True)

    results = [
        run_pure_disformal_static_dust_expected_fail(),
        run_conformal_canonical_newtonian(),
        run_conformal_kessence_deep_mond(),
        run_conformal_kessence_simple_interpolation(),
        run_insufficient_coupling_expected_fail(),
        run_excessive_gradient_stability_fail(),
        run_compact_bulge_deep_mond_profile(),
    ]

    df = pd.DataFrame([_result_to_row(r) for r in results])
    df.to_csv(tables / "kessence_source_viability_summary.csv", index=False)
    (reports / "kessence_source_viability_report.md").write_text(
        _build_report(results),
        encoding="utf-8",
    )

    _plot_source_terms(results, figures / "kessence_source_terms.png")
    _plot_profiles(
        results,
        figures / "kessence_tau_gradient_profiles.png",
        "sigma_prime",
        "|σ'(r)|",
        "τ gradient σ'(r)",
    )
    _plot_profiles(
        results,
        figures / "kessence_rotation_proxy.png",
        "v2_total",
        "v_total²",
        "Rotation proxy v²",
    )
    _plot_outer_slope(results, figures / "kessence_outer_slope.png")
    _plot_stability(results, figures / "kessence_stability_proxy.png")

    return df, results

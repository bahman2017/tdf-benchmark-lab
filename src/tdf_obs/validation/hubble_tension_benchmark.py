"""Phase 5F — CMB-safe Hubble tension benchmark (not observational validation)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

from tdf_obs.validation.cmb_acoustic_benchmark import (
    CosmologyParams,
    H_lcdm,
    acoustic_scale,
    comoving_distance,
    sound_horizon_proxy,
)

BENCHMARK_MODE = "cmb_safe_hubble_tension_benchmark"
BANNER_HUBBLE = "CMB-SAFE HUBBLE TENSION BENCHMARK — NOT REAL OBSERVATIONAL DATA"

DEFAULT_CMB_THRESHOLD_PERCENT = 1.0
HUBBLE_SHIFT_MIN_PERCENT = 2.0
HUBBLE_SHIFT_MAX_PERCENT = 10.0

LOW_Z_SAMPLES = (0.0, 0.5, 1.0, 2.0)

REQUIRED_SUMMARY_COLUMNS: tuple[str, ...] = (
    "case_name",
    "tau_model",
    "H0_lcdm",
    "H0_tdf",
    "H0_shift_percent",
    "H_zstar_error_percent",
    "r_s_error_percent",
    "D_M_error_percent",
    "ell_A_error_percent",
    "H_z_0p5_shift_percent",
    "H_z_1_shift_percent",
    "H_z_2_shift_percent",
    "cmb_safe_pass",
    "hubble_shift_active",
    "overall_success",
    "expected_status",
    "warnings",
)


@dataclass(frozen=True)
class HubbleTensionCase:
    case_name: str
    tau_model: str
    tau_params: dict[str, float]
    expected_status: str  # success | cmb_safe_only | fail_cmb | fail_shift
    description: str = ""


@dataclass
class HubbleShiftMetrics:
    H0_lcdm: float
    H0_tdf: float
    H0_shift_percent: float
    H_z_0p5_shift_percent: float
    H_z_1_shift_percent: float
    H_z_2_shift_percent: float


@dataclass
class CmbSafetyMetrics:
    H_zstar_lcdm: float
    H_zstar_tdf: float
    H_zstar_error_percent: float
    r_s_lcdm: float
    r_s_tdf: float
    r_s_error_percent: float
    D_M_lcdm: float
    D_M_tdf: float
    D_M_error_percent: float
    ell_A_lcdm: float
    ell_A_tdf: float
    ell_A_error_percent: float


@dataclass
class HubbleTensionResult:
    case_name: str
    tau_model: str
    shift: HubbleShiftMetrics
    cmb: CmbSafetyMetrics
    cmb_safe_pass: bool
    hubble_shift_active: bool
    overall_success: bool
    expected_status: str
    warnings: list[str] = field(default_factory=list)


BENCHMARK_CASE_REGISTRY: dict[str, HubbleTensionCase] = {
    "lcdm_control": HubbleTensionCase(
        case_name="lcdm_control",
        tau_model="zero_tau",
        tau_params={},
        expected_status="cmb_safe_only",
        description="ΛCDM control — CMB-safe, no H₀ shift",
    ),
    "zero_tau": HubbleTensionCase(
        case_name="zero_tau",
        tau_model="zero_tau",
        tau_params={},
        expected_status="cmb_safe_only",
        description="ε_τ = 0",
    ),
    "late_mild_3_percent": HubbleTensionCase(
        case_name="late_mild_3_percent",
        tau_model="late_time_tanh_mild",
        tau_params={"A": 0.062, "z_transition": 0.5, "width": 0.28},
        expected_status="success",
        description="~3% H₀ shift, sharp late-time tanh (low-z only)",
    ),
    "late_moderate_6_percent": HubbleTensionCase(
        case_name="late_moderate_6_percent",
        tau_model="late_time_tanh_mild",
        tau_params={"A": 0.125, "z_transition": 0.5, "width": 0.28},
        expected_status="success",
        description="~6% H₀ shift, sharp late-time tanh",
    ),
    "late_strong_10_percent": HubbleTensionCase(
        case_name="late_strong_10_percent",
        tau_model="sharp_low_z_transition",
        tau_params={"A": 0.205, "z_cut": 0.3},
        expected_status="success",
        description="~10% H₀ shift, sharp low-z Gaussian (CMB-safe profile)",
    ),
    "sharp_low_z_5_percent": HubbleTensionCase(
        case_name="sharp_low_z_5_percent",
        tau_model="sharp_low_z_transition",
        tau_params={"A": 0.102, "z_cut": 0.3},
        expected_status="success",
        description="Sharp activation below z ≈ 0.3 (~5% H₀ shift)",
    ),
    "early_leakage_bad": HubbleTensionCase(
        case_name="early_leakage_bad",
        tau_model="early_leakage_bad",
        tau_params={"A": 0.09, "z_transition": 600.0, "width": 250.0},
        expected_status="fail_cmb",
        description="Late model with high-z leakage (intentional fail)",
    ),
    "recombination_leakage_bad": HubbleTensionCase(
        case_name="recombination_leakage_bad",
        tau_model="recombination_leakage_bad",
        tau_params={"A": 0.14, "sigma": 55.0},
        expected_status="fail_cmb",
        description="Bump near z_* (intentional fail)",
    ),
}


def list_benchmark_cases() -> list[str]:
    return list(BENCHMARK_CASE_REGISTRY.keys())


def get_benchmark_case(name: str) -> HubbleTensionCase:
    if name not in BENCHMARK_CASE_REGISTRY:
        raise KeyError(f"Unknown Hubble case {name!r}; available: {list_benchmark_cases()}")
    return BENCHMARK_CASE_REGISTRY[name]


def epsilon_tau_hubble_model(
    z: np.ndarray | float,
    model_name: str,
    tau_params: dict[str, float],
    *,
    z_star: float,
) -> np.ndarray:
    """ε_τ(z) models for Phase 5F late-time / leakage tests."""
    z = np.asarray(z, dtype=float)
    name = model_name.lower()

    if name == "zero_tau":
        return np.zeros_like(z)

    if name == "constant_tau_small":
        return np.full_like(z, float(tau_params.get("epsilon0", 1e-5)))

    if name in ("late_time_tanh_mild", "late_time_tanh_strong", "early_leakage_bad"):
        A = float(tau_params.get("A", 0.0))
        z_tr = float(tau_params.get("z_transition", 2.0))
        width = max(float(tau_params.get("width", 1.0)), 1e-6)
        return A * 0.5 * (1.0 - np.tanh((z - z_tr) / width))

    if name == "late_time_gaussian":
        A = float(tau_params.get("A", 0.0))
        sigma = max(float(tau_params.get("sigma", 1.0)), 1e-6)
        return A * np.exp(-0.5 * (z / sigma) ** 2)

    if name == "sharp_low_z_transition":
        A = float(tau_params.get("A", 0.0))
        z_cut = max(float(tau_params.get("z_cut", 0.3)), 1e-4)
        return A * np.exp(-0.5 * (z / z_cut) ** 2)

    if name == "recombination_leakage_bad":
        A = float(tau_params.get("A", 0.0))
        sigma = max(float(tau_params.get("sigma", 50.0)), 1e-6)
        z_s = float(tau_params.get("z_star", z_star))
        return A * np.exp(-0.5 * ((z - z_s) / sigma) ** 2)

    raise ValueError(f"Unknown Hubble tau model {model_name!r}")


def H_tdf(
    z: np.ndarray | float,
    params: CosmologyParams,
    tau_model: str,
    tau_params: dict[str, float],
) -> np.ndarray:
    """H_TDF² = H_LCDM² [1 + ε_τ(z)]."""
    h = H_lcdm(z, params)
    eps = epsilon_tau_hubble_model(z, tau_model, tau_params, z_star=params.z_star)
    return h * np.sqrt(np.maximum(1.0 + eps, 1e-12))


def _rel_error_percent(true: float, approx: float) -> float:
    denom = max(abs(true), 1e-30)
    return float(abs(approx - true) / denom * 100.0)


def _H_shift_percent(H_lcdm_val: float, H_tdf_val: float) -> float:
    denom = max(abs(H_lcdm_val), 1e-30)
    return float((H_tdf_val - H_lcdm_val) / denom * 100.0)


def compute_hubble_shift_metrics(
    params: CosmologyParams,
    tau_model: str,
    tau_params: dict[str, float],
) -> HubbleShiftMetrics:
    """Low-z H(z) shifts and effective H₀ shift at z=0."""
    H0_l = float(H_lcdm(0.0, params))
    H0_t = float(H_tdf(0.0, params, tau_model, tau_params))

    shifts: dict[float, float] = {}
    for z_s in LOW_Z_SAMPLES:
        if z_s == 0.0:
            continue
        H_l = float(H_lcdm(z_s, params))
        H_t = float(H_tdf(z_s, params, tau_model, tau_params))
        shifts[z_s] = _H_shift_percent(H_l, H_t)

    return HubbleShiftMetrics(
        H0_lcdm=H0_l,
        H0_tdf=H0_t,
        H0_shift_percent=_H_shift_percent(H0_l, H0_t),
        H_z_0p5_shift_percent=shifts[0.5],
        H_z_1_shift_percent=shifts[1.0],
        H_z_2_shift_percent=shifts[2.0],
    )


def compute_cmb_safety_metrics(
    params: CosmologyParams,
    tau_model: str,
    tau_params: dict[str, float],
) -> CmbSafetyMetrics:
    """Recombination-era proxy comparison vs ΛCDM teacher."""
    H_lcdm_func = lambda z: H_lcdm(z, params)
    H_tdf_func = lambda z: H_tdf(z, params, tau_model, tau_params)

    z_s = params.z_star
    H_z_l = float(H_lcdm(z_s, params))
    H_z_t = float(H_tdf(z_s, params, tau_model, tau_params))

    rs_l = sound_horizon_proxy(z_s, params.z_max_horizon, H_lcdm_func, params)
    rs_t = sound_horizon_proxy(z_s, params.z_max_horizon, H_tdf_func, params)
    DM_l = comoving_distance(z_s, H_lcdm_func)
    DM_t = comoving_distance(z_s, H_tdf_func)
    ell_l = acoustic_scale(DM_l, rs_l)
    ell_t = acoustic_scale(DM_t, rs_t)

    return CmbSafetyMetrics(
        H_zstar_lcdm=H_z_l,
        H_zstar_tdf=H_z_t,
        H_zstar_error_percent=_rel_error_percent(H_z_l, H_z_t),
        r_s_lcdm=rs_l,
        r_s_tdf=rs_t,
        r_s_error_percent=_rel_error_percent(rs_l, rs_t),
        D_M_lcdm=DM_l,
        D_M_tdf=DM_t,
        D_M_error_percent=_rel_error_percent(DM_l, DM_t),
        ell_A_lcdm=ell_l,
        ell_A_tdf=ell_t,
        ell_A_error_percent=_rel_error_percent(ell_l, ell_t),
    )


def classify_hubble_case(
    shift: HubbleShiftMetrics,
    cmb: CmbSafetyMetrics,
    *,
    cmb_threshold_percent: float = DEFAULT_CMB_THRESHOLD_PERCENT,
    hubble_min_percent: float = HUBBLE_SHIFT_MIN_PERCENT,
    hubble_max_percent: float = HUBBLE_SHIFT_MAX_PERCENT,
) -> tuple[bool, bool, bool]:
    """
    Return (cmb_safe_pass, hubble_shift_active, overall_success).

    overall_success = cmb_safe and H₀ shift in [hubble_min, hubble_max] (absolute value).
    """
    cmb_safe = (
        cmb.H_zstar_error_percent < cmb_threshold_percent
        and cmb.r_s_error_percent < cmb_threshold_percent
        and cmb.D_M_error_percent < cmb_threshold_percent
        and cmb.ell_A_error_percent < cmb_threshold_percent
    )
    h0_abs = abs(shift.H0_shift_percent)
    hubble_active = hubble_min_percent <= h0_abs <= hubble_max_percent
    overall = cmb_safe and hubble_active
    return cmb_safe, hubble_active, overall


def run_single_hubble_case(
    case: HubbleTensionCase,
    params: CosmologyParams | None = None,
    *,
    cmb_threshold_percent: float = DEFAULT_CMB_THRESHOLD_PERCENT,
    hubble_min_percent: float = HUBBLE_SHIFT_MIN_PERCENT,
    hubble_max_percent: float = HUBBLE_SHIFT_MAX_PERCENT,
) -> HubbleTensionResult:
    """Run one Phase 5F benchmark case."""
    params = params or CosmologyParams()
    warnings: list[str] = []

    shift = compute_hubble_shift_metrics(params, case.tau_model, case.tau_params)
    cmb = compute_cmb_safety_metrics(params, case.tau_model, case.tau_params)
    cmb_safe, hubble_active, overall = classify_hubble_case(
        shift,
        cmb,
        cmb_threshold_percent=cmb_threshold_percent,
        hubble_min_percent=hubble_min_percent,
        hubble_max_percent=hubble_max_percent,
    )

    if case.expected_status == "success" and not overall:
        warnings.append("Expected overall success but case did not meet both criteria.")
    elif case.expected_status == "cmb_safe_only" and not cmb_safe:
        warnings.append("Expected CMB-safe control but CMB proxies failed.")
    elif case.expected_status == "cmb_safe_only" and hubble_active:
        warnings.append("Control case shows unexpected Hubble-shift activity.")
    elif case.expected_status == "fail_cmb" and cmb_safe:
        warnings.append("Expected CMB failure but case passed CMB-safe checks.")
    elif case.expected_status == "fail_shift" and overall:
        warnings.append("Expected shift failure but case succeeded overall.")

    for label, val in [
        ("H0_lcdm", shift.H0_lcdm),
        ("H0_tdf", shift.H0_tdf),
        ("r_s_lcdm", cmb.r_s_lcdm),
        ("ell_A_lcdm", cmb.ell_A_lcdm),
    ]:
        if not np.isfinite(val) or val <= 0:
            warnings.append(f"Non-finite or non-positive {label}={val}")

    return HubbleTensionResult(
        case_name=case.case_name,
        tau_model=case.tau_model,
        shift=shift,
        cmb=cmb,
        cmb_safe_pass=cmb_safe,
        hubble_shift_active=hubble_active,
        overall_success=overall,
        expected_status=case.expected_status,
        warnings=warnings,
    )


def _plot_epsilon_cases(
    cases: Sequence[HubbleTensionCase],
    params: CosmologyParams,
    output_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    z = np.concatenate([
        np.linspace(0.0, 5.0, 200),
        np.linspace(5.0, params.z_star * 1.02, 150),
    ])

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for case in cases:
        eps = epsilon_tau_hubble_model(z, case.tau_model, case.tau_params, z_star=params.z_star)
        ax.plot(z, eps, label=case.case_name, lw=1.5)

    ax.axvline(params.z_star, color="k", ls="--", alpha=0.35, label="z_*")
    ax.set_xlabel("z")
    ax.set_ylabel("ε_τ(z)")
    ax.set_title(f"Late-time ε_τ — {BANNER_HUBBLE[:50]}...")
    ax.legend(fontsize=6, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_H_ratio_cases(
    cases: Sequence[HubbleTensionCase],
    params: CosmologyParams,
    output_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    z = np.linspace(0.0, min(50.0, params.z_star * 0.05), 300)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for case in cases:
        H_l = H_lcdm(z, params)
        H_t = H_tdf(z, params, case.tau_model, case.tau_params)
        ratio = H_t / np.maximum(H_l, 1e-30)
        ax.plot(z, ratio, label=case.case_name, lw=1.5)

    ax.axhline(1.0, color="k", ls=":", alpha=0.5)
    ax.set_xlabel("z")
    ax.set_ylabel("H_TDF / H_ΛCDM")
    ax.set_title("Background H ratio (low–mid z)")
    ax.legend(fontsize=6, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _result_to_row(res: HubbleTensionResult) -> dict[str, Any]:
    s, c = res.shift, res.cmb
    return {
        "case_name": res.case_name,
        "tau_model": res.tau_model,
        "H0_lcdm": s.H0_lcdm,
        "H0_tdf": s.H0_tdf,
        "H0_shift_percent": s.H0_shift_percent,
        "H_zstar_error_percent": c.H_zstar_error_percent,
        "r_s_error_percent": c.r_s_error_percent,
        "D_M_error_percent": c.D_M_error_percent,
        "ell_A_error_percent": c.ell_A_error_percent,
        "H_z_0p5_shift_percent": s.H_z_0p5_shift_percent,
        "H_z_1_shift_percent": s.H_z_1_shift_percent,
        "H_z_2_shift_percent": s.H_z_2_shift_percent,
        "cmb_safe_pass": res.cmb_safe_pass,
        "hubble_shift_active": res.hubble_shift_active,
        "overall_success": res.overall_success,
        "expected_status": res.expected_status,
        "warnings": "; ".join(res.warnings),
        "dataset_mode": BENCHMARK_MODE,
        "is_real_observational_data": False,
    }


def _build_report(
    results: list[HubbleTensionResult],
    *,
    cmb_threshold_percent: float,
    hubble_min_percent: float,
    hubble_max_percent: float,
) -> str:
    n = len(results)
    n_cmb = sum(1 for r in results if r.cmb_safe_pass)
    n_hub = sum(1 for r in results if r.hubble_shift_active)
    n_ok = sum(1 for r in results if r.overall_success)
    intentional = [r for r in results if r.expected_status == "fail_cmb"]
    worst_ell = max(results, key=lambda r: r.cmb.ell_A_error_percent)

    lines = [
        "# CMB-safe Hubble tension benchmark report (Phase 5F)",
        "",
        f"## ⚠️ {BANNER_HUBBLE}",
        "",
        "## Purpose",
        "",
        "Check whether **late-time** TDF background corrections `ε_τ(z)` can shift "
        "low-redshift expansion (effective H₀ and H(z)) while preserving **CMB acoustic-scale "
        "proxies** at z_*.",
        "",
        "> **NOT REAL OBSERVATIONAL DATA.** Does **not** solve the Hubble tension.",
        "",
        "## Equations",
        "",
        "```text",
        "H_ΛCDM²(z) = H0² [Ω_r(1+z)⁴ + Ω_m(1+z)³ + Ω_Λ]",
        "H_TDF²(z) = H_ΛCDM²(z) [1 + ε_τ(z)]",
        "r_s = ∫_{z_*}^{z_max} c_s/H dz",
        "D_M = ∫_0^{z_*} c/H dz",
        "ℓ_A = π D_M / r_s",
        "```",
        "",
        f"**CMB-safe:** each of H(z_*), r_s, D_M, ℓ_A errors < {cmb_threshold_percent}%.",
        f"**Hubble-shift-active:** |H₀ shift| in [{hubble_min_percent}, {hubble_max_percent}]%.",
        "**Overall success:** both.",
        "",
        "## Summary",
        "",
        f"- **Cases:** {n}",
        f"- **CMB-safe pass:** {n_cmb} / {n}",
        f"- **Hubble-shift-active:** {n_hub} / {n}",
        f"- **Overall success:** {n_ok} / {n}",
        f"- **Intentional fail cases:** {len(intentional)} "
        f"({', '.join(r.case_name for r in intentional) or 'none'})",
        f"- **Worst ℓ_A error:** {worst_ell.case_name} ({worst_ell.cmb.ell_A_error_percent:.4f}%)",
        "",
        "## Per-case table",
        "",
        "| Case | H₀ shift % | ℓ_A err % | CMB safe | H shift active | Overall | expected |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for res in results:
        lines.append(
            f"| {res.case_name} | {res.shift.H0_shift_percent:.3f} | "
            f"{res.cmb.ell_A_error_percent:.4f} | "
            f"{'✓' if res.cmb_safe_pass else '✗'} | "
            f"{'✓' if res.hubble_shift_active else '✗'} | "
            f"{'✓' if res.overall_success else '✗'} | {res.expected_status} |",
        )

    lines.extend(
        [
            "",
            "## Scientific interpretation",
            "",
            "- **Overall success** means only that the **configured** late-time ε_τ(z) preserves "
            "background acoustic proxies while shifting low-z H(z) in this ΛCDM-teacher scaffold.",
            "- This does **not** solve the Hubble tension or validate TDF against SH0ES, BAO, SNe, or CMB.",
            "- Real tests require joint likelihoods and **perturbation** evolution.",
            "",
            "## Failure modes",
            "",
            "- Background-only proxy; no Boltzmann perturbations.",
            "- No Planck/ACT/SPT, SH0ES, BAO, Pantheon, or cosmic-chronometer data.",
            "- No growth-of-structure constraints.",
            "- High-z leakage in ε_τ(z) can pass H₀ shift but fail CMB-safe (by design in fail cases).",
            "",
            "## Disclaimer",
            "",
            "- **NOT REAL OBSERVATIONAL DATA**",
            "- Controlled ΛCDM teacher benchmark only.",
            "",
        ],
    )
    return "\n".join(lines)


def run_hubble_tension_benchmark(
    cases: Sequence[str] | None = None,
    outputs_root: Path | None = None,
    *,
    cosmology: CosmologyParams | None = None,
    cmb_threshold_percent: float = DEFAULT_CMB_THRESHOLD_PERCENT,
    hubble_min_percent: float = HUBBLE_SHIFT_MIN_PERCENT,
    hubble_max_percent: float = HUBBLE_SHIFT_MAX_PERCENT,
) -> tuple[pd.DataFrame, list[HubbleTensionResult]]:
    """Run Phase 5F and write CSV, report, and figures."""
    root = Path(__file__).resolve().parents[3]
    outputs = outputs_root or (root / "outputs")
    tables_dir = outputs / "tables"
    reports_dir = outputs / "reports"
    figures_dir = outputs / "figures"
    for d in (tables_dir, reports_dir, figures_dir):
        d.mkdir(parents=True, exist_ok=True)

    params = cosmology or CosmologyParams()
    names = list(cases) if cases is not None else list_benchmark_cases()
    case_objs = [get_benchmark_case(n) for n in names]

    results = [
        run_single_hubble_case(
            c,
            params,
            cmb_threshold_percent=cmb_threshold_percent,
            hubble_min_percent=hubble_min_percent,
            hubble_max_percent=hubble_max_percent,
        )
        for c in case_objs
    ]

    out_df = pd.DataFrame([_result_to_row(r) for r in results])
    out_df.to_csv(tables_dir / "hubble_tension_benchmark_summary.csv", index=False)
    (reports_dir / "hubble_tension_benchmark_report.md").write_text(
        _build_report(
            results,
            cmb_threshold_percent=cmb_threshold_percent,
            hubble_min_percent=hubble_min_percent,
            hubble_max_percent=hubble_max_percent,
        ),
        encoding="utf-8",
    )
    _plot_epsilon_cases(case_objs, params, figures_dir / "hubble_tension_epsilon_tau_cases.png")
    _plot_H_ratio_cases(case_objs, params, figures_dir / "hubble_tension_H_ratio_cases.png")
    return out_df, results

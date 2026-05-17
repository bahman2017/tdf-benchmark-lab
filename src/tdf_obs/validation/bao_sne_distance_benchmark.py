"""Phase 5G — BAO/SNe late-time distance consistency benchmark (not validation)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

from tdf_obs.validation.cmb_acoustic_benchmark import C_KM_S, CosmologyParams, H_lcdm
from tdf_obs.validation.cmb_acoustic_benchmark import _integrate_trapz
from tdf_obs.validation.hubble_tension_benchmark import (
    HUBBLE_SHIFT_MAX_PERCENT,
    HUBBLE_SHIFT_MIN_PERCENT,
    epsilon_tau_hubble_model,
)

BENCHMARK_MODE = "bao_sne_distance_consistency_benchmark"
BANNER_BAO_SNE = "BAO/SNe DISTANCE CONSISTENCY BENCHMARK — NOT REAL OBSERVATIONAL DATA"

DEFAULT_DL_THRESHOLD_PERCENT = 2.0
DEFAULT_DM_DV_THRESHOLD_PERCENT = 2.0
DEFAULT_H_BAO_THRESHOLD_PERCENT = 5.0

SNE_REDSHIFTS: tuple[float, ...] = (0.01, 0.05, 0.1, 0.3, 0.5, 1.0, 1.5)
BAO_REDSHIFTS: tuple[float, ...] = (0.35, 0.57, 0.8, 1.0, 1.5, 2.0)

REQUIRED_SUMMARY_COLUMNS: tuple[str, ...] = (
    "case_name",
    "tau_model",
    "H0_lcdm",
    "H0_tdf",
    "H0_shift_percent",
    "max_rel_error_D_L_percent",
    "max_rel_error_D_M_percent",
    "max_rel_error_D_V_percent",
    "max_rel_error_H_percent",
    "distance_safe_pass",
    "hubble_shift_active",
    "overall_success",
    "expected_status",
    "warnings",
)


@dataclass(frozen=True)
class BaoSneDistanceCase:
    case_name: str
    tau_model: str
    tau_params: dict[str, float]
    expected_status: str  # success | distance_safe_only | fail_distance | fail_other
    description: str = ""


@dataclass
class DistanceMetrics:
    H0_lcdm: float
    H0_tdf: float
    H0_shift_percent: float
    max_rel_error_D_L_percent: float
    max_rel_error_D_M_percent: float
    max_rel_error_D_V_percent: float
    max_rel_error_H_percent: float


@dataclass
class BaoSneDistanceResult:
    case_name: str
    tau_model: str
    metrics: DistanceMetrics
    distance_safe_pass: bool
    hubble_shift_active: bool
    overall_success: bool
    expected_status: str
    warnings: list[str] = field(default_factory=list)


BENCHMARK_CASE_REGISTRY: dict[str, BaoSneDistanceCase] = {
    "lcdm_control": BaoSneDistanceCase(
        case_name="lcdm_control",
        tau_model="zero_tau",
        tau_params={},
        expected_status="distance_safe_only",
        description="ΛCDM control — distance-safe, no H₀ shift",
    ),
    "zero_tau": BaoSneDistanceCase(
        case_name="zero_tau",
        tau_model="zero_tau",
        tau_params={},
        expected_status="distance_safe_only",
        description="ε_τ = 0",
    ),
    "late_mild_3_percent": BaoSneDistanceCase(
        case_name="late_mild_3_percent",
        tau_model="late_time_tanh_mild",
        tau_params={"A": 0.0424, "z_transition": 0.5, "width": 0.28},
        expected_status="success",
        description="~2% H₀ shift with distance proxies within 2% (narrow band)",
    ),
    "late_moderate_6_percent": BaoSneDistanceCase(
        case_name="late_moderate_6_percent",
        tau_model="late_time_tanh_mild",
        tau_params={"A": 0.125, "z_transition": 0.5, "width": 0.28},
        expected_status="hubble_active_only",
        description="~6% H₀ shift — distances exceed 2% tolerance (informative)",
    ),
    "sharp_low_z_5_percent": BaoSneDistanceCase(
        case_name="sharp_low_z_5_percent",
        tau_model="sharp_low_z_transition",
        tau_params={"A": 0.102, "z_cut": 0.3},
        expected_status="hubble_active_only",
        description="~5% H₀ shift — distance proxies strained",
    ),
    "late_strong_10_percent": BaoSneDistanceCase(
        case_name="late_strong_10_percent",
        tau_model="sharp_low_z_transition",
        tau_params={"A": 0.205, "z_cut": 0.3},
        expected_status="hubble_active_only",
        description="~10% H₀ shift — distance proxies exceed tolerance",
    ),
    "early_leakage_bad": BaoSneDistanceCase(
        case_name="early_leakage_bad",
        tau_model="early_leakage_bad",
        tau_params={"A": 0.09, "z_transition": 600.0, "width": 250.0},
        expected_status="fail_distance",
        description="High-z leakage — distances distorted",
    ),
    "recombination_leakage_bad": BaoSneDistanceCase(
        case_name="recombination_leakage_bad",
        tau_model="recombination_leakage_bad",
        tau_params={"A": 0.14, "sigma": 55.0},
        expected_status="fail_distance",
        description="Recombination bump — distances distorted",
    ),
    "distance_distorting_bad": BaoSneDistanceCase(
        case_name="distance_distorting_bad",
        tau_model="distance_distorting_bad",
        tau_params={"epsilon0": 0.22},
        expected_status="fail_distance",
        description="Large constant ε_τ — intentional distance fail",
    ),
}


def list_benchmark_cases() -> list[str]:
    return list(BENCHMARK_CASE_REGISTRY.keys())


def get_benchmark_case(name: str) -> BaoSneDistanceCase:
    if name not in BENCHMARK_CASE_REGISTRY:
        raise KeyError(f"Unknown BAO/SNe case {name!r}; available: {list_benchmark_cases()}")
    return BENCHMARK_CASE_REGISTRY[name]


def epsilon_tau_distance_model(
    z: np.ndarray | float,
    model_name: str,
    tau_params: dict[str, float],
    *,
    z_star: float,
) -> np.ndarray:
    """ε_τ(z) for distance benchmark (extends Phase 5F models)."""
    if model_name == "distance_distorting_bad":
        z_arr = np.asarray(z, dtype=float)
        return np.full_like(z_arr, float(tau_params.get("epsilon0", 0.22)))
    return epsilon_tau_hubble_model(z, model_name, tau_params, z_star=z_star)


def H_tdf(
    z: np.ndarray | float,
    params: CosmologyParams,
    tau_model: str,
    tau_params: dict[str, float],
) -> np.ndarray:
    """H_TDF² = H_LCDM² [1 + ε_τ(z)]."""
    h = H_lcdm(z, params)
    eps = epsilon_tau_distance_model(z, tau_model, tau_params, z_star=params.z_star)
    return h * np.sqrt(np.maximum(1.0 + eps, 1e-12))


def E_ratio(z: np.ndarray | float, H_func: Callable[[np.ndarray], np.ndarray], H0: float) -> np.ndarray:
    """E(z) = H(z)/H0."""
    z = np.asarray(z, dtype=float)
    return H_func(z) / max(float(H0), 1e-30)


def comoving_distance(
    z: float,
    H_func: Callable[[np.ndarray], np.ndarray],
    *,
    n_steps: int | None = None,
) -> float:
    """D_C(z) = D_M(z) = ∫_0^z c/H(z') dz' [Mpc] (flat universe)."""
    z_val = float(z)
    if z_val <= 0.0:
        return 0.0
    n = n_steps or max(300, int(500 * z_val))
    z_grid = np.linspace(0.0, z_val, n)
    H = np.maximum(H_func(z_grid), 1e-30)
    return _integrate_trapz(z_grid, C_KM_S / H)


def luminosity_distance(
    z: float,
    H_func: Callable[[np.ndarray], np.ndarray],
) -> float:
    """D_L(z) = (1+z) D_M(z)."""
    dm = comoving_distance(z, H_func)
    return (1.0 + z) * dm


def angular_diameter_distance(
    z: float,
    H_func: Callable[[np.ndarray], np.ndarray],
) -> float:
    """D_A(z) = D_M(z)/(1+z)."""
    dm = comoving_distance(z, H_func)
    return dm / (1.0 + z)


def bao_DV_proxy(
    z: float,
    H_func: Callable[[np.ndarray], np.ndarray],
) -> float:
    """D_V(z) = [z D_M² c/H(z)]^(1/3) [Mpc]."""
    z_val = float(z)
    if z_val <= 0.0:
        return 0.0
    dm = comoving_distance(z_val, H_func)
    H_z = float(H_func(z_val))
    inner = z_val * dm**2 * C_KM_S / max(H_z, 1e-30)
    return float(inner ** (1.0 / 3.0))


def _rel_error_percent(true: float, approx: float) -> float:
    denom = max(abs(true), 1e-30)
    return float(abs(approx - true) / denom * 100.0)


def _max_rel_error_percent(true_vals: np.ndarray, approx_vals: np.ndarray) -> float:
    denom = np.maximum(np.abs(true_vals), 1e-30)
    return float(np.max(np.abs(approx_vals - true_vals) / denom) * 100.0)


def compute_distance_metrics(
    params: CosmologyParams,
    tau_model: str,
    tau_params: dict[str, float],
) -> DistanceMetrics:
    """Compare TDF vs ΛCDM distance proxies at SNe- and BAO-like redshifts."""
    H_lcdm_func = lambda z: H_lcdm(z, params)
    H_tdf_func = lambda z: H_tdf(z, params, tau_model, tau_params)

    H0_l = float(H_lcdm(0.0, params))
    H0_t = float(H_tdf(0.0, params, tau_model, tau_params))
    H0_shift = _rel_error_percent(H0_l, H0_t)

    z_sne = np.array(SNE_REDSHIFTS, dtype=float)
    z_bao = np.array(BAO_REDSHIFTS, dtype=float)

    DL_l = np.array([luminosity_distance(z, H_lcdm_func) for z in z_sne])
    DL_t = np.array([luminosity_distance(z, H_tdf_func) for z in z_sne])

    DM_l = np.array([comoving_distance(z, H_lcdm_func) for z in z_bao])
    DM_t = np.array([comoving_distance(z, H_tdf_func) for z in z_bao])

    DV_l = np.array([bao_DV_proxy(z, H_lcdm_func) for z in z_bao])
    DV_t = np.array([bao_DV_proxy(z, H_tdf_func) for z in z_bao])

    H_l = H_lcdm(z_bao, params)
    H_t = H_tdf(z_bao, params, tau_model, tau_params)

    return DistanceMetrics(
        H0_lcdm=H0_l,
        H0_tdf=H0_t,
        H0_shift_percent=H0_shift,
        max_rel_error_D_L_percent=_max_rel_error_percent(DL_l, DL_t),
        max_rel_error_D_M_percent=_max_rel_error_percent(DM_l, DM_t),
        max_rel_error_D_V_percent=_max_rel_error_percent(DV_l, DV_t),
        max_rel_error_H_percent=_max_rel_error_percent(H_l, H_t),
    )


def classify_distance_case(
    metrics: DistanceMetrics,
    *,
    dl_threshold_percent: float = DEFAULT_DL_THRESHOLD_PERCENT,
    dm_dv_threshold_percent: float = DEFAULT_DM_DV_THRESHOLD_PERCENT,
    h_bao_threshold_percent: float = DEFAULT_H_BAO_THRESHOLD_PERCENT,
    hubble_min_percent: float = HUBBLE_SHIFT_MIN_PERCENT,
    hubble_max_percent: float = HUBBLE_SHIFT_MAX_PERCENT,
) -> tuple[bool, bool, bool]:
    """Return (distance_safe_pass, hubble_shift_active, overall_success)."""
    distance_safe = (
        metrics.max_rel_error_D_L_percent < dl_threshold_percent
        and metrics.max_rel_error_D_M_percent < dm_dv_threshold_percent
        and metrics.max_rel_error_D_V_percent < dm_dv_threshold_percent
        and metrics.max_rel_error_H_percent < h_bao_threshold_percent
    )
    h0_abs = abs(metrics.H0_shift_percent)
    hubble_active = hubble_min_percent <= h0_abs <= hubble_max_percent
    overall = distance_safe and hubble_active
    return distance_safe, hubble_active, overall


def run_single_distance_case(
    case: BaoSneDistanceCase,
    params: CosmologyParams | None = None,
    **threshold_kwargs: Any,
) -> BaoSneDistanceResult:
    """Run one Phase 5G case."""
    params = params or CosmologyParams()
    warnings: list[str] = []

    metrics = compute_distance_metrics(params, case.tau_model, case.tau_params)
    dist_safe, hub_active, overall = classify_distance_case(metrics, **threshold_kwargs)

    if case.expected_status == "success" and not overall:
        warnings.append("Expected overall success but criteria not met.")
    elif case.expected_status == "hubble_active_only":
        if not hub_active:
            warnings.append("Expected Hubble-shift-active but |H₀ shift| outside [2%, 10%].")
        if dist_safe:
            warnings.append(
                "Expected distance proxies to exceed tolerance at this ε_τ amplitude; "
                "case passed distance-safe unexpectedly.",
            )
    elif case.expected_status == "distance_safe_only" and not dist_safe:
        warnings.append("Expected distance-safe control but distance proxies failed.")
    elif case.expected_status == "distance_safe_only" and hub_active:
        warnings.append("Control case shows unexpected Hubble-shift activity.")
    elif case.expected_status == "fail_distance" and dist_safe:
        warnings.append("Expected distance failure but case passed distance-safe checks.")

    return BaoSneDistanceResult(
        case_name=case.case_name,
        tau_model=case.tau_model,
        metrics=metrics,
        distance_safe_pass=dist_safe,
        hubble_shift_active=hub_active,
        overall_success=overall,
        expected_status=case.expected_status,
        warnings=warnings,
    )


def _plot_H_ratio(
    cases: Sequence[BaoSneDistanceCase],
    params: CosmologyParams,
    output_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    z = np.linspace(0.0, 2.2, 300)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for case in cases:
        H_l = H_lcdm(z, params)
        H_t = H_tdf(z, params, case.tau_model, case.tau_params)
        ax.plot(z, H_t / np.maximum(H_l, 1e-30), label=case.case_name, lw=1.4)
    ax.axhline(1.0, color="k", ls=":", alpha=0.5)
    ax.set_xlabel("z")
    ax.set_ylabel("H_TDF / H_ΛCDM")
    ax.set_title("Background H ratio")
    ax.legend(fontsize=6, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_DL_ratio(
    cases: Sequence[BaoSneDistanceCase],
    params: CosmologyParams,
    output_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    z_sne = np.array(SNE_REDSHIFTS, dtype=float)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for case in cases:
        H_l = lambda z: H_lcdm(z, params)
        H_t = lambda z: H_tdf(z, params, case.tau_model, case.tau_params)
        DL_l = np.array([luminosity_distance(z, H_l) for z in z_sne])
        DL_t = np.array([luminosity_distance(z, H_t) for z in z_sne])
        ax.plot(z_sne, DL_t / np.maximum(DL_l, 1e-30), "o-", ms=4, label=case.case_name, lw=1.2)
    ax.axhline(1.0, color="k", ls=":", alpha=0.5)
    ax.set_xlabel("z (SNe-like samples)")
    ax.set_ylabel("D_L^TDF / D_L^ΛCDM")
    ax.legend(fontsize=6, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_DV_ratio(
    cases: Sequence[BaoSneDistanceCase],
    params: CosmologyParams,
    output_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    z_bao = np.array(BAO_REDSHIFTS, dtype=float)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for case in cases:
        H_l = lambda z: H_lcdm(z, params)
        H_t = lambda z: H_tdf(z, params, case.tau_model, case.tau_params)
        DV_l = np.array([bao_DV_proxy(z, H_l) for z in z_bao])
        DV_t = np.array([bao_DV_proxy(z, H_t) for z in z_bao])
        ax.plot(z_bao, DV_t / np.maximum(DV_l, 1e-30), "s-", ms=4, label=case.case_name, lw=1.2)
    ax.axhline(1.0, color="k", ls=":", alpha=0.5)
    ax.set_xlabel("z (BAO-like samples)")
    ax.set_ylabel("D_V^TDF / D_V^ΛCDM")
    ax.legend(fontsize=6, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _result_to_row(res: BaoSneDistanceResult) -> dict[str, Any]:
    m = res.metrics
    return {
        "case_name": res.case_name,
        "tau_model": res.tau_model,
        "H0_lcdm": m.H0_lcdm,
        "H0_tdf": m.H0_tdf,
        "H0_shift_percent": m.H0_shift_percent,
        "max_rel_error_D_L_percent": m.max_rel_error_D_L_percent,
        "max_rel_error_D_M_percent": m.max_rel_error_D_M_percent,
        "max_rel_error_D_V_percent": m.max_rel_error_D_V_percent,
        "max_rel_error_H_percent": m.max_rel_error_H_percent,
        "distance_safe_pass": res.distance_safe_pass,
        "hubble_shift_active": res.hubble_shift_active,
        "overall_success": res.overall_success,
        "expected_status": res.expected_status,
        "warnings": "; ".join(res.warnings),
        "dataset_mode": BENCHMARK_MODE,
        "is_real_observational_data": False,
    }


def _build_report(
    results: list[BaoSneDistanceResult],
    *,
    dl_threshold_percent: float,
    dm_dv_threshold_percent: float,
    h_bao_threshold_percent: float,
    hubble_min_percent: float,
    hubble_max_percent: float,
) -> str:
    n = len(results)
    n_dist = sum(1 for r in results if r.distance_safe_pass)
    n_hub = sum(1 for r in results if r.hubble_shift_active)
    n_ok = sum(1 for r in results if r.overall_success)
    intentional = [r for r in results if r.expected_status == "fail_distance"]
    worst_dl = max(results, key=lambda r: r.metrics.max_rel_error_D_L_percent)

    lines = [
        "# BAO/SNe distance consistency report (Phase 5G)",
        "",
        f"## ⚠️ {BANNER_BAO_SNE}",
        "",
        "## Purpose",
        "",
        "Check whether **late-time** TDF background corrections `ε_τ(z)` that shift H₀ "
        "also preserve **BAO/SNe-like distance proxies** relative to a ΛCDM teacher.",
        "",
        "> **NOT REAL OBSERVATIONAL DATA.** Does **not** solve the Hubble tension or validate BAO/SNe.",
        "",
        "## Equations",
        "",
        "```text",
        "H_TDF²(z) = H_ΛCDM²(z) [1 + ε_τ(z)]",
        "D_C(z) = ∫_0^z c/H(z') dz' = D_M(z)   (flat)",
        "D_L(z) = (1+z) D_M(z)",
        "D_A(z) = D_M(z)/(1+z)",
        "D_V(z) = [z D_M² c/H(z)]^(1/3)",
        "```",
        "",
        f"**Distance-safe:** max ΔD_L &lt; {dl_threshold_percent}% (SNe z), "
        f"max ΔD_M, ΔD_V &lt; {dm_dv_threshold_percent}% (BAO z), "
        f"max ΔH &lt; {h_bao_threshold_percent}% (BAO z).",
        f"**Hubble-shift-active:** |H₀ shift| in [{hubble_min_percent}, {hubble_max_percent}]%.",
        "",
        "## Summary",
        "",
        f"- **Cases:** {n}",
        f"- **Distance-safe:** {n_dist} / {n}",
        f"- **Hubble-shift-active:** {n_hub} / {n}",
        f"- **Overall success:** {n_ok} / {n}",
        f"- **Intentional fail cases:** {len(intentional)} "
        f"({', '.join(r.case_name for r in intentional) or 'none'})",
        f"- **Worst D_L error:** {worst_dl.case_name} ({worst_dl.metrics.max_rel_error_D_L_percent:.3f}%)",
        "",
        "## Per-case table",
        "",
        "| Case | H₀ shift % | max ΔD_L % | max ΔD_M % | max ΔD_V % | dist safe | H active | overall |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for res in results:
        m = res.metrics
        lines.append(
            f"| {res.case_name} | {m.H0_shift_percent:.3f} | {m.max_rel_error_D_L_percent:.3f} | "
            f"{m.max_rel_error_D_M_percent:.3f} | {m.max_rel_error_D_V_percent:.3f} | "
            f"{'✓' if res.distance_safe_pass else '✗'} | "
            f"{'✓' if res.hubble_shift_active else '✗'} | "
            f"{'✓' if res.overall_success else '✗'} |",
        )

    lines.extend(
        [
            "",
            "## Scientific interpretation",
            "",
            "- **Overall success** (distance-safe **and** Hubble-shift-active) is achievable only "
            "in a **narrow** ε_τ amplitude band (~2% H₀ shift with 2% distance tolerances).",
            "- Larger late-time shifts (5–10% H₀) can remain CMB-safe (Phase 5F) yet **fail** "
            "strict 2% distance-proxy tolerances — reported as `hubble_active_only`.",
            "- **Passing** means the configured late-time ε_τ(z) preserves controlled distance "
            "proxies while shifting H₀ in the benchmark band.",
            "- This does **not** validate TDF with BAO, SNe, SH0ES, DESI, or chronometer data.",
            "- Real tests require likelihood-level fits with covariances, nuisances, CMB, and growth.",
            "",
            "## Failure modes",
            "",
            "- Background-only; no real datasets or covariance matrices.",
            "- No perturbation equations or growth-of-structure constraints.",
            "- Large uniform ε_τ or high-z leakage can pass H₀ shift but fail distance-safe (by design).",
            "",
            "## Disclaimer",
            "",
            "- **NOT REAL OBSERVATIONAL DATA**",
            "- Controlled ΛCDM teacher benchmark only.",
            "",
        ],
    )
    return "\n".join(lines)


def run_bao_sne_distance_benchmark(
    cases: Sequence[str] | None = None,
    outputs_root: Path | None = None,
    *,
    cosmology: CosmologyParams | None = None,
    dl_threshold_percent: float = DEFAULT_DL_THRESHOLD_PERCENT,
    dm_dv_threshold_percent: float = DEFAULT_DM_DV_THRESHOLD_PERCENT,
    h_bao_threshold_percent: float = DEFAULT_H_BAO_THRESHOLD_PERCENT,
    hubble_min_percent: float = HUBBLE_SHIFT_MIN_PERCENT,
    hubble_max_percent: float = HUBBLE_SHIFT_MAX_PERCENT,
) -> tuple[pd.DataFrame, list[BaoSneDistanceResult]]:
    """Run Phase 5G BAO/SNe distance benchmark."""
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

    thresh = {
        "dl_threshold_percent": dl_threshold_percent,
        "dm_dv_threshold_percent": dm_dv_threshold_percent,
        "h_bao_threshold_percent": h_bao_threshold_percent,
        "hubble_min_percent": hubble_min_percent,
        "hubble_max_percent": hubble_max_percent,
    }

    results = [run_single_distance_case(c, params, **thresh) for c in case_objs]

    out_df = pd.DataFrame([_result_to_row(r) for r in results])
    out_df.to_csv(tables_dir / "bao_sne_distance_benchmark_summary.csv", index=False)
    (reports_dir / "bao_sne_distance_benchmark_report.md").write_text(
        _build_report(
            results,
            dl_threshold_percent=dl_threshold_percent,
            dm_dv_threshold_percent=dm_dv_threshold_percent,
            h_bao_threshold_percent=h_bao_threshold_percent,
            hubble_min_percent=hubble_min_percent,
            hubble_max_percent=hubble_max_percent,
        ),
        encoding="utf-8",
    )
    _plot_H_ratio(case_objs, params, figures_dir / "bao_sne_distance_H_ratio_cases.png")
    _plot_DL_ratio(case_objs, params, figures_dir / "bao_sne_distance_DL_ratio_cases.png")
    _plot_DV_ratio(case_objs, params, figures_dir / "bao_sne_distance_DV_ratio_cases.png")
    return out_df, results

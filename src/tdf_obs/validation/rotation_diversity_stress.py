"""Phase 5B — Rotation-curve diversity stress test (not observational validation)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from tdf_obs.fitting.fit_rotation import N_PARAMS_TDF, fit_single_galaxy_rotation
from tdf_obs.io.schemas import RotationCurveData
from tdf_obs.models.dark_matter import v2_nfw_halo_only, v_nfw_simple
from tdf_obs.models.rotation import baryon_only_model, v_tdf_simple
from tdf_obs.validation.core_cusp_stress import (
    N_PARAMS_TDF_CORE,
    _fit_tdf_core_params,
    _model_metrics,
    v_tdf_core_proxy,
)
from tdf_obs.validation.nfw_surrogate import generate_baryon_profile, relative_curve_error_percent

BENCHMARK_MODE = "rotation_diversity_stress_test"
BANNER_ROTATION_DIVERSITY = "ROTATION-CURVE DIVERSITY STRESS TEST — NOT REAL OBSERVATIONAL DATA"

REQUIRED_SUMMARY_COLUMNS: tuple[str, ...] = (
    "case_name",
    "teacher_shape_type",
    "baryon_profile_type",
    "best_model_by_bic",
    "mse_baryon",
    "mse_tdf_simple",
    "mse_tdf_core",
    "mse_nfw",
    "bic_baryon",
    "bic_tdf_simple",
    "bic_tdf_core",
    "bic_nfw",
    "rel_err_tdf_simple_percent",
    "rel_err_tdf_core_percent",
    "rel_err_nfw_percent",
    "tdf_best_flag",
    "tdf_core_best_flag",
    "nfw_best_flag",
    "warnings",
)


@dataclass(frozen=True)
class DiversityCaseConfig:
    case_name: str
    r_min_kpc: float
    r_max_kpc: float
    n_points: int
    baryon_profile_type: str
    baryon_params: dict[str, float]
    teacher_shape_type: str
    teacher_params: dict[str, float]
    noise_std: float = 1.0
    random_seed: int = 0
    description: str = ""


@dataclass
class DiversityStressResult:
    case_name: str
    teacher_shape_type: str
    baryon_profile_type: str
    teacher_params: dict[str, float]
    dataframe: pd.DataFrame
    rotation_data: RotationCurveData
    best_model_by_bic: str
    mse_baryon: float
    mse_tdf_simple: float
    mse_tdf_core: float
    mse_nfw: float
    bic_baryon: float
    bic_tdf_simple: float
    bic_tdf_core: float
    bic_nfw: float
    rel_err_tdf_simple_percent: float
    rel_err_tdf_core_percent: float
    rel_err_nfw_percent: float
    tdf_best_flag: bool
    tdf_core_best_flag: bool
    nfw_best_flag: bool
    warnings: list[str] = field(default_factory=list)


def diversity_teacher_velocity(
    r: np.ndarray,
    v_baryon: np.ndarray,
    teacher_shape_type: str,
    params: dict[str, float],
) -> np.ndarray:
    """
    Synthetic diverse teacher v(r) [km/s], built from documented shape families.

    All shapes return finite, non-negative velocities.
    """
    r = np.asarray(r, dtype=float)
    v_baryon = np.asarray(v_baryon, dtype=float)
    p = params
    shape = teacher_shape_type

    if shape == "fast_rising_flat":
        V2 = p["V2"]
        r_core = max(p.get("r_core", 1.0), 1e-6)
        v2 = v_baryon**2 + V2 * r**2 / (r**2 + r_core**2)

    elif shape == "slow_rising_lsb":
        V2 = p["V2"]
        r0 = max(p.get("r0_large", p.get("r0", 4.0)), 1e-6)
        v2 = v_baryon**2 + V2 * r / (r + r0)

    elif shape == "compact_baryon_dominated":
        alpha = p.get("alpha", 0.06)
        r_scale = max(p.get("r_scale", 2.0), 1e-6)
        v = v_baryon * (1.0 + alpha * r / (r + r_scale))
        return np.maximum(v, 0.0)

    elif shape == "diffuse_baryon_dominated":
        V2 = p["V2"]
        r0 = max(p.get("r0", 5.0), 1e-6)
        v2 = v_baryon**2 + V2 * r / (r + r0)

    elif shape == "declining_outer_curve":
        V2 = p["V2"]
        r0 = max(p.get("r0", 2.0), 1e-6)
        d = p.get("d", 0.35)
        r_drop = max(p.get("r_drop", 12.0), 1e-6)
        v_base = np.sqrt(np.maximum(v_baryon**2 + V2 * r / (r + r0), 0.0))
        factor = np.maximum(1.0 - d * r / (r + r_drop), 0.15)
        return np.maximum(v_base * factor, 0.0)

    elif shape == "rising_outer_curve":
        V2 = p["V2"]
        r0 = max(p.get("r0", 3.0), 1e-6)
        v2 = v_baryon**2 + V2 * np.log1p(r / r0)

    elif shape == "flat_extended_curve":
        V2 = p["V2"]
        r_flat = max(p.get("r_flat", 8.0), 1e-6)
        v2 = v_baryon**2 + V2 * (1.0 - np.exp(-r / r_flat))

    elif shape == "inner_core_outer_flat":
        V2 = p["V2"]
        r_core = max(p.get("r_core", 0.8), 1e-6)
        r_flat = max(p.get("r_flat", 10.0), 1e-6)
        fade = p.get("fade", 0.25)
        v_inner = np.sqrt(np.maximum(v_baryon**2 + V2 * r**2 / (r**2 + r_core**2), 0.0))
        taper = np.maximum(1.0 - fade * r / (r + r_flat), 0.55)
        return np.maximum(v_inner * taper, 0.0)

    elif shape == "inner_cusp_outer_flat":
        Vh2 = p["Vh2"]
        rs = max(p.get("rs", 1.5), 1e-6)
        r_flat = max(p.get("r_flat", 14.0), 1e-6)
        v2 = v_baryon**2 + v2_nfw_halo_only(r, Vh2, rs)
        v = np.sqrt(np.maximum(v2, 0.0))
        outer = np.minimum(1.0, (r_flat + r_flat) / (r + r_flat))
        return np.maximum(v * outer, 0.0)

    elif shape == "low_velocity_dwarf_diverse":
        V2 = p["V2"]
        r_core = max(p.get("r_core", 1.2), 1e-6)
        boost = p.get("boost", 1.15)
        v2 = (v_baryon * boost) ** 2 + V2 * r**2 / (r**2 + r_core**2)

    else:
        raise ValueError(f"Unknown teacher_shape_type: {shape!r}")

    return np.sqrt(np.maximum(v2, 0.0))


BENCHMARK_CASE_REGISTRY: dict[str, DiversityCaseConfig] = {
    "fast_rising_flat": DiversityCaseConfig(
        case_name="fast_rising_flat",
        r_min_kpc=0.3,
        r_max_kpc=20.0,
        n_points=28,
        baryon_profile_type="saturating_disk",
        baryon_params={"v_max": 55.0, "r_disk": 1.2},
        teacher_shape_type="fast_rising_flat",
        teacher_params={"V2": 4500.0, "r_core": 1.0},
        noise_std=1.0,
        random_seed=301,
    ),
    "slow_rising_lsb": DiversityCaseConfig(
        case_name="slow_rising_lsb",
        r_min_kpc=0.5,
        r_max_kpc=24.0,
        n_points=30,
        baryon_profile_type="exponential_like",
        baryon_params={"v_max": 32.0, "r_disk": 4.5},
        teacher_shape_type="slow_rising_lsb",
        teacher_params={"V2": 3200.0, "r0_large": 6.0},
        noise_std=1.2,
        random_seed=302,
    ),
    "compact_baryon_dominated": DiversityCaseConfig(
        case_name="compact_baryon_dominated",
        r_min_kpc=0.2,
        r_max_kpc=16.0,
        n_points=26,
        baryon_profile_type="compact_bulge_disk",
        baryon_params={"v_bulge_max": 65.0, "r_bulge": 0.4, "v_disk_max": 45.0, "r_disk": 1.5},
        teacher_shape_type="compact_baryon_dominated",
        teacher_params={"alpha": 0.05, "r_scale": 2.0},
        noise_std=0.8,
        random_seed=303,
    ),
    "diffuse_baryon_dominated": DiversityCaseConfig(
        case_name="diffuse_baryon_dominated",
        r_min_kpc=0.5,
        r_max_kpc=28.0,
        n_points=30,
        baryon_profile_type="exponential_like",
        baryon_params={"v_max": 28.0, "r_disk": 5.0},
        teacher_shape_type="diffuse_baryon_dominated",
        teacher_params={"V2": 7500.0, "r0": 6.0},
        noise_std=1.5,
        random_seed=304,
    ),
    "declining_outer_curve": DiversityCaseConfig(
        case_name="declining_outer_curve",
        r_min_kpc=0.4,
        r_max_kpc=22.0,
        n_points=28,
        baryon_profile_type="saturating_disk",
        baryon_params={"v_max": 70.0, "r_disk": 2.0},
        teacher_shape_type="declining_outer_curve",
        teacher_params={"V2": 5000.0, "r0": 2.5, "d": 0.4, "r_drop": 14.0},
        noise_std=1.0,
        random_seed=305,
    ),
    "rising_outer_curve": DiversityCaseConfig(
        case_name="rising_outer_curve",
        r_min_kpc=0.5,
        r_max_kpc=25.0,
        n_points=30,
        baryon_profile_type="saturating_disk",
        baryon_params={"v_max": 60.0, "r_disk": 2.5},
        teacher_shape_type="rising_outer_curve",
        teacher_params={"V2": 1800.0, "r0": 4.0},
        noise_std=1.0,
        random_seed=306,
    ),
    "flat_extended_curve": DiversityCaseConfig(
        case_name="flat_extended_curve",
        r_min_kpc=0.5,
        r_max_kpc=30.0,
        n_points=32,
        baryon_profile_type="exponential_like",
        baryon_params={"v_max": 50.0, "r_disk": 3.5},
        teacher_shape_type="flat_extended_curve",
        teacher_params={"V2": 4200.0, "r_flat": 9.0},
        noise_std=1.0,
        random_seed=307,
    ),
    "inner_core_outer_flat": DiversityCaseConfig(
        case_name="inner_core_outer_flat",
        r_min_kpc=0.3,
        r_max_kpc=22.0,
        n_points=28,
        baryon_profile_type="saturating_disk",
        baryon_params={"v_max": 48.0, "r_disk": 1.5},
        teacher_shape_type="inner_core_outer_flat",
        teacher_params={"V2": 3800.0, "r_core": 0.9, "r_flat": 11.0, "fade": 0.28},
        noise_std=1.0,
        random_seed=308,
    ),
    "inner_cusp_outer_flat": DiversityCaseConfig(
        case_name="inner_cusp_outer_flat",
        r_min_kpc=0.2,
        r_max_kpc=20.0,
        n_points=28,
        baryon_profile_type="saturating_disk",
        baryon_params={"v_max": 58.0, "r_disk": 1.8},
        teacher_shape_type="inner_cusp_outer_flat",
        teacher_params={"Vh2": 9000.0, "rs": 1.2, "r_flat": 12.0},
        noise_std=1.0,
        random_seed=309,
    ),
    "low_velocity_dwarf_diverse": DiversityCaseConfig(
        case_name="low_velocity_dwarf_diverse",
        r_min_kpc=0.2,
        r_max_kpc=12.0,
        n_points=24,
        baryon_profile_type="exponential_like",
        baryon_params={"v_max": 22.0, "r_disk": 2.0},
        teacher_shape_type="low_velocity_dwarf_diverse",
        teacher_params={"V2": 1200.0, "r_core": 1.0, "boost": 1.1},
        noise_std=0.8,
        random_seed=310,
    ),
}


def list_benchmark_cases() -> list[str]:
    return list(BENCHMARK_CASE_REGISTRY.keys())


def get_benchmark_case(name: str) -> DiversityCaseConfig:
    if name not in BENCHMARK_CASE_REGISTRY:
        raise KeyError(f"Unknown diversity case {name!r}; available: {list_benchmark_cases()}")
    return BENCHMARK_CASE_REGISTRY[name]


def generate_diversity_case(case_name: str, config: DiversityCaseConfig | None = None) -> pd.DataFrame:
    """Build synthetic rotation table for one diversity case."""
    cfg = config or get_benchmark_case(case_name)

    r = np.linspace(cfg.r_min_kpc, cfg.r_max_kpc, cfg.n_points)
    v_baryon = generate_baryon_profile(r, kind=cfg.baryon_profile_type, **cfg.baryon_params)
    v_teacher = diversity_teacher_velocity(r, v_baryon, cfg.teacher_shape_type, cfg.teacher_params)

    rng = np.random.default_rng(cfg.random_seed)
    v_err = np.maximum(0.03 * v_teacher, 0.5)
    v_obs = (
        v_teacher + rng.normal(0.0, cfg.noise_std, size=len(r))
        if cfg.noise_std > 0
        else v_teacher.copy()
    )
    v_obs = np.maximum(v_obs, 0.0)

    return pd.DataFrame(
        {
            "galaxy_id": case_name,
            "r_kpc": r,
            "v_obs": v_obs,
            "v_err": v_err,
            "v_baryon": v_baryon,
            "v_teacher": v_teacher,
            "teacher_shape_type": cfg.teacher_shape_type,
            "teacher_type": cfg.teacher_shape_type,
            "dataset_mode": BENCHMARK_MODE,
        },
    )


def _to_rotation_data(df: pd.DataFrame) -> RotationCurveData:
    return RotationCurveData(
        galaxy_id=str(df["galaxy_id"].iloc[0]),
        r_kpc=df["r_kpc"].to_numpy(dtype=float),
        v_obs=df["v_obs"].to_numpy(dtype=float),
        v_err=df["v_err"].to_numpy(dtype=float),
        v_baryon=df["v_baryon"].to_numpy(dtype=float),
        metadata={
            "dataset_mode": BENCHMARK_MODE,
            "is_real_observational_data": False,
            "teacher_shape_type": str(df["teacher_shape_type"].iloc[0]),
            "v_teacher": df["v_teacher"].to_numpy(dtype=float),
        },
    )


def fit_diversity_case(
    case_name: str,
    df: pd.DataFrame,
    *,
    config: DiversityCaseConfig | None = None,
) -> DiversityStressResult:
    """Fit baryon, TDF simple, TDF core, NFW; compare to teacher via BIC."""
    cfg = config or get_benchmark_case(case_name)
    rot = _to_rotation_data(df)
    base = fit_single_galaxy_rotation(rot)

    r = rot.r_kpc
    v_obs = rot.v_obs
    v_err = np.maximum(rot.v_err, 1e-3)
    v_baryon = rot.v_baryon
    v_teacher = df["v_teacher"].to_numpy()
    warnings = list(base.warnings)

    C, rc_tau, cw = _fit_tdf_core_params(r, v_obs, v_err, v_baryon)
    warnings.extend(cw)
    if np.isfinite(C) and np.isfinite(rc_tau):
        v_core = v_tdf_core_proxy(r, v_baryon, C, rc_tau)
    else:
        warnings.append("TDF core fit failed.")
        v_core = baryon_only_model(v_baryon)

    mse_tc, _, _, bic_tc = _model_metrics(v_obs, v_core, v_err, N_PARAMS_TDF_CORE)

    v_simple = (
        v_tdf_simple(r, v_baryon, base.tdf_B, base.tdf_r0)
        if base.success_tdf
        else baryon_only_model(v_baryon)
    )
    v_nfw = (
        v_nfw_simple(r, v_baryon, base.nfw_Vh2, base.nfw_rs)
        if base.success_nfw
        else baryon_only_model(v_baryon)
    )

    rel_s = relative_curve_error_percent(v_teacher, v_simple)
    rel_c = relative_curve_error_percent(v_teacher, v_core)
    rel_n = relative_curve_error_percent(v_teacher, v_nfw)

    bic_scores = {
        "baryon_only": base.bic_baryon,
        "tdf_simple": base.bic_tdf,
        "tdf_core": bic_tc,
        "nfw_simple": base.bic_nfw,
    }
    best = min(bic_scores, key=bic_scores.get)  # type: ignore[arg-type]

    return DiversityStressResult(
        case_name=case_name,
        teacher_shape_type=cfg.teacher_shape_type,
        baryon_profile_type=cfg.baryon_profile_type,
        teacher_params=cfg.teacher_params,
        dataframe=df,
        rotation_data=rot,
        best_model_by_bic=best,
        mse_baryon=base.mse_baryon,
        mse_tdf_simple=base.mse_tdf,
        mse_tdf_core=mse_tc,
        mse_nfw=base.mse_nfw,
        bic_baryon=base.bic_baryon,
        bic_tdf_simple=base.bic_tdf,
        bic_tdf_core=bic_tc,
        bic_nfw=base.bic_nfw,
        rel_err_tdf_simple_percent=rel_s,
        rel_err_tdf_core_percent=rel_c,
        rel_err_nfw_percent=rel_n,
        tdf_best_flag=best == "tdf_simple",
        tdf_core_best_flag=best == "tdf_core",
        nfw_best_flag=best == "nfw_simple",
        warnings=warnings,
    )


def _plot_case(res: DiversityStressResult, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = res.dataframe
    r = df["r_kpc"].to_numpy()
    vb = df["v_baryon"].to_numpy()
    rot = res.rotation_data
    base = fit_single_galaxy_rotation(rot)
    r_f = np.linspace(r.min(), r.max(), 200)
    vb_f = np.interp(r_f, r, vb)
    C, rc_tau, _ = _fit_tdf_core_params(r, rot.v_obs, rot.v_err, vb)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.errorbar(r, df["v_obs"], yerr=df["v_err"], fmt="o", capsize=3, label="v_obs", color="C0")
    ax.plot(r, df["v_teacher"], "k-", lw=2, label=f"teacher ({res.teacher_shape_type})")
    ax.plot(r, vb, "--", color="C1", label="baryon")
    if base.success_tdf:
        ax.plot(r_f, v_tdf_simple(r_f, vb_f, base.tdf_B, base.tdf_r0), "-", color="C2", label="TDF simple")
    if np.isfinite(C):
        ax.plot(r_f, v_tdf_core_proxy(r_f, vb_f, C, rc_tau), "-", color="C4", label="TDF core")
    if base.success_nfw:
        ax.plot(r_f, v_nfw_simple(r_f, vb_f, base.nfw_Vh2, base.nfw_rs), "-.", color="C3", label="NFW")
    ax.set_xlabel("r [kpc]")
    ax.set_ylabel("v [km/s]")
    ax.set_title(f"{res.case_name}\n{BANNER_ROTATION_DIVERSITY[:48]}...")
    ax.legend(fontsize=7, loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _row(res: DiversityStressResult) -> dict[str, Any]:
    return {
        "case_name": res.case_name,
        "teacher_shape_type": res.teacher_shape_type,
        "baryon_profile_type": res.baryon_profile_type,
        "best_model_by_bic": res.best_model_by_bic,
        "mse_baryon": res.mse_baryon,
        "mse_tdf_simple": res.mse_tdf_simple,
        "mse_tdf_core": res.mse_tdf_core,
        "mse_nfw": res.mse_nfw,
        "bic_baryon": res.bic_baryon,
        "bic_tdf_simple": res.bic_tdf_simple,
        "bic_tdf_core": res.bic_tdf_core,
        "bic_nfw": res.bic_nfw,
        "rel_err_tdf_simple_percent": res.rel_err_tdf_simple_percent,
        "rel_err_tdf_core_percent": res.rel_err_tdf_core_percent,
        "rel_err_nfw_percent": res.rel_err_nfw_percent,
        "tdf_best_flag": res.tdf_best_flag,
        "tdf_core_best_flag": res.tdf_core_best_flag,
        "nfw_best_flag": res.nfw_best_flag,
        "warnings": "; ".join(res.warnings),
        "teacher_params": json.dumps(res.teacher_params),
        "dataset_mode": BENCHMARK_MODE,
        "is_real_observational_data": False,
    }


def _build_report(results: list[DiversityStressResult]) -> str:
    n = len(results)
    n_tdf = sum(1 for r in results if r.tdf_best_flag)
    n_tdf_c = sum(1 for r in results if r.tdf_core_best_flag)
    n_any_tdf = sum(1 for r in results if r.tdf_best_flag or r.tdf_core_best_flag)
    n_nfw = sum(1 for r in results if r.nfw_best_flag)

    med_s = float(np.median([r.rel_err_tdf_simple_percent for r in results]))
    med_c = float(np.median([r.rel_err_tdf_core_percent for r in results]))
    med_n = float(np.median([r.rel_err_nfw_percent for r in results]))
    worst = max(results, key=lambda r: min(r.rel_err_tdf_simple_percent, r.rel_err_tdf_core_percent))

    lines = [
        "# Rotation-curve diversity stress test (Phase 5B)",
        "",
        f"## ⚠️ {BANNER_ROTATION_DIVERSITY}",
        "",
        "## Purpose",
        "",
        "This stress test checks whether TDF variants can represent **diverse synthetic** "
        "rotation-curve shapes before real data are used.",
        "",
        "> **Not observational validation.**",
        "",
        "## Equations",
        "",
        "**TDF simple:** `v_TDF^2 = v_baryon^2 + B * r / (r + r0)`",
        "",
        "**TDF core (diagnostic):** `v_TDF_core^2 = v_baryon^2 + C * r^2 / (r^2 + rc_tau^2)`",
        "",
        "**NFW simple:** `v_NFW^2 = v_baryon^2 + Vh2 * [ln(1+x) - x/(1+x)] / x`",
        "",
        "**Teacher families** include: fast_rising_flat, slow_rising_lsb, compact/diffuse baryon-dominated, "
        "declining/rising outer, flat extended, inner core/cusp + outer flat, low-velocity dwarf.",
        "",
        "## Summary",
        "",
        f"- **Cases:** {n}",
        f"- **TDF simple best by BIC:** {n_tdf}",
        f"- **TDF core best by BIC:** {n_tdf_c}",
        f"- **Any TDF variant best:** {n_any_tdf}",
        f"- **NFW best by BIC:** {n_nfw}",
        f"- **Median rel. error TDF simple:** {med_s:.2f}%",
        f"- **Median rel. error TDF core:** {med_c:.2f}%",
        f"- **Median rel. error NFW:** {med_n:.2f}%",
        f"- **Hardest case (min TDF rel. err):** {worst.case_name}",
        "",
        "## Per-case table",
        "",
        "| Case | Shape | Best BIC | rel% simple | rel% core | rel% NFW |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for r in results:
        lines.append(
            f"| {r.case_name} | {r.teacher_shape_type} | {r.best_model_by_bic} | "
            f"{r.rel_err_tdf_simple_percent:.2f} | {r.rel_err_tdf_core_percent:.2f} | "
            f"{r.rel_err_nfw_percent:.2f} |",
        )

    lines.extend(
        [
            "",
            "## Scientific interpretation",
            "",
            "- Passing means TDF variants can represent **synthetic diversity patterns** in this registry.",
            "- It does **not** validate TDF against real galaxies or SPARC.",
            "- Cases where NFW wins are **informative**, not test failures.",
            "",
            "## Failure modes",
            "",
            "- Teacher curves are **synthetic** shape families only.",
            "- TDF core has a core-scale parameter and may overfit some cases.",
            "- NFW is a simple comparison baseline, not a full halo model survey.",
            "- Real diversity requires SPARC or similar data in a later phase.",
            "",
            "## Disclaimer",
            "",
            "- **NOT REAL OBSERVATIONAL DATA**",
            "",
        ],
    )
    return "\n".join(lines)


def run_rotation_diversity_stress(
    outputs_root: Path | None = None,
    *,
    case_names: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, list[DiversityStressResult]]:
    """Run Phase 5B pipeline and write CSV, report, figures."""
    root = Path(__file__).resolve().parents[3]
    outputs = outputs_root or (root / "outputs")
    tables = outputs / "tables"
    reports = outputs / "reports"
    figures = outputs / "figures"
    for d in (tables, reports, figures):
        d.mkdir(parents=True, exist_ok=True)

    names = list(case_names) if case_names is not None else list_benchmark_cases()
    results: list[DiversityStressResult] = []
    for name in names:
        cfg = get_benchmark_case(name)
        df = generate_diversity_case(name, cfg)
        res = fit_diversity_case(name, df, config=cfg)
        results.append(res)
        _plot_case(res, figures / f"rotation_diversity_{name}.png")

    out = pd.DataFrame([_row(r) for r in results])
    out.to_csv(tables / "rotation_diversity_stress_summary.csv", index=False)
    (reports / "rotation_diversity_stress_report.md").write_text(_build_report(results), encoding="utf-8")
    return out, results

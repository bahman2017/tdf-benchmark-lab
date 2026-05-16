"""Phase 5A — Core–cusp stress test (not observational validation)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from tdf_obs.fitting.fit_rotation import (
    N_PARAMS_BARYON,
    N_PARAMS_NFW,
    N_PARAMS_TDF,
    fit_single_galaxy_rotation,
)
from tdf_obs.fitting.metrics import aic, bic, chi_square, mse, reduced_chi_square
from tdf_obs.io.schemas import RotationCurveData
from tdf_obs.models.dark_matter import v2_nfw_halo_only, v_nfw_simple
from tdf_obs.models.rotation import baryon_only_model, v_tdf_simple
from tdf_obs.validation.nfw_surrogate import generate_baryon_profile, relative_curve_error_percent

BENCHMARK_MODE = "core_cusp_stress_test"
BANNER_CORE_CUSP = "CORE-CUSP STRESS TEST — NOT REAL OBSERVATIONAL DATA"

N_PARAMS_TDF_CORE = 2
N_PARAMS_CORED_PROXY = 2

TeacherType = Literal["cuspy", "cored"]

REQUIRED_SUMMARY_COLUMNS: tuple[str, ...] = (
    "case_name",
    "teacher_type",
    "baryon_profile_type",
    "teacher_params",
    "best_model_by_bic",
    "mse_baryon",
    "mse_tdf_simple",
    "mse_tdf_core",
    "mse_nfw",
    "bic_baryon",
    "bic_tdf_simple",
    "bic_tdf_core",
    "bic_nfw",
    "tdf_simple_relative_error_percent",
    "tdf_core_relative_error_percent",
    "nfw_relative_error_percent",
    "core_advantage_flag",
    "warnings",
)


@dataclass(frozen=True)
class CoreCuspCase:
    case_name: str
    teacher_type: TeacherType
    r_min_kpc: float
    r_max_kpc: float
    n_points: int
    baryon_profile_type: str
    baryon_params: dict[str, float]
    teacher_params: dict[str, float]
    noise_std: float = 1.0
    random_seed: int = 0
    description: str = ""


@dataclass
class CoreCuspStressResult:
    case_name: str
    teacher_type: TeacherType
    baryon_profile_type: str
    teacher_params: dict[str, float]
    dataframe: pd.DataFrame
    rotation_data: RotationCurveData
    best_model_by_bic: str
    mse_baryon: float
    mse_tdf_simple: float
    mse_tdf_core: float
    mse_nfw: float
    mse_cored_proxy: float | None
    bic_baryon: float
    bic_tdf_simple: float
    bic_tdf_core: float
    bic_nfw: float
    bic_cored_proxy: float | None
    tdf_simple_relative_error_percent: float
    tdf_core_relative_error_percent: float
    nfw_relative_error_percent: float
    core_advantage_flag: bool
    tdf_core_beats_nfw_in_cored: bool | None
    warnings: list[str] = field(default_factory=list)


def v2_core_proxy(r: np.ndarray, Vc2: float, rc: float) -> np.ndarray:
    """Cored halo proxy: v_core^2(r) = Vc2 * r^2 / (r^2 + rc^2) [km^2/s^2]."""
    r = np.asarray(r, dtype=float)
    rc = max(float(rc), 1e-6)
    return float(Vc2) * r**2 / (r**2 + rc**2)


def v_core_proxy(r: np.ndarray, Vc2: float, rc: float) -> np.ndarray:
    return np.sqrt(np.maximum(v2_core_proxy(r, Vc2, rc), 0.0))


def v2_tdf_core_proxy(
    r: np.ndarray,
    v_baryon: np.ndarray,
    C: float,
    rc_tau: float,
) -> np.ndarray:
    """TDF core stress diagnostic: v^2 = v_baryon^2 + C * r^2 / (r^2 + rc_tau^2)."""
    r = np.asarray(r, dtype=float)
    v_baryon = np.asarray(v_baryon, dtype=float)
    rc_tau = max(float(rc_tau), 1e-6)
    return v_baryon**2 + float(C) * r**2 / (r**2 + rc_tau**2)


def v_tdf_core_proxy(
    r: np.ndarray,
    v_baryon: np.ndarray,
    C: float,
    rc_tau: float,
) -> np.ndarray:
    return np.sqrt(np.maximum(v2_tdf_core_proxy(r, v_baryon, C, rc_tau), 0.0))


def cuspy_teacher_velocity(r: np.ndarray, v_baryon: np.ndarray, Vh2: float, rs: float) -> np.ndarray:
    v2 = v_baryon**2 + v2_nfw_halo_only(r, Vh2, rs)
    return np.sqrt(np.maximum(v2, 0.0))


def cored_teacher_velocity(r: np.ndarray, v_baryon: np.ndarray, Vc2: float, rc: float) -> np.ndarray:
    v2 = v_baryon**2 + v2_core_proxy(r, Vc2, rc)
    return np.sqrt(np.maximum(v2, 0.0))


def _model_metrics(
    v_obs: np.ndarray,
    v_pred: np.ndarray,
    v_err: np.ndarray,
    n_params: int,
) -> tuple[float, float, float, float]:
    n = len(v_obs)
    c2 = chi_square(v_obs, v_pred, v_err)
    return mse(v_obs, v_pred), c2, reduced_chi_square(c2, n, n_params), bic(c2, n, n_params)


def _fit_tdf_core_params(
    r: np.ndarray,
    v_obs: np.ndarray,
    v_err: np.ndarray,
    v_baryon: np.ndarray,
) -> tuple[float, float, list[str]]:
    warnings: list[str] = []
    v2_obs = v_obs**2
    v2_err = np.maximum(2.0 * v_obs * v_err, 1.0)

    def model_v2(r_fit: np.ndarray, C: float, rc_tau: float) -> np.ndarray:
        return v2_tdf_core_proxy(r_fit, v_baryon, C, rc_tau)

    r_outer = max(r[-1], 1.0)
    excess = max(v2_obs[-1] - v_baryon[-1] ** 2, 1.0)
    p0 = (excess * r_outer**2 / (r_outer**2 + 1.0), 1.0)
    bounds = ([0.0, 0.05], [1e6, 100.0])

    try:
        popt, _ = curve_fit(
            model_v2,
            r,
            v2_obs,
            p0=p0,
            bounds=bounds,
            sigma=v2_err,
            absolute_sigma=True,
            maxfev=20_000,
        )
        return float(popt[0]), float(popt[1]), warnings
    except Exception as exc:
        warnings.append(f"TDF core curve_fit failed: {exc}")
        return float("nan"), float("nan"), warnings


def _fit_cored_proxy_params(
    r: np.ndarray,
    v_obs: np.ndarray,
    v_err: np.ndarray,
    v_baryon: np.ndarray,
) -> tuple[float, float, list[str]]:
    warnings: list[str] = []
    v2_obs = v_obs**2
    v2_err = np.maximum(2.0 * v_obs * v_err, 1.0)

    def model_v2(r_fit: np.ndarray, Vc2: float, rc: float) -> np.ndarray:
        return v_baryon**2 + v2_core_proxy(r_fit, Vc2, rc)

    r_outer = max(r[-1], 1.0)
    excess = max(v2_obs[-1] - v_baryon[-1] ** 2, 1.0)
    p0 = (excess * 2.0, max(r_outer / 4.0, 0.3))
    bounds = ([0.0, 0.05], [1e6, 100.0])

    try:
        popt, _ = curve_fit(
            model_v2,
            r,
            v2_obs,
            p0=p0,
            bounds=bounds,
            sigma=v2_err,
            absolute_sigma=True,
            maxfev=20_000,
        )
        return float(popt[0]), float(popt[1]), warnings
    except Exception as exc:
        warnings.append(f"Cored proxy curve_fit failed: {exc}")
        return float("nan"), float("nan"), warnings


BENCHMARK_CASE_REGISTRY: dict[str, CoreCuspCase] = {
    "nfw_mild_cusp": CoreCuspCase(
        case_name="nfw_mild_cusp",
        teacher_type="cuspy",
        r_min_kpc=0.2,
        r_max_kpc=20.0,
        n_points=28,
        baryon_profile_type="saturating_disk",
        baryon_params={"v_max": 60.0, "r_disk": 1.5},
        teacher_params={"Vh2": 4000.0, "rs": 3.0},
        noise_std=1.0,
        random_seed=201,
    ),
    "nfw_strong_cusp": CoreCuspCase(
        case_name="nfw_strong_cusp",
        teacher_type="cuspy",
        r_min_kpc=0.1,
        r_max_kpc=15.0,
        n_points=30,
        baryon_profile_type="saturating_disk",
        baryon_params={"v_max": 45.0, "r_disk": 0.8},
        teacher_params={"Vh2": 8000.0, "rs": 0.8},
        noise_std=1.2,
        random_seed=202,
    ),
    "nfw_concentrated_inner": CoreCuspCase(
        case_name="nfw_concentrated_inner",
        teacher_type="cuspy",
        r_min_kpc=0.15,
        r_max_kpc=18.0,
        n_points=28,
        baryon_profile_type="compact_bulge_disk",
        baryon_params={"v_bulge_max": 55.0, "r_bulge": 0.3, "v_disk_max": 50.0, "r_disk": 1.2},
        teacher_params={"Vh2": 12000.0, "rs": 1.2},
        noise_std=1.0,
        random_seed=203,
    ),
    "nfw_extended_outer": CoreCuspCase(
        case_name="nfw_extended_outer",
        teacher_type="cuspy",
        r_min_kpc=0.5,
        r_max_kpc=30.0,
        n_points=32,
        baryon_profile_type="exponential_like",
        baryon_params={"v_max": 70.0, "r_disk": 3.0},
        teacher_params={"Vh2": 6000.0, "rs": 12.0},
        noise_std=1.5,
        random_seed=204,
    ),
    "core_small_rc": CoreCuspCase(
        case_name="core_small_rc",
        teacher_type="cored",
        r_min_kpc=0.2,
        r_max_kpc=18.0,
        n_points=28,
        baryon_profile_type="saturating_disk",
        baryon_params={"v_max": 50.0, "r_disk": 1.0},
        teacher_params={"Vc2": 3500.0, "rc": 0.8},
        noise_std=1.0,
        random_seed=205,
    ),
    "core_large_rc": CoreCuspCase(
        case_name="core_large_rc",
        teacher_type="cored",
        r_min_kpc=0.3,
        r_max_kpc=22.0,
        n_points=30,
        baryon_profile_type="saturating_disk",
        baryon_params={"v_max": 65.0, "r_disk": 2.0},
        teacher_params={"Vc2": 5000.0, "rc": 4.0},
        noise_std=1.0,
        random_seed=206,
    ),
    "core_lsb_like": CoreCuspCase(
        case_name="core_lsb_like",
        teacher_type="cored",
        r_min_kpc=0.3,
        r_max_kpc=24.0,
        n_points=30,
        baryon_profile_type="exponential_like",
        baryon_params={"v_max": 35.0, "r_disk": 4.0},
        teacher_params={"Vc2": 2800.0, "rc": 2.5},
        noise_std=1.2,
        random_seed=207,
    ),
    "core_diffuse_dwarf": CoreCuspCase(
        case_name="core_diffuse_dwarf",
        teacher_type="cored",
        r_min_kpc=0.2,
        r_max_kpc=14.0,
        n_points=26,
        baryon_profile_type="exponential_like",
        baryon_params={"v_max": 25.0, "r_disk": 2.5},
        teacher_params={"Vc2": 1500.0, "rc": 1.5},
        noise_std=1.0,
        random_seed=208,
    ),
}


def list_benchmark_cases() -> list[str]:
    return list(BENCHMARK_CASE_REGISTRY.keys())


def get_benchmark_case(name: str) -> CoreCuspCase:
    if name not in BENCHMARK_CASE_REGISTRY:
        raise KeyError(f"Unknown core-cusp case {name!r}; available: {list_benchmark_cases()}")
    return BENCHMARK_CASE_REGISTRY[name]


def generate_cuspy_teacher_case(case: CoreCuspCase) -> pd.DataFrame:
    """Build synthetic rotation table from cuspy NFW-like teacher."""
    if case.teacher_type != "cuspy":
        raise ValueError(f"generate_cuspy_teacher_case requires teacher_type='cuspy', got {case.teacher_type!r}")
    return _generate_teacher_table(case, cuspy=True)


def generate_cored_teacher_case(case: CoreCuspCase) -> pd.DataFrame:
    """Build synthetic rotation table from cored halo teacher."""
    if case.teacher_type != "cored":
        raise ValueError(f"generate_cored_teacher_case requires teacher_type='cored', got {case.teacher_type!r}")
    return _generate_teacher_table(case, cuspy=False)


def _generate_teacher_table(case: CoreCuspCase, *, cuspy: bool) -> pd.DataFrame:
    r = np.linspace(case.r_min_kpc, case.r_max_kpc, case.n_points)
    v_baryon = generate_baryon_profile(r, kind=case.baryon_profile_type, **case.baryon_params)
    tp = case.teacher_params
    if cuspy:
        v_teacher = cuspy_teacher_velocity(r, v_baryon, tp["Vh2"], tp["rs"])
    else:
        v_teacher = cored_teacher_velocity(r, v_baryon, tp["Vc2"], tp["rc"])

    rng = np.random.default_rng(case.random_seed)
    v_err = np.maximum(0.03 * v_teacher, 1.0)
    v_obs = (
        v_teacher + rng.normal(0.0, case.noise_std, size=len(r))
        if case.noise_std > 0
        else v_teacher.copy()
    )
    v_obs = np.maximum(v_obs, 0.0)

    return pd.DataFrame(
        {
            "galaxy_id": case.case_name,
            "case_name": case.case_name,
            "r_kpc": r,
            "v_obs": v_obs,
            "v_err": v_err,
            "v_baryon": v_baryon,
            "v_teacher": v_teacher,
            "teacher_type": case.teacher_type,
            "baryon_profile_type": case.baryon_profile_type,
            "teacher_params_json": json.dumps(tp),
            "dataset_mode": BENCHMARK_MODE,
        },
    )


def dataframe_to_rotation_data(df: pd.DataFrame) -> RotationCurveData:
    return RotationCurveData(
        galaxy_id=str(df["case_name"].iloc[0]),
        r_kpc=df["r_kpc"].to_numpy(dtype=float),
        v_obs=df["v_obs"].to_numpy(dtype=float),
        v_err=df["v_err"].to_numpy(dtype=float),
        v_baryon=df["v_baryon"].to_numpy(dtype=float),
        metadata={
            "dataset_mode": BENCHMARK_MODE,
            "is_real_observational_data": False,
            "teacher_type": str(df["teacher_type"].iloc[0]),
            "v_teacher": df["v_teacher"].to_numpy(dtype=float),
        },
    )


def fit_core_cusp_case(case: CoreCuspCase, df: pd.DataFrame) -> CoreCuspStressResult:
    """Fit all stress-test models and compare to teacher curve."""
    rot = dataframe_to_rotation_data(df)
    base = fit_single_galaxy_rotation(rot)

    r = rot.r_kpc
    v_obs = rot.v_obs
    v_err = np.maximum(rot.v_err, 1e-3)
    v_baryon = rot.v_baryon
    v_teacher = df["v_teacher"].to_numpy()

    warnings = list(base.warnings)

    C, rc_tau, core_w = _fit_tdf_core_params(r, v_obs, v_err, v_baryon)
    warnings.extend(core_w)
    if np.isfinite(C) and np.isfinite(rc_tau):
        v_tdf_core = v_tdf_core_proxy(r, v_baryon, C, rc_tau)
    else:
        warnings.append("TDF core fit failed; using baryon-only for core metrics.")
        v_tdf_core = baryon_only_model(v_baryon)

    mse_tc, _, _, bic_tc = _model_metrics(v_obs, v_tdf_core, v_err, N_PARAMS_TDF_CORE)

    mse_cp: float | None = None
    bic_cp: float | None = None
    if case.teacher_type == "cored":
        Vc2, rc, cp_w = _fit_cored_proxy_params(r, v_obs, v_err, v_baryon)
        warnings.extend(cp_w)
        if np.isfinite(Vc2) and np.isfinite(rc):
            v_cp = np.sqrt(np.maximum(v_baryon**2 + v2_core_proxy(r, Vc2, rc), 0.0))
            mse_cp, _, _, bic_cp = _model_metrics(v_obs, v_cp, v_err, N_PARAMS_CORED_PROXY)
        else:
            warnings.append("Cored teacher proxy refit failed.")

    v_simple_pred = (
        v_tdf_simple(r, v_baryon, base.tdf_B, base.tdf_r0)
        if base.success_tdf
        else baryon_only_model(v_baryon)
    )

    rel_simple = relative_curve_error_percent(v_teacher, v_simple_pred)
    rel_core = relative_curve_error_percent(v_teacher, v_tdf_core)
    rel_nfw = relative_curve_error_percent(
        v_teacher,
        v_nfw_simple(r, v_baryon, base.nfw_Vh2, base.nfw_rs) if base.success_nfw else baryon_only_model(v_baryon),
    )

    core_advantage = bic_tc < base.bic_tdf and rel_core <= rel_simple

    tdf_core_beats_nfw: bool | None = None
    if case.teacher_type == "cored":
        tdf_core_beats_nfw = bic_tc < base.bic_nfw

    bic_scores: dict[str, float] = {
        "baryon_only": base.bic_baryon,
        "tdf_simple": base.bic_tdf,
        "tdf_core": bic_tc,
        "nfw_simple": base.bic_nfw,
    }
    if bic_cp is not None:
        bic_scores["cored_proxy"] = bic_cp
    best_model = min(bic_scores, key=bic_scores.get)  # type: ignore[arg-type]

    if case.teacher_type == "cored" and rel_simple > rel_core * 1.05:
        warnings.append("TDF simple worse than TDF core vs cored teacher (expected in core regime).")
    if case.teacher_type == "cuspy" and rel_nfw < rel_simple and base.bic_nfw < base.bic_tdf:
        warnings.append("NFW refit tracks cuspy teacher better than TDF simple by BIC.")

    return CoreCuspStressResult(
        case_name=case.case_name,
        teacher_type=case.teacher_type,
        baryon_profile_type=case.baryon_profile_type,
        teacher_params=case.teacher_params,
        dataframe=df,
        rotation_data=rot,
        best_model_by_bic=best_model,
        mse_baryon=base.mse_baryon,
        mse_tdf_simple=base.mse_tdf,
        mse_tdf_core=mse_tc,
        mse_nfw=base.mse_nfw,
        mse_cored_proxy=mse_cp,
        bic_baryon=base.bic_baryon,
        bic_tdf_simple=base.bic_tdf,
        bic_tdf_core=bic_tc,
        bic_nfw=base.bic_nfw,
        bic_cored_proxy=bic_cp,
        tdf_simple_relative_error_percent=rel_simple,
        tdf_core_relative_error_percent=rel_core,
        nfw_relative_error_percent=rel_nfw,
        core_advantage_flag=core_advantage,
        tdf_core_beats_nfw_in_cored=tdf_core_beats_nfw,
        warnings=warnings,
    )


def _plot_case(result: CoreCuspStressResult, output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = result.dataframe
    r = df["r_kpc"].to_numpy()
    v_bary = df["v_baryon"].to_numpy()
    v_teacher = df["v_teacher"].to_numpy()
    rot = result.rotation_data

    base = fit_single_galaxy_rotation(rot)
    r_fine = np.linspace(r.min(), r.max(), 200)
    vb_fine = np.interp(r_fine, r, v_bary)

    C, rc_tau, _ = _fit_tdf_core_params(r, rot.v_obs, rot.v_err, v_bary)
    v_simple = (
        v_tdf_simple(r_fine, vb_fine, base.tdf_B, base.tdf_r0)
        if base.success_tdf
        else np.interp(r_fine, r, v_bary)
    )
    v_core = (
        v_tdf_core_proxy(r_fine, vb_fine, C, rc_tau)
        if np.isfinite(C)
        else np.interp(r_fine, r, v_bary)
    )
    v_nfw = (
        v_nfw_simple(r_fine, vb_fine, base.nfw_Vh2, base.nfw_rs)
        if base.success_nfw
        else np.interp(r_fine, r, v_bary)
    )

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.errorbar(r, df["v_obs"], yerr=df["v_err"], fmt="o", capsize=3, label="stress v_obs", color="C0")
    ax.plot(r, v_teacher, "k-", lw=2, label=f"teacher ({result.teacher_type})")
    ax.plot(r, v_bary, "--", color="C1", label="baryon")
    ax.plot(r_fine, v_simple, "-", color="C2", label="TDF simple")
    ax.plot(r_fine, v_core, "-", color="C4", label="TDF core proxy")
    ax.plot(r_fine, v_nfw, "-.", color="C3", label="NFW simple")
    ax.set_xlabel("r [kpc]")
    ax.set_ylabel("v [km/s]")
    ax.set_title(f"{result.case_name} — {BANNER_CORE_CUSP[:40]}...")
    ax.legend(loc="best", fontsize=7)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _result_to_row(res: CoreCuspStressResult) -> dict[str, Any]:
    return {
        "case_name": res.case_name,
        "teacher_type": res.teacher_type,
        "baryon_profile_type": res.baryon_profile_type,
        "teacher_params": json.dumps(res.teacher_params),
        "best_model_by_bic": res.best_model_by_bic,
        "mse_baryon": res.mse_baryon,
        "mse_tdf_simple": res.mse_tdf_simple,
        "mse_tdf_core": res.mse_tdf_core,
        "mse_nfw": res.mse_nfw,
        "bic_baryon": res.bic_baryon,
        "bic_tdf_simple": res.bic_tdf_simple,
        "bic_tdf_core": res.bic_tdf_core,
        "bic_nfw": res.bic_nfw,
        "tdf_simple_relative_error_percent": res.tdf_simple_relative_error_percent,
        "tdf_core_relative_error_percent": res.tdf_core_relative_error_percent,
        "nfw_relative_error_percent": res.nfw_relative_error_percent,
        "core_advantage_flag": res.core_advantage_flag,
        "warnings": "; ".join(res.warnings),
        "dataset_mode": BENCHMARK_MODE,
        "is_real_observational_data": False,
        "tdf_core_beats_nfw_in_cored": res.tdf_core_beats_nfw_in_cored,
    }


def _build_report(results: list[CoreCuspStressResult]) -> str:
    n = len(results)
    n_cuspy = sum(1 for r in results if r.teacher_type == "cuspy")
    n_cored = sum(1 for r in results if r.teacher_type == "cored")
    n_core_adv = sum(1 for r in results if r.core_advantage_flag)
    cored_results = [r for r in results if r.teacher_type == "cored"]
    n_tdf_core_beats_nfw_cored = sum(1 for r in cored_results if r.tdf_core_beats_nfw_in_cored)

    worst = max(results, key=lambda r: r.tdf_core_relative_error_percent)

    lines = [
        "# Core–cusp stress test report (Phase 5A)",
        "",
        f"## ⚠️ {BANNER_CORE_CUSP}",
        "",
        "## Purpose",
        "",
        "This test checks whether TDF-style **core smoothing** can represent cored inner rotation "
        "behavior in controlled stress cases, compared to cuspy NFW-like teachers.",
        "",
        "> **Not observational validation.** Synthetic teachers only.",
        "",
        "## Equations",
        "",
        "**Cuspy NFW-like teacher:**",
        "",
        "```text",
        "v_NFW^2(r) = Vh2 * [ln(1+x) - x/(1+x)] / x,  x = r/rs",
        "v_teacher^2 = v_baryon^2 + v_NFW^2",
        "```",
        "",
        "**Cored teacher proxy:**",
        "",
        "```text",
        "v_core^2(r) = Vc2 * r^2 / (r^2 + rc^2)",
        "v_teacher^2 = v_baryon^2 + v_core^2",
        "```",
        "",
        "**TDF simple (unchanged):**",
        "",
        "```text",
        "v_TDF^2 = v_baryon^2 + B * r / (r + r0)",
        "```",
        "",
        "**TDF core proxy (stress diagnostic only):**",
        "",
        "```text",
        "v_TDF_core^2 = v_baryon^2 + C * r^2 / (r^2 + rc_tau^2)",
        "```",
        "",
        "## Summary",
        "",
        f"- **Total cases:** {n}",
        f"- **Cuspy teachers:** {n_cuspy}",
        f"- **Cored teachers:** {n_cored}",
        f"- **TDF core beats TDF simple (BIC + error):** {n_core_adv} / {n}",
        f"- **TDF core beats NFW in cored cases:** {n_tdf_core_beats_nfw_cored} / {len(cored_results)}",
        f"- **Worst TDF core rel. error:** {worst.case_name} ({worst.tdf_core_relative_error_percent:.2f}%)",
        "",
        "## Results table",
        "",
        "| Case | Teacher | Best BIC | rel err TDF simple % | rel err TDF core % | rel err NFW % | core adv. |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for res in results:
        adv = "yes" if res.core_advantage_flag else "no"
        lines.append(
            f"| {res.case_name} | {res.teacher_type} | {res.best_model_by_bic} | "
            f"{res.tdf_simple_relative_error_percent:.2f} | {res.tdf_core_relative_error_percent:.2f} | "
            f"{res.nfw_relative_error_percent:.2f} | {adv} |",
        )

    lines.extend(
        [
            "",
            "## Scientific interpretation",
            "",
            "- Passing cored stress tests does **not** validate TDF against real galaxies.",
            "- It only suggests that tau-core smoothing can represent **core-like phenomenology** "
            "in these controlled benchmarks.",
            "- BIC penalizes extra parameters; compare TDF core (2 params) fairly against TDF simple and NFW.",
            "",
            "## Failure modes",
            "",
            "- TDF simple may fail near inner-core behavior (cuspy vs cored mismatch).",
            "- TDF core proxy adds a physically motivated core scale — more flexibility by design.",
            "- More parameters can improve fit; always compare **BIC**, not MSE alone.",
            "- Real galaxies and SPARC data are **not** tested here.",
            "",
            "## Disclaimer",
            "",
            "- **NOT REAL OBSERVATIONAL DATA**",
            "- Does not resolve the core–cusp problem observationally.",
            "",
        ],
    )
    return "\n".join(lines)


def run_core_cusp_stress_pipeline(
    outputs_root: Path | None = None,
    *,
    case_names: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, list[CoreCuspStressResult]]:
    """Run Phase 5A core–cusp stress test and write outputs."""
    root = Path(__file__).resolve().parents[3]
    outputs = outputs_root or (root / "outputs")
    tables_dir = outputs / "tables"
    reports_dir = outputs / "reports"
    figures_dir = outputs / "figures"
    for d in (tables_dir, reports_dir, figures_dir):
        d.mkdir(parents=True, exist_ok=True)

    names = list(case_names) if case_names is not None else list_benchmark_cases()
    results: list[CoreCuspStressResult] = []

    for name in names:
        case = get_benchmark_case(name)
        if case.teacher_type == "cuspy":
            df = generate_cuspy_teacher_case(case)
        else:
            df = generate_cored_teacher_case(case)
        result = fit_core_cusp_case(case, df)
        results.append(result)
        _plot_case(result, figures_dir / f"core_cusp_{case.case_name}.png")

    out_df = pd.DataFrame([_result_to_row(r) for r in results])
    out_df.to_csv(tables_dir / "core_cusp_stress_summary.csv", index=False)
    (reports_dir / "core_cusp_stress_report.md").write_text(_build_report(results), encoding="utf-8")
    return out_df, results

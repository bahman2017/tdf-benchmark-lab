"""Phase 5C — Same-τ multi-observable consistency benchmark (not observational validation)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from tdf_obs.fitting.fit_rotation import _fit_tdf_params
from tdf_obs.io.schemas import RotationCurveData
from tdf_obs.models.rotation import v_tdf_simple
from tdf_obs.validation.nfw_surrogate import generate_baryon_profile, relative_curve_error_percent

BENCHMARK_MODE = "same_tau_multi_observable_benchmark"
BANNER_SAME_TAU = "SAME-TAU MULTI-OBSERVABLE BENCHMARK — NOT REAL OBSERVATIONAL DATA"

DEFAULT_A_LENS = 1.0
DEFAULT_C_KM_S = 299_792.458
PASS_TOLERANCE_PERCENT = 5.0

REQUIRED_SUMMARY_COLUMNS: tuple[str, ...] = (
    "case_name",
    "true_B",
    "true_r0",
    "recovered_B",
    "recovered_r0",
    "B_recovery_error_percent",
    "r0_recovery_error_percent",
    "rotation_relative_error_percent",
    "lensing_relative_error_percent",
    "redshift_relative_error_percent",
    "same_tau_pass",
    "A_lens",
    "rotation_noise_std",
    "lensing_noise_std",
    "redshift_noise_std",
    "warnings",
)


@dataclass(frozen=True)
class RedshiftPairSpec:
    R_emit_kpc: float
    R_obs_kpc: float


@dataclass(frozen=True)
class SameTauCase:
    case_name: str
    true_B: float
    true_r0: float
    r_min_kpc: float
    r_max_kpc: float
    n_rotation_points: int
    R_min_kpc: float
    R_max_kpc: float
    n_lensing_points: int
    redshift_pairs: tuple[RedshiftPairSpec, ...]
    baryon_profile_type: str
    baryon_params: dict[str, float]
    rotation_noise_std: float = 1.0
    lensing_noise_std: float = 0.02
    redshift_noise_std: float = 1e-6
    A_lens: float = DEFAULT_A_LENS
    random_seed: int = 0
    description: str = ""


@dataclass
class SameTauTeacherData:
    case: SameTauCase
    rotation_df: pd.DataFrame
    lensing_df: pd.DataFrame
    redshift_df: pd.DataFrame


@dataclass
class SameTauConsistencyResult:
    case_name: str
    true_B: float
    true_r0: float
    recovered_B: float
    recovered_r0: float
    B_recovery_error_percent: float
    r0_recovery_error_percent: float
    rotation_relative_error_percent: float
    lensing_relative_error_percent: float
    redshift_relative_error_percent: float
    same_tau_pass: bool
    A_lens: float
    rotation_noise_std: float
    lensing_noise_std: float
    redshift_noise_std: float
    rotation_data: RotationCurveData
    lensing_df: pd.DataFrame
    redshift_df: pd.DataFrame
    warnings: list[str] = field(default_factory=list)


def phi_tau_log(r: np.ndarray, B: float, r0: float) -> np.ndarray:
    """Tau potential Φ_τ(r) = B log(1 + r/r0) [km²/s²]."""
    r = np.asarray(r, dtype=float)
    r0_safe = max(float(r0), 1e-12)
    return float(B) * np.log1p(r / r0_safe)


def tau_acceleration_term(r: np.ndarray, B: float, r0: float) -> np.ndarray:
    """r dΦ_τ/dr = B r/(r+r0) [km²/s²], matching the rotation correction term."""
    r = np.asarray(r, dtype=float)
    r0_safe = max(float(r0), 1e-12)
    return float(B) * r / (r + r0_safe)


def predict_lensing_proxy(
    R: np.ndarray,
    B: float,
    r0: float,
    *,
    A_lens: float = DEFAULT_A_LENS,
) -> np.ndarray:
    """Simplified lensing proxy α_τ(R) = A_lens B/(R+r0) [dimensionless benchmark units]."""
    R = np.asarray(R, dtype=float)
    r0_safe = max(float(r0), 1e-12)
    return float(A_lens) * float(B) / (R + r0_safe)


def predict_redshift_tau(
    R_emit: np.ndarray,
    R_obs: np.ndarray,
    B: float,
    r0: float,
    *,
    c_km_s: float = DEFAULT_C_KM_S,
) -> np.ndarray:
    """z_τ = [Φ_τ(R_emit) − Φ_τ(R_obs)] / c² with c in km/s and Φ in km²/s²."""
    R_emit = np.asarray(R_emit, dtype=float)
    R_obs = np.asarray(R_obs, dtype=float)
    c2 = float(c_km_s) ** 2
    return (phi_tau_log(R_emit, B, r0) - phi_tau_log(R_obs, B, r0)) / c2


def _relative_percent(true: np.ndarray, pred: np.ndarray, *, floor: float = 1e-12) -> float:
    true = np.asarray(true, dtype=float)
    pred = np.asarray(pred, dtype=float)
    denom = np.maximum(np.abs(true), floor)
    return float(np.mean(np.abs(pred - true) / denom) * 100.0)


def _param_recovery_error_percent(true: float, recovered: float) -> float:
    if not np.isfinite(recovered):
        return float("inf")
    denom = max(abs(true), 1e-12)
    return float(abs(recovered - true) / denom * 100.0)


BENCHMARK_CASE_REGISTRY: dict[str, SameTauCase] = {
    "same_tau_low_mass": SameTauCase(
        case_name="same_tau_low_mass",
        true_B=420.0,
        true_r0=1.8,
        r_min_kpc=0.3,
        r_max_kpc=12.0,
        n_rotation_points=24,
        R_min_kpc=0.5,
        R_max_kpc=14.0,
        n_lensing_points=20,
        redshift_pairs=(
            RedshiftPairSpec(2.0, 8.0),
            RedshiftPairSpec(4.0, 12.0),
            RedshiftPairSpec(6.0, 10.0),
        ),
        baryon_profile_type="saturating_disk",
        baryon_params={"v_max": 32.0, "r_disk": 0.9},
        rotation_noise_std=0.8,
        lensing_noise_std=0.015,
        redshift_noise_std=5e-7,
        random_seed=501,
        description="Low-mass dwarf-like baryon disk; modest τ amplitude",
    ),
    "same_tau_mid_mass": SameTauCase(
        case_name="same_tau_mid_mass",
        true_B=1800.0,
        true_r0=4.5,
        r_min_kpc=0.5,
        r_max_kpc=22.0,
        n_rotation_points=28,
        R_min_kpc=0.8,
        R_max_kpc=25.0,
        n_lensing_points=22,
        redshift_pairs=(
            RedshiftPairSpec(3.0, 15.0),
            RedshiftPairSpec(8.0, 20.0),
            RedshiftPairSpec(12.0, 22.0),
        ),
        baryon_profile_type="saturating_disk",
        baryon_params={"v_max": 72.0, "r_disk": 2.2},
        rotation_noise_std=1.0,
        lensing_noise_std=0.02,
        redshift_noise_std=8e-7,
        random_seed=502,
        description="Mid-mass spiral-like saturating disk",
    ),
    "same_tau_high_mass": SameTauCase(
        case_name="same_tau_high_mass",
        true_B=5200.0,
        true_r0=9.0,
        r_min_kpc=0.5,
        r_max_kpc=30.0,
        n_rotation_points=30,
        R_min_kpc=1.0,
        R_max_kpc=32.0,
        n_lensing_points=24,
        redshift_pairs=(
            RedshiftPairSpec(5.0, 18.0),
            RedshiftPairSpec(10.0, 28.0),
            RedshiftPairSpec(15.0, 30.0),
        ),
        baryon_profile_type="saturating_disk",
        baryon_params={"v_max": 105.0, "r_disk": 3.0},
        rotation_noise_std=1.2,
        lensing_noise_std=0.025,
        redshift_noise_std=1e-6,
        random_seed=503,
        description="High-mass disk with extended τ scale",
    ),
    "same_tau_lsb": SameTauCase(
        case_name="same_tau_lsb",
        true_B=950.0,
        true_r0=6.5,
        r_min_kpc=0.5,
        r_max_kpc=24.0,
        n_rotation_points=28,
        R_min_kpc=0.8,
        R_max_kpc=26.0,
        n_lensing_points=22,
        redshift_pairs=(
            RedshiftPairSpec(4.0, 16.0),
            RedshiftPairSpec(8.0, 24.0),
        ),
        baryon_profile_type="exponential_like",
        baryon_params={"v_max": 38.0, "r_disk": 4.0},
        rotation_noise_std=1.0,
        lensing_noise_std=0.018,
        redshift_noise_std=6e-7,
        random_seed=504,
        description="LSB exponential-like baryon profile",
    ),
    "same_tau_compact_baryon": SameTauCase(
        case_name="same_tau_compact_baryon",
        true_B=2400.0,
        true_r0=2.2,
        r_min_kpc=0.2,
        r_max_kpc=18.0,
        n_rotation_points=26,
        R_min_kpc=0.4,
        R_max_kpc=20.0,
        n_lensing_points=22,
        redshift_pairs=(
            RedshiftPairSpec(1.5, 10.0),
            RedshiftPairSpec(3.0, 14.0),
            RedshiftPairSpec(5.0, 16.0),
        ),
        baryon_profile_type="compact_bulge_disk",
        baryon_params={
            "v_bulge_max": 55.0,
            "r_bulge": 0.35,
            "v_disk_max": 70.0,
            "r_disk": 1.6,
        },
        rotation_noise_std=1.0,
        lensing_noise_std=0.02,
        redshift_noise_std=7e-7,
        random_seed=505,
        description="Compact bulge+disk baryons with concentrated τ",
    ),
    "same_tau_core_like": SameTauCase(
        case_name="same_tau_core_like",
        true_B=1100.0,
        true_r0=8.0,
        r_min_kpc=0.4,
        r_max_kpc=20.0,
        n_rotation_points=26,
        R_min_kpc=0.6,
        R_max_kpc=22.0,
        n_lensing_points=22,
        redshift_pairs=(
            RedshiftPairSpec(2.0, 12.0),
            RedshiftPairSpec(6.0, 18.0),
            RedshiftPairSpec(10.0, 20.0),
        ),
        baryon_profile_type="exponential_like",
        baryon_params={"v_max": 45.0, "r_disk": 2.5},
        rotation_noise_std=0.9,
        lensing_noise_std=0.018,
        redshift_noise_std=6e-7,
        random_seed=506,
        description="Softer τ scale (large r0) with slowly rising baryon",
    ),
}


def list_benchmark_cases() -> list[str]:
    return list(BENCHMARK_CASE_REGISTRY.keys())


def get_benchmark_case(name: str) -> SameTauCase:
    if name not in BENCHMARK_CASE_REGISTRY:
        raise KeyError(
            f"Unknown same-tau case {name!r}; available: {list_benchmark_cases()}",
        )
    return BENCHMARK_CASE_REGISTRY[name]


def generate_same_tau_teacher_case(case: SameTauCase) -> SameTauTeacherData:
    """Build synthetic rotation, lensing-proxy, and redshift-proxy teachers from one (B, r0)."""
    rng = np.random.default_rng(case.random_seed)
    B, r0 = case.true_B, case.true_r0

    r = np.linspace(case.r_min_kpc, case.r_max_kpc, case.n_rotation_points)
    v_baryon = generate_baryon_profile(r, kind=case.baryon_profile_type, **case.baryon_params)
    v_teacher = v_tdf_simple(r, v_baryon, B, r0)
    v_err = np.full_like(r, max(case.rotation_noise_std, 0.05))
    noise = rng.normal(0.0, case.rotation_noise_std, size=r.shape)
    v_obs = np.maximum(v_teacher + noise, 0.5)

    rotation_df = pd.DataFrame(
        {
            "r_kpc": r,
            "v_baryon": v_baryon,
            "v_teacher": v_teacher,
            "v_obs": v_obs,
            "v_err": v_err,
        },
    )

    R = np.linspace(case.R_min_kpc, case.R_max_kpc, case.n_lensing_points)
    alpha_teacher = predict_lensing_proxy(R, B, r0, A_lens=case.A_lens)
    alpha_err = np.maximum(np.abs(alpha_teacher) * 0.05, case.lensing_noise_std)
    alpha_noise = rng.normal(0.0, case.lensing_noise_std, size=R.shape)
    alpha_obs = np.maximum(alpha_teacher + alpha_noise, 1e-8)

    lensing_df = pd.DataFrame(
        {
            "R_kpc": R,
            "alpha_teacher": alpha_teacher,
            "alpha_obs": alpha_obs,
            "alpha_err": alpha_err,
        },
    )

    pairs = case.redshift_pairs
    R_emit = np.array([p.R_emit_kpc for p in pairs], dtype=float)
    R_obs = np.array([p.R_obs_kpc for p in pairs], dtype=float)
    z_teacher = predict_redshift_tau(R_emit, R_obs, B, r0)
    z_err = np.full_like(z_teacher, max(case.redshift_noise_std, 1e-12))
    z_noise = rng.normal(0.0, case.redshift_noise_std, size=z_teacher.shape)
    z_obs = z_teacher + z_noise

    redshift_df = pd.DataFrame(
        {
            "R_emit_kpc": R_emit,
            "R_obs_kpc": R_obs,
            "z_teacher": z_teacher,
            "z_obs": z_obs,
            "z_err": z_err,
        },
    )

    return SameTauTeacherData(case=case, rotation_df=rotation_df, lensing_df=lensing_df, redshift_df=redshift_df)


def fit_rotation_then_predict_lensing_redshift(
    teacher: SameTauTeacherData,
) -> SameTauConsistencyResult:
    """
    Fit (B, r0) from rotation only; freeze; predict lensing and redshift proxies.
    """
    case = teacher.case
    rot_df = teacher.rotation_df
    warnings: list[str] = []

    r = rot_df["r_kpc"].to_numpy()
    v_obs = rot_df["v_obs"].to_numpy()
    v_err = rot_df["v_err"].to_numpy()
    v_baryon = rot_df["v_baryon"].to_numpy()
    v_teacher = rot_df["v_teacher"].to_numpy()

    B_fit, r0_fit, fit_warnings = _fit_tdf_params(r, v_obs, v_err, v_baryon)
    warnings.extend(fit_warnings)

    if not (np.isfinite(B_fit) and np.isfinite(r0_fit)):
        warnings.append("Rotation-only TDF fit failed; predictions use NaN parameters.")

    v_pred = (
        v_tdf_simple(r, v_baryon, B_fit, r0_fit)
        if np.isfinite(B_fit) and np.isfinite(r0_fit)
        else np.full_like(v_teacher, np.nan)
    )
    rotation_rel = relative_curve_error_percent(v_teacher, v_pred)

    R = teacher.lensing_df["R_kpc"].to_numpy()
    alpha_teacher = teacher.lensing_df["alpha_teacher"].to_numpy()
    alpha_pred = predict_lensing_proxy(R, B_fit, r0_fit, A_lens=case.A_lens)
    lensing_rel = _relative_percent(alpha_teacher, alpha_pred, floor=1e-10)

    R_emit = teacher.redshift_df["R_emit_kpc"].to_numpy()
    R_obs = teacher.redshift_df["R_obs_kpc"].to_numpy()
    z_teacher = teacher.redshift_df["z_teacher"].to_numpy()
    z_pred = predict_redshift_tau(R_emit, R_obs, B_fit, r0_fit)
    redshift_rel = _relative_percent(z_teacher, z_pred, floor=1e-15)

    B_err = _param_recovery_error_percent(case.true_B, B_fit)
    r0_err = _param_recovery_error_percent(case.true_r0, r0_fit)

    same_tau_pass = (
        rotation_rel < PASS_TOLERANCE_PERCENT
        and lensing_rel < PASS_TOLERANCE_PERCENT
        and redshift_rel < PASS_TOLERANCE_PERCENT
    )

    rotation_data = RotationCurveData(
        galaxy_id=case.case_name,
        r_kpc=r,
        v_obs=v_obs,
        v_err=v_err,
        v_baryon=v_baryon,
        metadata={
            "dataset_mode": BENCHMARK_MODE,
            "is_real_observational_data": False,
            "warning_banner": BANNER_SAME_TAU,
            "true_B": case.true_B,
            "true_r0": case.true_r0,
        },
    )

    return SameTauConsistencyResult(
        case_name=case.case_name,
        true_B=case.true_B,
        true_r0=case.true_r0,
        recovered_B=B_fit,
        recovered_r0=r0_fit,
        B_recovery_error_percent=B_err,
        r0_recovery_error_percent=r0_err,
        rotation_relative_error_percent=rotation_rel,
        lensing_relative_error_percent=lensing_rel,
        redshift_relative_error_percent=redshift_rel,
        same_tau_pass=same_tau_pass,
        A_lens=case.A_lens,
        rotation_noise_std=case.rotation_noise_std,
        lensing_noise_std=case.lensing_noise_std,
        redshift_noise_std=case.redshift_noise_std,
        rotation_data=rotation_data,
        lensing_df=teacher.lensing_df.copy(),
        redshift_df=teacher.redshift_df.copy(),
        warnings=warnings,
    )


def _plot_rotation(res: SameTauConsistencyResult, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = res.rotation_data
    r = np.asarray(df.r_kpc, dtype=float)
    v_bary = np.asarray(df.v_baryon, dtype=float)
    v_obs = np.asarray(df.v_obs, dtype=float)
    v_err = np.asarray(df.v_err, dtype=float)

    r_fine = np.linspace(r.min(), r.max(), 200)
    vb_fine = np.interp(r_fine, r, v_bary)
    v_teacher = v_tdf_simple(r, v_bary, res.true_B, res.true_r0)
    v_pred = v_tdf_simple(r_fine, vb_fine, res.recovered_B, res.recovered_r0)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(r, v_obs, yerr=v_err, fmt="o", capsize=3, label="v_obs (rotation fit)", color="C0")
    ax.plot(r, v_teacher, "k-", lw=2, label="teacher (true B, r0)")
    ax.plot(r_fine, v_pred, "C2-", lw=1.8, label="TDF pred (frozen B, r0)")
    ax.plot(r, v_bary, "--", color="C1", alpha=0.7, label="baryon")
    ax.set_xlabel("r [kpc]")
    ax.set_ylabel("v [km/s]")
    ax.set_title(f"{res.case_name} — rotation")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_lensing(res: SameTauConsistencyResult, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = res.lensing_df
    R = df["R_kpc"].to_numpy()
    alpha_teacher = df["alpha_teacher"].to_numpy()
    alpha_pred = predict_lensing_proxy(R, res.recovered_B, res.recovered_r0, A_lens=res.A_lens)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(R, alpha_teacher, "k-o", lw=2, ms=4, label="teacher α_τ")
    ax.plot(R, alpha_pred, "C3-s", lw=1.8, ms=4, label="pred α_τ (frozen B, r0)")
    ax.set_xlabel("R [kpc]")
    ax.set_ylabel("α_τ proxy [arb.]")
    ax.set_title(f"{res.case_name} — lensing proxy")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_redshift(res: SameTauConsistencyResult, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = res.redshift_df
    labels = [f"{re:.1f}→{ro:.1f}" for re, ro in zip(df["R_emit_kpc"], df["R_obs_kpc"])]
    x = np.arange(len(labels))
    z_teacher = df["z_teacher"].to_numpy()
    z_pred = predict_redshift_tau(
        df["R_emit_kpc"].to_numpy(),
        df["R_obs_kpc"].to_numpy(),
        res.recovered_B,
        res.recovered_r0,
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    w = 0.35
    ax.bar(x - w / 2, z_teacher, width=w, label="teacher z_τ", color="C0")
    ax.bar(x + w / 2, z_pred, width=w, label="pred z_τ (frozen B, r0)", color="C2")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("z_τ")
    ax.set_title(f"{res.case_name} — redshift proxy pairs")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _result_to_row(res: SameTauConsistencyResult) -> dict[str, Any]:
    return {
        "case_name": res.case_name,
        "true_B": res.true_B,
        "true_r0": res.true_r0,
        "recovered_B": res.recovered_B,
        "recovered_r0": res.recovered_r0,
        "B_recovery_error_percent": res.B_recovery_error_percent,
        "r0_recovery_error_percent": res.r0_recovery_error_percent,
        "rotation_relative_error_percent": res.rotation_relative_error_percent,
        "lensing_relative_error_percent": res.lensing_relative_error_percent,
        "redshift_relative_error_percent": res.redshift_relative_error_percent,
        "same_tau_pass": res.same_tau_pass,
        "A_lens": res.A_lens,
        "rotation_noise_std": res.rotation_noise_std,
        "lensing_noise_std": res.lensing_noise_std,
        "redshift_noise_std": res.redshift_noise_std,
        "warnings": "; ".join(res.warnings),
        "dataset_mode": BENCHMARK_MODE,
        "is_real_observational_data": False,
    }


def _build_report(results: list[SameTauConsistencyResult]) -> str:
    n = len(results)
    n_pass = sum(1 for r in results if r.same_tau_pass)
    rot_med = float(np.median([r.rotation_relative_error_percent for r in results]))
    lens_med = float(np.median([r.lensing_relative_error_percent for r in results]))
    z_med = float(np.median([r.redshift_relative_error_percent for r in results]))
    worst = max(
        results,
        key=lambda r: max(
            r.rotation_relative_error_percent,
            r.lensing_relative_error_percent,
            r.redshift_relative_error_percent,
        ),
    )

    lines = [
        "# Same-τ multi-observable consistency report (Phase 5C)",
        "",
        f"## ⚠️ {BANNER_SAME_TAU}",
        "",
        "## Purpose",
        "",
        "Test whether **one** fitted τ profile (parameters **B**, **r0** from rotation only) can "
        "also predict lensing-proxy and redshift-proxy observables **without refitting** τ for those channels.",
        "",
        "> **Controlled synthetic benchmark only.** Not observational validation.",
        "",
        "## Equations",
        "",
        "```text",
        "Φ_τ(r) = B log(1 + r/r0)",
        "v_τ² = r dΦ_τ/dr = B r/(r + r0)",
        "v_TDF² = v_baryon² + v_τ²",
        "α_τ(R) = A_lens B/(R + r0)          [A_lens fixed; not fitted per galaxy]",
        "z_τ = [Φ_τ(R_emit) − Φ_τ(R_obs)] / c²   [c in km/s]",
        "```",
        "",
        "## Fitting rule",
        "",
        "- Fit **(B, r0)** from rotation data only.",
        "- **Do not** fit separate lensing or redshift parameters.",
        "- Freeze recovered **(B, r0)** before lensing/redshift predictions.",
        "",
        f"**Pass criterion:** rotation, lensing, and redshift relative errors all < {PASS_TOLERANCE_PERCENT:.0f}%.",
        "",
        "## Summary",
        "",
        f"- **Cases:** {n}",
        f"- **Same-τ pass:** {n_pass} / {n}",
        f"- **Median rotation rel. error:** {rot_med:.3f}%",
        f"- **Median lensing rel. error:** {lens_med:.3f}%",
        f"- **Median redshift rel. error:** {z_med:.3e}%",
        f"- **Worst case (max channel error):** {worst.case_name}",
        "",
        "## Per-case table",
        "",
        "| Case | rot % | lens % | z % | B err % | r0 err % | pass |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for res in results:
        p = "yes" if res.same_tau_pass else "no"
        lines.append(
            f"| {res.case_name} | {res.rotation_relative_error_percent:.3f} | "
            f"{res.lensing_relative_error_percent:.3f} | {res.redshift_relative_error_percent:.3e} | "
            f"{res.B_recovery_error_percent:.3f} | {res.r0_recovery_error_percent:.3f} | {p} |",
        )

    lines.extend(
        [
            "",
            "## Scientific interpretation",
            "",
            "- Passing means **same-τ consistency** in this controlled synthetic setup only.",
            "- It does **not** validate TDF observationally.",
            "- It does **not** prove a single τ field explains real lensing, clocks, or rotation jointly.",
            "",
            "## Failure modes",
            "",
            "- Lensing proxy is simplified; no real lensing geometry or surface-density integration.",
            "- Redshift proxy ignores peculiar velocities and full GR metric decomposition.",
            "- Noise or baryon complexity can degrade rotation recovery and cascade to other channels.",
            "- Future work must replace proxies with real multi-observable data (Phase 6+).",
            "",
            "## Disclaimer",
            "",
            "- **NOT REAL OBSERVATIONAL DATA**",
            "- Does not disprove ΛCDM or dark matter.",
            "",
        ],
    )
    return "\n".join(lines)


def run_same_tau_consistency_benchmark(
    outputs_root: Path | None = None,
    *,
    case_names: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, list[SameTauConsistencyResult]]:
    """Run Phase 5C same-τ benchmark and write CSV, report, and figures."""
    root = Path(__file__).resolve().parents[3]
    outputs = outputs_root or (root / "outputs")
    tables_dir = outputs / "tables"
    reports_dir = outputs / "reports"
    figures_dir = outputs / "figures"
    for d in (tables_dir, reports_dir, figures_dir):
        d.mkdir(parents=True, exist_ok=True)

    names = list(case_names) if case_names is not None else list_benchmark_cases()
    results: list[SameTauConsistencyResult] = []

    for name in names:
        case = get_benchmark_case(name)
        teacher = generate_same_tau_teacher_case(case)
        result = fit_rotation_then_predict_lensing_redshift(teacher)
        results.append(result)
        _plot_rotation(result, figures_dir / f"same_tau_{name}_rotation.png")
        _plot_lensing(result, figures_dir / f"same_tau_{name}_lensing.png")
        _plot_redshift(result, figures_dir / f"same_tau_{name}_redshift.png")

    out_df = pd.DataFrame([_result_to_row(r) for r in results])
    out_df.to_csv(tables_dir / "same_tau_consistency_summary.csv", index=False)
    (reports_dir / "same_tau_consistency_report.md").write_text(
        _build_report(results),
        encoding="utf-8",
    )
    return out_df, results

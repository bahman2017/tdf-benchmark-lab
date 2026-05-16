"""Phase 4A — ΛCDM/NFW rotation benchmark recovery (not observational validation)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from tdf_obs.fitting.fit_rotation import RotationFitResult, fit_single_galaxy_rotation
from tdf_obs.io.schemas import RotationCurveData
from tdf_obs.models.dark_matter import v2_nfw_halo_only, v_nfw_simple
from tdf_obs.models.rotation import baryon_only_model, v_tdf_simple

DATASET_MODE = "nfw_surrogate_validation"
BANNER_LCDM_NFW = "ΛCDM/NFW BENCHMARK — NOT REAL OBSERVATIONAL DATA"
BANNER_NFW_SURROGATE = BANNER_LCDM_NFW  # backward-compatible alias

TEACHER_EQUATION = (
    "v_teacher^2(r) = v_baryon^2(r) + v_NFW^2(r)  "
    "[NFW halo: Vh2 * (ln(1+x) - x/(1+x)) / x,  x = r/rs]"
)
STUDENT_EQUATION = "v_TDF^2(r) = v_baryon^2(r) + B * r / (r + r0)"

DEFAULT_MIMIC_TOLERANCE_PERCENT = 5.0

REQUIRED_SUMMARY_COLUMNS: tuple[str, ...] = (
    "case_name",
    "baryon_profile_type",
    "true_Vh2",
    "true_rs",
    "fitted_tdf_B",
    "fitted_tdf_r0",
    "fitted_nfw_Vh2",
    "fitted_nfw_rs",
    "teacher_student_mse",
    "relative_curve_error_percent",
    "max_fractional_error",
    "mimic_success",
    "best_model_by_bic",
    "tdf_beats_baryon_by_bic",
    "tdf_beats_nfw_by_bic",
)

# Legacy presets (Phase 3B); superseded by BENCHMARK_CASE_REGISTRY.
SURROGATE_GALAXY_PRESETS: dict[str, dict[str, float]] = {
    "nfw_surrogate_low_mass": {"Vh2": 1200.0, "rs": 2.5, "v_max": 55.0, "r_disk": 1.5},
    "nfw_surrogate_mid_mass": {"Vh2": 5500.0, "rs": 7.0, "v_max": 75.0, "r_disk": 2.0},
    "nfw_surrogate_high_mass": {"Vh2": 16000.0, "rs": 14.0, "v_max": 95.0, "r_disk": 2.5},
}


@dataclass(frozen=True)
class NfwBenchmarkCase:
    """One ΛCDM/NFW teacher benchmark configuration (synthetic, not real data)."""

    case_name: str
    r_min_kpc: float
    r_max_kpc: float
    n_points: int
    baryon_profile_type: str
    baryon_params: dict[str, float]
    Vh2: float
    rs: float
    noise_std: float
    random_seed: int
    description: str = ""

    def required_fields(self) -> dict[str, Any]:
        return {
            "case_name": self.case_name,
            "r_min_kpc": self.r_min_kpc,
            "r_max_kpc": self.r_max_kpc,
            "n_points": self.n_points,
            "baryon_profile_type": self.baryon_profile_type,
            "baryon_params": self.baryon_params,
            "Vh2": self.Vh2,
            "rs": self.rs,
            "noise_std": self.noise_std,
            "random_seed": self.random_seed,
        }


BENCHMARK_CASE_REGISTRY: dict[str, NfwBenchmarkCase] = {
    "dwarf_low_mass": NfwBenchmarkCase(
        case_name="dwarf_low_mass",
        r_min_kpc=0.3,
        r_max_kpc=12.0,
        n_points=24,
        baryon_profile_type="saturating_disk",
        baryon_params={"v_max": 35.0, "r_disk": 0.8},
        Vh2=800.0,
        rs=1.2,
        noise_std=1.0,
        random_seed=101,
        description="Low-mass dwarf; small disk and compact halo",
    ),
    "dwarf_extended_lsb": NfwBenchmarkCase(
        case_name="dwarf_extended_lsb",
        r_min_kpc=0.5,
        r_max_kpc=18.0,
        n_points=30,
        baryon_profile_type="exponential_like",
        baryon_params={"v_max": 28.0, "r_disk": 3.5},
        Vh2=600.0,
        rs=2.8,
        noise_std=1.5,
        random_seed=102,
        description="Extended LSB dwarf; slowly rising exponential disk",
    ),
    "compact_dwarf": NfwBenchmarkCase(
        case_name="compact_dwarf",
        r_min_kpc=0.2,
        r_max_kpc=8.0,
        n_points=22,
        baryon_profile_type="compact_bulge_disk",
        baryon_params={"v_bulge_max": 30.0, "r_bulge": 0.3, "v_disk_max": 25.0, "r_disk": 1.0},
        Vh2=1500.0,
        rs=0.9,
        noise_std=1.2,
        random_seed=103,
        description="Compact dwarf with bulge+disk quadrature",
    ),
    "lsb_diffuse": NfwBenchmarkCase(
        case_name="lsb_diffuse",
        r_min_kpc=0.5,
        r_max_kpc=22.0,
        n_points=28,
        baryon_profile_type="exponential_like",
        baryon_params={"v_max": 40.0, "r_disk": 4.5},
        Vh2=2500.0,
        rs=6.0,
        noise_std=2.0,
        random_seed=104,
        description="Diffuse LSB disk with extended halo",
    ),
    "spiral_mid_mass": NfwBenchmarkCase(
        case_name="spiral_mid_mass",
        r_min_kpc=0.5,
        r_max_kpc=25.0,
        n_points=28,
        baryon_profile_type="saturating_disk",
        baryon_params={"v_max": 75.0, "r_disk": 2.0},
        Vh2=5500.0,
        rs=7.0,
        noise_std=1.5,
        random_seed=105,
        description="Mid-mass spiral-like saturating disk",
    ),
    "milky_way_like": NfwBenchmarkCase(
        case_name="milky_way_like",
        r_min_kpc=0.5,
        r_max_kpc=30.0,
        n_points=32,
        baryon_profile_type="saturating_disk",
        baryon_params={"v_max": 85.0, "r_disk": 2.5},
        Vh2=9000.0,
        rs=10.0,
        noise_std=1.5,
        random_seed=106,
        description="Milky-Way-like disk and halo scale",
    ),
    "high_surface_brightness": NfwBenchmarkCase(
        case_name="high_surface_brightness",
        r_min_kpc=0.3,
        r_max_kpc=20.0,
        n_points=26,
        baryon_profile_type="compact_bulge_disk",
        baryon_params={"v_bulge_max": 70.0, "r_bulge": 0.4, "v_disk_max": 90.0, "r_disk": 1.8},
        Vh2=12000.0,
        rs=5.0,
        noise_std=1.0,
        random_seed=107,
        description="HSB galaxy; compact baryons and concentrated halo",
    ),
    "massive_spiral": NfwBenchmarkCase(
        case_name="massive_spiral",
        r_min_kpc=0.5,
        r_max_kpc=28.0,
        n_points=30,
        baryon_profile_type="saturating_disk",
        baryon_params={"v_max": 110.0, "r_disk": 3.0},
        Vh2=18000.0,
        rs=15.0,
        noise_std=2.0,
        random_seed=108,
        description="Massive spiral with large outer halo",
    ),
    "extended_disk": NfwBenchmarkCase(
        case_name="extended_disk",
        r_min_kpc=1.0,
        r_max_kpc=35.0,
        n_points=32,
        baryon_profile_type="exponential_like",
        baryon_params={"v_max": 90.0, "r_disk": 5.0},
        Vh2=8000.0,
        rs=12.0,
        noise_std=1.5,
        random_seed=109,
        description="Extended exponential disk; slowly rising outer curve",
    ),
    "concentrated_halo": NfwBenchmarkCase(
        case_name="concentrated_halo",
        r_min_kpc=0.5,
        r_max_kpc=22.0,
        n_points=28,
        baryon_profile_type="saturating_disk",
        baryon_params={"v_max": 80.0, "r_disk": 2.0},
        Vh2=14000.0,
        rs=3.5,
        noise_std=1.5,
        random_seed=110,
        description="Steep inner halo (small rs) vs disk",
    ),
}


@dataclass
class NfwSurrogateRecord:
    """One benchmark case with teacher truth and fit comparison."""

    galaxy_id: str
    case_name: str
    baryon_profile_type: str
    dataframe: pd.DataFrame
    rotation_data: RotationCurveData
    fit: RotationFitResult
    teacher_student_mse: float
    relative_curve_error_percent: float
    max_fractional_error: float
    mimic_success: bool
    mimic_tolerance_percent: float
    warnings: list[str] = field(default_factory=list)

    @property
    def tdf_mimics_teacher(self) -> bool:
        """Backward-compatible alias for mimic_success."""
        return self.mimic_success


def list_benchmark_cases() -> list[str]:
    return list(BENCHMARK_CASE_REGISTRY.keys())


def get_benchmark_case(name: str) -> NfwBenchmarkCase:
    if name not in BENCHMARK_CASE_REGISTRY:
        raise KeyError(f"Unknown benchmark case {name!r}; available: {list_benchmark_cases()}")
    return BENCHMARK_CASE_REGISTRY[name]


def generate_baryon_profile(
    r: np.ndarray,
    *,
    kind: str = "saturating_disk",
    v_max: float = 70.0,
    r_disk: float = 2.0,
    v_bulge_max: float = 40.0,
    r_bulge: float = 0.5,
    v_disk_max: float = 60.0,
    **kwargs: Any,
) -> np.ndarray:
    """
    Simple smooth baryonic circular-speed profiles [km/s].

    saturating_disk:
        v_b(r) = v_max * sqrt(r / (r + r_disk))

    exponential_like:
        v_b(r) = v_max * (1 - exp(-r / r_disk))

    compact_bulge_disk:
        v_b^2 = v_bulge^2 + v_disk^2
        v_bulge = v_bulge_max * sqrt(r / (r + r_bulge))
        v_disk  = v_disk_max  * sqrt(r / (r + r_disk))
    """
    r = np.asarray(r, dtype=float)
    if kwargs:
        v_max = float(kwargs.get("v_max", v_max))
        r_disk = float(kwargs.get("r_disk", r_disk))
        v_bulge_max = float(kwargs.get("v_bulge_max", v_bulge_max))
        r_bulge = float(kwargs.get("r_bulge", r_bulge))
        v_disk_max = float(kwargs.get("v_disk_max", v_disk_max))

    if kind == "saturating_disk":
        r_d = max(float(r_disk), 1e-6)
        return v_max * np.sqrt(r / (r + r_d))

    if kind == "exponential_like":
        r_d = max(float(r_disk), 1e-6)
        return v_max * (1.0 - np.exp(-r / r_d))

    if kind == "compact_bulge_disk":
        r_b = max(float(r_bulge), 1e-6)
        r_d = max(float(r_disk), 1e-6)
        v_bulge = v_bulge_max * np.sqrt(r / (r + r_b))
        v_disk = v_disk_max * np.sqrt(r / (r + r_d))
        return np.sqrt(v_bulge**2 + v_disk**2)

    raise ValueError(f"Unknown baryon profile kind: {kind!r}")


def generate_baryon_profile_from_case(case: NfwBenchmarkCase, r: np.ndarray) -> np.ndarray:
    return generate_baryon_profile(r, kind=case.baryon_profile_type, **case.baryon_params)


def teacher_velocity(
    r: np.ndarray,
    v_baryon: np.ndarray,
    Vh2: float,
    rs: float,
) -> np.ndarray:
    """Teacher speed v_teacher = sqrt(v_baryon^2 + v_NFW,halo^2)."""
    r = np.asarray(r, dtype=float)
    v_baryon = np.asarray(v_baryon, dtype=float)
    v2 = v_baryon**2 + v2_nfw_halo_only(r, Vh2, rs)
    return np.sqrt(np.maximum(v2, 0.0))


def teacher_student_mse(v_teacher: np.ndarray, v_student: np.ndarray) -> float:
    from tdf_obs.fitting.metrics import mse

    return mse(v_teacher, v_student)


def relative_curve_error_percent(v_teacher: np.ndarray, v_student: np.ndarray) -> float:
    """Mean |v_student - v_teacher| / v_teacher * 100 (%)."""
    v_teacher = np.asarray(v_teacher, dtype=float)
    v_student = np.asarray(v_student, dtype=float)
    denom = np.maximum(np.abs(v_teacher), 1e-3)
    return float(np.mean(np.abs(v_student - v_teacher) / denom) * 100.0)


def max_fractional_error(v_teacher: np.ndarray, v_student: np.ndarray) -> float:
    v_teacher = np.asarray(v_teacher, dtype=float)
    v_student = np.asarray(v_student, dtype=float)
    denom = np.maximum(np.abs(v_teacher), 1e-3)
    return float(np.max(np.abs(v_student - v_teacher) / denom))


def generate_nfw_surrogate_rotation_curve(
    galaxy_id: str,
    r_kpc: np.ndarray,
    v_baryon: np.ndarray,
    Vh2: float,
    rs: float,
    *,
    noise_std: float = 0.0,
    random_seed: int | None = None,
    baryon_profile_type: str | None = None,
) -> pd.DataFrame:
    """Build surrogate rotation-curve table from NFW teacher model."""
    r_kpc = np.asarray(r_kpc, dtype=float)
    v_baryon = np.asarray(v_baryon, dtype=float)
    v_teach = teacher_velocity(r_kpc, v_baryon, Vh2, rs)

    rng = np.random.default_rng(random_seed)
    v_err = np.maximum(0.03 * v_teach, 1.0)
    v_obs = v_teach + rng.normal(0.0, noise_std, size=len(r_kpc)) if noise_std > 0 else v_teach.copy()
    v_obs = np.maximum(v_obs, 0.0)

    out: dict[str, Any] = {
        "galaxy_id": galaxy_id,
        "case_name": galaxy_id,
        "r_kpc": r_kpc,
        "v_obs": v_obs,
        "v_err": v_err,
        "v_baryon": v_baryon,
        "v_teacher": v_teach,
        "nfw_Vh2_true": float(Vh2),
        "nfw_rs_true": float(rs),
        "dataset_mode": DATASET_MODE,
    }
    if baryon_profile_type is not None:
        out["baryon_profile_type"] = baryon_profile_type
    return pd.DataFrame(out)


def dataframe_to_rotation_data(df: pd.DataFrame) -> RotationCurveData:
    return RotationCurveData(
        galaxy_id=str(df["galaxy_id"].iloc[0]),
        r_kpc=df["r_kpc"].to_numpy(dtype=float),
        v_obs=df["v_obs"].to_numpy(dtype=float),
        v_err=df["v_err"].to_numpy(dtype=float),
        v_baryon=df["v_baryon"].to_numpy(dtype=float),
        metadata={
            "dataset_mode": DATASET_MODE,
            "is_real_observational_data": False,
            "v_teacher": df["v_teacher"].to_numpy(dtype=float),
            "nfw_Vh2_true": float(df["nfw_Vh2_true"].iloc[0]),
            "nfw_rs_true": float(df["nfw_rs_true"].iloc[0]),
            "baryon_profile_type": str(df["baryon_profile_type"].iloc[0])
            if "baryon_profile_type" in df.columns
            else "unknown",
        },
    )


def run_single_benchmark_case(
    case: NfwBenchmarkCase,
    *,
    mimic_tolerance_percent: float = DEFAULT_MIMIC_TOLERANCE_PERCENT,
    noise_std: float | None = None,
) -> NfwSurrogateRecord:
    """Generate, fit, and score one registry case."""
    r_kpc = np.linspace(case.r_min_kpc, case.r_max_kpc, case.n_points)
    v_baryon = generate_baryon_profile_from_case(case, r_kpc)
    ns = case.noise_std if noise_std is None else noise_std

    df = generate_nfw_surrogate_rotation_curve(
        case.case_name,
        r_kpc,
        v_baryon,
        case.Vh2,
        case.rs,
        noise_std=ns,
        random_seed=case.random_seed,
        baryon_profile_type=case.baryon_profile_type,
    )
    rot = dataframe_to_rotation_data(df)
    fit = fit_single_galaxy_rotation(rot)

    v_teacher = df["v_teacher"].to_numpy()
    v_tdf_at_data = (
        v_tdf_simple(rot.r_kpc, rot.v_baryon, fit.tdf_B, fit.tdf_r0)
        if fit.success_tdf
        else baryon_only_model(rot.v_baryon)
    )
    ts_mse = teacher_student_mse(v_teacher, v_tdf_at_data)
    rel_err = relative_curve_error_percent(v_teacher, v_tdf_at_data)
    max_frac = max_fractional_error(v_teacher, v_tdf_at_data)
    mimics = rel_err < mimic_tolerance_percent

    warnings: list[str] = list(fit.warnings)
    if not mimics:
        warnings.append(
            f"TDF student did not mimic teacher within {mimic_tolerance_percent}% "
            "mean relative error.",
        )

    return NfwSurrogateRecord(
        galaxy_id=case.case_name,
        case_name=case.case_name,
        baryon_profile_type=case.baryon_profile_type,
        dataframe=df,
        rotation_data=rot,
        fit=fit,
        teacher_student_mse=ts_mse,
        relative_curve_error_percent=rel_err,
        max_fractional_error=max_frac,
        mimic_success=mimics,
        mimic_tolerance_percent=mimic_tolerance_percent,
        warnings=warnings,
    )


def _record_to_summary_row(rec: NfwSurrogateRecord) -> dict[str, Any]:
    return {
        "case_name": rec.case_name,
        "baryon_profile_type": rec.baryon_profile_type,
        "true_Vh2": rec.dataframe["nfw_Vh2_true"].iloc[0],
        "true_rs": rec.dataframe["nfw_rs_true"].iloc[0],
        "fitted_tdf_B": rec.fit.tdf_B,
        "fitted_tdf_r0": rec.fit.tdf_r0,
        "fitted_nfw_Vh2": rec.fit.nfw_Vh2,
        "fitted_nfw_rs": rec.fit.nfw_rs,
        "teacher_student_mse": rec.teacher_student_mse,
        "relative_curve_error_percent": rec.relative_curve_error_percent,
        "max_fractional_error": rec.max_fractional_error,
        "mimic_success": rec.mimic_success,
        "best_model_by_bic": rec.fit.best_model_by_bic,
        "tdf_beats_baryon_by_bic": rec.fit.tdf_beats_baryon_by_bic,
        "tdf_beats_nfw_by_bic": rec.fit.tdf_beats_nfw_by_bic,
        "dataset_mode": DATASET_MODE,
        "is_real_observational_data": False,
        "mimic_tolerance_percent": rec.mimic_tolerance_percent,
        "tdf_mimics_teacher": rec.mimic_success,
        "galaxy_id": rec.galaxy_id,
    }


def _plot_surrogate(record: NfwSurrogateRecord, output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = record.dataframe
    r = df["r_kpc"].to_numpy()
    fit = record.fit
    r_fine = np.linspace(r.min(), r.max(), 200)
    v_bary_fine = np.interp(r_fine, r, df["v_baryon"].to_numpy())

    v_bary = baryon_only_model(df["v_baryon"].to_numpy())
    v_teacher = df["v_teacher"].to_numpy()
    v_tdf = (
        v_tdf_simple(r_fine, v_bary_fine, fit.tdf_B, fit.tdf_r0)
        if fit.success_tdf
        else np.interp(r_fine, r, v_bary)
    )
    v_nfw = (
        v_nfw_simple(r_fine, v_bary_fine, fit.nfw_Vh2, fit.nfw_rs)
        if fit.success_nfw
        else np.interp(r_fine, r, v_bary)
    )

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.errorbar(r, df["v_obs"], yerr=df["v_err"], fmt="o", capsize=3, label="benchmark v_obs", color="C0")
    ax.plot(r, v_teacher, "k-", linewidth=2, label="teacher (NFW+bar)")
    ax.plot(r, v_bary, "--", color="C1", label="baryon-only")
    ax.plot(r_fine, v_tdf, "-", color="C2", label="TDF student fit")
    ax.plot(r_fine, v_nfw, "-.", color="C3", label="NFW refit")
    ax.set_xlabel("r [kpc]")
    ax.set_ylabel("v [km/s]")
    mimic = "yes" if record.mimic_success else "no"
    ax.set_title(
        f"{record.case_name} ({record.baryon_profile_type})\n"
        f"{DATASET_MODE} — mimic: {mimic}",
    )
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _build_report(
    records: list[NfwSurrogateRecord],
    *,
    mimic_tolerance_percent: float,
) -> str:
    rel_errors = [r.relative_curve_error_percent for r in records]
    n_mimic = sum(1 for r in records if r.mimic_success)
    n_cases = len(records)
    pass_frac = n_mimic / n_cases if n_cases else 0.0
    median_err = float(np.median(rel_errors)) if rel_errors else float("nan")
    worst_err = float(np.max(rel_errors)) if rel_errors else float("nan")
    failed = [r.case_name for r in records if not r.mimic_success]

    lines = [
        "# ΛCDM/NFW rotation benchmark report (Phase 4A)",
        "",
        f"## ⚠️ {BANNER_LCDM_NFW}",
        "",
        "> **This is not observational validation.** Controlled ΛCDM/GR+DM teacher benchmarks only. "
        "> This does not validate TDF against real observations. "
        "> Builds on the Phase 3B NFW surrogate scaffold.",
        "",
        "## Scientific interpretation",
        "",
        "**Passing** means TDF can approximate NFW-like rotation phenomenology in these controlled "
        "benchmarks when the mean relative curve error vs the teacher is below the configured tolerance.",
        "",
        "**Passing does not mean** real observational validation, nor that TDF is correct in nature.",
        "",
        "## Models",
        "",
        "**Teacher (ΛCDM/NFW + baryon):**",
        "",
        "```text",
        TEACHER_EQUATION,
        "```",
        "",
        "**Student (TDF simple):**",
        "",
        "```text",
        STUDENT_EQUATION,
        "```",
        "",
        f"**Mimic tolerance:** mean relative curve error < {mimic_tolerance_percent}%",
        "",
        "## Summary",
        "",
        f"- **Number of cases:** {n_cases}",
        f"- **Mimicked by TDF:** {n_mimic}",
        f"- **Pass fraction:** {pass_frac:.1%}",
        f"- **Median relative curve error:** {median_err:.2f}%",
        f"- **Worst-case relative curve error:** {worst_err:.2f}%",
        "",
    ]
    if failed:
        lines.append(f"- **Failed mimic threshold:** {', '.join(failed)}")
    else:
        lines.append("- **Failed mimic threshold:** _(none)_")
    lines.append("")

    lines.extend(
        [
            "## Per-case table",
            "",
            "| Case | Baryon profile | Vh2 true | rs true | rel. error % | mimic | best BIC |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ],
    )
    for rec in records:
        df = rec.dataframe
        mim = "yes" if rec.mimic_success else "no"
        lines.append(
            f"| {rec.case_name} | {rec.baryon_profile_type} | "
            f"{df['nfw_Vh2_true'].iloc[0]:.4g} | {df['nfw_rs_true'].iloc[0]:.4g} | "
            f"{rec.relative_curve_error_percent:.2f} | {mim} | {rec.fit.best_model_by_bic} |",
        )

    lines.extend(
        [
            "",
            "## Failure modes",
            "",
            "The simple TDF ansatz `B·r/(r+r0)` may struggle when:",
            "",
            "- **Inner steep halos** — small r_s with rapid rise in v_NFW at small r.",
            "- **Very compact baryons** — bulge+disk quadrature dominates inner radii.",
            "- **Extreme concentration** — concentrated_halo-like teachers with small rs.",
            "- **Outer slowly rising curves** — extended exponential disks + extended radii.",
            "- **Noisy low-mass profiles** — dwarf cases with higher noise_std.",
            "",
            "A failed mimic does not imply TDF fails on real data; these are synthetic teachers only.",
            "",
            "## Per-case detail",
            "",
        ],
    )

    for rec in records:
        lines.append(f"### {rec.case_name}")
        lines.append("")
        lines.append(f"- Baryon profile: `{rec.baryon_profile_type}`")
        lines.append(
            f"- True NFW: Vh2 = {rec.dataframe['nfw_Vh2_true'].iloc[0]:.6g}, "
            f"rs = {rec.dataframe['nfw_rs_true'].iloc[0]:.6g} kpc",
        )
        lines.append(f"- Fitted TDF: B = {rec.fit.tdf_B:.6g} km²/s², r0 = {rec.fit.tdf_r0:.6g} kpc")
        lines.append(f"- Teacher–student MSE: {rec.teacher_student_mse:.6g}")
        lines.append(f"- Relative curve error: {rec.relative_curve_error_percent:.2f}%")
        lines.append(f"- Max fractional error: {rec.max_fractional_error:.4f}")
        lines.append(f"- Mimic success: **{rec.mimic_success}**")
        if rec.warnings:
            lines.append("- Warnings:")
            for w in rec.warnings:
                lines.append(f"  - {w}")
        lines.append("")

    lines.extend(
        [
            "## Disclaimer",
            "",
            "- **NOT REAL OBSERVATIONAL DATA** — ΛCDM/NFW benchmark teachers only.",
            "- Does **not** validate TDF against the real Universe.",
            "- Proceed to ΛCDM stress regimes (Phase 5) and real calibration (Phase 6) only after reviewing patterns here.",
            "",
        ],
    )
    return "\n".join(lines)


def run_nfw_surrogate_pipeline(
    outputs_root: Path | None = None,
    *,
    case_names: Sequence[str] | None = None,
    max_cases: int | None = None,
    mimic_tolerance_percent: float = DEFAULT_MIMIC_TOLERANCE_PERCENT,
    noise_std: float | None = None,
) -> list[NfwSurrogateRecord]:
    """Run registered ΛCDM/NFW benchmark cases and write Phase 4A outputs."""
    root = Path(__file__).resolve().parents[3]
    outputs = outputs_root or (root / "outputs")
    tables_dir = outputs / "tables"
    reports_dir = outputs / "reports"
    figures_dir = outputs / "figures"
    for d in (tables_dir, reports_dir, figures_dir):
        d.mkdir(parents=True, exist_ok=True)

    names = list(case_names) if case_names is not None else list_benchmark_cases()
    if max_cases is not None:
        names = names[: max_cases]

    records: list[NfwSurrogateRecord] = []
    for name in names:
        case = get_benchmark_case(name)
        record = run_single_benchmark_case(
            case,
            mimic_tolerance_percent=mimic_tolerance_percent,
            noise_std=noise_std,
        )
        records.append(record)
        _plot_surrogate(record, figures_dir / f"nfw_surrogate_{case.case_name}.png")

    summary_rows = [_record_to_summary_row(rec) for rec in records]
    pd.DataFrame(summary_rows).to_csv(tables_dir / "nfw_surrogate_fit_summary.csv", index=False)
    (reports_dir / "nfw_surrogate_report.md").write_text(
        _build_report(records, mimic_tolerance_percent=mimic_tolerance_percent),
        encoding="utf-8",
    )
    return records

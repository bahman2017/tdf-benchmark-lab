"""Phase 5E — CMB acoustic-scale compatibility benchmark (not observational validation)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

BENCHMARK_MODE = "cmb_acoustic_scale_benchmark"
BANNER_CMB = "CMB ACOUSTIC SCALE BENCHMARK — NOT REAL OBSERVATIONAL DATA"

C_KM_S = 299_792.458  # km/s

DEFAULT_PASS_THRESHOLD_PERCENT = 1.0
DEFAULT_N_Z_STEPS = 4000

REQUIRED_SUMMARY_COLUMNS: tuple[str, ...] = (
    "case_name",
    "tau_model",
    "H_zstar_lcdm",
    "H_zstar_tdf",
    "r_s_lcdm",
    "r_s_tdf",
    "D_M_lcdm",
    "D_M_tdf",
    "ell_A_lcdm",
    "ell_A_tdf",
    "rel_error_H_zstar_percent",
    "rel_error_r_s_percent",
    "rel_error_D_M_percent",
    "rel_error_ell_A_percent",
    "cmb_compatibility_pass",
    "warnings",
)


@dataclass(frozen=True)
class CosmologyParams:
    """Flat ΛCDM background parameters (teacher)."""

    H0: float = 67.4  # km/s/Mpc
    Omega_m: float = 0.315
    Omega_Lambda: float = 0.685
    Omega_r: float = 9.0e-5
    z_star: float = 1090.0
    z_max_horizon: float = 2.0e4
    Rb_at_zstar: float = 1025.0  # R_b(z_*) = Rb0/(1+z_*)

    @property
    def Rb0(self) -> float:
        return self.Rb_at_zstar * (1.0 + self.z_star)


@dataclass(frozen=True)
class CmbAcousticCase:
    case_name: str
    tau_model: str
    tau_params: dict[str, float]
    expected_pass: bool
    description: str = ""


@dataclass
class CmbAcousticResult:
    case_name: str
    tau_model: str
    H_zstar_lcdm: float
    H_zstar_tdf: float
    r_s_lcdm: float
    r_s_tdf: float
    D_M_lcdm: float
    D_M_tdf: float
    ell_A_lcdm: float
    ell_A_tdf: float
    rel_error_H_zstar_percent: float
    rel_error_r_s_percent: float
    rel_error_D_M_percent: float
    rel_error_ell_A_percent: float
    cmb_compatibility_pass: bool
    expected_pass: bool
    warnings: list[str] = field(default_factory=list)


BENCHMARK_CASE_REGISTRY: dict[str, CmbAcousticCase] = {
    "lcdm_control": CmbAcousticCase(
        case_name="lcdm_control",
        tau_model="zero_tau",
        tau_params={},
        expected_pass=True,
        description="ΛCDM teacher vs TDF with ε_τ=0 (control)",
    ),
    "zero_tau": CmbAcousticCase(
        case_name="zero_tau",
        tau_model="zero_tau",
        tau_params={},
        expected_pass=True,
        description="Explicit zero τ background correction",
    ),
    "tiny_tau_constant": CmbAcousticCase(
        case_name="tiny_tau_constant",
        tau_model="small_constant_tau",
        tau_params={"epsilon0": 1.0e-6},
        expected_pass=True,
        description="Tiny constant ε_τ",
    ),
    "small_tau_constant": CmbAcousticCase(
        case_name="small_tau_constant",
        tau_model="small_constant_tau",
        tau_params={"epsilon0": 5.0e-5},
        expected_pass=True,
        description="Small constant ε_τ",
    ),
    "recombination_bump_mild": CmbAcousticCase(
        case_name="recombination_bump_mild",
        tau_model="recombination_bump",
        tau_params={"A": 2.0e-5, "sigma": 90.0},
        expected_pass=True,
        description="Mild Gaussian bump at z_*",
    ),
    "recombination_bump_too_large": CmbAcousticCase(
        case_name="recombination_bump_too_large",
        tau_model="recombination_bump",
        tau_params={"A": 0.15, "sigma": 60.0},
        expected_pass=False,
        description="Intentional fail — oversized recombination bump",
    ),
    "early_tau_fraction_mild": CmbAcousticCase(
        case_name="early_tau_fraction_mild",
        tau_model="early_tau_fraction",
        tau_params={"f_tau": 1.0e-5, "decay_scale": 80.0},
        expected_pass=True,
        description="Mild early-universe τ fraction",
    ),
    "early_tau_fraction_too_large": CmbAcousticCase(
        case_name="early_tau_fraction_too_large",
        tau_model="early_tau_fraction",
        tau_params={"f_tau": 0.12, "decay_scale": 120.0},
        expected_pass=False,
        description="Intentional fail — large early τ fraction",
    ),
}


def list_benchmark_cases() -> list[str]:
    return list(BENCHMARK_CASE_REGISTRY.keys())


def get_benchmark_case(name: str) -> CmbAcousticCase:
    if name not in BENCHMARK_CASE_REGISTRY:
        raise KeyError(f"Unknown CMB case {name!r}; available: {list_benchmark_cases()}")
    return BENCHMARK_CASE_REGISTRY[name]


def H_lcdm(z: np.ndarray | float, params: CosmologyParams) -> np.ndarray:
    """ΛCDM H(z) in km/s/Mpc."""
    z = np.asarray(z, dtype=float)
    one_plus_z = 1.0 + z
    e2 = (
        params.Omega_r * one_plus_z**4
        + params.Omega_m * one_plus_z**3
        + params.Omega_Lambda
    )
    return params.H0 * np.sqrt(np.maximum(e2, 0.0))


def epsilon_tau_model(
    z: np.ndarray | float,
    model_name: str,
    tau_params: dict[str, float],
    *,
    z_star: float,
) -> np.ndarray:
    """Configurable ε_τ(z) background correction (dimensionless)."""
    z = np.asarray(z, dtype=float)
    name = model_name.lower()

    if name == "zero_tau":
        return np.zeros_like(z)

    if name == "small_constant_tau":
        return np.full_like(z, float(tau_params.get("epsilon0", 0.0)))

    if name == "recombination_bump":
        A = float(tau_params.get("A", 0.0))
        sigma = max(float(tau_params.get("sigma", 50.0)), 1e-6)
        z_s = float(tau_params.get("z_star", z_star))
        return A * np.exp(-0.5 * ((z - z_s) / sigma) ** 2)

    if name == "early_tau_fraction":
        f_tau = float(tau_params.get("f_tau", 0.0))
        decay_scale = max(float(tau_params.get("decay_scale", 50.0)), 1e-6)
        z_s = float(tau_params.get("z_star", z_star))
        out = np.zeros_like(z)
        high = z > z_s
        out[high] = f_tau
        low = ~high
        if np.any(low):
            out[low] = f_tau * np.exp(-0.5 * ((z_s - z[low]) / decay_scale) ** 2)
        return out

    raise ValueError(f"Unknown tau model {model_name!r}")


def H_tdf(
    z: np.ndarray | float,
    params: CosmologyParams,
    tau_model: str,
    tau_params: dict[str, float],
) -> np.ndarray:
    """H_TDF² = H_LCDM² [1 + ε_τ(z)] → H_TDF = H_LCDM √(1+ε_τ)."""
    h = H_lcdm(z, params)
    eps = epsilon_tau_model(z, tau_model, tau_params, z_star=params.z_star)
    factor = np.maximum(1.0 + eps, 1e-12)
    return h * np.sqrt(factor)


def R_baryon(z: np.ndarray | float, params: CosmologyParams) -> np.ndarray:
    """R_b(z) = Rb0/(1+z)."""
    z = np.asarray(z, dtype=float)
    return params.Rb0 / (1.0 + z)


def sound_speed_photon_baryon(z: np.ndarray | float, params: CosmologyParams) -> np.ndarray:
    """c_s = c / √(3(1+R_b))."""
    z = np.asarray(z, dtype=float)
    Rb = R_baryon(z, params)
    return C_KM_S / np.sqrt(3.0 * (1.0 + Rb))


def _integrate_trapz(
    z_grid: np.ndarray,
    integrand: np.ndarray,
) -> float:
    order = np.argsort(z_grid)
    z_sorted = z_grid[order]
    f_sorted = integrand[order]
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(f_sorted, z_sorted))
    return float(np.trapz(f_sorted, z_sorted))  # noqa: NPY201 — fallback for NumPy < 2


def sound_horizon_proxy(
    z_star: float,
    z_max: float,
    H_func: Callable[[np.ndarray], np.ndarray],
    params: CosmologyParams,
    *,
    n_steps: int = DEFAULT_N_Z_STEPS,
) -> float:
    """r_s = ∫_{z_*}^z_max c_s(z)/H(z) dz [Mpc]."""
    z_grid = np.linspace(float(z_star), float(z_max), n_steps)
    cs = sound_speed_photon_baryon(z_grid, params)
    H = np.maximum(H_func(z_grid), 1e-30)
    return _integrate_trapz(z_grid, cs / H)


def comoving_distance(
    z_star: float,
    H_func: Callable[[np.ndarray], np.ndarray],
    *,
    n_steps: int = DEFAULT_N_Z_STEPS,
) -> float:
    """D_M = ∫_0^{z_*} c/H(z) dz [Mpc]."""
    z_grid = np.linspace(0.0, float(z_star), n_steps)
    H = np.maximum(H_func(z_grid), 1e-30)
    return _integrate_trapz(z_grid, C_KM_S / H)


def acoustic_scale(D_M: float, r_s: float) -> float:
    """ℓ_A = π D_M / r_s."""
    if r_s <= 0 or not np.isfinite(r_s):
        return float("nan")
    return float(np.pi * D_M / r_s)


def _rel_error_percent(true: float, approx: float) -> float:
    denom = max(abs(true), 1e-30)
    return float(abs(approx - true) / denom * 100.0)


def run_single_cmb_case(
    case: CmbAcousticCase,
    params: CosmologyParams | None = None,
    *,
    pass_threshold_percent: float = DEFAULT_PASS_THRESHOLD_PERCENT,
) -> CmbAcousticResult:
    """Compare ΛCDM teacher vs TDF student for one benchmark case."""
    params = params or CosmologyParams()
    warnings: list[str] = []

    H_lcdm_func = lambda z: H_lcdm(z, params)
    H_tdf_func = lambda z: H_tdf(z, params, case.tau_model, case.tau_params)

    z_s = params.z_star
    H_z_lcdm = float(H_lcdm(z_s, params))
    H_z_tdf = float(H_tdf(z_s, params, case.tau_model, case.tau_params))

    rs_lcdm = sound_horizon_proxy(z_s, params.z_max_horizon, H_lcdm_func, params)
    rs_tdf = sound_horizon_proxy(z_s, params.z_max_horizon, H_tdf_func, params)
    DM_lcdm = comoving_distance(z_s, H_lcdm_func)
    DM_tdf = comoving_distance(z_s, H_tdf_func)
    ell_lcdm = acoustic_scale(DM_lcdm, rs_lcdm)
    ell_tdf = acoustic_scale(DM_tdf, rs_tdf)

    for label, val in [
        ("H_lcdm", H_z_lcdm),
        ("H_tdf", H_z_tdf),
        ("r_s_lcdm", rs_lcdm),
        ("r_s_tdf", rs_tdf),
        ("D_M_lcdm", DM_lcdm),
        ("D_M_tdf", DM_tdf),
        ("ell_A_lcdm", ell_lcdm),
        ("ell_A_tdf", ell_tdf),
    ]:
        if not np.isfinite(val) or val <= 0:
            warnings.append(f"Non-finite or non-positive {label}={val}")

    err_H = _rel_error_percent(H_z_lcdm, H_z_tdf)
    err_rs = _rel_error_percent(rs_lcdm, rs_tdf)
    err_DM = _rel_error_percent(DM_lcdm, DM_tdf)
    err_ell = _rel_error_percent(ell_lcdm, ell_tdf)

    passed = (
        err_H < pass_threshold_percent
        and err_rs < pass_threshold_percent
        and err_DM < pass_threshold_percent
        and err_ell < pass_threshold_percent
    )

    if passed != case.expected_pass:
        warnings.append(
            f"Outcome {'pass' if passed else 'fail'} differs from expected "
            f"{'pass' if case.expected_pass else 'fail'} for this case.",
        )

    return CmbAcousticResult(
        case_name=case.case_name,
        tau_model=case.tau_model,
        H_zstar_lcdm=H_z_lcdm,
        H_zstar_tdf=H_z_tdf,
        r_s_lcdm=rs_lcdm,
        r_s_tdf=rs_tdf,
        D_M_lcdm=DM_lcdm,
        D_M_tdf=DM_tdf,
        ell_A_lcdm=ell_lcdm,
        ell_A_tdf=ell_tdf,
        rel_error_H_zstar_percent=err_H,
        rel_error_r_s_percent=err_rs,
        rel_error_D_M_percent=err_DM,
        rel_error_ell_A_percent=err_ell,
        cmb_compatibility_pass=passed,
        expected_pass=case.expected_pass,
        warnings=warnings,
    )


def _plot_epsilon_cases(
    cases: Sequence[CmbAcousticCase],
    params: CosmologyParams,
    output_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    z = np.linspace(0.0, params.z_max_horizon * 0.15, 500)
    z = np.concatenate([z, np.linspace(z.max(), params.z_star * 1.05, 300)])
    z = np.unique(np.sort(z))

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for case in cases:
        eps = epsilon_tau_model(z, case.tau_model, case.tau_params, z_star=params.z_star)
        ax.plot(z, eps, label=case.case_name, lw=1.5)

    ax.axvline(params.z_star, color="k", ls="--", alpha=0.4, label="z_*")
    ax.set_xlabel("z")
    ax.set_ylabel("ε_τ(z)")
    ax.set_title(f"ε_τ models — {BANNER_CMB[:48]}...")
    ax.legend(fontsize=6, ncol=2, loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _result_to_row(res: CmbAcousticResult) -> dict[str, Any]:
    return {
        "case_name": res.case_name,
        "tau_model": res.tau_model,
        "H_zstar_lcdm": res.H_zstar_lcdm,
        "H_zstar_tdf": res.H_zstar_tdf,
        "r_s_lcdm": res.r_s_lcdm,
        "r_s_tdf": res.r_s_tdf,
        "D_M_lcdm": res.D_M_lcdm,
        "D_M_tdf": res.D_M_tdf,
        "ell_A_lcdm": res.ell_A_lcdm,
        "ell_A_tdf": res.ell_A_tdf,
        "rel_error_H_zstar_percent": res.rel_error_H_zstar_percent,
        "rel_error_r_s_percent": res.rel_error_r_s_percent,
        "rel_error_D_M_percent": res.rel_error_D_M_percent,
        "rel_error_ell_A_percent": res.rel_error_ell_A_percent,
        "cmb_compatibility_pass": res.cmb_compatibility_pass,
        "expected_pass": res.expected_pass,
        "warnings": "; ".join(res.warnings),
        "dataset_mode": BENCHMARK_MODE,
        "is_real_observational_data": False,
    }


def _build_report(
    results: list[CmbAcousticResult],
    *,
    pass_threshold_percent: float,
) -> str:
    n = len(results)
    n_pass = sum(1 for r in results if r.cmb_compatibility_pass)
    intentional_fail = [r for r in results if not r.expected_pass]
    worst = max(results, key=lambda r: r.rel_error_ell_A_percent)

    lines = [
        "# CMB acoustic-scale compatibility report (Phase 5E)",
        "",
        f"## ⚠️ {BANNER_CMB}",
        "",
        "## Purpose",
        "",
        "This benchmark checks whether **simple TDF background corrections** "
        "`H_TDF² = H_ΛCDM² [1 + ε_τ(z)]` preserve the **acoustic-scale proxy structure** "
        "produced by a flat ΛCDM teacher — **not** whether TDF explains the CMB.",
        "",
        "> **NOT REAL OBSERVATIONAL DATA.** No Planck/ACT/SPT likelihoods.",
        "",
        "## Equations",
        "",
        "```text",
        "H_ΛCDM²(z) = H0² [Ω_r(1+z)⁴ + Ω_m(1+z)³ + Ω_Λ]",
        "H_TDF²(z) = H_ΛCDM²(z) [1 + ε_τ(z)]",
        "R_b(z) = Rb0/(1+z)",
        "c_s(z) = c / √(3(1+R_b))",
        "r_s = ∫_{z_*}^{z_max} c_s/H dz",
        "D_M = ∫_0^{z_*} c/H dz",
        "ℓ_A = π D_M / r_s",
        "```",
        "",
        f"**Pass thresholds:** H, r_s, D_M, ℓ_A relative errors each < {pass_threshold_percent}%.",
        "",
        "## Summary",
        "",
        f"- **Cases:** {n}",
        f"- **Compatibility pass:** {n_pass} / {n}",
        f"- **Intentional fail cases:** {len(intentional_fail)} "
        f"({', '.join(r.case_name for r in intentional_fail) or 'none'})",
        f"- **Worst ℓ_A error:** {worst.case_name} ({worst.rel_error_ell_A_percent:.4f}%)",
        "",
        "## Results table",
        "",
        "| Case | τ model | ℓ_A err % | r_s err % | D_M err % | H err % | pass | expected |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for res in results:
        lines.append(
            f"| {res.case_name} | {res.tau_model} | {res.rel_error_ell_A_percent:.4f} | "
            f"{res.rel_error_r_s_percent:.4f} | {res.rel_error_D_M_percent:.4f} | "
            f"{res.rel_error_H_zstar_percent:.4f} | "
            f"{'✓' if res.cmb_compatibility_pass else '✗'} | "
            f"{'pass' if res.expected_pass else 'fail'} |",
        )

    lines.extend(
        [
            "",
            "## Scientific interpretation",
            "",
            "- **Passing** means TDF does **not spoil** acoustic-scale proxies in this controlled "
            "ΛCDM-teacher benchmark.",
            "- **Passing does not** validate TDF against CMB observations or prove TDF explains the CMB.",
            "- Real CMB constraints require **Boltzmann perturbation** calculations and data likelihoods.",
            "",
            "## Failure modes",
            "",
            "- Full photon–baryon perturbation equations are **not** solved.",
            "- No Planck/ACT/SPT data or likelihoods.",
            "- ε_τ(z) is a **background-only** proxy; peak heights and damping tail are not computed.",
            "- Units follow simplified integral proxies (c in km/s, H in km/s/Mpc, distances in Mpc).",
            "",
            "## Disclaimer",
            "",
            "- **NOT REAL OBSERVATIONAL DATA**",
            "- ΛCDM teacher benchmark only.",
            "",
        ],
    )
    return "\n".join(lines)


def run_cmb_acoustic_benchmark(
    cases: Sequence[str] | None = None,
    outputs_root: Path | None = None,
    *,
    cosmology: CosmologyParams | None = None,
    pass_threshold_percent: float = DEFAULT_PASS_THRESHOLD_PERCENT,
) -> tuple[pd.DataFrame, list[CmbAcousticResult]]:
    """Run Phase 5E CMB acoustic-scale benchmark and write outputs."""
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
        run_single_cmb_case(c, params, pass_threshold_percent=pass_threshold_percent)
        for c in case_objs
    ]

    out_df = pd.DataFrame([_result_to_row(r) for r in results])
    out_df.to_csv(tables_dir / "cmb_acoustic_benchmark_summary.csv", index=False)
    (reports_dir / "cmb_acoustic_benchmark_report.md").write_text(
        _build_report(results, pass_threshold_percent=pass_threshold_percent),
        encoding="utf-8",
    )
    _plot_epsilon_cases(case_objs, params, figures_dir / "cmb_acoustic_epsilon_tau_cases.png")
    return out_df, results

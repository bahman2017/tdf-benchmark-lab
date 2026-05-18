"""
SPARC Step 6A — inverse-design diagnostics for τ from dark-matter-like residuals.

Reverse-engineers required a_τ(r) from SPARC rotation curves vs baryons.
Inverse-design diagnostics only; not observational validation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit

from tdf_obs.validation.sparc_galaxy_class_analysis import (
    CLASS_ORDER,
    build_galaxy_properties,
    classify_galaxy_by_vmax,
)
from tdf_obs.validation.sparc_real_calibration import (
    A0_TDF_DEFAULT,
    _sq_sign_safe,
    galaxy_has_bulge,
    validate_sparc_input_schema,
)

BANNER_TAU_INVERSE = (
    "SPARC TAU INVERSE DESIGN — DARK-MATTER PHENOMENOLOGY PROXY, "
    "NOT FULL OBSERVATIONAL VALIDATION"
)

BANNER_CALIBRATION = (
    "These results are calibration diagnostics for a phenomenological TDF model. "
    "They do not constitute observational validation of TDF unless all required "
    "multi-channel constraints are passed."
)

CANONICAL_UPSILON_DISK = 0.5
CANONICAL_UPSILON_BULGE = 0.7
OUTER_FRAC = 0.7
INNER_FRAC = 0.3
R_MIN_KPC = 0.02
ACCEL_EPS = 1e-6

PROFILE_COLUMNS: tuple[str, ...] = (
    "galaxy_id",
    "r_kpc",
    "r_over_rmax",
    "v_obs",
    "v_baryon",
    "a_obs",
    "a_baryon",
    "a_tau_required",
    "rho_tau_eff_proxy",
    "rho_tau_cyl_proxy",
    "beta_eff",
)

HALO_SUMMARY_COLUMNS: tuple[str, ...] = (
    "galaxy_id",
    "galaxy_class",
    "vmax_obs",
    "rmax_kpc",
    "n_points",
    "Y_disk",
    "Y_bulge",
    "a0_used",
    "beta_over_M_fitted",
    "median_beta_eff",
    "iqr_beta_eff",
    "beta_eff_outer_median",
    "beta_eff_inner_median",
    "beta_eff_vs_ab_slope",
    "outer_slope_s",
    "outer_slope_status",
    "core_A_inv",
    "core_r_c_inv",
    "core_fit_rmse_inv",
    "core_A_soft",
    "core_r_c_soft",
    "core_fit_rmse_soft",
    "rho_tau_eff_median",
    "rho_tau_cyl_median",
    "low_accel_fraction",
)

CONSTRAINT_COLUMNS: tuple[str, ...] = (
    "galaxy_id",
    "needs_core_regularization",
    "outer_mond_like",
    "inner_transition_needed",
    "beta_eff_nonconstant",
    "compatible_with_single_global_beta",
    "dwarf_enhanced_tau_response",
    "massive_suppressed_tau_response",
)


@dataclass
class TauInverseDesignResult:
    profiles: pd.DataFrame
    halo_summary: pd.DataFrame
    design_constraints: pd.DataFrame
    input_run: Path


def v2_baryon_user(
    v_gas: np.ndarray,
    v_disk: np.ndarray,
    v_bulge: np.ndarray,
    y_disk: float,
    y_bulge: float,
) -> np.ndarray:
    """v_baryon² = Y_disk v_disk² + Y_bulge v_bulge² + |v_gas| v_gas."""
    return (
        float(y_disk) * _sq_sign_safe(v_disk)
        + float(y_bulge) * _sq_sign_safe(v_bulge)
        + np.abs(v_gas) * v_gas
    )


def compute_a_tau_required(
    r: np.ndarray,
    v_obs: np.ndarray,
    v2_baryon: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return a_obs, a_baryon, a_tau_required (nonnegative)."""
    r_safe = np.maximum(np.asarray(r, dtype=float), R_MIN_KPC)
    v_obs = np.asarray(v_obs, dtype=float)
    v2_b = np.maximum(np.asarray(v2_baryon, dtype=float), 0.0)
    a_obs = _sq_sign_safe(v_obs) / r_safe
    a_b = v2_b / r_safe
    a_tau = np.maximum(a_obs - a_b, 0.0)
    return a_obs, a_b, a_tau


def rho_tau_proxies(r: np.ndarray, a_tau: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Spherical and cylindrical density proxies from required acceleration.

    rho_tau_eff ∝ (1/r²) d/dr [r² a_τ]  (labeled proxy)
    rho_tau_cyl ∝ (1/r) d/dr [r a_τ]    (labeled proxy)
    """
    r = np.asarray(r, dtype=float)
    a_tau = np.asarray(a_tau, dtype=float)
    r_safe = np.maximum(r, R_MIN_KPC)
    g_sphere = r_safe**2 * a_tau
    g_cyl = r_safe * a_tau
    dg_dr_sphere = np.gradient(g_sphere, r_safe, edge_order=1)
    dg_dr_cyl = np.gradient(g_cyl, r_safe, edge_order=1)
    rho_eff = dg_dr_sphere / np.maximum(r_safe**2, R_MIN_KPC**2)
    rho_cyl = dg_dr_cyl / np.maximum(r_safe, R_MIN_KPC)
    rho_eff = np.where(np.isfinite(rho_eff), np.maximum(rho_eff, 0.0), np.nan)
    rho_cyl = np.where(np.isfinite(rho_cyl), np.maximum(rho_cyl, 0.0), np.nan)
    return rho_eff, rho_cyl


def beta_eff_profile(a_tau: np.ndarray, a_b: np.ndarray, a0: float) -> np.ndarray:
    a0 = max(float(a0), 1e-30)
    ab = np.maximum(np.asarray(a_b, dtype=float), ACCEL_EPS)
    denom = np.sqrt(ab * a0) + ACCEL_EPS
    return np.asarray(a_tau, dtype=float) / denom


def _core_model_inv(r: np.ndarray, a: float, rc: float) -> np.ndarray:
    return a / (np.maximum(r, R_MIN_KPC) + max(rc, R_MIN_KPC))


def _core_model_soft(r: np.ndarray, a: float, rc: float) -> np.ndarray:
    return a / np.sqrt(np.maximum(r, R_MIN_KPC) ** 2 + max(rc, R_MIN_KPC) ** 2)


def fit_core_regularization(
    r: np.ndarray,
    a_tau: np.ndarray,
    form: Literal["inv", "soft"] = "inv",
) -> tuple[float, float, float]:
    """Fit A, r_c and return RMSE; NaNs if fit fails."""
    mask = np.isfinite(r) & np.isfinite(a_tau) & (r > R_MIN_KPC) & (a_tau > ACCEL_EPS)
    if int(mask.sum()) < 4:
        return float("nan"), float("nan"), float("nan")
    rr = r[mask]
    aa = a_tau[mask]
    model = _core_model_inv if form == "inv" else _core_model_soft
    try:
        popt, _ = curve_fit(
            model,
            rr,
            aa,
            p0=[float(np.median(aa) * np.median(rr)), float(np.median(rr) * 0.2)],
            bounds=([1e-6, R_MIN_KPC], [1e12, 500.0]),
            maxfev=8000,
        )
        pred = model(rr, *popt)
        rmse = float(np.sqrt(np.mean((pred - aa) ** 2)))
        return float(popt[0]), float(popt[1]), rmse
    except (RuntimeError, ValueError):
        return float("nan"), float("nan"), float("nan")


def fit_outer_slope(r: np.ndarray, a_tau: np.ndarray, rmax: float) -> tuple[float, str]:
    """Fit a_tau ~ r^s for r > 0.7 rmax on points with a_tau > eps."""
    outer = (r > OUTER_FRAC * rmax) & (a_tau > ACCEL_EPS) & np.isfinite(a_tau)
    n = int(outer.sum())
    if n < 3:
        return float("nan"), "insufficient_points"
    log_r = np.log(np.maximum(r[outer], R_MIN_KPC))
    log_a = np.log(np.maximum(a_tau[outer], ACCEL_EPS))
    if np.std(log_r) < 1e-6:
        return float("nan"), "degenerate_radius"
    slope, _ = np.polyfit(log_r, log_a, 1)
    if not np.isfinite(slope):
        return float("nan"), "unstable"
    return float(slope), "ok"


def _beta_stats(beta: np.ndarray, r: np.ndarray, rmax: float) -> dict[str, float]:
    beta = beta[np.isfinite(beta)]
    if len(beta) == 0:
        return {
            "median_beta_eff": float("nan"),
            "iqr_beta_eff": float("nan"),
            "beta_eff_outer_median": float("nan"),
            "beta_eff_inner_median": float("nan"),
            "beta_eff_vs_ab_slope": float("nan"),
        }
    q25, med, q75 = np.percentile(beta, [25, 50, 75])
    outer = r > OUTER_FRAC * rmax
    inner = r < INNER_FRAC * rmax
    return {
        "median_beta_eff": float(med),
        "iqr_beta_eff": float(q75 - q25),
        "beta_eff_outer_median": float(np.median(beta[outer])) if outer.any() else float("nan"),
        "beta_eff_inner_median": float(np.median(beta[inner])) if inner.any() else float("nan"),
        "beta_eff_vs_ab_slope": float("nan"),
    }


def process_galaxy(
    gid: str,
    gdf: pd.DataFrame,
    tdf_row: pd.Series | None,
    props: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, bool]]:
    r = gdf["r_kpc"].to_numpy(dtype=float)
    v_obs = gdf["v_obs"].to_numpy(dtype=float)
    v_gas = gdf["v_gas"].to_numpy(dtype=float)
    v_disk = gdf["v_disk"].to_numpy(dtype=float)
    v_bulge = gdf["v_bulge"].to_numpy(dtype=float)

    if tdf_row is not None and bool(tdf_row.get("success", True)):
        y_disk = float(tdf_row.get("upsilon_disk", CANONICAL_UPSILON_DISK) or CANONICAL_UPSILON_DISK)
        y_bulge = float(tdf_row.get("upsilon_bulge", CANONICAL_UPSILON_BULGE) or 0.0)
        if not galaxy_has_bulge(v_bulge):
            y_bulge = 0.0
        beta_fit = float(tdf_row.get("beta_over_M", np.nan))
        a0 = float(tdf_row.get("a0", A0_TDF_DEFAULT) or A0_TDF_DEFAULT)
    else:
        y_disk = CANONICAL_UPSILON_DISK
        y_bulge = CANONICAL_UPSILON_BULGE if galaxy_has_bulge(v_bulge) else 0.0
        beta_fit = float("nan")
        a0 = A0_TDF_DEFAULT

    v2_b = v2_baryon_user(v_gas, v_disk, v_bulge, y_disk, y_bulge)
    v_b = np.sqrt(np.maximum(v2_b, 0.0))
    a_obs, a_b, a_tau = compute_a_tau_required(r, v_obs, v2_b)
    rho_eff, rho_cyl = rho_tau_proxies(r, a_tau)
    beta = beta_eff_profile(a_tau, a_b, a0)

    rmax = float(np.max(r))
    slope_s, slope_status = fit_outer_slope(r, a_tau, rmax)
    a_inv, rc_inv, rmse_inv = fit_core_regularization(r, a_tau, "inv")
    a_soft, rc_soft, rmse_soft = fit_core_regularization(r, a_tau, "soft")

    bstats = _beta_stats(beta, r, rmax)
    mask_ab = np.isfinite(a_b) & np.isfinite(beta) & (a_b > ACCEL_EPS)
    if int(mask_ab.sum()) >= 4:
        slope_ab, _, _, _, _ = stats.linregress(np.log10(a_b[mask_ab]), beta[mask_ab])
        bstats["beta_eff_vs_ab_slope"] = float(slope_ab)

    low_acc_frac = float(np.mean(a_b < a0))

    prof = pd.DataFrame(
        {
            "galaxy_id": gid,
            "r_kpc": r,
            "r_over_rmax": r / max(rmax, R_MIN_KPC),
            "v_obs": v_obs,
            "v_baryon": v_b,
            "a_obs": a_obs,
            "a_baryon": a_b,
            "a_tau_required": a_tau,
            "rho_tau_eff_proxy": rho_eff,
            "rho_tau_cyl_proxy": rho_cyl,
            "beta_eff": beta,
        },
    )

    summary = {
        "galaxy_id": gid,
        "galaxy_class": props.get("galaxy_class", classify_galaxy_by_vmax(float(props.get("vmax_obs", 0)))),
        "vmax_obs": float(props.get("vmax_obs", np.nan)),
        "rmax_kpc": rmax,
        "n_points": int(len(r)),
        "Y_disk": y_disk,
        "Y_bulge": y_bulge,
        "a0_used": a0,
        "beta_over_M_fitted": beta_fit,
        "outer_slope_s": slope_s,
        "outer_slope_status": slope_status,
        "core_A_inv": a_inv,
        "core_r_c_inv": rc_inv,
        "core_fit_rmse_inv": rmse_inv,
        "core_A_soft": a_soft,
        "core_r_c_soft": rc_soft,
        "core_fit_rmse_soft": rmse_soft,
        "rho_tau_eff_median": float(np.nanmedian(rho_eff)),
        "rho_tau_cyl_median": float(np.nanmedian(rho_cyl)),
        "low_accel_fraction": low_acc_frac,
        **bstats,
    }

    flags = _design_flags(summary, global_beta_median=float("nan"))
    return prof, summary, flags


def _design_flags(row: dict[str, Any], global_beta_median: float) -> dict[str, bool]:
    med = row.get("median_beta_eff", np.nan)
    iqr = row.get("iqr_beta_eff", np.nan)
    inner = row.get("beta_eff_inner_median", np.nan)
    outer = row.get("beta_eff_outer_median", np.nan)
    s = row.get("outer_slope_s", np.nan)
    rc = row.get("core_r_c_soft", row.get("core_r_c_inv", np.nan))
    rmse_soft = row.get("core_fit_rmse_soft", np.nan)
    gclass = str(row.get("galaxy_class", ""))

    needs_core = bool(np.isfinite(rc) and rc > 0.15 and np.isfinite(rmse_soft) and rmse_soft < 500)
    mond_like = bool(np.isfinite(s) and -1.5 <= s <= -0.5)
    inner_trans = bool(
        np.isfinite(inner)
        and np.isfinite(outer)
        and outer > ACCEL_EPS
        and inner > 1.4 * outer,
    )
    nonconst = bool(np.isfinite(med) and med > ACCEL_EPS and np.isfinite(iqr) and iqr / med > 0.35)
    global_ok = bool(
        np.isfinite(med)
        and np.isfinite(global_beta_median)
        and global_beta_median > ACCEL_EPS
        and abs(med - global_beta_median) / global_beta_median < 0.35,
    )
    dwarf_enh = gclass == "dwarf" and bool(np.isfinite(med) and med > 0.25)
    massive_sup = gclass == "massive" and bool(np.isfinite(med) and med < 0.2)

    return {
        "needs_core_regularization": needs_core,
        "outer_mond_like": mond_like,
        "inner_transition_needed": inner_trans,
        "beta_eff_nonconstant": nonconst,
        "compatible_with_single_global_beta": global_ok,
        "dwarf_enhanced_tau_response": dwarf_enh,
        "massive_suppressed_tau_response": massive_sup,
    }


def load_tdf_parameters(summary_df: pd.DataFrame) -> pd.DataFrame:
    sub = summary_df[summary_df["model"] == "tdf_kessence"].copy()
    return sub.set_index("galaxy_id", drop=False)


def run_tau_inverse_design(
    input_run: Path,
    sparc_data_path: Path,
    output_dir: Path,
    *,
    max_galaxies: int | None = None,
) -> TauInverseDesignResult:
    input_run = Path(input_run)
    output_dir = Path(output_dir)
    sparc_df = pd.read_csv(sparc_data_path)
    validate_sparc_input_schema(sparc_df)

    summary_path = input_run / "tables" / "sparc_real_calibration_summary.csv"
    if not summary_path.is_file():
        raise FileNotFoundError(f"Missing calibration summary: {summary_path}")
    summary_df = pd.read_csv(summary_path)
    tdf_by_gid = load_tdf_parameters(summary_df)
    props_df = build_galaxy_properties(sparc_df)
    props_map = props_df.set_index("galaxy_id").to_dict("index")

    profiles_list: list[pd.DataFrame] = []
    summaries: list[dict[str, Any]] = []
    constraints: list[dict[str, Any]] = []

    gids = sorted(sparc_df["galaxy_id"].unique())
    if max_galaxies is not None:
        gids = gids[: max_galaxies]

    for gid in gids:
        gdf = sparc_df[sparc_df["galaxy_id"] == gid].sort_values("r_kpc")
        if len(gdf) < 5:
            continue
        tdf_row = tdf_by_gid.loc[gid] if gid in tdf_by_gid.index else None
        props = props_map.get(gid, {"galaxy_id": gid, "vmax_obs": float(gdf["v_obs"].max())})
        prof, summ, flags = process_galaxy(str(gid), gdf, tdf_row, props)
        profiles_list.append(prof)
        summaries.append(summ)
        constraints.append({"galaxy_id": gid, **flags})

    profiles = pd.concat(profiles_list, ignore_index=True) if profiles_list else pd.DataFrame(columns=PROFILE_COLUMNS)
    halo_summary = pd.DataFrame(summaries)
    if halo_summary.empty:
        halo_summary = pd.DataFrame(columns=HALO_SUMMARY_COLUMNS)
    else:
        global_med = float(halo_summary["median_beta_eff"].median())
        for i, summ in enumerate(summaries):
            constraints[i] = {
                "galaxy_id": summ["galaxy_id"],
                **_design_flags(summ, global_beta_median=global_med),
            }

    design_constraints = pd.DataFrame(constraints)

    tables = output_dir / "tables"
    reports = output_dir / "reports"
    figures = output_dir / "figures"
    for d in (tables, reports, figures):
        d.mkdir(parents=True, exist_ok=True)

    profiles.to_csv(tables / "tau_inverse_profiles_by_galaxy.csv", index=False)
    halo_summary.to_csv(tables / "tau_effective_halo_summary.csv", index=False)
    design_constraints.to_csv(tables / "tau_design_constraints.csv", index=False)

    _write_figures(profiles, halo_summary, figures)
    report = _build_report(
        input_run=input_run,
        sparc_data_path=sparc_data_path,
        halo_summary=halo_summary,
        design_constraints=design_constraints,
    )
    (reports / "tau_inverse_design_report.md").write_text(report, encoding="utf-8")

    return TauInverseDesignResult(
        profiles=profiles,
        halo_summary=halo_summary,
        design_constraints=design_constraints,
        input_run=input_run,
    )


def _cohort_spearman(halo_summary: pd.DataFrame) -> dict[str, float]:
    sub = halo_summary.dropna(subset=["median_beta_eff", "beta_over_M_fitted"])
    if len(sub) < 8:
        return {"spearman_beta_eff_vs_beta_over_M": float("nan")}
    rho, _ = stats.spearmanr(sub["median_beta_eff"], sub["beta_over_M_fitted"])
    return {"spearman_beta_eff_vs_beta_over_M": float(rho)}


def _write_figures(
    profiles: pd.DataFrame,
    halo_summary: pd.DataFrame,
    figures_dir: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures_dir.mkdir(parents=True, exist_ok=True)
    if profiles.empty or halo_summary.empty:
        return

    sample = halo_summary["galaxy_id"].head(6).tolist()
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for ax, gid in zip(axes.ravel(), sample):
        sub = profiles[profiles["galaxy_id"] == gid]
        ax.plot(sub["r_kpc"], sub["a_tau_required"], "o-", ms=3, label="a_τ req")
        ax.set_xlabel("r [kpc]")
        ax.set_ylabel("a_τ [km²/s²/kpc]")
        ax.set_title(str(gid), fontsize=8)
    fig.suptitle("Required τ acceleration vs radius (sample)")
    fig.tight_layout()
    fig.savefig(figures_dir / "a_tau_required_vs_radius.png", dpi=150)
    plt.close(fig)

    subp = profiles[np.isfinite(profiles["beta_eff"]) & (profiles["a_baryon"] > ACCEL_EPS)]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(
        np.log10(subp["a_baryon"]),
        subp["beta_eff"],
        alpha=0.15,
        s=8,
        c="steelblue",
    )
    ax.set_xlabel("log10(a_baryon)")
    ax.set_ylabel("β_eff proxy")
    ax.set_title("Effective coupling proxy vs baryonic acceleration")
    fig.tight_layout()
    fig.savefig(figures_dir / "beta_eff_vs_baryonic_acceleration.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    data = [
        halo_summary.loc[halo_summary["galaxy_class"] == c, "median_beta_eff"].dropna().to_numpy()
        for c in CLASS_ORDER
    ]
    ax.boxplot(data, tick_labels=list(CLASS_ORDER))
    ax.set_ylabel("median β_eff")
    ax.set_title("Inverse-designed β_eff by galaxy class")
    fig.tight_layout()
    fig.savefig(figures_dir / "tau_inverse_dwarf_vs_massive.png", dpi=150)
    plt.close(fig)

    rc = halo_summary["core_r_c_soft"].dropna()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(rc, bins=25, color="C2", alpha=0.85)
    ax.set_xlabel("fitted r_c [kpc] (soft core)")
    ax.set_ylabel("Galaxies")
    ax.set_title("Required core radius distribution")
    fig.tight_layout()
    fig.savefig(figures_dir / "required_core_radius_distribution.png", dpi=150)
    plt.close(fig)

    slopes = halo_summary.loc[halo_summary["outer_slope_status"] == "ok", "outer_slope_s"].dropna()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(slopes, bins=25, color="C1", alpha=0.85)
    ax.axvline(-1.0, color="k", ls="--", label="s ≈ −1 (MOND-like)")
    ax.axvline(-2.0, color="gray", ls=":", label="s ≈ −2 (Newtonian-like)")
    ax.set_xlabel("outer slope s")
    ax.set_ylabel("Galaxies")
    ax.legend(fontsize=8)
    ax.set_title("Outer a_τ ~ r^s slope distribution")
    fig.tight_layout()
    fig.savefig(figures_dir / "outer_slope_distribution.png", dpi=150)
    plt.close(fig)


def _build_report(
    *,
    input_run: Path,
    sparc_data_path: Path,
    halo_summary: pd.DataFrame,
    design_constraints: pd.DataFrame,
) -> str:
    n = len(halo_summary)
    corr = _cohort_spearman(halo_summary)
    rho_bm = corr.get("spearman_beta_eff_vs_beta_over_M", float("nan"))

    med_slope = float(halo_summary.loc[halo_summary["outer_slope_status"] == "ok", "outer_slope_s"].median())
    frac_mond = float(design_constraints["outer_mond_like"].mean()) if n else 0.0
    frac_core = float(design_constraints["needs_core_regularization"].mean()) if n else 0.0
    frac_global = float(design_constraints["compatible_with_single_global_beta"].mean()) if n else 0.0
    frac_dwarf = float(design_constraints["dwarf_enhanced_tau_response"].mean()) if n else 0.0
    frac_massive = float(design_constraints["massive_suppressed_tau_response"].mean()) if n else 0.0

    med_beta_dwarf = float(
        halo_summary.loc[halo_summary["galaxy_class"] == "dwarf", "median_beta_eff"].median(),
    )
    med_beta_massive = float(
        halo_summary.loc[halo_summary["galaxy_class"] == "massive", "median_beta_eff"].median(),
    )
    med_rc = float(halo_summary["core_r_c_soft"].median())

    lines = [
        "# SPARC τ inverse-design report (Step 6A)",
        "",
        f"## ⚠️ {BANNER_TAU_INVERSE}",
        "",
        f"## ⚠️ {BANNER_CALIBRATION}",
        "",
        f"**Input run:** `{input_run}`",
        f"**SPARC data:** `{sparc_data_path}`",
        f"**Galaxies analyzed:** {n}",
        "",
        "## Key questions",
        "",
        "### What empirical a_τ(r) is required by SPARC?",
        "",
        "At each radius we define "
        "`a_τ,required = max(v_obs²/r − v_baryon²/r, 0)` using fitted Υ from the TDF row "
        "when available. This is the **extra acceleration** a τ-like field must supply "
        "relative to the baryonic proxy — a dark-matter phenomenology diagnostic, not a unique τ potential.",
        "",
        "### Is the required outer τ response close to 1/r?",
        "",
        f"Outer-region power-law slopes `a_τ ∝ r^s` (r > 0.7 r_max) have median **s ≈ {med_slope:.2f}** "
        f"among galaxies with stable fits. **{frac_mond:.0%}** of galaxies have "
        "`-1.5 ≤ s ≤ -0.5` (MOND-like flat-curve band). "
        "`s ≈ -1` indicates nearly constant circular speed contribution from τ; "
        "`s ≈ -2` is closer to Newtonian falloff.",
        "",
        "### Does τ require a core radius r_c?",
        "",
        f"Soft-core fits `a_τ ≈ A/√(r²+r_c²)` yield median **r_c ≈ {med_rc:.2f} kpc** "
        f"when defined. **{frac_core:.0%}** of galaxies flag `needs_core_regularization` "
        "(finite r_c ≳ 0.15 kpc with acceptable fit RMSE). This supports testing a **cored τ response** "
        "similar to pseudo-isothermal/cored-halo phenomenology — not a proof of a specific τ potential.",
        "",
        "### Is β_eff larger in dwarf galaxies?",
        "",
        f"Median β_eff (inverse-designed) is **{med_beta_dwarf:.3f}** (dwarf) vs "
        f"**{med_beta_massive:.3f}** (massive). "
        + (
            "Dwarfs tend toward **stronger** required τ coupling in this proxy."
            if med_beta_dwarf > med_beta_massive * 1.1
            else "No strong dwarf–massive split in median β_eff."
        ),
        "",
        "### Is β_eff suppressing in massive galaxies?",
        "",
        f"**{frac_massive:.0%}** of massive galaxies flag suppressed τ response; "
        f"**{frac_dwarf:.0%}** of dwarfs flag enhanced response (heuristic thresholds).",
        "",
        "### Can one global β explain most galaxies?",
        "",
        f"**{frac_global:.0%}** lie within ~35% of the sample median β_eff "
        "(heuristic `compatible_with_single_global_beta`). "
        f"Spearman(β_eff median, fitted β/M) = **{rho_bm:.3f}** across galaxies with both defined. "
        + (
            "A **single global coupling is insufficient** for most galaxies without class-dependent modifiers."
            if frac_global < 0.45
            else "A **single global β** may approximate many galaxies, but class and radius dependence remain."
        ),
        "",
        "## Suggested formula revision (candidate inverse-designed response)",
        "",
        "**Candidate inverse-designed response** (not final physics):",
        "",
        "```",
        "a_τ(r) = β_eff(a_b, Σ_b, class) * sqrt(a_b * a0) * R_core(r)",
        "R_core(r) = r / sqrt(r² + r_c²)",
        "```",
        "",
        "Interpretation: retain the MOND-like **sqrt(a_b a0)** driver, multiply by a "
        "**class- and surface-density-dependent** effective β_eff, and apply a **core regularizer** "
        "R_core to avoid inner over-shooting seen in residual diagnostics. "
        "This is a phenomenological target for the next τ profile iteration — **not** observational validation "
        "and **not** a claim that TDF replaces dark matter.",
        "",
        "## Limitations",
        "",
        "- `ρ_τ` columns are labeled **proxies** from differentiated acceleration; not a unique Poisson source.",
        "- Υ from TDF fits couples M/L flexibility into `a_τ,required`.",
        "- No lensing, cosmology, or solar-system channels.",
        "- Does not validate TDF observationally.",
        "",
    ]
    return "\n".join(lines)

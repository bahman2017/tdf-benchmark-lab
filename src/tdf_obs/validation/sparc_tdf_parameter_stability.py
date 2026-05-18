"""
SPARC Step 5 — TDF fitted-parameter stability analysis.

Tests whether per-galaxy TDF parameters (especially beta_over_M) are coherent
or act as nuisance freedom. Analysis only; not observational validation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import least_squares

from tdf_obs.fitting.metrics import bic, chi_square
from tdf_obs.validation.sparc_boundary_filtered_analysis import (
    _param_at_bound,
    load_comparison_table,
)
from tdf_obs.validation.sparc_galaxy_class_analysis import (
    CLASS_ORDER,
    build_galaxy_properties,
    classify_galaxy_by_vmax,
    load_summary_table,
)
from tdf_obs.validation.sparc_real_calibration import (
    A0_MOND_DEFAULT,
    A0_TDF_DEFAULT,
    V_ERR_FLOOR,
    _prepare_galaxy_frame,
    compute_v_baryon,
    galaxy_has_bulge,
    v_tdf_kessence_disk_proxy,
)

BANNER_TDF_STABILITY = (
    "SPARC TDF PARAMETER STABILITY ANALYSIS — NOT FULL OBSERVATIONAL VALIDATION"
)

BANNER_CALIBRATION = (
    "These results are calibration diagnostics for a phenomenological TDF model. "
    "They do not constitute observational validation of TDF unless all required "
    "multi-channel constraints are passed."
)

BETA_LO = 1e-4
BETA_HI = 50.0
UPSILON_DISK_LO = 0.05
UPSILON_DISK_HI = 3.0
UPSILON_BULGE_LO = 0.05
UPSILON_BULGE_HI = 3.0

SUMMARY_COLUMNS: tuple[str, ...] = (
    "metric",
    "value",
    "notes",
)

BY_GALAXY_COLUMNS: tuple[str, ...] = (
    "galaxy_id",
    "galaxy_class",
    "n_points",
    "vmax_obs",
    "vbaryon_proxy_kms",
    "rmax_kpc",
    "outer_slope_v_per_kpc",
    "low_acceleration_fraction",
    "beta_over_M",
    "coupling_beta_over_M",
    "upsilon_disk",
    "upsilon_bulge",
    "a0_tdf",
    "n_params_tdf",
    "chi2_tdf",
    "reduced_chi2_tdf",
    "bic_tdf",
    "rmse_tdf",
    "beta_over_M_at_bound",
    "upsilon_disk_at_bound",
    "upsilon_bulge_at_bound",
    "any_tdf_boundary_hit",
    "delta_bic_tdf_vs_nfw",
    "best_model_by_bic",
    "bic_global_beta_approx",
    "delta_bic_global_beta_approx",
    "is_beta_iqr_outlier",
    "is_chi2_outlier",
    "is_parameter_outlier",
)

OUTLIER_COLUMNS: tuple[str, ...] = (
    "galaxy_id",
    "outlier_reason",
    "beta_over_M",
    "reduced_chi2_tdf",
    "bic_tdf",
    "galaxy_class",
    "low_acceleration_fraction",
)


@dataclass
class TdfParameterStabilityResult:
    summary: pd.DataFrame
    by_galaxy: pd.DataFrame
    outliers: pd.DataFrame
    input_run: Path


def extract_tdf_parameters(summary_df: pd.DataFrame) -> pd.DataFrame:
    """TDF rows from calibration summary with normalized parameter columns."""
    sub = summary_df[summary_df["model"] == "tdf_kessence"].copy()
    if sub.empty:
        raise ValueError("no tdf_kessence rows in summary table")
    sub["galaxy_id"] = sub["galaxy_id"].astype(str)
    sub["coupling_beta_over_M"] = sub["beta_over_M"]
    sub["a0_tdf"] = sub["a0"].fillna(A0_TDF_DEFAULT)
    return sub


def compute_galaxy_observables(
    sparc_df: pd.DataFrame,
    tdf_df: pd.DataFrame,
) -> pd.DataFrame:
    """Derived SPARC observables merged with TDF fit metadata."""
    props = build_galaxy_properties(sparc_df)
    rows: list[dict[str, Any]] = []

    for gid in tdf_df["galaxy_id"].astype(str):
        gdf = sparc_df[sparc_df["galaxy_id"].astype(str) == gid]
        gdf = _prepare_galaxy_frame(gdf, min_points=3)
        if gdf is None or len(gdf) < 3:
            continue

        r = gdf["r_kpc"].to_numpy(dtype=float)
        v_obs = gdf["v_obs"].to_numpy(dtype=float)
        v_gas = gdf["v_gas"].to_numpy(dtype=float)
        v_disk = gdf["v_disk"].to_numpy(dtype=float)
        v_bulge = gdf["v_bulge"].to_numpy(dtype=float)

        tdf_row = tdf_df[tdf_df["galaxy_id"] == gid].iloc[0]
        ud = float(tdf_row.get("upsilon_disk", 0.5))
        ub = float(tdf_row.get("upsilon_bulge", 0.0) or 0.0)
        v_b = compute_v_baryon(v_gas, v_disk, v_bulge, ud, ub)
        g_b = np.maximum(v_b**2 / np.maximum(r, 1e-3), 0.0)
        low_acc_frac = float(np.mean(g_b < A0_MOND_DEFAULT))

        n_outer = max(3, len(r) // 2)
        r_out = r[-n_outer:]
        v_out = v_obs[-n_outer:]
        if len(r_out) >= 2 and np.ptp(r_out) > 0:
            slope, _, _, _, _ = stats.linregress(r_out, v_out)
            outer_slope = float(slope)
        else:
            outer_slope = float("nan")

        prop = props[props["galaxy_id"] == gid]
        vmax = float(prop["vmax_obs"].iloc[0]) if len(prop) else float(v_obs.max())
        rmax = float(prop["rmax_kpc"].iloc[0]) if len(prop) else float(r.max())
        gcls = str(prop["galaxy_class"].iloc[0]) if len(prop) else classify_galaxy_by_vmax(vmax)

        rows.append(
            {
                "galaxy_id": gid,
                "galaxy_class": gcls,
                "n_points": int(len(gdf)),
                "vmax_obs": vmax,
                "vbaryon_proxy_kms": float(v_b[-1]),
                "rmax_kpc": rmax,
                "outer_slope_v_per_kpc": outer_slope,
                "low_acceleration_fraction": low_acc_frac,
            },
        )

    return pd.DataFrame(rows)


def tdf_boundary_flags(row: pd.Series, has_bulge: bool) -> dict[str, bool]:
    beta_at = _param_at_bound(row.get("beta_over_M"), BETA_LO, BETA_HI)
    ud_at = _param_at_bound(row.get("upsilon_disk"), UPSILON_DISK_LO, UPSILON_DISK_HI)
    ub_at = (
        has_bulge
        and _param_at_bound(row.get("upsilon_bulge"), UPSILON_BULGE_LO, UPSILON_BULGE_HI)
    )
    return {
        "beta_over_M_at_bound": beta_at,
        "upsilon_disk_at_bound": ud_at,
        "upsilon_bulge_at_bound": ub_at,
        "any_tdf_boundary_hit": beta_at or ud_at or ub_at,
    }


def _distribution_stats(values: np.ndarray) -> dict[str, float]:
    v = values[np.isfinite(values)]
    if len(v) == 0:
        return {
            "median": np.nan,
            "q25": np.nan,
            "q75": np.nan,
            "iqr": np.nan,
            "min": np.nan,
            "max": np.nan,
            "std": np.nan,
            "n": 0,
        }
    q25, q75 = np.percentile(v, [25, 75])
    return {
        "median": float(np.median(v)),
        "q25": float(q25),
        "q75": float(q75),
        "iqr": float(q75 - q25),
        "min": float(np.min(v)),
        "max": float(np.max(v)),
        "std": float(np.std(v)),
        "n": int(len(v)),
    }


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 5:
        return float("nan")
    if np.std(x[mask]) < 1e-12 or np.std(y[mask]) < 1e-12:
        return float("nan")
    return float(stats.spearmanr(x[mask], y[mask]).correlation)


def fit_tdf_fixed_beta_bic(
    gdf: pd.DataFrame,
    beta_fixed: float,
    has_bulge: bool,
) -> tuple[float, float]:
    """
    Refit TDF with fixed beta_over_M; free M/L only.

    Returns (chi2, bic). Approximate global-coupling diagnostic.
    """
    gdf = _prepare_galaxy_frame(gdf, min_points=3)
    if gdf is None:
        return float("nan"), float("nan")

    r = gdf["r_kpc"].to_numpy()
    v_obs = gdf["v_obs"].to_numpy()
    v_err = np.maximum(gdf["v_err"].to_numpy(), V_ERR_FLOOR)
    v_gas = gdf["v_gas"].to_numpy()
    v_disk = gdf["v_disk"].to_numpy()
    v_bulge = gdf["v_bulge"].to_numpy()
    n = len(r)
    beta_fixed = float(np.clip(beta_fixed, BETA_LO, BETA_HI))

    if has_bulge:
        x0 = np.array([0.5, 0.7])
        lb = np.array([UPSILON_DISK_LO, UPSILON_BULGE_LO])
        ub = np.array([UPSILON_DISK_HI, UPSILON_BULGE_HI])
        n_par = 2
    else:
        x0 = np.array([0.5])
        lb = np.array([UPSILON_DISK_LO])
        ub = np.array([UPSILON_DISK_HI])
        n_par = 1

    def predict(p: np.ndarray) -> np.ndarray:
        ud = float(p[0])
        ub = float(p[1]) if has_bulge else 0.0
        return v_tdf_kessence_disk_proxy(
            r, v_gas, v_disk, v_bulge, ud, ub, beta_fixed, A0_TDF_DEFAULT, "deep_mond",
        )

    def residuals(p: np.ndarray) -> np.ndarray:
        return (predict(p) - v_obs) / v_err

    try:
        res = least_squares(residuals, x0, bounds=(lb, ub), max_nfev=2000)
        p_opt = res.x
    except Exception:  # noqa: BLE001
        p_opt = x0

    v_pred = predict(p_opt)
    c2 = chi_square(v_obs, v_pred, v_err)
    return float(c2), float(bic(c2, n_par, n))


def estimate_global_beta_degradation(
    sparc_df: pd.DataFrame,
    by_galaxy: pd.DataFrame,
    global_beta: float,
    *,
    max_galaxies: int | None = None,
) -> pd.DataFrame:
    """Per-galaxy ΔBIC when beta_over_M is fixed to a global value."""
    rows: list[dict[str, Any]] = []
    gids = by_galaxy["galaxy_id"].astype(str).tolist()
    if max_galaxies is not None:
        gids = gids[: int(max_galaxies)]

    for gid in gids:
        gdf = sparc_df[sparc_df["galaxy_id"].astype(str) == gid]
        row = by_galaxy[by_galaxy["galaxy_id"] == gid].iloc[0]
        has_bulge = bool(
            np.isfinite(row.get("upsilon_bulge", np.nan))
            and float(row.get("upsilon_bulge", 0)) > 0.01,
        )
        _, bic_fix = fit_tdf_fixed_beta_bic(gdf, global_beta, has_bulge)
        bic_free = float(row["bic_tdf"])
        rows.append(
            {
                "galaxy_id": gid,
                "bic_global_beta_approx": bic_fix,
                "delta_bic_global_beta_approx": bic_fix - bic_free,
            },
        )
    return pd.DataFrame(rows)


def classify_outliers(by_galaxy: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Flag IQR / χ² outliers and build outlier table."""
    df = by_galaxy.copy()
    beta = df["beta_over_M"].to_numpy(dtype=float)
    chi2r = df["reduced_chi2_tdf"].to_numpy(dtype=float)

    bstats = _distribution_stats(beta)
    lo_fence = bstats["q25"] - 1.5 * bstats["iqr"]
    hi_fence = bstats["q75"] + 1.5 * bstats["iqr"]
    df["is_beta_iqr_outlier"] = (beta < lo_fence) | (beta > hi_fence)

    cstats = _distribution_stats(chi2r)
    chi_hi = cstats["q75"] + 1.5 * cstats["iqr"]
    df["is_chi2_outlier"] = chi2r > chi_hi

    df["is_parameter_outlier"] = (
        df["is_beta_iqr_outlier"]
        | df["is_chi2_outlier"]
        | df["any_tdf_boundary_hit"]
    )

    out_rows: list[dict[str, Any]] = []
    for _, row in df[df["is_parameter_outlier"]].iterrows():
        reasons: list[str] = []
        if row["is_beta_iqr_outlier"]:
            reasons.append("beta_IQR")
        if row["is_chi2_outlier"]:
            reasons.append("high_reduced_chi2")
        if row["any_tdf_boundary_hit"]:
            reasons.append("boundary_hit")
        out_rows.append(
            {
                "galaxy_id": row["galaxy_id"],
                "outlier_reason": ";".join(reasons),
                "beta_over_M": row["beta_over_M"],
                "reduced_chi2_tdf": row["reduced_chi2_tdf"],
                "bic_tdf": row["bic_tdf"],
                "galaxy_class": row["galaxy_class"],
                "low_acceleration_fraction": row["low_acceleration_fraction"],
            },
        )

    return df, pd.DataFrame(out_rows)


def build_stability_summary(by_galaxy: pd.DataFrame) -> pd.DataFrame:
    """Aggregate stability metrics for summary CSV."""
    beta = by_galaxy["beta_over_M"].to_numpy(dtype=float)
    bstats = _distribution_stats(beta)
    low_acc = by_galaxy["low_acceleration_fraction"].to_numpy(dtype=float)
    vmax = by_galaxy["vmax_obs"].to_numpy(dtype=float)
    chi2r = by_galaxy["reduced_chi2_tdf"].to_numpy(dtype=float)

    corr_vmax = _safe_corr(beta, vmax)
    corr_low_acc = _safe_corr(beta, low_acc)
    corr_chi2 = _safe_corr(beta, chi2r)

    class_corrs: list[str] = []
    for cls in CLASS_ORDER:
        sub = by_galaxy[by_galaxy["galaxy_class"] == cls]
        if len(sub) >= 5:
            med = float(sub["beta_over_M"].median())
            class_corrs.append(f"{cls} median β/M={med:.3f} (n={len(sub)})")

    delta_global = by_galaxy["delta_bic_global_beta_approx"].to_numpy(dtype=float)
    dg = delta_global[np.isfinite(delta_global)]

    rows = [
        ("n_galaxies_tdf", len(by_galaxy), ""),
        ("beta_over_M_median", bstats["median"], ""),
        ("beta_over_M_q25", bstats["q25"], ""),
        ("beta_over_M_q75", bstats["q75"], ""),
        ("beta_over_M_iqr", bstats["iqr"], ""),
        ("beta_over_M_min", bstats["min"], ""),
        ("beta_over_M_max", bstats["max"], ""),
        ("beta_over_M_std", bstats["std"], ""),
        (
            "beta_over_M_at_bound_fraction",
            float(by_galaxy["beta_over_M_at_bound"].mean()),
            f"bounds [{BETA_LO}, {BETA_HI}]",
        ),
        (
            "upsilon_disk_at_bound_fraction",
            float(by_galaxy["upsilon_disk_at_bound"].mean()),
            "",
        ),
        (
            "any_tdf_boundary_hit_fraction",
            float(by_galaxy["any_tdf_boundary_hit"].mean()),
            "",
        ),
        ("spearman_corr_beta_vs_vmax", corr_vmax, ""),
        ("spearman_corr_beta_vs_low_accel_fraction", corr_low_acc, ""),
        ("spearman_corr_beta_vs_reduced_chi2", corr_chi2, ""),
        ("global_beta_median_used", bstats["median"], "fixed-β refit diagnostic"),
        (
            "median_delta_bic_global_beta_approx",
            float(np.median(dg)) if len(dg) else np.nan,
            "ΔBIC = BIC(fixed β) − BIC(free β); approximate",
        ),
        (
            "mean_delta_bic_global_beta_approx",
            float(np.mean(dg)) if len(dg) else np.nan,
            "approximate global-β test",
        ),
        (
            "fraction_delta_bic_global_within_2",
            float(np.mean(dg < 2.0)) if len(dg) else np.nan,
            "share with mild BIC penalty under global β",
        ),
        ("parameter_outlier_count", int(by_galaxy["is_parameter_outlier"].sum()), ""),
        ("beta_iqr_outlier_count", int(by_galaxy["is_beta_iqr_outlier"].sum()), ""),
        ("class_beta_medians", "; ".join(class_corrs), ""),
    ]
    return pd.DataFrame(rows, columns=list(SUMMARY_COLUMNS))


def merge_parameter_table(
    tdf_df: pd.DataFrame,
    observables: pd.DataFrame,
    comparison_df: pd.DataFrame,
    sparc_df: pd.DataFrame,
    *,
    run_global_beta_test: bool = True,
    global_beta_max_galaxies: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Full per-galaxy parameter stability table and outlier list."""
    tdf = tdf_df.rename(
        columns={
            "chi2": "chi2_tdf",
            "reduced_chi2": "reduced_chi2_tdf",
            "bic": "bic_tdf",
            "rmse": "rmse_tdf",
            "n_params": "n_params_tdf",
        },
    )
    comp = comparison_df.copy()
    comp["galaxy_id"] = comp["galaxy_id"].astype(str)

    merged = observables.merge(
        tdf[
            [
                "galaxy_id",
                "beta_over_M",
                "coupling_beta_over_M",
                "upsilon_disk",
                "upsilon_bulge",
                "a0_tdf",
                "n_params_tdf",
                "chi2_tdf",
                "reduced_chi2_tdf",
                "bic_tdf",
                "rmse_tdf",
                "success",
            ]
        ],
        on="galaxy_id",
        how="inner",
    ).merge(
        comp[["galaxy_id", "best_model_by_bic", "delta_bic_tdf_vs_nfw"]],
        on="galaxy_id",
        how="left",
    )

    flags: list[dict[str, bool]] = []
    for _, row in merged.iterrows():
        gid = str(row["galaxy_id"])
        gdf = sparc_df[sparc_df["galaxy_id"].astype(str) == gid]
        has_bulge = galaxy_has_bulge(gdf["v_bulge"].to_numpy()) if len(gdf) else False
        flags.append(tdf_boundary_flags(row, has_bulge))
    flag_df = pd.DataFrame(flags)
    merged = pd.concat([merged.reset_index(drop=True), flag_df], axis=1)

    if run_global_beta_test:
        global_beta = float(np.median(merged["beta_over_M"]))
        deg = estimate_global_beta_degradation(
            sparc_df,
            merged,
            global_beta,
            max_galaxies=global_beta_max_galaxies,
        )
        merged = merged.merge(deg, on="galaxy_id", how="left")
    else:
        merged["bic_global_beta_approx"] = np.nan
        merged["delta_bic_global_beta_approx"] = np.nan

    merged, outliers = classify_outliers(merged)
    return merged, outliers


def run_tdf_parameter_stability(
    input_run: Path,
    sparc_data_path: Path,
    output_dir: Path,
    *,
    run_global_beta_test: bool = True,
    global_beta_max_galaxies: int | None = None,
) -> TdfParameterStabilityResult:
    input_run = Path(input_run)
    output_dir = Path(output_dir)
    sparc_data_path = Path(sparc_data_path)

    tables = output_dir / "tables"
    reports = output_dir / "reports"
    figures = output_dir / "figures"
    metadata = output_dir / "metadata"
    for d in (tables, reports, figures, metadata):
        d.mkdir(parents=True, exist_ok=True)

    summary_path = input_run / "tables" / "sparc_real_calibration_summary.csv"
    comparison_path = input_run / "tables" / "sparc_model_comparison_by_galaxy.csv"

    summary_df = load_summary_table(summary_path)
    comparison_df = load_comparison_table(comparison_path)
    sparc_df = pd.read_csv(sparc_data_path)

    tdf_df = extract_tdf_parameters(summary_df)
    observables = compute_galaxy_observables(sparc_df, tdf_df)
    by_galaxy, outliers = merge_parameter_table(
        tdf_df,
        observables,
        comparison_df,
        sparc_df,
        run_global_beta_test=run_global_beta_test,
        global_beta_max_galaxies=global_beta_max_galaxies,
    )

    stability_summary = build_stability_summary(by_galaxy)

    stability_summary.to_csv(tables / "tdf_parameter_stability_summary.csv", index=False)
    by_galaxy.to_csv(tables / "tdf_parameter_by_galaxy.csv", index=False)
    outliers.to_csv(tables / "tdf_parameter_outliers.csv", index=False)

    _write_figures(by_galaxy, figures)
    report = _build_report(
        input_run=input_run,
        sparc_data_path=sparc_data_path,
        stability_summary=stability_summary,
        by_galaxy=by_galaxy,
        outliers=outliers,
    )
    (reports / "tdf_parameter_stability_report.md").write_text(report, encoding="utf-8")

    return TdfParameterStabilityResult(
        summary=stability_summary,
        by_galaxy=by_galaxy,
        outliers=outliers,
        input_run=input_run,
    )


def _metric(summary: pd.DataFrame, name: str) -> float:
    sub = summary[summary["metric"] == name]
    if sub.empty:
        return float("nan")
    return float(sub["value"].iloc[0])


def _write_figures(by_galaxy: pd.DataFrame, figures_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if by_galaxy.empty:
        return

    beta = by_galaxy["beta_over_M"].dropna()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(beta, bins=25, color="steelblue", edgecolor="white", alpha=0.9)
    med = float(beta.median())
    ax.axvline(med, color="crimson", ls="--", label=f"median={med:.3f}")
    ax.set_xlabel("β/M (fitted)")
    ax.set_ylabel("Galaxies")
    ax.set_title("TDF coupling β/M distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "beta_over_M_distribution.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(by_galaxy["vmax_obs"], by_galaxy["beta_over_M"], alpha=0.6, s=24)
    ax.set_xlabel("v_max,obs [km/s]")
    ax.set_ylabel("β/M")
    ax.set_title("β/M vs v_max")
    fig.tight_layout()
    fig.savefig(figures_dir / "beta_over_M_vs_vmax.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    data = [
        by_galaxy.loc[by_galaxy["galaxy_class"] == c, "beta_over_M"].dropna().to_numpy()
        for c in CLASS_ORDER
    ]
    ax.boxplot(data, tick_labels=list(CLASS_ORDER))
    ax.set_ylabel("β/M")
    ax.set_title("β/M by galaxy class")
    fig.tight_layout()
    fig.savefig(figures_dir / "beta_over_M_by_galaxy_class.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    labels = ["β/M", "Υ_disk", "Υ_bulge", "any"]
    fracs = [
        float(by_galaxy["beta_over_M_at_bound"].mean()),
        float(by_galaxy["upsilon_disk_at_bound"].mean()),
        float(by_galaxy["upsilon_bulge_at_bound"].mean()),
        float(by_galaxy["any_tdf_boundary_hit"].mean()),
    ]
    ax.bar(labels, fracs, color=["C0", "C1", "C2", "C3"])
    ax.set_ylabel("Fraction at bound")
    ax.set_ylim(0, 1)
    ax.set_title("TDF parameter boundary hits")
    fig.tight_layout()
    fig.savefig(figures_dir / "tdf_parameter_boundary_flags.png", dpi=150)
    plt.close(fig)


def _build_report(
    *,
    input_run: Path,
    sparc_data_path: Path,
    stability_summary: pd.DataFrame,
    by_galaxy: pd.DataFrame,
    outliers: pd.DataFrame,
) -> str:
    n = len(by_galaxy)
    med_beta = _metric(stability_summary, "beta_over_M_median")
    iqr_beta = _metric(stability_summary, "beta_over_M_iqr")
    bound_frac = _metric(stability_summary, "any_tdf_boundary_hit_fraction")
    corr_low = _metric(stability_summary, "spearman_corr_beta_vs_low_accel_fraction")
    corr_vmax = _metric(stability_summary, "spearman_corr_beta_vs_vmax")
    med_dbic = _metric(stability_summary, "median_delta_bic_global_beta_approx")
    frac_ok = _metric(stability_summary, "fraction_delta_bic_global_within_2")
    n_out = int(len(outliers))

    stable = iqr_beta < 0.25 and bound_frac < 0.2
    global_ok = np.isfinite(med_dbic) and med_dbic < 5.0 and frac_ok > 0.5

    outlier_list = ", ".join(outliers["galaxy_id"].astype(str).head(12).tolist())
    if len(outliers) > 12:
        outlier_list += ", …"

    lines = [
        "# SPARC TDF parameter stability report (Step 5)",
        "",
        f"## ⚠️ {BANNER_TDF_STABILITY}",
        "",
        f"## ⚠️ {BANNER_CALIBRATION}",
        "",
        f"**Input run:** `{input_run}`",
        f"**SPARC data:** `{sparc_data_path}`",
        "",
        "## Summary metrics",
        "",
        stability_summary.to_string(index=False),
        "",
        "## Key questions",
        "",
        f"**Is β/M stable or scattered?** Median β/M={med_beta:.3f}, IQR={iqr_beta:.3f} "
        f"(range {_metric(stability_summary, 'beta_over_M_min'):.4f}–"
        f"{_metric(stability_summary, 'beta_over_M_max'):.3f}). "
        + (
            "Coupling is **moderately clustered** but not a single universal value."
            if not stable
            else "Coupling is **relatively stable** across the sample."
        ),
        "",
        f"**Is TDF using boundary values often?** Any TDF parameter at bound on "
        f"{bound_frac:.0%} of galaxies; β/M at bound on "
        f"{_metric(stability_summary, 'beta_over_M_at_bound_fraction'):.0%}.",
        "",
        f"**Does β/M correlate with low-acceleration systems?** "
        f"Spearman(β/M, low-a fraction)={corr_low:.3f}; "
        f"Spearman(β/M, v_max)={corr_vmax:.3f}. "
        + (
            "Weak trend toward higher β/M in low-acceleration-dominated systems."
            if np.isfinite(corr_low) and corr_low > 0.15
            else "No strong monotonic correlation with low-acceleration fraction in this sample."
        ),
        "",
        f"**Is a shared/global TDF parameter plausible?** "
        f"Approximate fixed-β refit (β fixed to sample median): "
        f"median ΔBIC≈{med_dbic:.2f}, "
        f"{frac_ok:.0%} of galaxies within ΔBIC<2. "
        + (
            "A **single global β/M is plausible** as a first-pass phenomenological choice."
            if global_ok
            else "Per-galaxy β/M **still helps** BIC for many galaxies; global β is only approximate."
        ),
        "",
        f"**Which galaxies are outliers?** {n_out} flagged ({outlier_list or 'none'}). "
        "See `tdf_parameter_outliers.csv`.",
        "",
        "## Limitations",
        "",
        "- β/M is an effective disk-proxy coupling, not a fundamental constant.",
        "- Global-β test refits M/L only; labeled approximate, not a full hierarchical fit.",
        "- Does not validate TDF observationally or replace dark matter.",
        "",
    ]
    return "\n".join(lines)

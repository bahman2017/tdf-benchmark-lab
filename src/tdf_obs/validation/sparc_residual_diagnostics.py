"""
SPARC Step 6 — rotation-curve residual diagnostics.

Identifies where TDF and baselines succeed or fail by radius, galaxy class, and
residual magnitude. Analysis only; not observational validation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy import stats

from tdf_obs.validation.sparc_cored_halo_baseline import v_nfw_total
from tdf_obs.validation.sparc_galaxy_class_analysis import (
    CLASS_ORDER,
    build_galaxy_properties,
    classify_galaxy_by_vmax,
    load_comparison_table,
    load_summary_table,
)
from tdf_obs.validation.sparc_real_calibration import (
    A0_MOND_DEFAULT,
    A0_TDF_DEFAULT,
    V_ERR_FLOOR,
    _prepare_galaxy_frame,
    compute_v_baryon,
    galaxy_has_bulge,
    validate_sparc_input_schema,
    v_mond_analytic,
    v_tdf_kessence_disk_proxy,
)

BANNER_RESIDUAL = (
    "SPARC RESIDUAL DIAGNOSTICS — NOT FULL OBSERVATIONAL VALIDATION"
)

BANNER_CALIBRATION = (
    "These results are calibration diagnostics for a phenomenological TDF model. "
    "They do not constitute observational validation of TDF unless all required "
    "multi-channel constraints are passed."
)

DiagnosticModel = Literal["baryon_only", "corrected_mond", "nfw", "tdf_kessence"]
DIAGNOSTIC_MODELS: tuple[DiagnosticModel, ...] = (
    "baryon_only",
    "corrected_mond",
    "nfw",
    "tdf_kessence",
)

INNER_FRAC = 0.3
OUTER_FRAC = 0.7

BY_POINT_COLUMNS: tuple[str, ...] = (
    "galaxy_id",
    "model",
    "r_kpc",
    "r_over_rmax",
    "radial_zone",
    "v_obs",
    "v_err",
    "v_model",
    "residual",
    "weighted_residual",
    "abs_weighted_residual",
)

BY_GALAXY_COLUMNS: tuple[str, ...] = (
    "galaxy_id",
    "galaxy_class",
    "vmax_obs",
    "model",
    "n_points",
    "inner_residual_score",
    "middle_residual_score",
    "outer_residual_score",
    "residual_slope_vs_radius",
    "worst_radius_kpc",
    "max_abs_weighted_residual",
    "rms_weighted_residual",
)

FAILURE_MODE_COLUMNS: tuple[str, ...] = (
    "failure_mode",
    "n_galaxies",
    "fraction_of_sample",
    "description",
)


@dataclass
class ResidualDiagnosticsResult:
    by_point: pd.DataFrame
    by_galaxy: pd.DataFrame
    failure_modes: pd.DataFrame
    input_run: Path


def radial_zone(r: np.ndarray, rmax: float) -> np.ndarray:
    """Classify each radius as inner / middle / outer."""
    rmax = max(float(rmax), 1e-6)
    x = r / rmax
    zones = np.full(len(r), "middle", dtype=object)
    zones[x < INNER_FRAC] = "inner"
    zones[x > OUTER_FRAC] = "outer"
    return zones


def zone_mask(zones: np.ndarray, name: str) -> np.ndarray:
    return zones == name


def predict_velocity(
    model: DiagnosticModel,
    r: np.ndarray,
    v_gas: np.ndarray,
    v_disk: np.ndarray,
    v_bulge: np.ndarray,
    row: pd.Series,
    has_bulge: bool,
) -> np.ndarray:
    """Reconstruct v_model from fitted parameters in summary row."""
    ud = float(row.get("upsilon_disk", 0.5))
    ub = float(row.get("upsilon_bulge", 0.0) or 0.0) if has_bulge else 0.0

    if model == "baryon_only":
        return compute_v_baryon(v_gas, v_disk, v_bulge, ud, ub)
    if model == "corrected_mond":
        a0 = float(row.get("a0", A0_MOND_DEFAULT) or A0_MOND_DEFAULT)
        return v_mond_analytic(r, v_gas, v_disk, v_bulge, ud, ub, a0)
    if model == "nfw":
        v200 = float(row.get("v200", 100.0))
        rs = float(row.get("r_s", 1.0))
        return v_nfw_total(r, v_gas, v_disk, v_bulge, ud, ub, v200, rs)
    if model == "tdf_kessence":
        beta = float(row.get("beta_over_M", 1.0))
        a0 = float(row.get("a0", A0_TDF_DEFAULT) or A0_TDF_DEFAULT)
        return v_tdf_kessence_disk_proxy(
            r, v_gas, v_disk, v_bulge, ud, ub, beta, a0, "deep_mond",
        )
    raise ValueError(model)


def _zone_score(wres: np.ndarray, mask: np.ndarray) -> float:
    if not np.any(mask):
        return float("nan")
    return float(np.sqrt(np.mean(wres[mask] ** 2)))


def galaxy_residual_metrics(
    r: np.ndarray,
    v_obs: np.ndarray,
    v_err: np.ndarray,
    v_model: np.ndarray,
) -> dict[str, Any]:
    v_err = np.maximum(v_err, V_ERR_FLOOR)
    residual = v_model - v_obs
    wres = residual / v_err
    aw = np.abs(wres)
    rmax = float(np.max(r))
    zones = radial_zone(r, rmax)

    inner_m = zone_mask(zones, "inner")
    mid_m = zone_mask(zones, "middle")
    outer_m = zone_mask(zones, "outer")

    if np.sum(aw > 0):
        worst_i = int(np.argmax(aw))
        worst_r = float(r[worst_i])
    else:
        worst_r = float("nan")

    slope = float("nan")
    if len(r) >= 3 and np.ptp(r) > 0:
        slope, _, _, _, _ = stats.linregress(r, wres)

    return {
        "inner_residual_score": _zone_score(wres, inner_m),
        "middle_residual_score": _zone_score(wres, mid_m),
        "outer_residual_score": _zone_score(wres, outer_m),
        "residual_slope_vs_radius": slope,
        "worst_radius_kpc": worst_r,
        "max_abs_weighted_residual": float(np.max(aw)) if len(aw) else float("nan"),
        "rms_weighted_residual": float(np.sqrt(np.mean(wres**2))) if len(wres) else float("nan"),
        "residual": residual,
        "weighted_residual": wres,
        "abs_weighted_residual": aw,
        "radial_zone": zones,
        "r_over_rmax": r / rmax,
    }


def compute_residuals_for_galaxy(
    galaxy_id: str,
    gdf: pd.DataFrame,
    summary_galaxy: pd.DataFrame,
    galaxy_class: str,
    vmax_obs: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    gdf = _prepare_galaxy_frame(gdf, min_points=3)
    if gdf is None:
        return [], []

    r = gdf["r_kpc"].to_numpy(dtype=float)
    v_obs = gdf["v_obs"].to_numpy(dtype=float)
    v_err = gdf["v_err"].to_numpy(dtype=float)
    v_gas = gdf["v_gas"].to_numpy(dtype=float)
    v_disk = gdf["v_disk"].to_numpy(dtype=float)
    v_bulge = gdf["v_bulge"].to_numpy(dtype=float)
    has_bulge = galaxy_has_bulge(v_bulge)

    point_rows: list[dict[str, Any]] = []
    galaxy_rows: list[dict[str, Any]] = []

    for model in DIAGNOSTIC_MODELS:
        sub = summary_galaxy[summary_galaxy["model"] == model]
        if sub.empty:
            continue
        row = sub.iloc[0]
        v_model = predict_velocity(model, r, v_gas, v_disk, v_bulge, row, has_bulge)
        v_model = np.where(np.isfinite(v_model), v_model, v_obs)
        met = galaxy_residual_metrics(r, v_obs, v_err, v_model)

        galaxy_rows.append(
            {
                "galaxy_id": galaxy_id,
                "galaxy_class": galaxy_class,
                "vmax_obs": vmax_obs,
                "model": model,
                "n_points": len(r),
                "inner_residual_score": met["inner_residual_score"],
                "middle_residual_score": met["middle_residual_score"],
                "outer_residual_score": met["outer_residual_score"],
                "residual_slope_vs_radius": met["residual_slope_vs_radius"],
                "worst_radius_kpc": met["worst_radius_kpc"],
                "max_abs_weighted_residual": met["max_abs_weighted_residual"],
                "rms_weighted_residual": met["rms_weighted_residual"],
            },
        )

        for i in range(len(r)):
            point_rows.append(
                {
                    "galaxy_id": galaxy_id,
                    "model": model,
                    "r_kpc": float(r[i]),
                    "r_over_rmax": float(met["r_over_rmax"][i]),
                    "radial_zone": str(met["radial_zone"][i]),
                    "v_obs": float(v_obs[i]),
                    "v_err": float(v_err[i]),
                    "v_model": float(v_model[i]),
                    "residual": float(met["residual"][i]),
                    "weighted_residual": float(met["weighted_residual"][i]),
                    "abs_weighted_residual": float(met["abs_weighted_residual"][i]),
                },
            )

    return point_rows, galaxy_rows


def build_tdf_nfw_comparison(by_galaxy: pd.DataFrame) -> pd.DataFrame:
    """Wide per-galaxy TDF vs NFW residual comparison."""
    rows: list[dict[str, Any]] = []
    for gid in by_galaxy["galaxy_id"].unique():
        sub = by_galaxy[by_galaxy["galaxy_id"] == gid]
        tdf = sub[sub["model"] == "tdf_kessence"]
        nfw = sub[sub["model"] == "nfw"]
        if tdf.empty or nfw.empty:
            continue
        t = tdf.iloc[0]
        n = nfw.iloc[0]
        rows.append(
            {
                "galaxy_id": gid,
                "galaxy_class": t["galaxy_class"],
                "vmax_obs": t["vmax_obs"],
                "tdf_inner_score": t["inner_residual_score"],
                "nfw_inner_score": n["inner_residual_score"],
                "tdf_outer_score": t["outer_residual_score"],
                "nfw_outer_score": n["outer_residual_score"],
                "tdf_rms": t["rms_weighted_residual"],
                "nfw_rms": n["rms_weighted_residual"],
                "tdf_minus_nfw_inner": t["inner_residual_score"] - n["inner_residual_score"],
                "tdf_minus_nfw_outer": t["outer_residual_score"] - n["outer_residual_score"],
                "tdf_minus_nfw_rms": t["rms_weighted_residual"] - n["rms_weighted_residual"],
                "tdf_better_than_nfw": t["rms_weighted_residual"] < n["rms_weighted_residual"],
                "tdf_worse_region": (
                    "outer"
                    if t["outer_residual_score"] > n["outer_residual_score"]
                    and t["outer_residual_score"] >= t["inner_residual_score"]
                    else (
                        "inner"
                        if t["inner_residual_score"] > n["inner_residual_score"]
                        else "middle"
                    )
                ),
            },
        )
    return pd.DataFrame(rows)


def build_failure_modes(
    comparison: pd.DataFrame,
    tdf_nfw: pd.DataFrame,
    properties: pd.DataFrame,
) -> pd.DataFrame:
    n = len(tdf_nfw)
    if n == 0:
        return pd.DataFrame(columns=list(FAILURE_MODE_COLUMNS))

    def frac(mask: pd.Series) -> float:
        return float(mask.sum()) / n

    tdf_worse = ~tdf_nfw["tdf_better_than_nfw"]
    modes = [
        (
            "tdf_worse_than_nfw_overall",
            int(tdf_worse.sum()),
            "TDF RMS weighted residual exceeds NFW",
        ),
        (
            "tdf_worse_outer_than_nfw",
            int((tdf_nfw["tdf_minus_nfw_outer"] > 0).sum()),
            "TDF outer-zone RMS residual worse than NFW",
        ),
        (
            "tdf_worse_inner_than_nfw",
            int((tdf_nfw["tdf_minus_nfw_inner"] > 0).sum()),
            "TDF inner-zone RMS residual worse than NFW",
        ),
        (
            "tdf_better_outer_than_nfw",
            int((tdf_nfw["tdf_minus_nfw_outer"] < 0).sum()),
            "TDF better in outer regions vs NFW",
        ),
        (
            "tdf_better_inner_than_nfw",
            int((tdf_nfw["tdf_minus_nfw_inner"] < 0).sum()),
            "TDF better in inner regions vs NFW",
        ),
        (
            "tdf_failures_massive",
            int(
                (
                    tdf_worse & (tdf_nfw["galaxy_class"] == "massive")
                ).sum(),
            ),
            "TDF worse than NFW in massive galaxies",
        ),
        (
            "tdf_failures_dwarf",
            int(
                (
                    tdf_worse & (tdf_nfw["galaxy_class"] == "dwarf")
                ).sum(),
            ),
            "TDF worse than NFW in dwarf galaxies",
        ),
        (
            "tdf_success_dwarf",
            int(
                (
                    (~tdf_worse) & (tdf_nfw["galaxy_class"] == "dwarf")
                ).sum(),
            ),
            "TDF better than NFW in dwarf galaxies",
        ),
        (
            "tdf_success_massive",
            int(
                (
                    (~tdf_worse) & (tdf_nfw["galaxy_class"] == "massive")
                ).sum(),
            ),
            "TDF better than NFW in massive galaxies",
        ),
    ]

    rows = [
        {
            "failure_mode": name,
            "n_galaxies": count,
            "fraction_of_sample": count / n,
            "description": desc,
        }
        for name, count, desc in modes
    ]
    return pd.DataFrame(rows)


def _model_zone_means(by_galaxy: pd.DataFrame) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for model in DIAGNOSTIC_MODELS:
        sub = by_galaxy[by_galaxy["model"] == model]
        out[model] = {
            "inner": float(sub["inner_residual_score"].median()),
            "middle": float(sub["middle_residual_score"].median()),
            "outer": float(sub["outer_residual_score"].median()),
        }
    return out


def run_residual_diagnostics(
    input_run: Path,
    sparc_data_path: Path,
    output_dir: Path,
    *,
    max_galaxies: int | None = None,
) -> ResidualDiagnosticsResult:
    input_run = Path(input_run)
    output_dir = Path(output_dir)
    sparc_data_path = Path(sparc_data_path)

    tables = output_dir / "tables"
    reports = output_dir / "reports"
    figures = output_dir / "figures"
    metadata = output_dir / "metadata"
    for d in (tables, reports, figures, metadata):
        d.mkdir(parents=True, exist_ok=True)

    summary_df = load_summary_table(
        input_run / "tables" / "sparc_real_calibration_summary.csv",
    )
    comparison_df = load_comparison_table(
        input_run / "tables" / "sparc_model_comparison_by_galaxy.csv",
    )
    sparc_df = pd.read_csv(sparc_data_path)
    validate_sparc_input_schema(sparc_df)
    properties = build_galaxy_properties(sparc_df)

    galaxy_ids = sorted(summary_df["galaxy_id"].astype(str).unique())
    if max_galaxies is not None:
        galaxy_ids = galaxy_ids[: int(max_galaxies)]

    all_points: list[dict[str, Any]] = []
    all_galaxy: list[dict[str, Any]] = []

    for gid in galaxy_ids:
        gdf = sparc_df[sparc_df["galaxy_id"].astype(str) == gid]
        sg = summary_df[summary_df["galaxy_id"].astype(str) == gid]
        prop = properties[properties["galaxy_id"].astype(str) == gid]
        if prop.empty:
            continue
        gcls = str(prop["galaxy_class"].iloc[0])
        vmax = float(prop["vmax_obs"].iloc[0])
        pts, gals = compute_residuals_for_galaxy(gid, gdf, sg, gcls, vmax)
        all_points.extend(pts)
        all_galaxy.extend(gals)

    by_point = pd.DataFrame(all_points)
    by_galaxy = pd.DataFrame(all_galaxy)
    tdf_nfw = build_tdf_nfw_comparison(by_galaxy)
    failure_modes = build_failure_modes(comparison_df, tdf_nfw, properties)

    by_point.to_csv(tables / "residuals_by_point.csv", index=False)
    by_galaxy.to_csv(tables / "residuals_by_galaxy.csv", index=False)
    failure_modes.to_csv(tables / "residual_failure_modes.csv", index=False)

    _write_figures(by_point, by_galaxy, tdf_nfw, sparc_df, summary_df, figures)
    report = _build_report(
        input_run=input_run,
        sparc_data_path=sparc_data_path,
        by_galaxy=by_galaxy,
        tdf_nfw=tdf_nfw,
        failure_modes=failure_modes,
    )
    (reports / "residual_diagnostics_report.md").write_text(report, encoding="utf-8")

    return ResidualDiagnosticsResult(
        by_point=by_point,
        by_galaxy=by_galaxy,
        failure_modes=failure_modes,
        input_run=input_run,
    )


def _write_figures(
    by_point: pd.DataFrame,
    by_galaxy: pd.DataFrame,
    tdf_nfw: pd.DataFrame,
    sparc_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    figures_dir: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if by_point.empty:
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    for model in DIAGNOSTIC_MODELS:
        sub = by_point[by_point["model"] == model]
        if len(sub):
            ax.hist(
                sub["weighted_residual"].clip(-10, 10),
                bins=40,
                alpha=0.4,
                label=model,
                density=True,
            )
    ax.axvline(0, color="k", ls="--")
    ax.set_xlabel("Weighted residual (v_model − v_obs) / σ")
    ax.set_ylabel("Density")
    ax.legend(fontsize=8)
    ax.set_title("Weighted residual distribution by model")
    fig.tight_layout()
    fig.savefig(figures_dir / "residual_distribution_by_model.png", dpi=150)
    plt.close(fig)

    tdf_pts = by_point[by_point["model"] == "tdf_kessence"][
        ["galaxy_id", "r_over_rmax", "weighted_residual"]
    ].rename(columns={"weighted_residual": "w_tdf"})
    nfw_pts = by_point[by_point["model"] == "nfw"][
        ["galaxy_id", "r_over_rmax", "weighted_residual"]
    ].rename(columns={"weighted_residual": "w_nfw"})
    merged = tdf_pts.merge(nfw_pts, on=["galaxy_id", "r_over_rmax"], how="inner")
    merged["delta"] = merged["w_tdf"] - merged["w_nfw"]
    bins = np.linspace(0, 1, 12)
    merged["r_bin"] = np.digitize(merged["r_over_rmax"], bins) - 1
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    med_delta = merged.groupby("r_bin")["delta"].median()
    fig, ax = plt.subplots(figsize=(8, 5))
    valid = med_delta.index[med_delta.index < len(bin_centers)]
    ax.plot(bin_centers[valid], med_delta.loc[valid], "o-", color="C0")
    ax.axhline(0, color="k", ls="--")
    ax.set_xlabel("r / r_max")
    ax.set_ylabel("median(weighted residual_TDF − residual_NFW)")
    ax.set_title("TDF − NFW residual vs radius")
    fig.tight_layout()
    fig.savefig(figures_dir / "tdf_minus_nfw_residual_by_radius.png", dpi=150)
    plt.close(fig)

    zone_means = _model_zone_means(by_galaxy)
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(3)
    width = 0.18
    zones = ["inner", "middle", "outer"]
    for i, model in enumerate(DIAGNOSTIC_MODELS):
        vals = [zone_means[model][z] for z in zones]
        ax.bar(x + (i - 1.5) * width, vals, width, label=model)
    ax.set_xticks(x)
    ax.set_xticklabels(zones)
    ax.set_ylabel("Median zone RMS weighted residual")
    ax.legend(fontsize=7)
    ax.set_title("Inner / middle / outer residual scores")
    fig.tight_layout()
    fig.savefig(figures_dir / "inner_outer_residual_scores.png", dpi=150)
    plt.close(fig)

    _plot_worst_galaxies(by_point, sparc_df, summary_df, figures_dir)


def _plot_worst_galaxies(
    by_point: pd.DataFrame,
    sparc_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    figures_dir: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tdf_gal = by_point[by_point["model"] == "tdf_kessence"].groupby("galaxy_id").agg(
        max_aw=("abs_weighted_residual", "max"),
    ).reset_index()
    worst = tdf_gal.nlargest(6, "max_aw")["galaxy_id"].tolist()
    if not worst:
        return

    n = len(worst)
    ncol = 3
    nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3.5 * nrow))
    axes_flat = np.atleast_1d(axes).flatten()

    for ax, gid in zip(axes_flat, worst):
        gdf = sparc_df[sparc_df["galaxy_id"].astype(str) == gid].sort_values("r_kpc")
        r = gdf["r_kpc"].to_numpy()
        ax.errorbar(r, gdf["v_obs"], yerr=gdf["v_err"], fmt="ko", ms=3, capsize=2)
        for model, color in (
            ("nfw", "C1"),
            ("tdf_kessence", "C2"),
        ):
            pts = by_point[
                (by_point["galaxy_id"] == gid) & (by_point["model"] == model)
            ].sort_values("r_kpc")
            if len(pts):
                ax.plot(pts["r_kpc"], pts["v_model"], color=color, label=model)
        ax.set_title(str(gid), fontsize=9)
        ax.set_xlabel("r [kpc]")
        ax.set_ylabel("v [km/s]")
        ax.legend(fontsize=6)

    for ax in axes_flat[len(worst) :]:
        ax.axis("off")

    fig.suptitle("Largest |weighted residual| TDF galaxies (sample)", fontsize=11)
    fig.tight_layout()
    fig.savefig(figures_dir / "worst_tdf_residual_galaxies.png", dpi=150)
    plt.close(fig)


def _top_galaxy_list(tdf_nfw: pd.DataFrame, *, failures: bool, n: int = 10) -> str:
    if tdf_nfw.empty:
        return "none"
    df = tdf_nfw.copy()
    if failures:
        df = df[~df["tdf_better_than_nfw"]].nlargest(n, "tdf_minus_nfw_rms")
    else:
        df = df[df["tdf_better_than_nfw"]].nsmallest(n, "tdf_minus_nfw_rms")
    return ", ".join(df["galaxy_id"].astype(str).tolist())


def _build_report(
    *,
    input_run: Path,
    sparc_data_path: Path,
    by_galaxy: pd.DataFrame,
    tdf_nfw: pd.DataFrame,
    failure_modes: pd.DataFrame,
) -> str:
    zone_means = _model_zone_means(by_galaxy)
    tdf_inner = zone_means["tdf_kessence"]["inner"]
    tdf_outer = zone_means["tdf_kessence"]["outer"]
    nfw_inner = zone_means["nfw"]["inner"]
    nfw_outer = zone_means["nfw"]["outer"]

    tdf_fail_outer = int((tdf_nfw["tdf_minus_nfw_outer"] > 0).sum()) if len(tdf_nfw) else 0
    tdf_fail_inner = int((tdf_nfw["tdf_minus_nfw_inner"] > 0).sum()) if len(tdf_nfw) else 0
    n = len(tdf_nfw)

    fail_massive = failure_modes.loc[
        failure_modes["failure_mode"] == "tdf_failures_massive", "n_galaxies",
    ]
    fail_massive_n = int(fail_massive.iloc[0]) if len(fail_massive) else 0
    succ_dwarf = failure_modes.loc[
        failure_modes["failure_mode"] == "tdf_success_dwarf", "n_galaxies",
    ]
    succ_dwarf_n = int(succ_dwarf.iloc[0]) if len(succ_dwarf) else 0

    lines = [
        "# SPARC residual diagnostics report (Step 6)",
        "",
        f"## ⚠️ {BANNER_RESIDUAL}",
        "",
        f"## ⚠️ {BANNER_CALIBRATION}",
        "",
        f"**Input run:** `{input_run}`",
        f"**SPARC data:** `{sparc_data_path}`",
        "",
        "Predictions recomputed from fitted parameters in the calibration summary "
        "(no per-point prediction archive in input run).",
        "",
        "## Failure-mode summary",
        "",
        failure_modes.to_string(index=False) if len(failure_modes) else "_none_",
        "",
        "## Key questions",
        "",
        f"**Does TDF fail more in inner or outer regions?** "
        f"Median zone scores — inner={tdf_inner:.3f}, outer={tdf_outer:.3f}. "
        + (
            "TDF residuals are **larger in the outer** regions on average."
            if tdf_outer > tdf_inner
            else "TDF residuals are **larger in the inner** regions on average."
        ),
        "",
        f"**Does NFW fail more in inner or outer regions?** "
        f"Median zone scores — inner={nfw_inner:.3f}, outer={nfw_outer:.3f}. "
        + (
            "NFW is **worse outward**."
            if nfw_outer > nfw_inner
            else "NFW is **worse inward**."
        ),
        "",
        f"**Are TDF failures concentrated in massive galaxies?** "
        f"{fail_massive_n}/{n} massive galaxies have TDF worse than NFW (by RMS weighted residual).",
        "",
        f"**Are TDF successes concentrated in dwarf galaxies?** "
        f"{succ_dwarf_n}/{n} dwarf galaxies have TDF better than NFW.",
        "",
        f"**Top 10 TDF failure galaxies (vs NFW):** "
        f"{_top_galaxy_list(tdf_nfw, failures=True, n=10)}.",
        "",
        f"**Top 10 TDF success galaxies (vs NFW):** "
        f"{_top_galaxy_list(tdf_nfw, failures=False, n=10)}.",
        "",
        f"Head-to-head: TDF worse outer than NFW on {tdf_fail_outer}/{n} galaxies; "
        f"worse inner on {tdf_fail_inner}/{n}.",
        "",
        "## Limitations",
        "",
        "- Point residuals use refitted curves from tabulated parameters; not independent hold-out.",
        "- Zone scores are RMS of weighted residuals; not spatially covariant errors.",
        "- Does not validate TDF observationally or replace dark matter.",
        "",
    ]
    return "\n".join(lines)

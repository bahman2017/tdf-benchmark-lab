"""
Phase 8A.1 — MOND baseline and parameter boundary audit for real SPARC calibration.

Audits v0.21.0 / Phase 8A outputs without modifying prior benchmark tables.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from tdf_obs.fitting.metrics import aic, bic
from tdf_obs.validation.sparc_real_calibration import (
    A0_MOND_DEFAULT,
    V_ERR_FLOOR,
    compute_v_baryon,
    galaxy_has_bulge,
    validate_sparc_input_schema,
    v_nfw_total,
    v_tdf_kessence_disk_proxy,
)

BANNER_SPARC_AUDIT = "SPARC PARAMETER AUDIT — NOT FULL OBSERVATIONAL VALIDATION"

BANNER_CALIBRATION = (
    "These results are calibration diagnostics for a phenomenological TDF model. "
    "They do not constitute observational validation of TDF unless all required "
    "multi-channel constraints are passed."
)

BOUND_REL_TOL = 0.02
BOUND_ABS_TOL = 0.08
BIC_MISMATCH_TOL = 1e-3
MOND_DV_TOL = 0.5  # km/s median |v_mond - v_baryon|
MOND_RATIO_TOL = 1.01
A0_MOND_KMS2_KPC = A0_MOND_DEFAULT

ModelName = Literal["baryon_only", "nfw", "mond", "tdf_kessence"]

MOND_AUDIT_COLUMNS = (
    "galaxy_id",
    "n_points",
    "has_bulge",
    "upsilon_disk",
    "upsilon_bulge",
    "a0_kms2_kpc",
    "mond_vs_baryon_max_dv",
    "mond_vs_baryon_median_dv",
    "mond_vs_baryon_max_frac_v2_boost",
    "mond_active_flag",
    "mond_inactive_or_unit_bug",
    "n_low_acc_points",
    "median_mond_boost_low_acc",
    "mond_low_acc_pass",
    "g_mond_ge_gb_all",
    "v_mond_ge_vb_all",
)

BOUNDARY_COLUMNS = (
    "galaxy_id",
    "model",
    "any_boundary_hit",
    "boundary_hit_count",
    "boundary_hit_fields",
    "likely_overfit_or_bound_limited",
    "upsilon_disk",
    "upsilon_bulge",
    "v200",
    "r_s",
    "beta_over_M",
    "a0",
    "a0_fixed",
    "upsilon_disk_at_bound",
    "upsilon_bulge_at_bound",
    "v200_at_bound",
    "r_s_at_bound",
    "beta_over_M_at_bound",
)

PARAM_COUNT_COLUMNS = (
    "model",
    "expected_n_params_rule",
    "observed_n_params_min",
    "observed_n_params_max",
    "parameter_count_consistent",
    "a0_fitted_in_calibration",
)


@dataclass
class BoundaryFitResult:
    galaxy_id: str
    model: ModelName
    params: dict[str, float]
    param_names: list[str]
    lower_bounds: np.ndarray
    upper_bounds: np.ndarray
    boundary_hits: list[str]
    n_params: int
    chi2: float
    n_points: int


def mond_g_baryon_analytic(g_b: np.ndarray, a0: float = A0_MOND_KMS2_KPC) -> np.ndarray:
    """
    Analytic deep-MOND solution for μ(x)=x/(1+x):

    g_mond = 0.5 * (g_b + sqrt(g_b² + 4 g_b a0))

    Units: g_b, a0, g_mond in (km/s)²/kpc; r in kpc; v in km/s.
    """
    g_b = np.maximum(np.asarray(g_b, dtype=float), 0.0)
    a0 = max(float(a0), 1e-30)
    inner = g_b**2 + 4.0 * g_b * a0
    return 0.5 * (g_b + np.sqrt(np.maximum(inner, 0.0)))


def v_mond_analytic(
    r: np.ndarray,
    v_gas: np.ndarray,
    v_disk: np.ndarray,
    v_bulge: np.ndarray,
    upsilon_disk: float,
    upsilon_bulge: float,
    a0: float = A0_MOND_KMS2_KPC,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (v_mond, v_baryon, g_b) arrays."""
    r = np.asarray(r, dtype=float)
    v_b = compute_v_baryon(v_gas, v_disk, v_bulge, upsilon_disk, upsilon_bulge)
    g_b = np.maximum(v_b**2 / np.maximum(r, 1e-3), 0.0)
    g_m = mond_g_baryon_analytic(g_b, a0)
    v_m = np.sqrt(np.maximum(r * g_m, 0.0))
    return v_m, v_b, g_b


def _at_bound(value: float, lo: float, hi: float) -> bool:
    """True if value is within tolerance of a bound (scale-aware, not span-only)."""
    eps_lo = max(BOUND_ABS_TOL, BOUND_REL_TOL * max(abs(lo), 1e-6))
    eps_hi = max(BOUND_ABS_TOL, BOUND_REL_TOL * max(abs(hi), 1e-6))
    return value <= lo + eps_lo or value >= hi - eps_hi


def expected_n_params(model: ModelName, has_bulge: bool, *, a0_fitted: bool = False) -> int:
    if model == "baryon_only":
        return 2 if has_bulge else 1
    if model == "mond":
        base = 2 if has_bulge else 1
        return base + (1 if a0_fitted else 0)
    if model == "nfw":
        return (4 if has_bulge else 3)
    if model == "tdf_kessence":
        return (3 if has_bulge else 2)
    raise ValueError(model)


def _fit_bounds_setup(
    model: ModelName,
    has_bulge: bool,
    r: np.ndarray,
    v_obs: np.ndarray,
) -> tuple[list[str], np.ndarray, np.ndarray, Any, int]:
    """Return param_names, lb, ub, predict_fn, n_params."""
    v2000 = float(np.sqrt(max(np.mean(v_obs**2), 1.0)))
    rout = max(float(r[-1]), 0.1)

    if model == "baryon_only":
        names = ["upsilon_disk", "upsilon_bulge"] if has_bulge else ["upsilon_disk"]

        def predict(p: np.ndarray, v_gas, v_disk, v_bulge, r_arr):
            ud = p[0]
            ub = p[1] if has_bulge else 0.0
            return compute_v_baryon(v_gas, v_disk, v_bulge, ud, ub)

        lb = np.array([0.05, 0.05] if has_bulge else [0.05])
        ub = np.array([3.0, 3.0] if has_bulge else [3.0])
        n_par = len(names)
    elif model == "nfw":
        names = (
            ["upsilon_disk", "upsilon_bulge", "v200", "r_s"]
            if has_bulge
            else ["upsilon_disk", "v200", "r_s"]
        )

        def predict(p: np.ndarray, v_gas, v_disk, v_bulge, r_arr):
            if has_bulge:
                ud, ub, v200, rs = p
            else:
                ud, v200, rs = p
                ub = 0.0
            return v_nfw_total(r_arr, v_gas, v_disk, v_bulge, ud, ub, v200, rs)

        lb = np.array([0.05, 0.05, 1.0, 0.1] if has_bulge else [0.05, 1.0, 0.1])
        ub = np.array([3.0, 3.0, 500.0, 100.0] if has_bulge else [3.0, 500.0, 100.0])
        n_par = len(names)
    elif model == "mond":
        names = ["upsilon_disk", "upsilon_bulge"] if has_bulge else ["upsilon_disk"]

        def predict(p: np.ndarray, v_gas, v_disk, v_bulge, r_arr):
            ud = p[0]
            ub = p[1] if has_bulge else 0.0
            return v_mond_analytic(r_arr, v_gas, v_disk, v_bulge, ud, ub, A0_MOND_KMS2_KPC)[0]

        lb = np.array([0.05, 0.05] if has_bulge else [0.05])
        ub = np.array([3.0, 3.0] if has_bulge else [3.0])
        n_par = len(names)
    elif model == "tdf_kessence":
        names = (
            ["upsilon_disk", "upsilon_bulge", "beta_over_M"]
            if has_bulge
            else ["upsilon_disk", "beta_over_M"]
        )

        def predict(p: np.ndarray, v_gas, v_disk, v_bulge, r_arr):
            if has_bulge:
                ud, ub, beta = p
            else:
                ud, beta = p
                ub = 0.0
            return v_tdf_kessence_disk_proxy(
                r_arr, v_gas, v_disk, v_bulge, ud, ub, beta, A0_MOND_KMS2_KPC, "deep_mond",
            )

        lb = np.array([0.05, 0.05, 1e-4] if has_bulge else [0.05, 1e-4])
        ub = np.array([3.0, 3.0, 50.0] if has_bulge else [3.0, 50.0])
        n_par = len(names)
    else:
        raise ValueError(model)

    x0 = np.clip((lb + ub) / 2.0, lb, ub)
    return names, lb, ub, predict, n_par


def fit_model_with_boundary_audit(
    galaxy_id: str,
    r: np.ndarray,
    v_obs: np.ndarray,
    v_err: np.ndarray,
    v_gas: np.ndarray,
    v_disk: np.ndarray,
    v_bulge: np.ndarray,
    has_bulge: bool,
    model: ModelName,
) -> BoundaryFitResult:
    """Refit one model and record parameter boundary hits (audit only)."""
    v_err = np.maximum(v_err, V_ERR_FLOOR)
    names, lb, ub, predict, n_par = _fit_bounds_setup(model, has_bulge, r, v_obs)
    x0 = np.clip((lb + ub) / 2.0, lb, ub)

    def residuals(p: np.ndarray) -> np.ndarray:
        return (predict(p, v_gas, v_disk, v_bulge, r) - v_obs) / v_err

    res = least_squares(residuals, x0, bounds=(lb, ub), max_nfev=4000)
    p_opt = res.x
    hits = [names[i] for i in range(len(names)) if _at_bound(float(p_opt[i]), float(lb[i]), float(ub[i]))]

    v_pred = predict(p_opt, v_gas, v_disk, v_bulge, r)
    chi2 = float(np.sum(((v_pred - v_obs) / v_err) ** 2))

    params = {names[i]: float(p_opt[i]) for i in range(len(names))}
    if model == "mond":
        params["a0"] = A0_MOND_KMS2_KPC
    if model == "tdf_kessence":
        params["a0"] = A0_MOND_KMS2_KPC
        params["coupling"] = params.get("beta_over_M", np.nan)

    return BoundaryFitResult(
        galaxy_id=galaxy_id,
        model=model,
        params=params,
        param_names=names,
        lower_bounds=lb,
        upper_bounds=ub,
        boundary_hits=hits,
        n_params=n_par,
        chi2=chi2,
        n_points=len(r),
    )


def audit_mond_baseline_galaxy(
    galaxy_id: str,
    gdf: pd.DataFrame,
    upsilon_disk: float,
    upsilon_bulge: float,
    a0: float = A0_MOND_KMS2_KPC,
) -> dict[str, Any]:
    """MOND vs baryon audit using analytic MOND at stored Υ."""
    gdf = gdf.sort_values("r_kpc")
    r = gdf["r_kpc"].to_numpy()
    v_gas = gdf["v_gas"].to_numpy()
    v_disk = gdf["v_disk"].to_numpy()
    v_bulge = gdf["v_bulge"].to_numpy()
    has_bulge = galaxy_has_bulge(v_bulge)

    v_m, v_b, g_b = v_mond_analytic(r, v_gas, v_disk, v_bulge, upsilon_disk, upsilon_bulge, a0)
    g_m = mond_g_baryon_analytic(g_b, a0)

    dv = np.abs(v_m - v_b)
    v2_b = np.maximum(v_b**2, 1e-12)
    frac_boost = (np.maximum(v_m**2, 0) - v2_b) / v2_b

    low_mask = g_b < a0
    n_low = int(np.sum(low_mask))
    if n_low > 0:
        boost_low = float(np.median(v_m[low_mask] / np.maximum(v_b[low_mask], 1e-12)))
    else:
        boost_low = float("nan")

    median_dv = float(np.median(dv))
    mond_active = median_dv >= MOND_DV_TOL
    if n_low > 0 and np.isfinite(boost_low):
        mond_active = mond_active and boost_low >= MOND_RATIO_TOL

    mond_inactive = (median_dv < MOND_DV_TOL) or (
        n_low > 0 and np.isfinite(boost_low) and boost_low < MOND_RATIO_TOL
    )

    return {
        "galaxy_id": galaxy_id,
        "n_points": len(r),
        "has_bulge": has_bulge,
        "upsilon_disk": upsilon_disk,
        "upsilon_bulge": upsilon_bulge if has_bulge else np.nan,
        "a0_kms2_kpc": a0,
        "mond_vs_baryon_max_dv": float(np.max(dv)),
        "mond_vs_baryon_median_dv": median_dv,
        "mond_vs_baryon_max_frac_v2_boost": float(np.max(frac_boost)),
        "mond_active_flag": bool(mond_active),
        "mond_inactive_or_unit_bug": bool(mond_inactive),
        "n_low_acc_points": n_low,
        "median_mond_boost_low_acc": boost_low,
        "mond_low_acc_pass": bool(n_low > 0 and np.isfinite(boost_low) and boost_low >= MOND_RATIO_TOL),
        "g_mond_ge_gb_all": bool(np.all(g_m >= g_b - 1e-9)),
        "v_mond_ge_vb_all": bool(np.all(v_m >= v_b - 1e-9)),
    }


def _galaxy_has_bulge_from_summary(summary_df: pd.DataFrame, galaxy_id: str) -> bool:
    b = summary_df[
        (summary_df["galaxy_id"] == galaxy_id) & (summary_df["model"] == "baryon_only")
    ]
    if b.empty:
        return False
    ub = b.iloc[0].get("upsilon_bulge", np.nan)
    return pd.notna(ub) and float(ub) > 0.01


def audit_parameter_count(summary_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model in ("baryon_only", "nfw", "mond", "tdf_kessence"):
        sub = summary_df[summary_df["model"] == model]
        if sub.empty:
            continue
        obs_min = int(sub["n_params"].min())
        obs_max = int(sub["n_params"].max())
        a0_fitted = False
        if model == "baryon_only":
            rule = "1 without bulge, 2 with bulge (Υ_disk [, Υ_bulge])"
        elif model == "mond":
            rule = "baryon Υ only; a0 fixed at 3700 (km/s)²/kpc in v0.21.0"
        elif model == "nfw":
            rule = "Υ + v200 + r_s (3 or 4 params)"
        else:
            rule = "Υ + beta_over_M (2 or 3 params); a0 fixed"
        consistent = True
        for _, row in sub.iterrows():
            gid = str(row["galaxy_id"])
            hb = _galaxy_has_bulge_from_summary(summary_df, gid)
            exp = expected_n_params(model, hb, a0_fitted=a0_fitted)  # type: ignore[arg-type]
            if int(row["n_params"]) != exp:
                consistent = False
        rows.append(
            {
                "model": model,
                "expected_n_params_rule": rule,
                "observed_n_params_min": obs_min,
                "observed_n_params_max": obs_max,
                "parameter_count_consistent": consistent,
                "a0_fitted_in_calibration": a0_fitted,
            },
        )
    return pd.DataFrame(rows)


def _galaxy_class(vmax: float) -> str:
    if vmax < 80.0:
        return "dwarf"
    if vmax < 160.0:
        return "intermediate"
    return "massive"


def run_sparc_parameter_audit(
    input_csv: Path,
    calibration_summary_csv: Path,
    comparison_csv: Path,
    output_dir: Path,
    *,
    refit_for_boundaries: bool = True,
    max_galaxies: int | None = None,
) -> dict[str, Any]:
    """
    Run full parameter audit; write new tables/figures/report only.
    """
    input_csv = Path(input_csv)
    output_dir = Path(output_dir)
    tables = output_dir / "tables"
    reports = output_dir / "reports"
    figures = output_dir / "figures"
    for d in (tables, reports, figures):
        d.mkdir(parents=True, exist_ok=True)

    sparc = pd.read_csv(input_csv)
    validate_sparc_input_schema(sparc)
    summary = pd.read_csv(calibration_summary_csv)
    comparison = pd.read_csv(comparison_csv)

    galaxy_ids = sorted(comparison["galaxy_id"].unique())
    if max_galaxies is not None:
        galaxy_ids = galaxy_ids[: max(1, int(max_galaxies))]
        comparison = comparison[comparison["galaxy_id"].isin(galaxy_ids)]

    # --- MOND baseline audit (analytic, stored Υ from mond row) ---
    mond_rows = []
    for gid in galaxy_ids:
        gdf = sparc[sparc["galaxy_id"] == gid]
        mond_fit = summary[(summary["galaxy_id"] == gid) & (summary["model"] == "mond")]
        if mond_fit.empty:
            continue
        row = mond_fit.iloc[0]
        ud = float(row["upsilon_disk"])
        ub = float(row["upsilon_bulge"]) if pd.notna(row["upsilon_bulge"]) else 0.0
        mond_rows.append(audit_mond_baseline_galaxy(gid, gdf, ud, ub))
    mond_df = pd.DataFrame(mond_rows)

    # --- Boundary refit audit ---
    boundary_records = []
    bic_audit_rows = []
    if refit_for_boundaries:
        for gid in galaxy_ids:
            gdf = sparc[sparc["galaxy_id"] == gid].sort_values("r_kpc")
            if len(gdf) < 5:
                continue
            r = gdf["r_kpc"].to_numpy()
            v_obs = gdf["v_obs"].to_numpy()
            v_err = gdf["v_err"].to_numpy()
            v_gas = gdf["v_gas"].to_numpy()
            v_disk = gdf["v_disk"].to_numpy()
            v_bulge = gdf["v_bulge"].to_numpy()
            hb = galaxy_has_bulge(v_bulge)
            for model in ("baryon_only", "nfw", "mond", "tdf_kessence"):
                bfit = fit_model_with_boundary_audit(
                    gid, r, v_obs, v_err, v_gas, v_disk, v_bulge, hb, model,
                )
                stored = summary[(summary["galaxy_id"] == gid) & (summary["model"] == model)]
                if stored.empty:
                    continue
                st = stored.iloc[0]
                bic_rec = bic(float(st["chi2"]), int(st["n_points"]), int(st["n_params"]))
                aic_rec = aic(float(st["chi2"]), int(st["n_params"]))
                boundary_records.append(
                    {
                        "galaxy_id": gid,
                        "model": model,
                        "any_boundary_hit": len(bfit.boundary_hits) > 0,
                        "boundary_hit_count": len(bfit.boundary_hits),
                        "boundary_hit_fields": ";".join(bfit.boundary_hits),
                        "likely_overfit_or_bound_limited": len(bfit.boundary_hits) >= 2,
                        "upsilon_disk": bfit.params.get("upsilon_disk", np.nan),
                        "upsilon_bulge": bfit.params.get("upsilon_bulge", np.nan),
                        "v200": bfit.params.get("v200", np.nan),
                        "r_s": bfit.params.get("r_s", np.nan),
                        "beta_over_M": bfit.params.get("beta_over_M", np.nan),
                        "a0": bfit.params.get("a0", np.nan),
                        "a0_fixed": model in ("mond", "tdf_kessence"),
                        "upsilon_disk_at_bound": "upsilon_disk" in bfit.boundary_hits,
                        "upsilon_bulge_at_bound": "upsilon_bulge" in bfit.boundary_hits,
                        "v200_at_bound": "v200" in bfit.boundary_hits,
                        "r_s_at_bound": "r_s" in bfit.boundary_hits,
                        "beta_over_M_at_bound": "beta_over_M" in bfit.boundary_hits,
                    },
                )
                bic_audit_rows.append(
                    {
                        "galaxy_id": gid,
                        "model": model,
                        "n_points": int(st["n_points"]),
                        "n_params": int(st["n_params"]),
                        "chi2": float(st["chi2"]),
                        "bic_original": float(st["bic"]),
                        "bic_recomputed": bic_rec,
                        "bic_mismatch": abs(bic_rec - float(st["bic"])) > BIC_MISMATCH_TOL,
                        "aic_original": float(st["aic"]),
                        "aic_recomputed": aic_rec,
                        "aic_mismatch": abs(aic_rec - float(st["aic"])) > BIC_MISMATCH_TOL,
                    },
                )

    boundary_df = pd.DataFrame(boundary_records)
    param_count_df = audit_parameter_count(summary)

    # Galaxies with severe NFW/TDF boundary pressure (≥2 params at bound on either model)
    bound_galaxies: set[str] = set()
    if len(boundary_df):
        for gid in galaxy_ids:
            sub = boundary_df[
                (boundary_df["galaxy_id"] == gid)
                & (boundary_df["model"].isin(["nfw", "tdf_kessence"]))
            ]
            if sub["likely_overfit_or_bound_limited"].any():
                bound_galaxies.add(gid)

    comp_clean = comparison[~comparison["galaxy_id"].isin(bound_galaxies)].copy()
    comp_all = comparison.copy()

    def _comparison_stats(comp: pd.DataFrame) -> dict[str, Any]:
        if comp.empty:
            return {}
        delta = comp["delta_bic_tdf_vs_nfw"]
        return {
            "n_galaxies": len(comp),
            "tdf_bic_wins": int((comp["best_model_by_bic"] == "tdf_kessence").sum()),
            "nfw_bic_wins": int((comp["best_model_by_bic"] == "nfw").sum()),
            "tdf_competitive_delta2": int(comp["tdf_bic_competitive"].sum()),
            "tdf_strong_win_vs_nfw": int((delta < -6).sum()),
            "nfw_strong_win_vs_tdf": int((delta > 6).sum()),
            "median_delta_bic_tdf_minus_nfw": float(delta.median()),
            "fraction_boundary_nfw_or_tdf": (
                len(bound_galaxies) / max(len(galaxy_ids), 1)
            ),
        }

    stats_before = _comparison_stats(comp_all)
    stats_after = _comparison_stats(comp_clean)

    # Galaxy class
    vmax_map = sparc.groupby("galaxy_id")["v_obs"].max()
    comp_all = comp_all.copy()
    comp_all["galaxy_class"] = comp_all["galaxy_id"].map(
        lambda g: _galaxy_class(float(vmax_map.get(g, 0))),
    )

    class_rows = []
    for cls in ("dwarf", "intermediate", "massive"):
        sub = comp_all[comp_all["galaxy_class"] == cls]
        if len(sub) == 0:
            continue
        tdf_chi2 = []
        nfw_chi2 = []
        for gid in sub["galaxy_id"]:
            t = summary[(summary["galaxy_id"] == gid) & (summary["model"] == "tdf_kessence")]
            n = summary[(summary["galaxy_id"] == gid) & (summary["model"] == "nfw")]
            if len(t):
                tdf_chi2.append(float(t.iloc[0]["reduced_chi2"]))
            if len(n):
                nfw_chi2.append(float(n.iloc[0]["reduced_chi2"]))
        class_rows.append(
            {
                "galaxy_class": cls,
                "n_galaxies": len(sub),
                "median_delta_bic_tdf_minus_nfw": float(sub["delta_bic_tdf_vs_nfw"].median()),
                "tdf_bic_wins": int((sub["best_model_by_bic"] == "tdf_kessence").sum()),
                "nfw_bic_wins": int((sub["best_model_by_bic"] == "nfw").sum()),
                "median_reduced_chi2_tdf": float(np.median(tdf_chi2)) if tdf_chi2 else np.nan,
                "median_reduced_chi2_nfw": float(np.median(nfw_chi2)) if nfw_chi2 else np.nan,
            },
        )
    class_df = pd.DataFrame(class_rows)

    # Stored MOND vs baryon identity check from summary
    mond_inactive_count = int(mond_df["mond_inactive_or_unit_bug"].sum()) if len(mond_df) else 0
    mond_identical_chi2 = 0
    for gid in galaxy_ids:
        b = summary[(summary["galaxy_id"] == gid) & (summary["model"] == "baryon_only")]
        m = summary[(summary["galaxy_id"] == gid) & (summary["model"] == "mond")]
        if len(b) and len(m) and abs(float(b.iloc[0]["chi2"]) - float(m.iloc[0]["chi2"])) < 1e-6:
            mond_identical_chi2 += 1

    if mond_identical_chi2 > 0.5 * max(len(mond_df), 1):
        recommendation = "B"
        recommendation_text = (
            "Rerun Phase 8A calibration with corrected analytic MOND "
            "(stored fits match baryon-only χ² for most galaxies; iterative solver "
            "often returned g_obs ≈ g_b while analytic MOND is active at same Υ)."
        )
    elif mond_inactive_count > 0.5 * len(mond_df):
        recommendation = "B"
        recommendation_text = (
            "Rerun Phase 8A calibration with corrected MOND units or solver; "
            "analytic MOND boost is inactive for many galaxies."
        )
    elif len(bound_galaxies) > 0.35 * len(galaxy_ids):
        recommendation = "C"
        recommendation_text = (
            "Rerun with tighter/fairer parameter bounds; many NFW/TDF fits hit limits."
        )
    elif stats_after.get("tdf_bic_wins", 0) >= stats_before.get("tdf_bic_wins", 0) * 0.7:
        recommendation = "A"
        recommendation_text = (
            "v0.21.0 comparison remains qualitatively stable after audit "
            "(analytic MOND is active; boundary-excluded subset retains TDF competitiveness)."
        )
    else:
        recommendation = "D"
        recommendation_text = "Do not use v0.21.0 SPARC comparison in paper until recalibration."

    audit_summary = pd.DataFrame(
        [
            {
                "metric": "galaxies_in_comparison",
                "value": len(galaxy_ids),
            },
            {
                "metric": "mond_galaxies_inactive_or_unit_bug",
                "value": mond_inactive_count,
            },
            {
                "metric": "mond_chi2_identical_to_baryon_count",
                "value": mond_identical_chi2,
            },
            {
                "metric": "galaxies_nfw_or_tdf_boundary_hit",
                "value": len(bound_galaxies),
            },
            {
                "metric": "bic_mismatch_count",
                "value": int(pd.DataFrame(bic_audit_rows)["bic_mismatch"].sum())
                if bic_audit_rows
                else 0,
            },
            {
                "metric": "recommendation_code",
                "value": recommendation,
            },
            {
                "metric": "tdf_bic_wins_all",
                "value": stats_before.get("tdf_bic_wins", 0),
            },
            {
                "metric": "tdf_bic_wins_audited_clean",
                "value": stats_after.get("tdf_bic_wins", 0),
            },
            {
                "metric": "nfw_bic_wins_all",
                "value": stats_before.get("nfw_bic_wins", 0),
            },
            {
                "metric": "nfw_bic_wins_audited_clean",
                "value": stats_after.get("nfw_bic_wins", 0),
            },
            {
                "metric": "median_delta_bic_tdf_minus_nfw",
                "value": stats_before.get("median_delta_bic_tdf_minus_nfw", np.nan),
            },
        ],
    )

    # Save tables
    mond_df.to_csv(tables / "sparc_mond_baseline_audit.csv", index=False)
    boundary_df.to_csv(tables / "sparc_parameter_boundary_flags.csv", index=False)
    param_count_df.to_csv(tables / "sparc_parameter_count_audit.csv", index=False)
    audit_summary.to_csv(tables / "sparc_parameter_audit_summary.csv", index=False)
    if bic_audit_rows:
        pd.DataFrame(bic_audit_rows).to_csv(
            tables / "sparc_bic_recompute_audit.csv", index=False,
        )
    class_df.to_csv(tables / "sparc_galaxy_class_summary.csv", index=False)

    _write_audit_figures(mond_df, boundary_df, comp_all, comp_clean, figures)
    report = _build_audit_report(
        recommendation,
        recommendation_text,
        mond_df,
        param_count_df,
        boundary_df,
        stats_before,
        stats_after,
        class_df,
        bic_audit_rows,
        mond_identical_chi2,
    )
    (reports / "sparc_parameter_audit_report.md").write_text(report, encoding="utf-8")

    return {
        "mond_df": mond_df,
        "boundary_df": boundary_df,
        "param_count_df": param_count_df,
        "audit_summary": audit_summary,
        "recommendation": recommendation,
        "stats_before": stats_before,
        "stats_after": stats_after,
    }


def _write_audit_figures(
    mond_df: pd.DataFrame,
    boundary_df: pd.DataFrame,
    comp_all: pd.DataFrame,
    comp_clean: pd.DataFrame,
    figures_dir: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures_dir.mkdir(parents=True, exist_ok=True)

    if len(mond_df):
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(
            mond_df["mond_vs_baryon_median_dv"],
            mond_df["median_mond_boost_low_acc"],
            c=mond_df["mond_active_flag"].map({True: "C2", False: "C3"}),
            alpha=0.7,
        )
        ax.axhline(MOND_RATIO_TOL, color="gray", ls="--", lw=1)
        ax.axvline(MOND_DV_TOL, color="gray", ls="--", lw=1)
        ax.set_xlabel("Median |v_mond − v_baryon| [km/s]")
        ax.set_ylabel("Median v_mond/v_baryon (low g_b region)")
        ax.set_title("MOND baseline activity audit")
        fig.tight_layout()
        fig.savefig(figures_dir / "sparc_mond_vs_baryon_boost.png", dpi=150)
        plt.close(fig)

    if len(boundary_df):
        fig, ax = plt.subplots(figsize=(8, 4))
        counts = boundary_df.groupby("model")["any_boundary_hit"].sum()
        ax.bar(counts.index.astype(str), counts.values)
        ax.set_ylabel("Galaxy-model fits with boundary hit")
        ax.set_title("Parameter boundary hits (audit refit)")
        fig.tight_layout()
        fig.savefig(figures_dir / "sparc_parameter_boundary_counts.png", dpi=150)
        plt.close(fig)

    if len(comp_all) and len(comp_clean):
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(
            comp_all["delta_bic_tdf_vs_nfw"],
            bins=25,
            alpha=0.45,
            label="all galaxies",
        )
        ax.hist(
            comp_clean["delta_bic_tdf_vs_nfw"],
            bins=25,
            alpha=0.45,
            label="exclude boundary-limited NFW/TDF",
        )
        ax.axvline(0, color="k", ls="--")
        ax.axvline(-6, color="gray", ls=":")
        ax.axvline(6, color="gray", ls=":")
        ax.set_xlabel("ΔBIC (TDF − NFW)")
        ax.set_ylabel("Count")
        ax.legend()
        ax.set_title("Audited TDF vs NFW BIC delta")
        fig.tight_layout()
        fig.savefig(figures_dir / "sparc_delta_bic_tdf_nfw_audited.png", dpi=150)
        plt.close(fig)

    if "galaxy_class" in comp_all.columns:
        fig, ax = plt.subplots(figsize=(7, 4))
        for cls in comp_all["galaxy_class"].unique():
            sub = comp_all[comp_all["galaxy_class"] == cls]
            ax.scatter(
                sub.index,
                sub["delta_bic_tdf_vs_nfw"],
                label=cls,
                alpha=0.5,
                s=20,
            )
        ax.axhline(0, color="k", ls="--")
        ax.set_ylabel("ΔBIC TDF − NFW")
        ax.set_title("By galaxy class (v_max proxy)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(figures_dir / "sparc_audit_by_galaxy_class.png", dpi=150)
        plt.close(fig)


def _fmt_num(x: Any) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    return f"{float(x):.2f}"


def _build_audit_report(
    recommendation: str,
    recommendation_text: str,
    mond_df: pd.DataFrame,
    param_count_df: pd.DataFrame,
    boundary_df: pd.DataFrame,
    stats_before: dict[str, Any],
    stats_after: dict[str, Any],
    class_df: pd.DataFrame,
    bic_audit_rows: list[dict[str, Any]],
    mond_identical_chi2: int,
) -> str:
    lines = [
        "# SPARC parameter audit report (Phase 8A.1)",
        "",
        f"## ⚠️ {BANNER_SPARC_AUDIT}",
        "",
        f"## ⚠️ {BANNER_CALIBRATION}",
        "",
        "## Why this audit was needed",
        "",
        "Phase 8A (v0.21.0) reported TDF BIC wins over NFW on 171 galaxies, but "
        "**MOND median BIC and reduced χ² were nearly identical to baryon-only**. "
        "That pattern is consistent with the calibration MOND solver returning "
        "`g_obs ≈ g_b` whenever `g_b ≳ 0.5 a₀`, making MOND inactive for many SPARC points.",
        "",
        "## MOND unit check",
        "",
        f"- Acceleration units: **g = v²/r** in **(km/s)²/kpc**",
        f"- MOND scale: **a₀ = {A0_MOND_KMS2_KPC} (km/s)²/kpc**",
        "- Analytic simple-μ solution used in audit:",
        "  `g_mond = 0.5 (g_b + sqrt(g_b² + 4 g_b a₀))`",
        f"- Galaxies flagged `mond_inactive_or_unit_bug`: "
        f"**{int(mond_df['mond_inactive_or_unit_bug'].sum()) if len(mond_df) else 0}** / {len(mond_df)}",
        f"- Galaxies with **identical χ²** (stored) for MOND and baryon-only: **{mond_identical_chi2}**",
        "",
        "## Parameter count audit (BIC fairness)",
        "",
        "| Model | Rule | n_min | n_max | Consistent |",
        "| --- | --- | --- | --- | --- |",
    ]
    for _, row in param_count_df.iterrows():
        lines.append(
            f"| {row['model']} | {row['expected_n_params_rule']} | "
            f"{row['observed_n_params_min']} | {row['observed_n_params_max']} | "
            f"{row['parameter_count_consistent']} |",
        )

    lines.extend(
        [
            "",
            "## BIC recomputation",
            "",
        ],
    )
    if bic_audit_rows:
        n_mis = sum(1 for r in bic_audit_rows if r["bic_mismatch"])
        lines.append(
            f"- Stored vs recomputed BIC mismatches: **{n_mis}** / {len(bic_audit_rows)} "
            f"(tolerance {BIC_MISMATCH_TOL})",
        )
    else:
        lines.append("- BIC recomputation not run.")

    lines.extend(
        [
            "",
            "## Comparison summary (TDF vs NFW)",
            "",
            "| Metric | All galaxies | Excluding NFW/TDF boundary hits |",
            "| --- | --- | --- |",
            f"| N galaxies | {stats_before.get('n_galaxies', '—')} | {stats_after.get('n_galaxies', '—')} |",
            f"| TDF BIC wins | {stats_before.get('tdf_bic_wins', '—')} | {stats_after.get('tdf_bic_wins', '—')} |",
            f"| NFW BIC wins | {stats_before.get('nfw_bic_wins', '—')} | {stats_after.get('nfw_bic_wins', '—')} |",
            f"| TDF competitive (ΔBIC<2 vs best) | {stats_before.get('tdf_competitive_delta2', '—')} | "
            f"{stats_after.get('tdf_competitive_delta2', '—')} |",
            f"| TDF strong win (ΔBIC<-6) | {stats_before.get('tdf_strong_win_vs_nfw', '—')} | "
            f"{stats_after.get('tdf_strong_win_vs_nfw', '—')} |",
            f"| NFW strong win (ΔBIC>6) | {stats_before.get('nfw_strong_win_vs_tdf', '—')} | "
            f"{stats_after.get('nfw_strong_win_vs_tdf', '—')} |",
            f"| Median ΔBIC (TDF−NFW) | {_fmt_num(stats_before.get('median_delta_bic_tdf_minus_nfw'))} | "
            f"{_fmt_num(stats_after.get('median_delta_bic_tdf_minus_nfw'))} |",
            "",
            "## Galaxy class (v_max proxy)",
            "",
        ],
    )
    if len(class_df):
        lines.append(
            "| Class | N | Median ΔBIC TDF−NFW | Median χ²_red TDF | Median χ²_red NFW | TDF wins | NFW wins |",
        )
        lines.append("| --- | --- | --- | --- | --- | --- | --- |")
        for _, row in class_df.iterrows():
            lines.append(
                f"| {row['galaxy_class']} | {row['n_galaxies']} | "
                f"{row['median_delta_bic_tdf_minus_nfw']:.2f} | "
                f"{_fmt_num(row.get('median_reduced_chi2_tdf'))} | "
                f"{_fmt_num(row.get('median_reduced_chi2_nfw'))} | "
                f"{row['tdf_bic_wins']} | {row['nfw_bic_wins']} |",
            )

    if len(boundary_df):
        hit_rate = boundary_df.groupby("model")["any_boundary_hit"].mean()
        lines.extend(
            [
                "",
                "## Boundary hits (audit refit)",
                "",
            ],
        )
        for model, rate in hit_rate.items():
            lines.append(f"- **{model}**: {100*rate:.1f}% of fits hit a bound")

    lines.extend(
        [
            "",
            "## Final recommendation",
            "",
            f"**{recommendation}.** {recommendation_text}",
            "",
            "## Limitations",
            "",
            "- Rotation curves only; no lensing or cosmology.",
            "- No full 3D disk solver; TDF remains a calibration proxy.",
            "- Does not prove dark-matter replacement or full SPARC validation.",
            "",
        ],
    )
    return "\n".join(lines)

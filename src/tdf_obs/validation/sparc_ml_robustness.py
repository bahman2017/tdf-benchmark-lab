"""
SPARC Step 3 — mass-to-light (M/L) robustness analysis.

Reruns corrected-MOND calibration under multiple Υ_disk / Υ_bulge priors.
Analysis only; not observational validation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from tdf_obs.fitting.metrics import aic, bic, chi_square, mse, reduced_chi_square
from tdf_obs.validation.sparc_boundary_filtered_analysis import _at_bound
from tdf_obs.validation.sparc_real_calibration import (
    A0_MOND_DEFAULT,
    A0_TDF_DEFAULT,
    BIC_COMPETITIVE_DELTA,
    ModelFitResult,
    V_ERR_FLOOR,
    _prepare_galaxy_frame,
    check_mond_activity,
    compute_v_baryon,
    galaxy_has_bulge,
    validate_sparc_input_schema,
    v_mond_analytic,
    v_nfw_total,
    v_tdf_kessence_disk_proxy,
)

BANNER_ML_ROBUSTNESS = (
    "SPARC MASS-TO-LIGHT ROBUSTNESS ANALYSIS — NOT FULL OBSERVATIONAL VALIDATION"
)

BANNER_CALIBRATION = (
    "These results are calibration diagnostics for a phenomenological TDF model. "
    "They do not constitute observational validation of TDF unless all required "
    "multi-channel constraints are passed."
)

ModelName = Literal["baryon_only", "corrected_mond", "nfw", "tdf_kessence"]
RegimeId = Literal["A_fixed_canonical", "B_narrow_prior", "C_standard_prior", "D_shared_prior"]

SUMMARY_COLUMNS: tuple[str, ...] = (
    "ml_regime",
    "ml_regime_label",
    "galaxies_fitted",
    "bic_win_baryon_only",
    "bic_win_corrected_mond",
    "bic_win_nfw",
    "bic_win_tdf_kessence",
    "median_bic_baryon_only",
    "median_bic_corrected_mond",
    "median_bic_nfw",
    "median_bic_tdf_kessence",
    "median_reduced_chi2_baryon_only",
    "median_reduced_chi2_corrected_mond",
    "median_reduced_chi2_nfw",
    "median_reduced_chi2_tdf_kessence",
    "tdf_vs_nfw_win_count",
    "tdf_vs_corrected_mond_win_count",
    "tdf_competitive_count",
    "boundary_hit_fraction_baryon_only",
    "boundary_hit_fraction_corrected_mond",
    "boundary_hit_fraction_nfw",
    "boundary_hit_fraction_tdf_kessence",
    "median_upsilon_disk_baryon_only",
    "median_upsilon_disk_corrected_mond",
    "median_upsilon_disk_nfw",
    "median_upsilon_disk_tdf_kessence",
    "fraction_upsilon_disk_at_bound_baryon_only",
    "fraction_upsilon_disk_at_bound_corrected_mond",
    "fraction_upsilon_disk_at_bound_nfw",
    "fraction_upsilon_disk_at_bound_tdf_kessence",
    "mond_active_fraction",
)

BY_GALAXY_COLUMNS: tuple[str, ...] = (
    "ml_regime",
    "galaxy_id",
    "n_points",
    "has_bulge",
    "best_model_by_bic",
    "bic_baryon_only",
    "bic_corrected_mond",
    "bic_nfw",
    "bic_tdf_kessence",
    "delta_bic_tdf_vs_nfw",
    "delta_bic_tdf_vs_corrected_mond",
    "tdf_beats_nfw",
    "tdf_beats_corrected_mond",
    "tdf_bic_competitive",
    "upsilon_disk_baryon_only",
    "upsilon_disk_corrected_mond",
    "upsilon_disk_nfw",
    "upsilon_disk_tdf_kessence",
    "upsilon_disk_at_bound_any_model",
    "mond_active_flag",
)


@dataclass(frozen=True)
class MLRegimeConfig:
    regime_id: RegimeId
    label: str
    fixed_upsilon: bool
    fixed_disk: float = 0.5
    fixed_bulge: float = 0.7
    disk_bounds: tuple[float, float] = (0.05, 3.0)
    bulge_bounds: tuple[float, float] = (0.05, 3.0)


ML_REGIMES: dict[RegimeId, MLRegimeConfig] = {
    "A_fixed_canonical": MLRegimeConfig(
        regime_id="A_fixed_canonical",
        label="A: fixed canonical (Υ_disk=0.5, Υ_bulge=0.7)",
        fixed_upsilon=True,
        fixed_disk=0.5,
        fixed_bulge=0.7,
    ),
    "B_narrow_prior": MLRegimeConfig(
        regime_id="B_narrow_prior",
        label="B: narrow prior Υ_disk∈[0.3,0.8], Υ_bulge∈[0.5,1.0]",
        fixed_upsilon=False,
        disk_bounds=(0.3, 0.8),
        bulge_bounds=(0.5, 1.0),
    ),
    "C_standard_prior": MLRegimeConfig(
        regime_id="C_standard_prior",
        label="C: standard prior Υ∈[0.05, 3.0]",
        fixed_upsilon=False,
        disk_bounds=(0.05, 3.0),
        bulge_bounds=(0.05, 3.0),
    ),
    "D_shared_prior": MLRegimeConfig(
        regime_id="D_shared_prior",
        label="D: shared prior (same bounds as C for all models)",
        fixed_upsilon=False,
        disk_bounds=(0.05, 3.0),
        bulge_bounds=(0.05, 3.0),
    ),
}


@dataclass
class MLFitResult(ModelFitResult):
    upsilon_disk_at_bound: bool = False
    upsilon_bulge_at_bound: bool = False
    any_boundary_hit: bool = False
    ml_regime: str = ""


def _metrics(v_obs, v_pred, v_err, n_params):
    v_err = np.maximum(np.asarray(v_err, dtype=float), V_ERR_FLOOR)
    c2 = chi_square(v_obs, v_pred, v_err)
    n = len(v_obs)
    return (
        c2,
        reduced_chi_square(c2, n, n_params),
        float(np.sqrt(mse(v_obs, v_pred))),
        aic(c2, n_params),
        bic(c2, n, n_params),
    )


def fit_model_ml_regime(
    r: np.ndarray,
    v_obs: np.ndarray,
    v_err: np.ndarray,
    v_gas: np.ndarray,
    v_disk: np.ndarray,
    v_bulge: np.ndarray,
    has_bulge: bool,
    model: ModelName,
    regime: MLRegimeConfig,
) -> MLFitResult:
    """Fit one model under a mass-to-light regime."""
    v_err = np.maximum(v_err, V_ERR_FLOOR)
    d_lo, d_hi = regime.disk_bounds
    b_lo, b_hi = regime.bulge_bounds

    def get_upsilon(p_ml: np.ndarray) -> tuple[float, float]:
        if regime.fixed_upsilon:
            return (
                float(regime.fixed_disk),
                float(regime.fixed_bulge if has_bulge else 0.0),
            )
        if has_bulge:
            return float(p_ml[0]), float(p_ml[1])
        return float(p_ml[0]), 0.0

    if regime.fixed_upsilon:
        ml_lb = np.array([])
        ml_ub = np.array([])
        ml_x0 = np.array([])
        n_ml = 0
    else:
        if has_bulge:
            ml_lb = np.array([d_lo, b_lo])
            ml_ub = np.array([d_hi, b_hi])
            ml_x0 = np.array([(d_lo + d_hi) / 2, (b_lo + b_hi) / 2])
            n_ml = 2
        else:
            ml_lb = np.array([d_lo])
            ml_ub = np.array([d_hi])
            ml_x0 = np.array([(d_lo + d_hi) / 2])
            n_ml = 1

    def ml_extra(p_ml: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return (
            p_ml[:n_ml] if n_ml else np.array([]),
            p_ml[n_ml:],
        )

    if model == "baryon_only":

        def predict_ml(p_ml: np.ndarray) -> np.ndarray:
            ml_p, _ = ml_extra(p_ml)
            ud, ub = get_upsilon(ml_p)
            return compute_v_baryon(v_gas, v_disk, v_bulge, ud, ub)

        extra_lb = np.array([])
        extra_ub = np.array([])
        extra_x0 = np.array([])
        n_extra = 0
    elif model == "nfw":
        v2000 = float(np.sqrt(max(np.mean(v_obs**2), 1.0)))

        def predict_ml(p_ml: np.ndarray) -> np.ndarray:
            ml_p, extra = ml_extra(p_ml)
            ud, ub = get_upsilon(ml_p)
            v200, rs = extra[0], extra[1]
            return v_nfw_total(r, v_gas, v_disk, v_bulge, ud, ub, v200, rs)

        if has_bulge:
            extra_lb = np.array([1.0, 0.1])
            extra_ub = np.array([500.0, 100.0])
            extra_x0 = np.array([v2000, max(r[-1] / 3.0, 0.5)])
        else:
            extra_lb = np.array([1.0, 0.1])
            extra_ub = np.array([500.0, 100.0])
            extra_x0 = np.array([v2000, max(r[-1] / 3.0, 0.5)])
        n_extra = 2
    elif model == "corrected_mond":

        def predict_ml(p_ml: np.ndarray) -> np.ndarray:
            ml_p, _ = ml_extra(p_ml)
            ud, ub = get_upsilon(ml_p)
            return v_mond_analytic(r, v_gas, v_disk, v_bulge, ud, ub, A0_MOND_DEFAULT)

        extra_lb = np.array([])
        extra_ub = np.array([])
        extra_x0 = np.array([])
        n_extra = 0
    elif model == "tdf_kessence":

        def predict_ml(p_ml: np.ndarray) -> np.ndarray:
            ml_p, extra = ml_extra(p_ml)
            ud, ub = get_upsilon(ml_p)
            beta = extra[0]
            return v_tdf_kessence_disk_proxy(
                r, v_gas, v_disk, v_bulge, ud, ub, beta, A0_TDF_DEFAULT, "deep_mond",
            )

        if has_bulge:
            extra_lb = np.array([1e-4])
            extra_ub = np.array([50.0])
            extra_x0 = np.array([1.0])
        else:
            extra_lb = np.array([1e-4])
            extra_ub = np.array([50.0])
            extra_x0 = np.array([1.0])
        n_extra = 1
    else:
        raise ValueError(model)

    lb = np.concatenate([ml_lb, extra_lb])
    ub = np.concatenate([ml_ub, extra_ub])
    x0 = np.clip(np.concatenate([ml_x0, extra_x0]), lb, ub)
    n_par = n_ml + n_extra

    def residuals(p: np.ndarray) -> np.ndarray:
        return (predict_ml(p) - v_obs) / v_err

    if n_extra == 0 and regime.fixed_upsilon:
        p_opt = np.array([])
        success = True
        reason = ""
    elif regime.fixed_upsilon and n_ml == 0:
        p_opt = np.array(list(extra_x0))
        try:
            res = least_squares(residuals, x0, bounds=(lb, ub), max_nfev=4000)
            p_opt = res.x
            success = bool(res.success)
            reason = "" if success else str(res.message)
        except Exception as exc:  # noqa: BLE001
            p_opt = x0
            success = False
            reason = str(exc)
    else:
        try:
            res = least_squares(residuals, x0, bounds=(lb, ub), max_nfev=4000)
            p_opt = res.x
            success = bool(res.success and res.cost < 1e12)
            reason = "" if success else str(res.message)
        except Exception as exc:  # noqa: BLE001
            p_opt = x0
            success = False
            reason = str(exc)

    ud, ub = get_upsilon(p_opt[:n_ml] if n_ml else np.array([]))
    v_pred = predict_ml(p_opt if len(p_opt) else x0)
    if not np.all(np.isfinite(v_pred)):
        success = False
        reason = reason or "non-finite velocities"
        v_pred = np.where(np.isfinite(v_pred), v_pred, v_obs)

    c2, chi2_red, rmse, aic_v, bic_v = _metrics(v_obs, v_pred, v_err, n_par)

    disk_bound = (
        not regime.fixed_upsilon and _at_bound(ud, d_lo, d_hi)
    )
    bulge_bound = (
        has_bulge
        and not regime.fixed_upsilon
        and _at_bound(ub, b_lo, b_hi)
    )

    params: dict[str, float] = {
        "upsilon_disk": ud,
        "upsilon_bulge": ub if has_bulge else np.nan,
    }
    if model == "nfw":
        extra = p_opt[n_ml:]
        params["v200"] = float(extra[0]) if len(extra) > 0 else np.nan
        params["r_s"] = float(extra[1]) if len(extra) > 1 else np.nan
    if model == "corrected_mond":
        params["a0"] = A0_MOND_DEFAULT
    if model == "tdf_kessence":
        params["beta_over_M"] = float(p_opt[-1]) if len(p_opt) else np.nan
        params["a0"] = A0_TDF_DEFAULT

    return MLFitResult(
        galaxy_id="",
        model=model,
        n_points=len(r),
        n_params=n_par,
        chi2=c2,
        reduced_chi2=chi2_red,
        rmse=rmse,
        aic=aic_v,
        bic=bic_v,
        success=success,
        failure_reason=reason,
        params=params,
        v_pred=v_pred,
        upsilon_disk_at_bound=disk_bound,
        upsilon_bulge_at_bound=bulge_bound,
        any_boundary_hit=disk_bound or bulge_bound,
        ml_regime=regime.regime_id,
    )


def fit_galaxy_ml_regime(
    galaxy_id: str,
    gdf: pd.DataFrame,
    regime: MLRegimeConfig,
    *,
    min_points: int = 5,
) -> tuple[list[MLFitResult], dict[str, Any] | None]:
    gdf = _prepare_galaxy_frame(gdf, min_points)
    if gdf is None:
        return [], None

    r = gdf["r_kpc"].to_numpy()
    v_obs = gdf["v_obs"].to_numpy()
    v_err = gdf["v_err"].to_numpy()
    v_gas = gdf["v_gas"].to_numpy()
    v_disk = gdf["v_disk"].to_numpy()
    v_bulge = gdf["v_bulge"].to_numpy()
    has_bulge = galaxy_has_bulge(v_bulge)

    models: tuple[ModelName, ...] = (
        "baryon_only",
        "corrected_mond",
        "nfw",
        "tdf_kessence",
    )
    results: list[MLFitResult] = []
    for model in models:
        fr = fit_model_ml_regime(
            r, v_obs, v_err, v_gas, v_disk, v_bulge, has_bulge, model, regime,
        )
        fr.galaxy_id = galaxy_id
        results.append(fr)

    by = {r.model: r for r in results}
    bic_tdf = by["tdf_kessence"].bic
    bic_b = by["baryon_only"].bic
    bic_n = by["nfw"].bic
    bic_m = by["corrected_mond"].bic
    best = min(by, key=lambda m: by[m].bic)

    mond_act = check_mond_activity(
        r, v_gas, v_disk, v_bulge,
        by["corrected_mond"].params["upsilon_disk"],
        by["corrected_mond"].params.get("upsilon_bulge", 0.0) or 0.0,
    )

    comp = {
        "ml_regime": regime.regime_id,
        "galaxy_id": galaxy_id,
        "n_points": len(r),
        "has_bulge": has_bulge,
        "best_model_by_bic": best,
        "bic_baryon_only": bic_b,
        "bic_corrected_mond": bic_m,
        "bic_nfw": bic_n,
        "bic_tdf_kessence": bic_tdf,
        "delta_bic_tdf_vs_nfw": bic_tdf - bic_n,
        "delta_bic_tdf_vs_corrected_mond": bic_tdf - bic_m,
        "tdf_beats_nfw": bic_tdf < bic_n,
        "tdf_beats_corrected_mond": bic_tdf < bic_m,
        "tdf_bic_competitive": (bic_tdf - min(bic_b, bic_n, bic_m)) < BIC_COMPETITIVE_DELTA,
        "upsilon_disk_baryon_only": by["baryon_only"].params["upsilon_disk"],
        "upsilon_disk_corrected_mond": by["corrected_mond"].params["upsilon_disk"],
        "upsilon_disk_nfw": by["nfw"].params["upsilon_disk"],
        "upsilon_disk_tdf_kessence": by["tdf_kessence"].params["upsilon_disk"],
        "upsilon_disk_at_bound_any_model": any(r.upsilon_disk_at_bound for r in results),
        "mond_active_flag": bool(mond_act["mond_active_flag"]),
    }
    return results, comp


def aggregate_regime_stats(
    regime: MLRegimeConfig,
    comparisons: list[dict[str, Any]],
    fit_rows: list[MLFitResult],
) -> dict[str, Any]:
    comp_df = pd.DataFrame(comparisons)
    if comp_df.empty:
        return {"ml_regime": regime.regime_id, "ml_regime_label": regime.label, "galaxies_fitted": 0}

    fit_df = pd.DataFrame(
        [
            {
                "model": fr.model,
                "bic": fr.bic,
                "reduced_chi2": fr.reduced_chi2,
                "upsilon_disk": fr.params.get("upsilon_disk"),
                "upsilon_disk_at_bound": fr.upsilon_disk_at_bound,
                "any_boundary_hit": fr.any_boundary_hit,
            }
            for fr in fit_rows
        ],
    )

    row: dict[str, Any] = {
        "ml_regime": regime.regime_id,
        "ml_regime_label": regime.label,
        "galaxies_fitted": len(comp_df),
        "bic_win_baryon_only": int((comp_df["best_model_by_bic"] == "baryon_only").sum()),
        "bic_win_corrected_mond": int((comp_df["best_model_by_bic"] == "corrected_mond").sum()),
        "bic_win_nfw": int((comp_df["best_model_by_bic"] == "nfw").sum()),
        "bic_win_tdf_kessence": int((comp_df["best_model_by_bic"] == "tdf_kessence").sum()),
        "tdf_vs_nfw_win_count": int(comp_df["tdf_beats_nfw"].sum()),
        "tdf_vs_corrected_mond_win_count": int(comp_df["tdf_beats_corrected_mond"].sum()),
        "tdf_competitive_count": int(comp_df["tdf_bic_competitive"].sum()),
        "mond_active_fraction": float(comp_df["mond_active_flag"].mean()),
    }

    for model in ("baryon_only", "corrected_mond", "nfw", "tdf_kessence"):
        sub = fit_df[fit_df["model"] == model]
        row[f"median_bic_{model}"] = float(sub["bic"].median()) if len(sub) else np.nan
        row[f"median_reduced_chi2_{model}"] = (
            float(sub["reduced_chi2"].median()) if len(sub) else np.nan
        )
        row[f"median_upsilon_disk_{model}"] = (
            float(sub["upsilon_disk"].median()) if len(sub) else np.nan
        )
        row[f"boundary_hit_fraction_{model}"] = (
            float(sub["any_boundary_hit"].mean()) if len(sub) else np.nan
        )
        row[f"fraction_upsilon_disk_at_bound_{model}"] = (
            float(sub["upsilon_disk_at_bound"].mean()) if len(sub) else np.nan
        )

    return row


@dataclass
class MLRobustnessRunResult:
    summary: pd.DataFrame
    by_galaxy: pd.DataFrame
    regimes: list[RegimeId]


def run_ml_robustness_analysis(
    sparc_csv: Path,
    output_dir: Path,
    *,
    max_galaxies: int | None = None,
    quality_min_points: int = 5,
    regimes: tuple[RegimeId, ...] | None = None,
) -> MLRobustnessRunResult:
    output_dir = Path(output_dir)
    tables = output_dir / "tables"
    reports = output_dir / "reports"
    figures = output_dir / "figures"
    metadata = output_dir / "metadata"
    for d in (tables, reports, figures, metadata):
        d.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(sparc_csv)
    validate_sparc_input_schema(df)

    regime_list = list(regimes) if regimes else list(ML_REGIMES.keys())
    galaxy_ids = sorted(df["galaxy_id"].unique())
    if max_galaxies is not None:
        galaxy_ids = galaxy_ids[: int(max_galaxies)]

    summary_rows: list[dict[str, Any]] = []
    by_galaxy_rows: list[dict[str, Any]] = []
    all_fits_by_regime: dict[str, list[MLFitResult]] = {}

    for rid in regime_list:
        regime = ML_REGIMES[rid]
        comparisons: list[dict[str, Any]] = []
        regime_fits: list[MLFitResult] = []
        for gid in galaxy_ids:
            gdf = df[df["galaxy_id"] == gid]
            if len(gdf) < quality_min_points:
                continue
            try:
                fits, comp = fit_galaxy_ml_regime(
                    gid, gdf, regime, min_points=quality_min_points,
                )
                if comp is None:
                    continue
                comparisons.append(comp)
                regime_fits.extend(fits)
                by_galaxy_rows.append(comp)
            except Exception:
                continue
        all_fits_by_regime[rid] = regime_fits
        summary_rows.append(aggregate_regime_stats(regime, comparisons, regime_fits))

    summary_df = pd.DataFrame(summary_rows)
    by_galaxy_df = pd.DataFrame(by_galaxy_rows)

    summary_df.to_csv(tables / "ml_robustness_summary.csv", index=False)
    by_galaxy_df.to_csv(tables / "ml_robustness_by_galaxy.csv", index=False)

    _write_figures(summary_df, by_galaxy_df, all_fits_by_regime, figures)
    report = _build_report(summary_df, sparc_csv)
    (reports / "ml_robustness_report.md").write_text(report, encoding="utf-8")

    return MLRobustnessRunResult(
        summary=summary_df,
        by_galaxy=by_galaxy_df,
        regimes=regime_list,
    )


def _write_figures(
    summary_df: pd.DataFrame,
    by_galaxy_df: pd.DataFrame,
    all_fits: dict[str, list[MLFitResult]],
    figures_dir: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if summary_df.empty:
        return

    regimes = summary_df["ml_regime"].astype(str).tolist()
    x = np.arange(len(regimes))
    width = 0.18

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (col, label) in enumerate(
        [
            ("bic_win_baryon_only", "baryon"),
            ("bic_win_corrected_mond", "MOND"),
            ("bic_win_nfw", "NFW"),
            ("bic_win_tdf_kessence", "TDF"),
        ],
    ):
        ax.bar(x + (i - 1.5) * width, summary_df[col], width, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels([r.replace("_", "\n") for r in regimes], fontsize=7)
    ax.set_ylabel("BIC wins")
    ax.legend()
    ax.set_title("BIC wins by M/L regime")
    fig.tight_layout()
    fig.savefig(figures_dir / "bic_wins_by_ml_regime.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    for rid in regimes:
        sub = by_galaxy_df[by_galaxy_df["ml_regime"] == rid]
        if len(sub):
            ax.hist(sub["delta_bic_tdf_vs_nfw"], bins=20, alpha=0.45, label=rid[:12])
    ax.axvline(0, color="k", ls="--")
    ax.set_xlabel("ΔBIC (TDF − NFW)")
    ax.legend(fontsize=7)
    ax.set_title("TDF vs NFW by M/L regime")
    fig.tight_layout()
    fig.savefig(figures_dir / "tdf_vs_nfw_delta_bic_by_ml_regime.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    for model, color in zip(
        ("baryon_only", "corrected_mond", "nfw", "tdf_kessence"),
        ("C0", "C1", "C2", "C3"),
    ):
        vals = []
        for rid in regimes:
            fits = [f for f in all_fits.get(rid, []) if f.model == model]
            vals.extend([f.params.get("upsilon_disk", np.nan) for f in fits])
        vals = [v for v in vals if np.isfinite(v)]
        if vals:
            ax.hist(vals, bins=20, alpha=0.4, label=model, color=color)
    ax.set_xlabel("Υ_disk (fitted or fixed)")
    ax.legend(fontsize=8)
    ax.set_title("Υ_disk distribution (all regimes)")
    fig.tight_layout()
    fig.savefig(figures_dir / "upsilon_disk_distribution_by_model.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4))
    models = ["baryon_only", "corrected_mond", "nfw", "tdf_kessence"]
    for i, model in enumerate(models):
        fracs = [
            float(summary_df.loc[summary_df["ml_regime"] == rid, f"boundary_hit_fraction_{model}"].iloc[0])
            for rid in regimes
        ]
        ax.bar(x + (i - 1.5) * width, fracs, width, label=model)
    ax.set_xticks(x)
    ax.set_xticklabels([r[:10] for r in regimes], fontsize=7)
    ax.set_ylabel("Fraction with Υ at bound")
    ax.legend(fontsize=7)
    ax.set_title("M/L boundary hits by regime and model")
    fig.tight_layout()
    fig.savefig(figures_dir / "boundary_hits_by_ml_regime.png", dpi=150)
    plt.close(fig)


def _safe_int(val: Any) -> int:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return 0
    return int(val)


def _build_report(summary_df: pd.DataFrame, sparc_csv: Path) -> str:
    lines = [
        "# SPARC mass-to-light robustness report (Step 3)",
        "",
        f"## ⚠️ {BANNER_ML_ROBUSTNESS}",
        "",
        f"## ⚠️ {BANNER_CALIBRATION}",
        "",
        f"**Input:** `{sparc_csv}`",
        "",
        "Corrected analytic MOND: "
        "`g_mond = 0.5 (g_b + sqrt(g_b² + 4 g_b a₀))`, `a₀ = 3700 (km/s)²/kpc`.",
        "",
        "## Regimes",
        "",
        "- **A:** fixed Υ_disk=0.5, Υ_bulge=0.7 (no fitted M/L)",
        "- **B:** narrow Υ priors",
        "- **C:** standard wide Υ priors (baseline calibration)",
        "- **D:** shared bounds (same as C across all models)",
        "",
        "## Summary by regime",
        "",
        "| Regime | N | TDF wins | NFW wins | Med ΔBIC TDF−NFW | TDF vs NFW wins | "
        "Υ_disk bound frac (TDF) |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for _, r in summary_df.iterrows():
        lines.append(
            f"| {r['ml_regime']} | {r['galaxies_fitted']} | {r['bic_win_tdf_kessence']} | "
            f"{r['bic_win_nfw']} | {r['median_bic_tdf_kessence'] - r['median_bic_nfw']:.1f} | "
            f"{r['tdf_vs_nfw_win_count']} | {r['fraction_upsilon_disk_at_bound_tdf_kessence']:.2f} |",
        )

    def _safe_row(regime_id: str) -> pd.Series | None:
        sub = summary_df[summary_df["ml_regime"] == regime_id]
        return sub.iloc[0] if len(sub) else None

    a = _safe_row("A_fixed_canonical")
    b = _safe_row("B_narrow_prior")
    c = _safe_row("C_standard_prior")

    if a is not None and b is not None:

        lines.extend(
            [
                "",
                "## Key questions",
                "",
                f"**Does TDF remain competitive when M/L is fixed?** "
                f"Regime A: TDF BIC wins={_safe_int(a['bic_win_tdf_kessence'])}, "
                f"NFW wins={_safe_int(a['bic_win_nfw'])}, "
                f"TDF beats NFW on {_safe_int(a['tdf_vs_nfw_win_count'])}/"
                f"{_safe_int(a['galaxies_fitted'])} galaxies. "
                + (
                    "TDF **remains competitive** under fixed canonical M/L."
                    if _safe_int(a["tdf_vs_nfw_win_count"])
                    > _safe_int(a["galaxies_fitted"]) * 0.4
                    else "TDF **weakens** under fixed canonical M/L."
                ),
                "",
                f"**Does TDF remain competitive when M/L priors are narrow?** "
                f"Regime B: TDF wins={_safe_int(b['bic_win_tdf_kessence'])}, "
                f"median ΔBIC TDF−NFW={float(b['median_bic_tdf_kessence']) - float(b['median_bic_nfw']):.2f} "
                "(aggregate medians).",
                "",
                f"**Does TDF require unrealistic M/L values?** "
                + (
                    f"Under regime C, median Υ_disk (TDF)="
                    f"{float(c['median_upsilon_disk_tdf_kessence']):.2f} vs NFW="
                    f"{float(c['median_upsilon_disk_nfw']):.2f}; "
                    f"fraction at bounds (TDF)="
                    f"{float(c['fraction_upsilon_disk_at_bound_tdf_kessence']):.2f}, "
                    f"(NFW)={float(c['fraction_upsilon_disk_at_bound_nfw']):.2f}."
                    if c is not None
                    else "Regime C not in this run."
                ),
                "",
                f"**Does NFW or TDF hit M/L boundaries more often?** "
                "See `boundary_hit_fraction_*` and Υ_disk at-bound fractions in the summary table. "
                "Typically **NFW and TDF both** hit bounds under wide priors; "
                "fixed and narrow regimes reduce this.",
                "",
                "## Limitations",
                "",
                "- Rotation-curve calibration only; no new physics claim.",
                "- Does not validate TDF observationally or replace dark matter.",
                "",
            ],
        )
    return "\n".join(lines)

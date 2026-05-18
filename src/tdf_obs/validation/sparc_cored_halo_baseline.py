"""
SPARC Step 4 — cored dark-matter halo baseline comparison.

Adds Burkert and pseudo-isothermal halos alongside NFW and TDF.
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
from tdf_obs.models.dark_matter import (
    v2_burkert_halo_only,
    v2_nfw_halo_only,
    v2_pseudo_isothermal_halo_only,
)
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
    v_tdf_kessence_disk_proxy,
)

BANNER_CORED_HALO = (
    "SPARC CORED-HALO BASELINE COMPARISON — NOT FULL OBSERVATIONAL VALIDATION"
)

BANNER_CALIBRATION = (
    "These results are calibration diagnostics for a phenomenological TDF model. "
    "They do not constitute observational validation of TDF unless all required "
    "multi-channel constraints are passed."
)

ModelName = Literal[
    "baryon_only",
    "corrected_mond",
    "nfw",
    "burkert",
    "pseudo_isothermal",
    "tdf_kessence",
]

HALO_MODELS: tuple[ModelName, ...] = ("nfw", "burkert", "pseudo_isothermal")
CORED_HALO_MODELS: tuple[ModelName, ...] = ("burkert", "pseudo_isothermal")

SUMMARY_COLUMNS: tuple[str, ...] = (
    "model",
    "galaxies_fitted",
    "bic_win_count",
    "median_bic",
    "median_reduced_chi2",
    "median_chi2",
    "tdf_vs_model_win_count",
    "tdf_vs_model_median_delta_bic",
)

COMPARISON_COLUMNS: tuple[str, ...] = (
    "galaxy_id",
    "n_points",
    "has_bulge",
    "best_model_by_bic",
    "bic_baryon_only",
    "bic_corrected_mond",
    "bic_nfw",
    "bic_burkert",
    "bic_pseudo_isothermal",
    "bic_tdf_kessence",
    "chi2_baryon_only",
    "chi2_corrected_mond",
    "chi2_nfw",
    "chi2_burkert",
    "chi2_pseudo_isothermal",
    "chi2_tdf_kessence",
    "reduced_chi2_baryon_only",
    "reduced_chi2_corrected_mond",
    "reduced_chi2_nfw",
    "reduced_chi2_burkert",
    "reduced_chi2_pseudo_isothermal",
    "reduced_chi2_tdf_kessence",
    "delta_bic_tdf_vs_nfw",
    "delta_bic_tdf_vs_burkert",
    "delta_bic_tdf_vs_pseudo_isothermal",
    "delta_bic_tdf_vs_best_cored_halo",
    "best_cored_halo_model",
    "bic_best_cored_halo",
    "tdf_beats_nfw",
    "tdf_beats_burkert",
    "tdf_beats_best_cored_halo",
    "tdf_bic_competitive",
    "mond_active_flag",
)


@dataclass(frozen=True)
class MLPriorConfig:
    disk_bounds: tuple[float, float] = (0.05, 3.0)
    bulge_bounds: tuple[float, float] = (0.05, 3.0)


ML_STANDARD = MLPriorConfig()
ML_NARROW = MLPriorConfig(disk_bounds=(0.3, 0.8), bulge_bounds=(0.5, 1.0))


def v_nfw_total(
    r: np.ndarray,
    v_gas: np.ndarray,
    v_disk: np.ndarray,
    v_bulge: np.ndarray,
    upsilon_disk: float,
    upsilon_bulge: float,
    v200: float,
    r_s: float,
) -> np.ndarray:
    v_b = compute_v_baryon(v_gas, v_disk, v_bulge, upsilon_disk, upsilon_bulge)
    v200 = max(float(v200), 0.0)
    r_s = max(float(r_s), 1e-3)
    v2_halo = v2_nfw_halo_only(r, v200**2, r_s)
    return np.sqrt(np.maximum(v_b**2 + v2_halo, 0.0))


def v_burkert_total(
    r: np.ndarray,
    v_gas: np.ndarray,
    v_disk: np.ndarray,
    v_bulge: np.ndarray,
    upsilon_disk: float,
    upsilon_bulge: float,
    v_core: float,
    r_core: float,
) -> np.ndarray:
    v_b = compute_v_baryon(v_gas, v_disk, v_bulge, upsilon_disk, upsilon_bulge)
    v2_halo = v2_burkert_halo_only(r, v_core, r_core)
    return np.sqrt(np.maximum(v_b**2 + v2_halo, 0.0))


def v_pseudo_isothermal_total(
    r: np.ndarray,
    v_gas: np.ndarray,
    v_disk: np.ndarray,
    v_bulge: np.ndarray,
    upsilon_disk: float,
    upsilon_bulge: float,
    v_inf: float,
    r_core: float,
) -> np.ndarray:
    v_b = compute_v_baryon(v_gas, v_disk, v_bulge, upsilon_disk, upsilon_bulge)
    v2_halo = v2_pseudo_isothermal_halo_only(r, v_inf, r_core)
    return np.sqrt(np.maximum(v_b**2 + v2_halo, 0.0))


def _metrics(
    v_obs: np.ndarray,
    v_pred: np.ndarray,
    v_err: np.ndarray,
    n_params: int,
) -> tuple[float, float, float, float, float]:
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


def fit_model_cored_baseline(
    r: np.ndarray,
    v_obs: np.ndarray,
    v_err: np.ndarray,
    v_gas: np.ndarray,
    v_disk: np.ndarray,
    v_bulge: np.ndarray,
    has_bulge: bool,
    model: ModelName,
    ml_prior: MLPriorConfig = ML_STANDARD,
) -> ModelFitResult:
    """Fit one rotation model with shared M/L priors."""
    v_err = np.maximum(v_err, V_ERR_FLOOR)
    d_lo, d_hi = ml_prior.disk_bounds
    b_lo, b_hi = ml_prior.bulge_bounds

    def get_upsilon(p_ml: np.ndarray) -> tuple[float, float]:
        if has_bulge:
            return float(p_ml[0]), float(p_ml[1])
        return float(p_ml[0]), 0.0

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

    def ml_extra(p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return p[:n_ml], p[n_ml:]

    v_scale = float(np.sqrt(max(np.mean(v_obs**2), 1.0)))
    r_scale = max(float(r[-1]) / 3.0, 0.5)

    if model == "baryon_only":

        def predict(p: np.ndarray) -> np.ndarray:
            ml_p, _ = ml_extra(p)
            ud, ub = get_upsilon(ml_p)
            return compute_v_baryon(v_gas, v_disk, v_bulge, ud, ub)

        extra_lb = np.array([])
        extra_ub = np.array([])
        extra_x0 = np.array([])
        n_extra = 0
    elif model == "corrected_mond":

        def predict(p: np.ndarray) -> np.ndarray:
            ml_p, _ = ml_extra(p)
            ud, ub = get_upsilon(ml_p)
            return v_mond_analytic(r, v_gas, v_disk, v_bulge, ud, ub, A0_MOND_DEFAULT)

        extra_lb = np.array([])
        extra_ub = np.array([])
        extra_x0 = np.array([])
        n_extra = 0
    elif model == "nfw":

        def predict(p: np.ndarray) -> np.ndarray:
            ml_p, extra = ml_extra(p)
            ud, ub = get_upsilon(ml_p)
            v200, rs = extra[0], extra[1]
            return v_nfw_total(r, v_gas, v_disk, v_bulge, ud, ub, v200, rs)

        extra_lb = np.array([1.0, 0.1])
        extra_ub = np.array([500.0, 100.0])
        extra_x0 = np.array([v_scale, r_scale])
        n_extra = 2
    elif model == "burkert":

        def predict(p: np.ndarray) -> np.ndarray:
            ml_p, extra = ml_extra(p)
            ud, ub = get_upsilon(ml_p)
            v_core, r_core = extra[0], extra[1]
            return v_burkert_total(r, v_gas, v_disk, v_bulge, ud, ub, v_core, r_core)

        extra_lb = np.array([1.0, 0.1])
        extra_ub = np.array([500.0, 100.0])
        extra_x0 = np.array([v_scale, r_scale])
        n_extra = 2
    elif model == "pseudo_isothermal":

        def predict(p: np.ndarray) -> np.ndarray:
            ml_p, extra = ml_extra(p)
            ud, ub = get_upsilon(ml_p)
            v_inf, r_core = extra[0], extra[1]
            return v_pseudo_isothermal_total(r, v_gas, v_disk, v_bulge, ud, ub, v_inf, r_core)

        extra_lb = np.array([1.0, 0.1])
        extra_ub = np.array([500.0, 100.0])
        extra_x0 = np.array([v_scale, r_scale])
        n_extra = 2
    elif model == "tdf_kessence":

        def predict(p: np.ndarray) -> np.ndarray:
            ml_p, extra = ml_extra(p)
            ud, ub = get_upsilon(ml_p)
            beta = extra[0]
            return v_tdf_kessence_disk_proxy(
                r, v_gas, v_disk, v_bulge, ud, ub, beta, A0_TDF_DEFAULT, "deep_mond",
            )

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
        return (predict(p) - v_obs) / v_err

    try:
        res = least_squares(residuals, x0, bounds=(lb, ub), max_nfev=4000)
        p_opt = res.x
        success = bool(res.success and res.cost < 1e12)
        reason = "" if success else str(res.message)
    except Exception as exc:  # noqa: BLE001
        p_opt = x0
        success = False
        reason = str(exc)

    ml_p, extra = ml_extra(p_opt)
    ud, ub = get_upsilon(ml_p)
    v_pred = predict(p_opt)
    if not np.all(np.isfinite(v_pred)):
        success = False
        reason = reason or "non-finite velocities"
        v_pred = np.where(np.isfinite(v_pred), v_pred, v_obs)

    c2, chi2_red, rmse, aic_v, bic_v = _metrics(v_obs, v_pred, v_err, n_par)

    params: dict[str, float] = {
        "upsilon_disk": ud,
        "upsilon_bulge": ub if has_bulge else np.nan,
    }
    if model == "nfw":
        params["v200"] = float(extra[0])
        params["r_s"] = float(extra[1])
    if model == "burkert":
        params["v_core"] = float(extra[0])
        params["r_core"] = float(extra[1])
    if model == "pseudo_isothermal":
        params["v_inf"] = float(extra[0])
        params["r_core"] = float(extra[1])
    if model == "corrected_mond":
        params["a0"] = A0_MOND_DEFAULT
    if model == "tdf_kessence":
        params["beta_over_M"] = float(extra[0])
        params["a0"] = A0_TDF_DEFAULT

    return ModelFitResult(
        galaxy_id="",
        model=model,  # type: ignore[assignment]
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
    )


def fit_galaxy_cored_baseline(
    galaxy_id: str,
    gdf: pd.DataFrame,
    *,
    min_points: int = 5,
    ml_prior: MLPriorConfig = ML_STANDARD,
    include_pseudo_isothermal: bool = True,
) -> tuple[list[ModelFitResult], dict[str, Any] | None]:
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

    models: list[ModelName] = [
        "baryon_only",
        "corrected_mond",
        "nfw",
        "burkert",
    ]
    if include_pseudo_isothermal:
        models.append("pseudo_isothermal")
    models.append("tdf_kessence")

    results: list[ModelFitResult] = []
    for model in models:
        fr = fit_model_cored_baseline(
            r, v_obs, v_err, v_gas, v_disk, v_bulge, has_bulge, model, ml_prior,
        )
        fr.galaxy_id = galaxy_id
        results.append(fr)

    by = {r.model: r for r in results}
    bic_tdf = by["tdf_kessence"].bic
    bic_n = by["nfw"].bic
    bic_b = by["burkert"].bic
    bic_m = by["corrected_mond"].bic
    bic_p = by["pseudo_isothermal"].bic if include_pseudo_isothermal else np.inf

    cored_bics = {"burkert": bic_b}
    if include_pseudo_isothermal:
        cored_bics["pseudo_isothermal"] = bic_p
    best_cored = min(cored_bics, key=cored_bics.get)  # type: ignore[arg-type]
    bic_best_cored = cored_bics[best_cored]

    all_bics = {m: by[m].bic for m in by}
    best = min(all_bics, key=all_bics.get)  # type: ignore[arg-type]

    mond_act = check_mond_activity(
        r, v_gas, v_disk, v_bulge,
        by["corrected_mond"].params["upsilon_disk"],
        by["corrected_mond"].params.get("upsilon_bulge", 0.0) or 0.0,
    )

    competitors = [by["baryon_only"].bic, bic_m, bic_n, bic_b]
    if include_pseudo_isothermal:
        competitors.append(bic_p)

    comp: dict[str, Any] = {
        "galaxy_id": galaxy_id,
        "n_points": len(r),
        "has_bulge": has_bulge,
        "best_model_by_bic": best,
        "bic_baryon_only": by["baryon_only"].bic,
        "bic_corrected_mond": bic_m,
        "bic_nfw": bic_n,
        "bic_burkert": bic_b,
        "bic_pseudo_isothermal": bic_p if include_pseudo_isothermal else np.nan,
        "bic_tdf_kessence": bic_tdf,
        "chi2_baryon_only": by["baryon_only"].chi2,
        "chi2_corrected_mond": by["corrected_mond"].chi2,
        "chi2_nfw": by["nfw"].chi2,
        "chi2_burkert": by["burkert"].chi2,
        "chi2_pseudo_isothermal": (
            by["pseudo_isothermal"].chi2 if include_pseudo_isothermal else np.nan
        ),
        "chi2_tdf_kessence": by["tdf_kessence"].chi2,
        "reduced_chi2_baryon_only": by["baryon_only"].reduced_chi2,
        "reduced_chi2_corrected_mond": by["corrected_mond"].reduced_chi2,
        "reduced_chi2_nfw": by["nfw"].reduced_chi2,
        "reduced_chi2_burkert": by["burkert"].reduced_chi2,
        "reduced_chi2_pseudo_isothermal": (
            by["pseudo_isothermal"].reduced_chi2 if include_pseudo_isothermal else np.nan
        ),
        "reduced_chi2_tdf_kessence": by["tdf_kessence"].reduced_chi2,
        "delta_bic_tdf_vs_nfw": bic_tdf - bic_n,
        "delta_bic_tdf_vs_burkert": bic_tdf - bic_b,
        "delta_bic_tdf_vs_pseudo_isothermal": (
            bic_tdf - bic_p if include_pseudo_isothermal else np.nan
        ),
        "delta_bic_tdf_vs_best_cored_halo": bic_tdf - bic_best_cored,
        "best_cored_halo_model": best_cored,
        "bic_best_cored_halo": bic_best_cored,
        "tdf_beats_nfw": bic_tdf < bic_n,
        "tdf_beats_burkert": bic_tdf < bic_b,
        "tdf_beats_best_cored_halo": bic_tdf < bic_best_cored,
        "tdf_bic_competitive": (bic_tdf - min(competitors)) < BIC_COMPETITIVE_DELTA,
        "mond_active_flag": bool(mond_act["mond_active_flag"]),
    }
    return results, comp


def aggregate_model_summary(
    comparisons: list[dict[str, Any]],
    fit_rows: list[ModelFitResult],
) -> pd.DataFrame:
    comp_df = pd.DataFrame(comparisons)
    fit_df = pd.DataFrame(
        [
            {
                "model": fr.model,
                "galaxy_id": fr.galaxy_id,
                "bic": fr.bic,
                "chi2": fr.chi2,
                "reduced_chi2": fr.reduced_chi2,
            }
            for fr in fit_rows
        ],
    )
    rows: list[dict[str, Any]] = []
    for model in fit_df["model"].unique():
        sub = fit_df[fit_df["model"] == model]
        wins = int((comp_df["best_model_by_bic"] == model).sum())
        if model == "tdf_kessence":
            tdf_wins = np.nan
            med_delta = np.nan
        elif model == "nfw":
            tdf_wins = int(comp_df["tdf_beats_nfw"].sum())
            med_delta = float(comp_df["delta_bic_tdf_vs_nfw"].median())
        elif model == "burkert":
            tdf_wins = int(comp_df["tdf_beats_burkert"].sum())
            med_delta = float(comp_df["delta_bic_tdf_vs_burkert"].median())
        elif model == "pseudo_isothermal":
            tdf_wins = int((comp_df["delta_bic_tdf_vs_pseudo_isothermal"] < 0).sum())
            med_delta = float(comp_df["delta_bic_tdf_vs_pseudo_isothermal"].median())
        else:
            tdf_wins = np.nan
            med_delta = np.nan

        rows.append(
            {
                "model": model,
                "galaxies_fitted": len(comp_df),
                "bic_win_count": wins,
                "median_bic": float(sub["bic"].median()),
                "median_reduced_chi2": float(sub["reduced_chi2"].median()),
                "median_chi2": float(sub["chi2"].median()),
                "tdf_vs_model_win_count": tdf_wins,
                "tdf_vs_model_median_delta_bic": med_delta,
            },
        )
    return pd.DataFrame(rows)


@dataclass
class CoredHaloRunResult:
    model_summary: pd.DataFrame
    by_galaxy: pd.DataFrame
    fit_details: list[ModelFitResult]


def run_cored_halo_baseline(
    sparc_csv: Path,
    output_dir: Path,
    *,
    max_galaxies: int | None = None,
    quality_min_points: int = 5,
    narrow_ml_prior: bool = False,
    include_pseudo_isothermal: bool = True,
) -> CoredHaloRunResult:
    output_dir = Path(output_dir)
    tables = output_dir / "tables"
    reports = output_dir / "reports"
    figures = output_dir / "figures"
    metadata = output_dir / "metadata"
    for d in (tables, reports, figures, metadata):
        d.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(sparc_csv)
    validate_sparc_input_schema(df)

    ml_prior = ML_NARROW if narrow_ml_prior else ML_STANDARD
    galaxy_ids = sorted(df["galaxy_id"].unique())
    if max_galaxies is not None:
        galaxy_ids = galaxy_ids[: int(max_galaxies)]

    comparisons: list[dict[str, Any]] = []
    all_fits: list[ModelFitResult] = []

    for gid in galaxy_ids:
        gdf = df[df["galaxy_id"] == gid]
        if len(gdf) < quality_min_points:
            continue
        try:
            fits, comp = fit_galaxy_cored_baseline(
                gid,
                gdf,
                min_points=quality_min_points,
                ml_prior=ml_prior,
                include_pseudo_isothermal=include_pseudo_isothermal,
            )
            if comp is None:
                continue
            comparisons.append(comp)
            all_fits.extend(fits)
        except Exception:
            continue

    summary_df = aggregate_model_summary(comparisons, all_fits)
    by_galaxy_df = pd.DataFrame(comparisons)

    summary_df.to_csv(tables / "cored_halo_model_summary.csv", index=False)
    by_galaxy_df.to_csv(tables / "cored_halo_comparison_by_galaxy.csv", index=False)

    _write_figures(summary_df, by_galaxy_df, all_fits, df, figures)
    report = _build_report(summary_df, by_galaxy_df, sparc_csv, narrow_ml_prior)
    (reports / "cored_halo_baseline_report.md").write_text(report, encoding="utf-8")

    return CoredHaloRunResult(
        model_summary=summary_df,
        by_galaxy=by_galaxy_df,
        fit_details=all_fits,
    )


def _write_figures(
    summary_df: pd.DataFrame,
    by_galaxy_df: pd.DataFrame,
    all_fits: list[ModelFitResult],
    sparc_df: pd.DataFrame,
    figures_dir: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if summary_df.empty:
        return

    models = summary_df["model"].astype(str).tolist()
    wins = summary_df["bic_win_count"].to_numpy()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(len(models)), wins, color="steelblue")
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("BIC wins")
    ax.set_title("BIC wins with cored halo baselines")
    fig.tight_layout()
    fig.savefig(figures_dir / "bic_wins_with_cored_halo.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(by_galaxy_df["delta_bic_tdf_vs_burkert"].dropna(), bins=25, color="C1", alpha=0.85)
    ax.axvline(0, color="k", ls="--")
    ax.set_xlabel("ΔBIC (TDF − Burkert)")
    ax.set_ylabel("Galaxies")
    ax.set_title("TDF vs Burkert")
    fig.tight_layout()
    fig.savefig(figures_dir / "tdf_vs_burkert_delta_bic.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(models))
    med_chi2 = summary_df["median_reduced_chi2"].to_numpy()
    ax.bar(x, med_chi2, color="C2")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("Median reduced χ²")
    ax.set_title("Reduced χ² by model")
    fig.tight_layout()
    fig.savefig(figures_dir / "reduced_chi2_with_cored_halo.png", dpi=150)
    plt.close(fig)

    _plot_example_fits(sparc_df, all_fits, by_galaxy_df, figures_dir)


def _plot_example_fits(
    sparc_df: pd.DataFrame,
    all_fits: list[ModelFitResult],
    by_galaxy_df: pd.DataFrame,
    figures_dir: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if by_galaxy_df.empty:
        return

    picks: list[str] = []
    for criterion in (
        by_galaxy_df.nsmallest(1, "delta_bic_tdf_vs_burkert")["galaxy_id"],
        by_galaxy_df.nlargest(1, "delta_bic_tdf_vs_burkert")["galaxy_id"],
        by_galaxy_df[by_galaxy_df["best_model_by_bic"] == "burkert"].head(1)["galaxy_id"],
        by_galaxy_df[by_galaxy_df["best_model_by_bic"] == "tdf_kessence"].head(1)["galaxy_id"],
    ):
        gid = str(criterion.iloc[0]) if len(criterion) else ""
        if gid and gid not in picks:
            picks.append(gid)
        if len(picks) >= 4:
            break

    if not picks:
        picks = [str(by_galaxy_df["galaxy_id"].iloc[0])]

    n = len(picks)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes_flat = axes.flatten()

    plot_models = ("baryon_only", "corrected_mond", "nfw", "burkert", "tdf_kessence")
    labels = {
        "baryon_only": "baryon",
        "corrected_mond": "MOND",
        "nfw": "NFW",
        "burkert": "Burkert",
        "tdf_kessence": "TDF",
    }

    for ax, gid in zip(axes_flat, picks):
        gdf = sparc_df[sparc_df["galaxy_id"] == gid].sort_values("r_kpc")
        r = gdf["r_kpc"].to_numpy()
        ax.errorbar(
            r, gdf["v_obs"], yerr=gdf["v_err"], fmt="ko", ms=3, capsize=2, label="obs",
        )
        fits = [f for f in all_fits if f.galaxy_id == gid and f.v_pred is not None]
        for fr in fits:
            if fr.model not in plot_models:
                continue
            ax.plot(r, fr.v_pred, label=labels.get(fr.model, fr.model))
        ax.set_xlabel("r [kpc]")
        ax.set_ylabel("v [km/s]")
        ax.set_title(str(gid))
        ax.legend(fontsize=6)

    for ax in axes_flat[len(picks) :]:
        ax.axis("off")

    fig.suptitle("Example rotation fits (incl. Burkert)", fontsize=11)
    fig.tight_layout()
    fig.savefig(figures_dir / "example_rotation_fits_with_burkert.png", dpi=150)
    plt.close(fig)


def _summary_table_markdown(summary_df: pd.DataFrame) -> str:
    try:
        return summary_df.to_markdown(index=False)
    except ImportError:
        return "```\n" + summary_df.to_string(index=False) + "\n```"


def _safe_int(val: Any) -> int:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return 0
    return int(val)


def _build_report(
    summary_df: pd.DataFrame,
    by_galaxy_df: pd.DataFrame,
    sparc_csv: Path,
    narrow_ml: bool,
) -> str:
    n = len(by_galaxy_df)
    tdf_row = summary_df[summary_df["model"] == "tdf_kessence"]
    nfw_row = summary_df[summary_df["model"] == "nfw"]
    burk_row = summary_df[summary_df["model"] == "burkert"]

    tdf_wins = _safe_int(tdf_row["bic_win_count"].iloc[0]) if len(tdf_row) else 0
    nfw_wins = _safe_int(nfw_row["bic_win_count"].iloc[0]) if len(nfw_row) else 0
    burk_wins = _safe_int(burk_row["bic_win_count"].iloc[0]) if len(burk_row) else 0

    tdf_vs_nfw = int(by_galaxy_df["tdf_beats_nfw"].sum()) if n else 0
    tdf_vs_burk = int(by_galaxy_df["tdf_beats_burkert"].sum()) if n else 0
    tdf_vs_cored = int(by_galaxy_df["tdf_beats_best_cored_halo"].sum()) if n else 0
    burk_vs_nfw = int((by_galaxy_df["bic_burkert"] < by_galaxy_df["bic_nfw"]).sum()) if n else 0

    lines = [
        "# SPARC cored-halo baseline report (Step 4)",
        "",
        f"## ⚠️ {BANNER_CORED_HALO}",
        "",
        f"## ⚠️ {BANNER_CALIBRATION}",
        "",
        f"**Input:** `{sparc_csv}`",
        f"**M/L prior:** {'narrow' if narrow_ml else 'standard'} Υ_disk∈"
        f"{ML_NARROW.disk_bounds if narrow_ml else ML_STANDARD.disk_bounds}",
        "",
        "Models: baryon-only, corrected analytic MOND, NFW, Burkert, "
        "pseudo-isothermal, TDF K-essence proxy.",
        "",
        "## Model summary",
        "",
        _summary_table_markdown(summary_df),
        "",
        "## Key questions",
        "",
        f"**Does TDF remain competitive after adding a cored halo baseline?** "
        f"TDF BIC wins={tdf_wins}/{n}; TDF beats best cored halo on {tdf_vs_cored}/{n} galaxies; "
        f"beats Burkert on {tdf_vs_burk}/{n}. "
        + (
            "TDF **remains competitive** vs cored baselines under wide M/L."
            if tdf_vs_cored > n * 0.45
            else "TDF **weakens** relative to cored halos when they are included."
        ),
        "",
        f"**Does Burkert outperform NFW?** Burkert BIC wins={burk_wins} vs NFW={nfw_wins}; "
        f"Burkert lower BIC than NFW on {burk_vs_nfw}/{n} galaxies.",
        "",
        f"**Does Burkert outperform TDF?** TDF beats Burkert on {tdf_vs_burk}/{n} galaxies "
        f"(head-to-head BIC).",
        "",
        f"**Are TDF wins mostly against NFW but not cored halos?** "
        f"TDF vs NFW wins={tdf_vs_nfw}, vs best cored={tdf_vs_cored}. "
        + (
            "TDF wins are **similar** against NFW and cored halos."
            if abs(tdf_vs_nfw - tdf_vs_cored) < 0.1 * max(n, 1)
            else (
                "TDF wins are **concentrated vs NFW** more than vs cored halos."
                if tdf_vs_nfw > tdf_vs_cored + 5
                else "TDF does **better vs cored halos** than vs NFW in this sample."
            )
        ),
        "",
        "## Limitations",
        "",
        "- Phenomenological halo and TDF proxies; not a full 3D Poisson solve.",
        "- Does not validate TDF observationally or replace dark matter.",
        "",
    ]
    return "\n".join(lines)

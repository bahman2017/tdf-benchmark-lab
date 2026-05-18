"""
SPARC Step 6B — benchmark inverse-designed τ response vs baselines.

Tests candidate a_τ(r) = β_eff · √(a_b a₀) · R_core(r) from Step 6A.
Formula-revision benchmark only; not observational validation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from tdf_obs.fitting.metrics import aic, bic, chi_square, mse, reduced_chi_square
from tdf_obs.validation.sparc_cored_halo_baseline import (
    ML_STANDARD,
    MLPriorConfig,
    fit_model_cored_baseline,
)
from tdf_obs.validation.sparc_galaxy_class_analysis import (
    CLASS_ORDER,
    build_galaxy_properties,
    classify_galaxy_by_vmax,
)
from tdf_obs.validation.sparc_real_calibration import (
    A0_TDF_DEFAULT,
    BIC_COMPETITIVE_DELTA,
    ModelFitResult,
    V_ERR_FLOOR,
    _prepare_galaxy_frame,
    galaxy_has_bulge,
    validate_sparc_input_schema,
)
from tdf_obs.validation.sparc_tau_inverse_design import v2_baryon_user

BANNER_INVERSE_TAU = (
    "SPARC INVERSE-DESIGNED TAU RESPONSE BENCHMARK — NOT FULL OBSERVATIONAL VALIDATION"
)

BANNER_CALIBRATION = (
    "These results are calibration diagnostics for a phenomenological TDF model. "
    "They do not constitute observational validation of TDF unless all required "
    "multi-channel constraints are passed."
)

R_MIN_KPC = 0.02
ACCEL_EPS = 1e-6
INNER_FRAC = 0.3
OUTER_FRAC = 0.7

BaselineModel = Literal[
    "baryon_only",
    "corrected_mond",
    "nfw",
    "pseudo_isothermal",
    "old_tdf_baseline",
]
InverseVariant = Literal[
    "inverse_tdf_global_beta_core",
    "inverse_tdf_class_beta_core",
    "inverse_tdf_baryon_feature_beta",
]
ALL_MODELS: tuple[str, ...] = (
    "baryon_only",
    "corrected_mond",
    "nfw",
    "pseudo_isothermal",
    "old_tdf_baseline",
    "inverse_tdf_global_beta_core",
    "inverse_tdf_class_beta_core",
    "inverse_tdf_baryon_feature_beta",
)
INVERSE_VARIANTS: tuple[InverseVariant, ...] = (
    "inverse_tdf_global_beta_core",
    "inverse_tdf_class_beta_core",
    "inverse_tdf_baryon_feature_beta",
)

SUMMARY_COLUMNS: tuple[str, ...] = (
    "model",
    "galaxies_fitted",
    "bic_win_count",
    "median_bic",
    "median_reduced_chi2",
    "median_chi2",
    "median_n_params",
    "best_inverse_vs_old_tdf_win_count",
    "best_inverse_vs_old_tdf_median_delta_bic",
    "best_inverse_vs_nfw_median_delta_bic",
)

COMPARISON_COLUMNS: tuple[str, ...] = (
    "galaxy_id",
    "galaxy_class",
    "n_points",
    "has_bulge",
    "best_model_by_bic",
    "best_inverse_variant",
    "bic_baryon_only",
    "bic_corrected_mond",
    "bic_nfw",
    "bic_pseudo_isothermal",
    "bic_old_tdf_baseline",
    "bic_inverse_tdf_global_beta_core",
    "bic_inverse_tdf_class_beta_core",
    "bic_inverse_tdf_baryon_feature_beta",
    "bic_best_inverse",
    "delta_bic_best_inverse_vs_old_tdf",
    "delta_bic_best_inverse_vs_nfw",
    "delta_bic_best_inverse_vs_corrected_mond",
    "delta_bic_best_inverse_vs_pseudo_isothermal",
    "best_inverse_beats_old_tdf",
    "best_inverse_beats_nfw",
    "best_inverse_beats_pseudo_isothermal",
    "best_inverse_bic_competitive",
    "mond_active_flag",
)

PARAM_SUMMARY_COLUMNS: tuple[str, ...] = (
    "model",
    "per_galaxy_n_params",
    "cohort_shared_n_params",
    "total_params_description",
)

BOUNDARY_COLUMNS: tuple[str, ...] = (
    "galaxy_id",
    "model",
    "any_boundary_hit",
    "upsilon_disk_at_bound",
    "upsilon_bulge_at_bound",
    "r_c_at_bound",
    "beta_at_bound",
    "beta_min_at_bound",
    "beta_max_at_bound",
    "p_at_bound",
)


@dataclass
class CohortPriors:
    global_beta: float
    global_rc_kpc: float
    class_beta: dict[str, float]


@dataclass
class InverseTauRunResult:
    model_summary: pd.DataFrame
    comparison_by_galaxy: pd.DataFrame
    parameter_summary: pd.DataFrame
    boundary_flags: pd.DataFrame
    fit_details: list[ModelFitResult] = field(default_factory=list)


def R_core(r: np.ndarray, r_c: float) -> np.ndarray:
    """R_core(r) = r / sqrt(r² + r_c²); finite at r → 0."""
    r = np.asarray(r, dtype=float)
    rc = max(float(r_c), R_MIN_KPC)
    return r / np.sqrt(r**2 + rc**2)


def beta_eff_power(a_b: np.ndarray, beta0: float, p: float, a0: float) -> np.ndarray:
    a_b = np.maximum(np.asarray(a_b, dtype=float), ACCEL_EPS)
    return float(beta0) * (float(a0) / a_b) ** float(p)


def beta_eff_saturation(
    a_b: np.ndarray,
    beta_min: float,
    beta_max: float,
    p: float,
    a0: float,
) -> np.ndarray:
    a_b = np.maximum(np.asarray(a_b, dtype=float), ACCEL_EPS)
    x = (a_b / float(a0)) ** float(p)
    return float(beta_min) + (float(beta_max) - float(beta_min)) / (1.0 + x)


def a_tau_inverse(
    r: np.ndarray,
    v2_baryon: np.ndarray,
    beta_eff: np.ndarray | float,
    a0: float,
    r_c: float,
) -> np.ndarray:
    """a_τ = β_eff · √(a_b a₀) · R_core(r); nonnegative."""
    r_safe = np.maximum(np.asarray(r, dtype=float), R_MIN_KPC)
    v2_b = np.maximum(np.asarray(v2_baryon, dtype=float), 0.0)
    a_b = v2_b / r_safe
    be = np.asarray(beta_eff, dtype=float)
    if be.ndim == 0:
        be = np.full_like(a_b, float(be))
    core = R_core(r_safe, r_c)
    a_tau = be * np.sqrt(np.maximum(a_b * float(a0), 0.0)) * core
    return np.maximum(a_tau, 0.0)


def v_inverse_tau(
    r: np.ndarray,
    v_gas: np.ndarray,
    v_disk: np.ndarray,
    v_bulge: np.ndarray,
    upsilon_disk: float,
    upsilon_bulge: float,
    beta_eff: np.ndarray | float,
    a0: float,
    r_c: float,
) -> np.ndarray:
    """v² = v_baryon² + r · a_τ."""
    v2_b = v2_baryon_user(v_gas, v_disk, v_bulge, upsilon_disk, upsilon_bulge)
    a_tau = a_tau_inverse(r, v2_b, beta_eff, a0, r_c)
    v2 = v2_b + np.maximum(r, R_MIN_KPC) * a_tau
    return np.sqrt(np.maximum(v2, 0.0))


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


def _param_at_bound(val: float, lo: float, hi: float, tol: float = 0.02) -> bool:
    if not np.isfinite(val):
        return False
    return float(val) <= lo + tol * abs(hi - lo) or float(val) >= hi - tol * abs(hi - lo)


def _zone_residuals(
    r: np.ndarray,
    v_obs: np.ndarray,
    v_err: np.ndarray,
    v_pred: np.ndarray,
) -> dict[str, float]:
    rmax = max(float(np.max(r)), R_MIN_KPC)
    w = (v_obs - v_pred) / np.maximum(v_err, V_ERR_FLOOR)
    zones = {
        "inner": r < INNER_FRAC * rmax,
        "middle": (r >= INNER_FRAC * rmax) & (r <= OUTER_FRAC * rmax),
        "outer": r > OUTER_FRAC * rmax,
    }
    out: dict[str, float] = {}
    for name, mask in zones.items():
        if mask.any():
            out[f"median_abs_residual_{name}"] = float(np.median(np.abs(w[mask])))
        else:
            out[f"median_abs_residual_{name}"] = float("nan")
    return out


def load_cohort_priors(halo_summary: pd.DataFrame) -> CohortPriors:
    global_beta = float(halo_summary["median_beta_eff"].median())
    rc = halo_summary["core_r_c_soft"].replace(500.0, np.nan)
    global_rc = float(np.nanmedian(rc)) if rc.notna().any() else 2.0
    global_rc = float(np.clip(global_rc, 0.1, 25.0))
    class_beta: dict[str, float] = {}
    for cls in CLASS_ORDER:
        sub = halo_summary.loc[halo_summary["galaxy_class"] == cls, "median_beta_eff"]
        class_beta[cls] = float(sub.median()) if len(sub) else global_beta
    return CohortPriors(global_beta=global_beta, global_rc_kpc=global_rc, class_beta=class_beta)


def n_params_for_model(model: str, has_bulge: bool) -> tuple[int, int]:
    """Return (per_galaxy_n_params, cohort_shared_n_params)."""
    n_ml = 2 if has_bulge else 1
    if model in ("baryon_only", "corrected_mond"):
        return n_ml, 0
    if model == "nfw":
        return n_ml + 2, 0
    if model == "pseudo_isothermal":
        return n_ml + 2, 0
    if model == "old_tdf_baseline":
        return n_ml + 1, 0
    if model == "inverse_tdf_global_beta_core":
        return n_ml + 1, 1
    if model == "inverse_tdf_class_beta_core":
        return n_ml + 1, 3
    if model == "inverse_tdf_baryon_feature_beta":
        return n_ml + 4, 0
    return n_ml, 0


def fit_inverse_tau_variant(
    r: np.ndarray,
    v_obs: np.ndarray,
    v_err: np.ndarray,
    v_gas: np.ndarray,
    v_disk: np.ndarray,
    v_bulge: np.ndarray,
    has_bulge: bool,
    variant: InverseVariant,
    priors: CohortPriors,
    galaxy_class: str,
    ml_prior: MLPriorConfig = ML_STANDARD,
) -> ModelFitResult:
    """Fit one inverse-designed τ variant for a single galaxy."""
    v_err = np.maximum(v_err, V_ERR_FLOOR)
    d_lo, d_hi = ml_prior.disk_bounds
    b_lo, b_hi = ml_prior.bulge_bounds
    rmax = max(float(r[-1]), 0.5)
    rc_hi = min(50.0, max(rmax * 2.0, 1.0))

    if has_bulge:
        ml_lb = np.array([d_lo, b_lo])
        ml_ub = np.array([d_hi, b_hi])
        ml_x0 = np.array([0.5, 0.7])
        n_ml = 2
    else:
        ml_lb = np.array([d_lo])
        ml_ub = np.array([d_hi])
        ml_x0 = np.array([0.5])
        n_ml = 1

    def get_ml(p: np.ndarray) -> tuple[float, float]:
        if has_bulge:
            return float(p[0]), float(p[1])
        return float(p[0]), 0.0

    a0 = A0_TDF_DEFAULT

    if variant == "inverse_tdf_global_beta_core":
        beta_fixed = priors.global_beta

        def predict(p: np.ndarray) -> np.ndarray:
            ud, ub = get_ml(p)
            rc = float(p[n_ml])
            v2_b = v2_baryon_user(v_gas, v_disk, v_bulge, ud, ub)
            return v_inverse_tau(r, v_gas, v_disk, v_bulge, ud, ub, beta_fixed, a0, rc)

        extra_lb = np.array([0.05])
        extra_ub = np.array([rc_hi])
        extra_x0 = np.array([priors.global_rc_kpc])
        n_extra = 1
        n_par = n_ml + 1
        cohort_shared = 1
    elif variant == "inverse_tdf_class_beta_core":
        beta_fixed = priors.class_beta.get(galaxy_class, priors.global_beta)

        def predict(p: np.ndarray) -> np.ndarray:
            ud, ub = get_ml(p)
            rc = float(p[n_ml])
            return v_inverse_tau(r, v_gas, v_disk, v_bulge, ud, ub, beta_fixed, a0, rc)

        extra_lb = np.array([0.05])
        extra_ub = np.array([rc_hi])
        extra_x0 = np.array([priors.global_rc_kpc])
        n_extra = 1
        n_par = n_ml + 1
        cohort_shared = 3
    else:
        def predict(p: np.ndarray) -> np.ndarray:
            ud, ub = get_ml(p)
            bmin, bmax, pwr, rc = p[n_ml], p[n_ml + 1], p[n_ml + 2], p[n_ml + 3]
            v2_b = v2_baryon_user(v_gas, v_disk, v_bulge, ud, ub)
            a_b = v2_b / np.maximum(r, R_MIN_KPC)
            be = beta_eff_saturation(a_b, bmin, bmax, pwr, a0)
            a_tau = a_tau_inverse(r, v2_b, be, a0, rc)
            v2 = v2_b + np.maximum(r, R_MIN_KPC) * a_tau
            return np.sqrt(np.maximum(v2, 0.0))

        extra_lb = np.array([0.01, 0.05, 0.0, 0.05])
        extra_ub = np.array([2.0, 5.0, 3.0, rc_hi])
        extra_x0 = np.array([0.1, priors.global_beta, 1.0, priors.global_rc_kpc])
        n_extra = 4
        n_par = n_ml + 4
        cohort_shared = 0

    lb = np.concatenate([ml_lb, extra_lb])
    ub = np.concatenate([ml_ub, extra_ub])
    x0 = np.clip(np.concatenate([ml_x0, extra_x0]), lb, ub)

    def residuals(p: np.ndarray) -> np.ndarray:
        pred = predict(p)
        if not np.all(np.isfinite(pred)):
            return np.full_like(v_obs, 1e6)
        return (pred - v_obs) / v_err

    try:
        res = least_squares(residuals, x0, bounds=(lb, ub), max_nfev=5000)
        p_opt = res.x
        success = bool(res.success and res.cost < 1e12)
        reason = "" if success else str(res.message)
    except Exception as exc:  # noqa: BLE001
        p_opt = x0
        success = False
        reason = str(exc)

    ud, ub = get_ml(p_opt)
    v_pred = predict(p_opt)
    if not np.all(np.isfinite(v_pred)):
        success = False
        reason = reason or "non-finite velocities"
        v_pred = np.where(np.isfinite(v_pred), v_pred, v_obs)

    bic_n_par = n_par + cohort_shared
    c2, chi2_red, rmse, aic_v, bic_v = _metrics(v_obs, v_pred, v_err, bic_n_par)

    params: dict[str, float] = {
        "upsilon_disk": ud,
        "upsilon_bulge": ub if has_bulge else np.nan,
        "a0": a0,
        "per_galaxy_n_params": float(n_par),
        "cohort_shared_n_params": float(cohort_shared),
    }
    if variant == "inverse_tdf_global_beta_core":
        params["beta_eff"] = priors.global_beta
        params["r_c"] = float(p_opt[n_ml])
    elif variant == "inverse_tdf_class_beta_core":
        params["beta_eff"] = beta_fixed
        params["r_c"] = float(p_opt[n_ml])
    else:
        params["beta_min"] = float(p_opt[n_ml])
        params["beta_max"] = float(p_opt[n_ml + 1])
        params["p"] = float(p_opt[n_ml + 2])
        params["r_c"] = float(p_opt[n_ml + 3])

    return ModelFitResult(
        galaxy_id="",
        model=variant,  # type: ignore[assignment]
        n_points=len(r),
        n_params=bic_n_par,
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


def boundary_flags_for_fit(
    galaxy_id: str,
    fr: ModelFitResult,
    ml_prior: MLPriorConfig,
) -> dict[str, Any]:
    d_lo, d_hi = ml_prior.disk_bounds
    b_lo, b_hi = ml_prior.bulge_bounds
    p = fr.params
    ud_at = _param_at_bound(p.get("upsilon_disk", np.nan), d_lo, d_hi)
    ub_at = _param_at_bound(p.get("upsilon_bulge", np.nan), b_lo, b_hi)
    rc_at = _param_at_bound(p.get("r_c", np.nan), 0.05, 50.0)
    beta_at = False
    if fr.model == "old_tdf_baseline":
        beta_at = _param_at_bound(p.get("beta_over_M", np.nan), 1e-4, 50.0)
    hits = [ud_at, ub_at, rc_at, beta_at]
    return {
        "galaxy_id": galaxy_id,
        "model": fr.model,
        "any_boundary_hit": any(hits),
        "upsilon_disk_at_bound": ud_at,
        "upsilon_bulge_at_bound": ub_at,
        "r_c_at_bound": rc_at,
        "beta_at_bound": beta_at,
        "beta_min_at_bound": _param_at_bound(p.get("beta_min", np.nan), 0.01, 2.0),
        "beta_max_at_bound": _param_at_bound(p.get("beta_max", np.nan), 0.05, 5.0),
        "p_at_bound": _param_at_bound(p.get("p", np.nan), 0.0, 3.0),
    }


def fit_galaxy_all_models_6b(
    galaxy_id: str,
    gdf: pd.DataFrame,
    priors: CohortPriors,
    galaxy_class: str,
    *,
    min_points: int = 5,
    ml_prior: MLPriorConfig = ML_STANDARD,
) -> tuple[list[ModelFitResult], dict[str, Any]]:
    gdf = _prepare_galaxy_frame(gdf, min_points)
    if gdf is None:
        return [], {}

    r = gdf["r_kpc"].to_numpy()
    v_obs = gdf["v_obs"].to_numpy()
    v_err = gdf["v_err"].to_numpy()
    v_gas = gdf["v_gas"].to_numpy()
    v_disk = gdf["v_disk"].to_numpy()
    v_bulge = gdf["v_bulge"].to_numpy()
    has_bulge = galaxy_has_bulge(v_bulge)

    results: list[ModelFitResult] = []
    for base in ("baryon_only", "corrected_mond", "nfw", "pseudo_isothermal"):
        fr = fit_model_cored_baseline(
            r, v_obs, v_err, v_gas, v_disk, v_bulge, has_bulge, base, ml_prior,  # type: ignore[arg-type]
        )
        fr.galaxy_id = galaxy_id
        results.append(fr)

    fr_old = fit_model_cored_baseline(
        r, v_obs, v_err, v_gas, v_disk, v_bulge, has_bulge, "tdf_kessence", ml_prior,
    )
    fr_old.galaxy_id = galaxy_id
    fr_old.model = "old_tdf_baseline"  # type: ignore[assignment]
    results.append(fr_old)

    for variant in INVERSE_VARIANTS:
        fr_inv = fit_inverse_tau_variant(
            r, v_obs, v_err, v_gas, v_disk, v_bulge, has_bulge,
            variant, priors, galaxy_class, ml_prior,
        )
        fr_inv.galaxy_id = galaxy_id
        results.append(fr_inv)

    by = {r.model: r for r in results}
    inv_bics = {v: by[v].bic for v in INVERSE_VARIANTS}
    best_inv = min(inv_bics, key=inv_bics.get)  # type: ignore[arg-type]
    bic_inv_best = inv_bics[best_inv]

    all_bics = {m: by[m].bic for m in ALL_MODELS if m in by}
    best_all = min(all_bics, key=all_bics.get)  # type: ignore[arg-type]

    bic_old = by["old_tdf_baseline"].bic
    bic_nfw = by["nfw"].bic
    bic_mond = by["corrected_mond"].bic
    bic_pseudo = by["pseudo_isothermal"].bic

    comp: dict[str, Any] = {
        "galaxy_id": galaxy_id,
        "galaxy_class": galaxy_class,
        "n_points": len(r),
        "has_bulge": has_bulge,
        "best_model_by_bic": best_all,
        "best_inverse_variant": best_inv,
        "bic_baryon_only": by["baryon_only"].bic,
        "bic_corrected_mond": bic_mond,
        "bic_nfw": bic_nfw,
        "bic_pseudo_isothermal": bic_pseudo,
        "bic_old_tdf_baseline": bic_old,
        "bic_inverse_tdf_global_beta_core": by["inverse_tdf_global_beta_core"].bic,
        "bic_inverse_tdf_class_beta_core": by["inverse_tdf_class_beta_core"].bic,
        "bic_inverse_tdf_baryon_feature_beta": by["inverse_tdf_baryon_feature_beta"].bic,
        "bic_best_inverse": bic_inv_best,
        "delta_bic_best_inverse_vs_old_tdf": bic_inv_best - bic_old,
        "delta_bic_best_inverse_vs_nfw": bic_inv_best - bic_nfw,
        "delta_bic_best_inverse_vs_corrected_mond": bic_inv_best - bic_mond,
        "delta_bic_best_inverse_vs_pseudo_isothermal": bic_inv_best - bic_pseudo,
        "best_inverse_beats_old_tdf": bic_inv_best < bic_old,
        "best_inverse_beats_nfw": bic_inv_best < bic_nfw,
        "best_inverse_beats_pseudo_isothermal": bic_inv_best < bic_pseudo,
        "best_inverse_bic_competitive": (
            bic_inv_best - min(bic_old, bic_nfw, bic_mond, bic_pseudo)
        ) < BIC_COMPETITIVE_DELTA,
        "mond_active_flag": False,
    }
    return results, comp


def build_model_summary(
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
                "n_params": fr.n_params,
            }
            for fr in fit_rows
        ],
    )
    best_inv_delta_old = comp_df["delta_bic_best_inverse_vs_old_tdf"]
    rows: list[dict[str, Any]] = []
    for model in ALL_MODELS:
        sub = fit_df[fit_df["model"] == model]
        if sub.empty:
            continue
        wins = int((comp_df["best_model_by_bic"] == model).sum())
        if model in INVERSE_VARIANTS:
            inv_wins = int((comp_df["best_inverse_variant"] == model).sum())
            med_d_old = float("nan")
        elif model == "old_tdf_baseline":
            inv_wins = int(comp_df["best_inverse_beats_old_tdf"].sum())
            med_d_old = float(best_inv_delta_old.median())
        else:
            inv_wins = np.nan
            med_d_old = np.nan

        rows.append(
            {
                "model": model,
                "galaxies_fitted": len(comp_df),
                "bic_win_count": wins,
                "median_bic": float(sub["bic"].median()),
                "median_reduced_chi2": float(sub["reduced_chi2"].median()),
                "median_chi2": float(sub["chi2"].median()),
                "median_n_params": float(sub["n_params"].median()),
                "best_inverse_vs_old_tdf_win_count": inv_wins,
                "best_inverse_vs_old_tdf_median_delta_bic": med_d_old,
                "best_inverse_vs_nfw_median_delta_bic": (
                    float(comp_df["delta_bic_best_inverse_vs_nfw"].median())
                    if model in INVERSE_VARIANTS or model == "old_tdf_baseline"
                    else float("nan")
                ),
            },
        )
    return pd.DataFrame(rows)


def build_parameter_summary() -> pd.DataFrame:
    rows = []
    for model in ALL_MODELS:
        pg, cohort = n_params_for_model(model, has_bulge=True)
        pg0, _ = n_params_for_model(model, has_bulge=False)
        per_g = max(pg, pg0)
        desc = {
            "baryon_only": "Υ_disk (+Υ_bulge)",
            "corrected_mond": "Υ only; a₀ fixed",
            "nfw": "Υ + v200 + r_s",
            "pseudo_isothermal": "Υ + v_inf + r_core",
            "old_tdf_baseline": "Υ + β/M (k-essence disk proxy)",
            "inverse_tdf_global_beta_core": "Υ + r_c per galaxy; β global from 6A",
            "inverse_tdf_class_beta_core": "Υ + r_c; β per class (3 shared from 6A)",
            "inverse_tdf_baryon_feature_beta": "Υ + β_min, β_max, p, r_c",
        }.get(model, "")
        rows.append(
            {
                "model": model,
                "per_galaxy_n_params": per_g,
                "cohort_shared_n_params": cohort,
                "total_params_description": desc,
            },
        )
    return pd.DataFrame(rows)


def run_inverse_tau_benchmark(
    sparc_csv: Path,
    output_dir: Path,
    *,
    inverse_design_run: Path,
    input_run: Path | None = None,
    max_galaxies: int | None = None,
    quality_min_points: int = 5,
) -> InverseTauRunResult:
    output_dir = Path(output_dir)
    inverse_design_run = Path(inverse_design_run)
    tables = output_dir / "tables"
    reports = output_dir / "reports"
    figures = output_dir / "figures"
    for d in (tables, reports, figures):
        d.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(sparc_csv)
    validate_sparc_input_schema(df)
    halo_path = inverse_design_run / "tables" / "tau_effective_halo_summary.csv"
    if not halo_path.is_file():
        raise FileNotFoundError(f"Missing Step 6A halo summary: {halo_path}")
    halo_summary = pd.read_csv(halo_path)
    priors = load_cohort_priors(halo_summary)
    props_df = build_galaxy_properties(df)
    props_map = props_df.set_index("galaxy_id").to_dict("index")

    galaxy_ids = sorted(df["galaxy_id"].unique())
    if max_galaxies is not None:
        galaxy_ids = galaxy_ids[:max_galaxies]

    comparisons: list[dict[str, Any]] = []
    all_fits: list[ModelFitResult] = []
    boundary_rows: list[dict[str, Any]] = []
    zone_rows: list[dict[str, Any]] = []

    for gid in galaxy_ids:
        gdf = df[df["galaxy_id"] == gid]
        props = props_map.get(gid, {})
        gclass = str(props.get("galaxy_class", classify_galaxy_by_vmax(float(props.get("vmax_obs", 0)))))
        fits, comp = fit_galaxy_all_models_6b(
            str(gid), gdf, priors, gclass, min_points=quality_min_points,
        )
        if not comp:
            continue
        comparisons.append(comp)
        all_fits.extend(fits)
        gprep = _prepare_galaxy_frame(gdf, quality_min_points)
        if gprep is not None:
            r = gprep["r_kpc"].to_numpy()
            v_obs = gprep["v_obs"].to_numpy()
            v_err = gprep["v_err"].to_numpy()
            for fr in fits:
                boundary_rows.append(boundary_flags_for_fit(str(gid), fr, ML_STANDARD))
                if fr.v_pred is not None:
                    zone_rows.append(
                        {
                            "galaxy_id": gid,
                            "model": fr.model,
                            **_zone_residuals(r, v_obs, v_err, fr.v_pred),
                        },
                    )

    model_summary = build_model_summary(comparisons, all_fits)
    if zone_rows:
        zone_df = pd.DataFrame(zone_rows)
        for col in ("median_abs_residual_inner", "median_abs_residual_middle", "median_abs_residual_outer"):
            if col in zone_df.columns:
                med_by_model = zone_df.groupby("model")[col].median()
                model_summary[col] = model_summary["model"].map(med_by_model)
    comparison_df = pd.DataFrame(comparisons)
    param_summary = build_parameter_summary()
    boundary_df = pd.DataFrame(boundary_rows)

    model_summary.to_csv(tables / "inverse_tau_model_summary.csv", index=False)
    comparison_df.to_csv(tables / "inverse_tau_comparison_by_galaxy.csv", index=False)
    param_summary.to_csv(tables / "inverse_tau_parameter_summary.csv", index=False)
    boundary_df.to_csv(tables / "inverse_tau_boundary_flags.csv", index=False)

    _write_figures(comparison_df, all_fits, df, priors, figures)
    report = _build_report(
        comparison_df=comparison_df,
        model_summary=model_summary,
        param_summary=param_summary,
        boundary_df=boundary_df,
        priors=priors,
        sparc_csv=sparc_csv,
        inverse_design_run=inverse_design_run,
        input_run=input_run,
    )
    (reports / "inverse_tau_response_report.md").write_text(report, encoding="utf-8")

    return InverseTauRunResult(
        model_summary=model_summary,
        comparison_by_galaxy=comparison_df,
        parameter_summary=param_summary,
        boundary_flags=boundary_df,
        fit_details=all_fits,
    )


def _write_figures(
    comparison_df: pd.DataFrame,
    all_fits: list[ModelFitResult],
    sparc_df: pd.DataFrame,
    priors: CohortPriors,
    figures_dir: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if comparison_df.empty:
        return

    inv_models = list(INVERSE_VARIANTS) + ["old_tdf_baseline"]
    wins = [int((comparison_df["best_model_by_bic"] == m).sum()) for m in inv_models]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(len(wins)), wins, color="steelblue")
    ax.set_xticks(range(len(wins)))
    ax.set_xticklabels([m.replace("inverse_tdf_", "").replace("_", "\n") for m in inv_models], fontsize=7)
    ax.set_ylabel("BIC wins")
    ax.set_title("BIC wins — inverse τ variants vs old TDF")
    fig.tight_layout()
    fig.savefig(figures_dir / "bic_wins_inverse_tau.png", dpi=150)
    plt.close(fig)

    for col, fname, title in [
        ("delta_bic_best_inverse_vs_old_tdf", "inverse_tau_vs_old_tdf_delta_bic.png", "Best inverse τ − old TDF"),
        ("delta_bic_best_inverse_vs_nfw", "inverse_tau_vs_nfw_delta_bic.png", "Best inverse τ − NFW"),
    ]:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(comparison_df[col].dropna(), bins=25, color="C1", alpha=0.85)
        ax.axvline(0, color="k", ls="--")
        ax.set_xlabel("ΔBIC")
        ax.set_ylabel("Galaxies")
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(figures_dir / fname, dpi=150)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    for cls in CLASS_ORDER:
        sub = comparison_df[comparison_df["galaxy_class"] == cls]
        ax.scatter(
            sub["delta_bic_best_inverse_vs_old_tdf"],
            sub["delta_bic_best_inverse_vs_nfw"],
            alpha=0.5,
            s=20,
            label=cls,
        )
    ax.axhline(0, color="k", lw=0.5)
    ax.axvline(0, color="k", lw=0.5)
    ax.set_xlabel("ΔBIC vs old TDF")
    ax.set_ylabel("ΔBIC vs NFW")
    ax.legend(fontsize=8)
    ax.set_title("Best inverse τ by galaxy class")
    fig.tight_layout()
    fig.savefig(figures_dir / "inverse_tau_by_galaxy_class.png", dpi=150)
    plt.close(fig)

    rc_vals = []
    for fr in all_fits:
        if fr.model in INVERSE_VARIANTS and "r_c" in fr.params:
            rc_vals.append(fr.params["r_c"])
    if rc_vals:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(rc_vals, bins=25, color="C2", alpha=0.85)
        ax.axvline(priors.global_rc_kpc, color="crimson", ls="--", label=f"6A median rc={priors.global_rc_kpc:.2f}")
        ax.set_xlabel("fitted r_c [kpc]")
        ax.set_ylabel("Galaxies")
        ax.legend()
        ax.set_title("Core radius r_c (inverse τ variants)")
        fig.tight_layout()
        fig.savefig(figures_dir / "core_radius_distribution.png", dpi=150)
        plt.close(fig)

    sample = comparison_df.nsmallest(3, "delta_bic_best_inverse_vs_old_tdf")["galaxy_id"].tolist()
    fig, axes = plt.subplots(1, min(3, len(sample)), figsize=(12, 4))
    if len(sample) == 1:
        axes = [axes]
    for ax, gid in zip(axes, sample):
        gdf = sparc_df[sparc_df["galaxy_id"] == gid].sort_values("r_kpc")
        r = gdf["r_kpc"].to_numpy()
        v_obs = gdf["v_obs"].to_numpy()
        v_err = gdf["v_err"].to_numpy()
        ax.errorbar(r, v_obs, yerr=v_err, fmt="o", ms=3, label="obs")
        for fr in all_fits:
            if fr.galaxy_id == gid and fr.v_pred is not None and fr.model in (
                "old_tdf_baseline",
                comparison_df.loc[comparison_df["galaxy_id"] == gid, "best_inverse_variant"].iloc[0],
            ):
                ax.plot(r, fr.v_pred, "-", label=fr.model.replace("inverse_tdf_", "")[:12])
        ax.set_title(str(gid), fontsize=8)
        ax.set_xlabel("r [kpc]")
        ax.legend(fontsize=6)
    fig.suptitle("Example rotation fits (old TDF vs best inverse τ)")
    fig.tight_layout()
    fig.savefig(figures_dir / "example_inverse_tau_rotation_fits.png", dpi=150)
    plt.close(fig)


def _build_report(
    *,
    comparison_df: pd.DataFrame,
    model_summary: pd.DataFrame,
    param_summary: pd.DataFrame,
    boundary_df: pd.DataFrame,
    priors: CohortPriors,
    sparc_csv: Path,
    inverse_design_run: Path,
    input_run: Path | None,
) -> str:
    n = len(comparison_df)
    beats_old = float(comparison_df["best_inverse_beats_old_tdf"].mean()) if n else 0.0
    beats_nfw = float(comparison_df["best_inverse_beats_nfw"].mean()) if n else 0.0
    beats_pseudo = float(comparison_df["best_inverse_beats_pseudo_isothermal"].mean()) if n else 0.0
    med_d_old = float(comparison_df["delta_bic_best_inverse_vs_old_tdf"].median()) if n else float("nan")
    med_d_nfw = float(comparison_df["delta_bic_best_inverse_vs_nfw"].median()) if n else float("nan")

    inv_wins = {
        v: int((comparison_df["best_inverse_variant"] == v).sum()) for v in INVERSE_VARIANTS
    }
    best_variant = max(inv_wins, key=inv_wins.get) if inv_wins else ""

    ml_bound = float(boundary_df["upsilon_disk_at_bound"].mean()) if len(boundary_df) else 0.0

    ready = (
        med_d_old < -2
        and beats_old > 0.5
        and beats_pseudo < 0.45
    )
    synthesis_ready = "needs more formula work" if not ready else "candidate for final synthesis update"

    improve_old_txt = (
        "The **R_core · class/global β** revision **improves** BIC vs the k-essence disk proxy for many galaxies."
        if med_d_old < 0
        else "Improvement over old TDF is **limited** under strict BIC accounting."
    )
    pseudo_txt = (
        "Inverse τ **does not** routinely beat pseudo-isothermal; cored-halo baselines remain strong."
        if beats_pseudo < 0.4
        else "Inverse τ is **sometimes competitive** with pseudo-isothermal depending on variant."
    )

    lines = [
        "# SPARC inverse-designed τ response report (Step 6B)",
        "",
        f"## ⚠️ {BANNER_INVERSE_TAU}",
        "",
        f"## ⚠️ {BANNER_CALIBRATION}",
        "",
        f"**SPARC data:** `{sparc_csv}`",
        f"**Step 6A priors:** `{inverse_design_run}`",
        f"**Calibration reference:** `{input_run or 'n/a'}`",
        f"**Galaxies:** {n}",
        "",
        f"**Cohort priors (6A):** β_global={priors.global_beta:.3f}, "
        f"r_c≈{priors.global_rc_kpc:.2f} kpc; class β: "
        + ", ".join(f"{k}={v:.3f}" for k, v in priors.class_beta.items()),
        "",
        "## Model summary",
        "",
        model_summary.to_string(index=False),
        "",
        "## Parameter accounting (BIC penalties)",
        "",
        param_summary.to_string(index=False),
        "",
        "## Key questions",
        "",
        "### Does the inverse-designed τ response improve over old TDF?",
        "",
        f"Best-of-three inverse variants beats old TDF on **{beats_old:.0%}** of galaxies; "
        f"median ΔBIC(best inverse − old TDF) = **{med_d_old:.2f}**. "
        f"{improve_old_txt}",
        "",
        "### Does adding R_core reduce inner residuals?",
        "",
        "Cored factor R_core(r) is applied in all inverse variants; compare inner zone median |residual| "
        "in fit tables (not repeated here). Step 6A flagged core regularization for most galaxies; "
        "this benchmark tests that explicitly.",
        "",
        "### Does class-dependent beta improve dwarf/intermediate performance?",
        "",
        f"Variant wins: global={inv_wins.get('inverse_tdf_global_beta_core', 0)}, "
        f"class={inv_wins.get('inverse_tdf_class_beta_core', 0)}, "
        f"baryon-feature={inv_wins.get('inverse_tdf_baryon_feature_beta', 0)}. "
        f"Most frequent best inverse: **{best_variant}**.",
        "",
        "### Does baryon-feature beta reduce per-galaxy β freedom?",
        "",
        "Variant C uses β_min, β_max, p(r) with 4 extra parameters per galaxy; "
        "it trades per-galaxy β/M for a baryon-acceleration shape — check BIC vs old TDF with +3 DOF.",
        "",
        "### Does inverse TDF remain competitive after BIC parameter penalty?",
        "",
        f"Median ΔBIC(best inverse − NFW) = **{med_d_nfw:.2f}**; beats NFW on **{beats_nfw:.0%}**. "
        f"Beats pseudo-isothermal on **{beats_pseudo:.0%}**.",
        "",
        "### Does it outperform pseudo-isothermal or only old TDF/NFW?",
        "",
        pseudo_txt,
        "",
        "### Ready for final synthesis?",
        "",
        f"**{synthesis_ready}** — inverse τ is a formula-revision benchmark only, "
        "not observational validation and not a dark-matter replacement claim.",
        "",
        "## Candidate response (from 6A)",
        "",
        "```",
        "a_τ(r) = β_eff(a_b, Σ_b, class) * sqrt(a_b * a0) * R_core(r)",
        "R_core(r) = r / sqrt(r² + r_c²)",
        "```",
        "",
        "## Limitations",
        "",
        "- M/L fitted per galaxy on [0.05, 3.0]; Υ at bound on "
        f"{ml_bound:.0%} of model-galaxy rows.",
        "- Does not replace Step 7 synthesis; does not validate TDF observationally.",
        "",
    ]
    return "\n".join(lines)

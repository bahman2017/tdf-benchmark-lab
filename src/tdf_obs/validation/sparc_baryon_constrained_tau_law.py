"""
SPARC Step 6C — baryon-constrained τ coupling laws.

Tests global β_eff laws with R_core vs baselines and Step 6B inverse τ.
Formula-revision diagnostics only; not observational validation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import least_squares, minimize

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
from tdf_obs.validation.sparc_inverse_tau_response import R_core, load_cohort_priors
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

BANNER_TAU_LAW = (
    "SPARC BARYON-CONSTRAINED TAU COUPLING LAW — NOT FULL OBSERVATIONAL VALIDATION"
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

TauLawId = Literal["A", "B", "C", "D"]
RcPolicy = Literal["global_r_c", "bounded_galaxy_r_c"]

TAU_LAW_MODELS: tuple[str, ...] = (
    "tau_law_A",
    "tau_law_B",
    "tau_law_C",
    "tau_law_D",
)

BASELINE_MODELS: tuple[str, ...] = (
    "baryon_only",
    "corrected_mond",
    "nfw",
    "pseudo_isothermal",
    "old_tdf_baseline",
    "inverse_tdf_baryon_feature_beta",
)

ALL_COMPARE_MODELS: tuple[str, ...] = BASELINE_MODELS + TAU_LAW_MODELS

SUMMARY_COLUMNS: tuple[str, ...] = (
    "model",
    "galaxies_fitted",
    "bic_win_count",
    "median_bic",
    "median_reduced_chi2",
    "median_chi2",
    "median_n_params",
    "median_abs_residual_inner",
    "median_abs_residual_middle",
    "median_abs_residual_outer",
    "tau_law_vs_old_tdf_median_delta_bic",
    "tau_law_vs_step6b_median_delta_bic",
    "tau_law_vs_nfw_median_delta_bic",
    "tau_law_vs_pseudo_median_delta_bic",
)


@dataclass
class GalaxyFrame:
    galaxy_id: str
    galaxy_class: str
    r: np.ndarray
    v_obs: np.ndarray
    v_err: np.ndarray
    v_gas: np.ndarray
    v_disk: np.ndarray
    v_bulge: np.ndarray
    has_bulge: bool
    rmax: float


@dataclass
class TauLawRunResult:
    model_summary: pd.DataFrame
    comparison_by_galaxy: pd.DataFrame
    parameter_summary: pd.DataFrame
    boundary_flags: pd.DataFrame
    class_summary: pd.DataFrame
    fit_details: list[ModelFitResult] = field(default_factory=list)
    cohort_params: dict[str, dict[str, float]] = field(default_factory=dict)


def sigma_proxy(
    r: np.ndarray,
    v_disk: np.ndarray,
    v_bulge: np.ndarray,
) -> np.ndarray:
    r_safe = np.maximum(np.asarray(r, dtype=float), R_MIN_KPC)
    sig = np.maximum(_sq_disk_bulge(v_disk, v_bulge), ACCEL_EPS) / r_safe
    return np.maximum(sig, ACCEL_EPS)


def _sq_disk_bulge(v_disk: np.ndarray, v_bulge: np.ndarray) -> np.ndarray:
    from tdf_obs.validation.sparc_real_calibration import _sq_sign_safe

    return _sq_sign_safe(v_disk) + _sq_sign_safe(v_bulge)


def baryon_kinematics(
    r: np.ndarray,
    v_gas: np.ndarray,
    v_disk: np.ndarray,
    v_bulge: np.ndarray,
    upsilon_disk: float,
    upsilon_bulge: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return v2_baryon, a_b, Sigma_proxy."""
    v2_b = v2_baryon_user(v_gas, v_disk, v_bulge, upsilon_disk, upsilon_bulge)
    r_safe = np.maximum(np.asarray(r, dtype=float), R_MIN_KPC)
    a_b = np.maximum(v2_b, 0.0) / r_safe
    sig = sigma_proxy(r_safe, v_disk, v_bulge)
    return v2_b, a_b, sig


def beta_eff_law_A(a_b: np.ndarray, beta0: float, p: float, a0: float) -> np.ndarray:
    x = np.maximum(a_b / float(a0), ACCEL_EPS)
    return np.maximum(float(beta0) / (1.0 + x ** float(p)), 0.0)


def beta_eff_law_B(
    a_b: np.ndarray,
    beta_min: float,
    beta_max: float,
    p: float,
    a0: float,
) -> np.ndarray:
    x = np.maximum(a_b / float(a0), ACCEL_EPS) ** float(p)
    return np.maximum(
        float(beta_min) + (float(beta_max) - float(beta_min)) / (1.0 + x),
        0.0,
    )


def beta_eff_law_C(
    sigma: np.ndarray,
    beta0: float,
    sigma0: float,
    q: float,
) -> np.ndarray:
    s = np.maximum(sigma / max(float(sigma0), ACCEL_EPS), ACCEL_EPS)
    return np.maximum(float(beta0) / (1.0 + s ** float(q)), 0.0)


def beta_eff_law_D(
    a_b: np.ndarray,
    sigma: np.ndarray,
    beta0: float,
    p: float,
    sigma0: float,
    q: float,
    a0: float,
) -> np.ndarray:
    x = np.maximum(a_b / float(a0), ACCEL_EPS) ** float(p)
    s = np.maximum(sigma / max(float(sigma0), ACCEL_EPS), ACCEL_EPS) ** float(q)
    return np.maximum(float(beta0) / ((1.0 + x) * (1.0 + s)), 0.0)


def beta_eff_from_law(
    law: TauLawId,
    a_b: np.ndarray,
    sigma: np.ndarray,
    theta: np.ndarray,
    a0: float = A0_TDF_DEFAULT,
) -> np.ndarray:
    if law == "A":
        return beta_eff_law_A(a_b, theta[0], theta[1], a0)
    if law == "B":
        return beta_eff_law_B(a_b, theta[0], theta[1], theta[2], a0)
    if law == "C":
        return beta_eff_law_C(sigma, theta[0], theta[1], theta[2])
    return beta_eff_law_D(a_b, sigma, theta[0], theta[1], theta[2], theta[3], a0)


def v_tau_law(
    r: np.ndarray,
    v_gas: np.ndarray,
    v_disk: np.ndarray,
    v_bulge: np.ndarray,
    upsilon_disk: float,
    upsilon_bulge: float,
    law: TauLawId,
    theta: np.ndarray,
    r_c: float,
    a0: float = A0_TDF_DEFAULT,
) -> np.ndarray:
    v2_b, a_b, sig = baryon_kinematics(r, v_gas, v_disk, v_bulge, upsilon_disk, upsilon_bulge)
    be = beta_eff_from_law(law, a_b, sig, theta, a0)
    core = R_core(r, r_c)
    a_tau = be * np.sqrt(np.maximum(a_b * a0, 0.0)) * core
    v2 = v2_b + np.maximum(r, R_MIN_KPC) * np.maximum(a_tau, 0.0)
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
    out: dict[str, float] = {}
    for name, mask_fn in (
        ("inner", lambda rr: rr < INNER_FRAC * rmax),
        ("middle", lambda rr: (rr >= INNER_FRAC * rmax) & (rr <= OUTER_FRAC * rmax)),
        ("outer", lambda rr: rr > OUTER_FRAC * rmax),
    ):
        m = mask_fn(r)
        out[f"median_abs_residual_{name}"] = float(np.median(np.abs(w[m]))) if m.any() else float("nan")
    return out


def law_global_dim(law: TauLawId) -> int:
    return {"A": 2, "B": 3, "C": 3, "D": 4}[law]


def law_theta_bounds(law: TauLawId) -> tuple[np.ndarray, np.ndarray]:
    if law == "A":
        lb, ub = np.array([0.01, 0.0]), np.array([3.0, 4.0])
    elif law == "B":
        lb, ub = np.array([0.0, 0.05, 0.0]), np.array([0.5, 3.0, 4.0])
    elif law == "C":
        lb, ub = np.array([0.01, 1.0, 0.0]), np.array([3.0, 1e4, 4.0])
    else:
        lb, ub = np.array([0.01, 0.0, 1.0, 0.0]), np.array([3.0, 4.0, 1e4, 4.0])
    return lb, ub


def law_theta_x0(law: TauLawId, priors: Any) -> np.ndarray:
    b = priors.global_beta
    if law == "A":
        return np.array([b, 1.0])
    if law == "B":
        return np.array([0.05, b, 1.0])
    if law == "C":
        return np.array([b, 200.0, 1.0])
    return np.array([b, 1.0, 200.0, 1.0])


def n_params_tau_law(
    law: TauLawId,
    has_bulge: bool,
    rc_policy: RcPolicy,
    n_global: int | None = None,
) -> int:
    n_ml = 2 if has_bulge else 1
    k = n_global if n_global is not None else law_global_dim(law)
    n_rc = 0 if rc_policy == "global_r_c" else 1
    return n_ml + k + n_rc + (1 if rc_policy == "global_r_c" else 0)


def _fit_ml_only(
    gf: GalaxyFrame,
    law: TauLawId,
    theta: np.ndarray,
    r_c: float,
    ml_prior: MLPriorConfig,
) -> tuple[np.ndarray, float, dict[str, float], np.ndarray]:
    d_lo, d_hi = ml_prior.disk_bounds
    b_lo, b_hi = ml_prior.bulge_bounds
    if gf.has_bulge:
        ml_lb, ml_ub, ml_x0 = np.array([d_lo, b_lo]), np.array([d_hi, b_hi]), np.array([0.5, 0.7])
    else:
        ml_lb, ml_ub, ml_x0 = np.array([d_lo]), np.array([d_hi]), np.array([0.5])

    def predict(ml_p: np.ndarray) -> np.ndarray:
        ud = float(ml_p[0])
        ub = float(ml_p[1]) if gf.has_bulge else 0.0
        return v_tau_law(gf.r, gf.v_gas, gf.v_disk, gf.v_bulge, ud, ub, law, theta, r_c)

    def residuals(ml_p: np.ndarray) -> np.ndarray:
        pred = predict(ml_p)
        if not np.all(np.isfinite(pred)):
            return np.full_like(gf.v_obs, 1e6)
        return (pred - gf.v_obs) / gf.v_err

    res = least_squares(
        residuals,
        ml_x0,
        bounds=(ml_lb, ml_ub),
        max_nfev=3000,
    )
    ml_p = res.x
    ud = float(ml_p[0])
    ub = float(ml_p[1]) if gf.has_bulge else 0.0
    v_pred = predict(ml_p)
    c2 = chi_square(gf.v_obs, v_pred, gf.v_err)
    return ml_p, c2, {"upsilon_disk": ud, "upsilon_bulge": ub if gf.has_bulge else np.nan}, v_pred


def _galaxy_chi2_with_globals(
    gf: GalaxyFrame,
    law: TauLawId,
    theta: np.ndarray,
    r_c: float,
    ml_prior: MLPriorConfig,
    rc_policy: RcPolicy,
) -> float:
    if rc_policy == "bounded_galaxy_r_c":
        rc_hi = min(50.0, max(gf.rmax * 2.0, 0.5))
        d_lo, d_hi = ml_prior.disk_bounds
        if gf.has_bulge:
            ml_lb = np.array([d_lo, ml_prior.bulge_bounds[0], 0.05])
            ml_ub = np.array([d_hi, ml_prior.bulge_bounds[1], rc_hi])
            x0 = np.array([0.5, 0.7, min(float(r_c), rc_hi)])
        else:
            ml_lb = np.array([d_lo, 0.05])
            ml_ub = np.array([d_hi, rc_hi])
            x0 = np.array([0.5, min(float(r_c), rc_hi)])
        x0 = np.clip(x0, ml_lb, ml_ub)

        def predict(p: np.ndarray) -> np.ndarray:
            if gf.has_bulge:
                ud, ub, rc = float(p[0]), float(p[1]), float(p[2])
            else:
                ud, ub, rc = float(p[0]), 0.0, float(p[1])
            return v_tau_law(gf.r, gf.v_gas, gf.v_disk, gf.v_bulge, ud, ub, law, theta, rc)

        def residuals(p: np.ndarray) -> np.ndarray:
            pred = predict(p)
            if not np.all(np.isfinite(pred)):
                return np.full_like(gf.v_obs, 1e6)
            return (pred - gf.v_obs) / gf.v_err

        res = least_squares(residuals, x0, bounds=(ml_lb, ml_ub), max_nfev=2500)
        pred = predict(res.x)
    else:
        _, c2, _, pred = _fit_ml_only(gf, law, theta, r_c, ml_prior)
        return float(chi_square(gf.v_obs, pred, gf.v_err))
    return float(chi_square(gf.v_obs, pred, gf.v_err))


def fit_cohort_tau_law(
    galaxies: list[GalaxyFrame],
    law: TauLawId,
    rc_policy: RcPolicy,
    global_rc: float,
    priors: Any,
    ml_prior: MLPriorConfig = ML_STANDARD,
) -> tuple[np.ndarray, float, list[ModelFitResult], RcPolicy]:
    """Optimize global law parameters; return theta, r_c used, per-galaxy fits."""
    lb, ub = law_theta_bounds(law)
    x0 = np.clip(law_theta_x0(law, priors), lb, ub)
    rc_use = float(np.clip(global_rc, 0.1, 25.0))

    def objective(theta: np.ndarray) -> float:
        total = 0.0
        for gf in galaxies:
            total += _galaxy_chi2_with_globals(gf, law, theta, rc_use, ml_prior, rc_policy)
        return total

    opt = minimize(objective, x0, bounds=list(zip(lb, ub)), method="L-BFGS-B", options={"maxiter": 60})
    theta_opt = np.clip(opt.x, lb, ub)
    rc_final = rc_use

    results: list[ModelFitResult] = []
    model_name = f"tau_law_{law}"

    for gf in galaxies:
        n_par = n_params_tau_law(law, gf.has_bulge, rc_policy)
        if rc_policy == "bounded_galaxy_r_c":
            rc_hi = min(50.0, max(gf.rmax * 2.0, 0.5))
            d_lo, d_hi = ml_prior.disk_bounds
            if gf.has_bulge:
                ml_lb = np.array([d_lo, ml_prior.bulge_bounds[0], 0.05])
                ml_ub = np.array([d_hi, ml_prior.bulge_bounds[1], rc_hi])
                x0g = np.clip(np.array([0.5, 0.7, min(rc_final, rc_hi)]), ml_lb, ml_ub)
                n_ml = 2
            else:
                ml_lb = np.array([d_lo, 0.05])
                ml_ub = np.array([d_hi, rc_hi])
                x0g = np.clip(np.array([0.5, min(rc_final, rc_hi)]), ml_lb, ml_ub)
                n_ml = 1

            def predict(p: np.ndarray) -> np.ndarray:
                if gf.has_bulge:
                    ud, ub, rc = float(p[0]), float(p[1]), float(p[2])
                else:
                    ud, ub, rc = float(p[0]), 0.0, float(p[1])
                return v_tau_law(gf.r, gf.v_gas, gf.v_disk, gf.v_bulge, ud, ub, law, theta_opt, rc)

            def residuals(p: np.ndarray) -> np.ndarray:
                pred = predict(p)
                if not np.all(np.isfinite(pred)):
                    return np.full_like(gf.v_obs, 1e6)
                return (pred - gf.v_obs) / gf.v_err

            res = least_squares(residuals, x0g, bounds=(ml_lb, ml_ub), max_nfev=3000)
            p_opt = res.x
            if gf.has_bulge:
                ud, ub, rc_g = float(p_opt[0]), float(p_opt[1]), float(p_opt[2])
            else:
                ud, ub, rc_g = float(p_opt[0]), 0.0, float(p_opt[1])
            v_pred = predict(p_opt)
            params = {
                "upsilon_disk": ud,
                "upsilon_bulge": ub if gf.has_bulge else np.nan,
                "r_c": rc_g,
                **{f"theta_{i}": float(theta_opt[i]) for i in range(len(theta_opt))},
            }
        else:
            ml_p, _, params, v_pred = _fit_ml_only(gf, law, theta_opt, rc_final, ml_prior)
            params = dict(params)
            params["r_c"] = rc_final
            for i, v in enumerate(theta_opt):
                params[f"theta_{i}"] = float(v)

        c2, chi2_red, rmse, aic_v, bic_v = _metrics(gf.v_obs, v_pred, gf.v_err, n_par)
        results.append(
            ModelFitResult(
                galaxy_id=gf.galaxy_id,
                model=model_name,  # type: ignore[assignment]
                n_points=len(gf.r),
                n_params=n_par,
                chi2=c2,
                reduced_chi2=chi2_red,
                rmse=rmse,
                aic=aic_v,
                bic=bic_v,
                success=bool(np.all(np.isfinite(v_pred))),
                params=params,
                v_pred=v_pred,
            ),
        )

    return theta_opt, rc_final, results, rc_policy


def fit_galaxy_baselines(
    gf: GalaxyFrame,
    ml_prior: MLPriorConfig = ML_STANDARD,
) -> list[ModelFitResult]:
    results: list[ModelFitResult] = []
    for base in ("baryon_only", "corrected_mond", "nfw", "pseudo_isothermal"):
        fr = fit_model_cored_baseline(
            gf.r,
            gf.v_obs,
            gf.v_err,
            gf.v_gas,
            gf.v_disk,
            gf.v_bulge,
            gf.has_bulge,
            base,  # type: ignore[arg-type]
            ml_prior,
        )
        fr.galaxy_id = gf.galaxy_id
        results.append(fr)
    fr_old = fit_model_cored_baseline(
        gf.r,
        gf.v_obs,
        gf.v_err,
        gf.v_gas,
        gf.v_disk,
        gf.v_bulge,
        gf.has_bulge,
        "tdf_kessence",
        ml_prior,
    )
    fr_old.galaxy_id = gf.galaxy_id
    fr_old.model = "old_tdf_baseline"  # type: ignore[assignment]
    results.append(fr_old)
    return results


def build_parameter_summary(
    cohort_params: dict[str, dict[str, float]],
    rc_policies: dict[str, RcPolicy],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    desc = {
        "tau_law_A": "β₀/(1+(a_b/a₀)^p); global β₀,p + global r_c",
        "tau_law_B": "β_min+(β_max-β_min)/(1+(a_b/a₀)^p); global 3 + r_c",
        "tau_law_C": "β₀/(1+(Σ/Σ₀)^q); global 3 + r_c",
        "tau_law_D": "β₀/[(1+(a_b/a₀)^p)(1+(Σ/Σ₀)^q)]; global 4 + r_c",
    }
    for m in TAU_LAW_MODELS:
        law = m.split("_")[-1]
        k = law_global_dim(law)  # type: ignore[arg-type]
        rc_pol = rc_policies.get(m, "global_r_c")
        per_g = 1 + k + (1 if rc_pol == "global_r_c" else 1)
        cohort = k + (1 if rc_pol == "global_r_c" else 0)
        rows.append(
            {
                "model": m,
                "per_galaxy_n_params_ml": 2,
                "per_galaxy_n_params_local": per_g,
                "cohort_global_n_params": cohort,
                "r_c_policy": rc_pol,
                "total_params_description": desc.get(m, ""),
                **cohort_params.get(m, {}),
            },
        )
    for m in BASELINE_MODELS:
        if m == "inverse_tdf_baryon_feature_beta":
            rows.append(
                {
                    "model": m,
                    "per_galaxy_n_params_ml": 2,
                    "per_galaxy_n_params_local": 5,
                    "cohort_global_n_params": 0,
                    "r_c_policy": "per_galaxy",
                    "total_params_description": "Step 6B Variant C (loaded from prior run)",
                },
            )
        elif m == "old_tdf_baseline":
            rows.append(
                {
                    "model": m,
                    "per_galaxy_n_params_ml": 2,
                    "per_galaxy_n_params_local": 1,
                    "cohort_global_n_params": 0,
                    "r_c_policy": "n/a",
                    "total_params_description": "Υ + β/M k-essence",
                },
            )
    return pd.DataFrame(rows)


def run_baryon_constrained_tau_law(
    sparc_csv: Path,
    output_dir: Path,
    *,
    inverse_design_run: Path,
    inverse_response_run: Path | None = None,
    input_run: Path | None = None,
    max_galaxies: int | None = None,
) -> TauLawRunResult:
    output_dir = Path(output_dir)
    tables = output_dir / "tables"
    reports = output_dir / "reports"
    figures = output_dir / "figures"
    for d in (tables, reports, figures):
        d.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(sparc_csv)
    validate_sparc_input_schema(df)
    halo_path = Path(inverse_design_run) / "tables" / "tau_effective_halo_summary.csv"
    halo_summary = pd.read_csv(halo_path)
    priors = load_cohort_priors(halo_summary)

    step6b: pd.DataFrame | None = None
    if inverse_response_run is not None:
        p6b = Path(inverse_response_run) / "tables" / "inverse_tau_comparison_by_galaxy.csv"
        if p6b.is_file():
            step6b = pd.read_csv(p6b)

    props_df = build_galaxy_properties(df)
    props_map = props_df.set_index("galaxy_id").to_dict("index")

    galaxies: list[GalaxyFrame] = []
    gids = sorted(df["galaxy_id"].unique())
    if max_galaxies is not None:
        gids = gids[:max_galaxies]

    for gid in gids:
        gdf = df[df["galaxy_id"] == gid].sort_values("r_kpc")
        prep = _prepare_galaxy_frame(gdf, 5)
        if prep is None:
            continue
        props = props_map.get(gid, {})
        galaxies.append(
            GalaxyFrame(
                galaxy_id=str(gid),
                galaxy_class=str(
                    props.get("galaxy_class", classify_galaxy_by_vmax(float(props.get("vmax_obs", 0)))),
                ),
                r=prep["r_kpc"].to_numpy(dtype=float),
                v_obs=prep["v_obs"].to_numpy(dtype=float),
                v_err=np.maximum(prep["v_err"].to_numpy(dtype=float), V_ERR_FLOOR),
                v_gas=prep["v_gas"].to_numpy(dtype=float),
                v_disk=prep["v_disk"].to_numpy(dtype=float),
                v_bulge=prep["v_bulge"].to_numpy(dtype=float),
                has_bulge=galaxy_has_bulge(prep["v_bulge"].to_numpy()),
                rmax=float(prep["r_kpc"].max()),
            ),
        )

    all_fits: list[ModelFitResult] = []
    comparisons: list[dict[str, Any]] = []
    boundary_rows: list[dict[str, Any]] = []
    zone_rows: list[dict[str, Any]] = []
    cohort_params: dict[str, dict[str, float]] = {}
    rc_policies: dict[str, RcPolicy] = {}

    law_rc_policy: dict[TauLawId, RcPolicy] = {}
    pilot = galaxies[: min(12, len(galaxies))]
    for law in ("A", "B", "C", "D"):
        chi_global = sum(
            _galaxy_chi2_with_globals(
                gf, law, law_theta_x0(law, priors), priors.global_rc_kpc, ML_STANDARD, "global_r_c",  # type: ignore[arg-type]
            )
            for gf in pilot
        )
        chi_bounded = sum(
            _galaxy_chi2_with_globals(
                gf, law, law_theta_x0(law, priors), priors.global_rc_kpc, ML_STANDARD, "bounded_galaxy_r_c",  # type: ignore[arg-type]
            )
            for gf in pilot
        )
        law_rc_policy[law] = (
            "bounded_galaxy_r_c" if chi_bounded < chi_global * 0.98 else "global_r_c"
        )

    for law in ("A", "B", "C", "D"):
        policy = law_rc_policy[law]
        model_name = f"tau_law_{law}"
        rc_policies[model_name] = policy
        theta, rc, law_fits, _ = fit_cohort_tau_law(
            galaxies,
            law,  # type: ignore[arg-type]
            policy,
            priors.global_rc_kpc,
            priors,
        )
        cohort_params[model_name] = {
            "global_r_c_kpc": rc,
            **{f"theta_{i}": float(theta[i]) for i in range(len(theta))},
        }
        all_fits.extend(law_fits)

    for gf in galaxies:
        comp: dict[str, Any] = {
            "galaxy_id": gf.galaxy_id,
            "galaxy_class": gf.galaxy_class,
            "n_points": len(gf.r),
            "has_bulge": gf.has_bulge,
        }
        base_fits = fit_galaxy_baselines(gf)
        all_fits.extend(base_fits)
        by = {f.model: f for f in base_fits}
        for m in ("baryon_only", "corrected_mond", "nfw", "pseudo_isothermal", "old_tdf_baseline"):
            comp[f"bic_{m}"] = by[m].bic
            comp[f"reduced_chi2_{m}"] = by[m].reduced_chi2

        if step6b is not None and gf.galaxy_id in step6b["galaxy_id"].values:
            row6b = step6b[step6b["galaxy_id"] == gf.galaxy_id].iloc[0]
            comp["bic_inverse_tdf_baryon_feature_beta"] = float(
                row6b["bic_inverse_tdf_baryon_feature_beta"],
            )
        else:
            comp["bic_inverse_tdf_baryon_feature_beta"] = float("nan")

        for law in ("A", "B", "C", "D"):
            mname = f"tau_law_{law}"
            fr = next((f for f in all_fits if f.galaxy_id == gf.galaxy_id and f.model == mname), None)
            if fr:
                comp[f"bic_{mname}"] = fr.bic
                comp[f"reduced_chi2_{mname}"] = fr.reduced_chi2
                boundary_rows.append(_boundary_row(gf.galaxy_id, fr))
                if fr.v_pred is not None:
                    zone_rows.append(
                        {"galaxy_id": gf.galaxy_id, "model": mname, **_zone_residuals(
                            gf.r, gf.v_obs, gf.v_err, fr.v_pred,
                        )},
                    )

        law_bics = {f"tau_law_{law}": comp.get(f"bic_tau_law_{law}", np.inf) for law in ("A", "B", "C", "D")}
        best_law = min(law_bics, key=law_bics.get)  # type: ignore[arg-type]
        comp["best_tau_law"] = best_law
        comp["bic_best_tau_law"] = law_bics[best_law]
        comp["delta_bic_best_tau_law_vs_old_tdf"] = (
            comp["bic_best_tau_law"] - comp["bic_old_tdf_baseline"]
        )
        comp["delta_bic_best_tau_law_vs_step6b"] = (
            comp["bic_best_tau_law"] - comp.get("bic_inverse_tdf_baryon_feature_beta", np.nan)
        )
        comp["delta_bic_best_tau_law_vs_nfw"] = comp["bic_best_tau_law"] - comp["bic_nfw"]
        comp["delta_bic_best_tau_law_vs_pseudo"] = (
            comp["bic_best_tau_law"] - comp["bic_pseudo_isothermal"]
        )
        comp["best_tau_law_beats_old_tdf"] = comp["bic_best_tau_law"] < comp["bic_old_tdf_baseline"]
        comp["best_tau_law_beats_step6b"] = (
            comp["bic_best_tau_law"] < comp.get("bic_inverse_tdf_baryon_feature_beta", np.inf)
        )
        comp["best_tau_law_beats_nfw"] = comp["bic_best_tau_law"] < comp["bic_nfw"]
        comp["best_tau_law_beats_pseudo"] = (
            comp["bic_best_tau_law"] < comp["bic_pseudo_isothermal"]
        )

        all_bics = {
            k[4:]: float(v)
            for k, v in comp.items()
            if k.startswith("bic_") and np.isfinite(v)
        }
        comp["best_model_by_bic"] = min(all_bics, key=all_bics.get)  # type: ignore[arg-type]
        comparisons.append(comp)

        for fr in base_fits:
            boundary_rows.append(_boundary_row(gf.galaxy_id, fr))
            if fr.v_pred is not None:
                zone_rows.append(
                    {
                        "galaxy_id": gf.galaxy_id,
                        "model": fr.model,
                        **_zone_residuals(gf.r, gf.v_obs, gf.v_err, fr.v_pred),
                    },
                )

    comparison_df = pd.DataFrame(comparisons)
    model_summary = _build_model_summary(comparison_df, all_fits, zone_rows)
    param_summary = build_parameter_summary(cohort_params, rc_policies)
    boundary_df = pd.DataFrame(boundary_rows)
    class_summary = _build_class_summary(comparison_df)

    model_summary.to_csv(tables / "tau_law_model_summary.csv", index=False)
    comparison_df.to_csv(tables / "tau_law_comparison_by_galaxy.csv", index=False)
    param_summary.to_csv(tables / "tau_law_parameter_summary.csv", index=False)
    boundary_df.to_csv(tables / "tau_law_boundary_flags.csv", index=False)
    class_summary.to_csv(tables / "tau_law_class_summary.csv", index=False)

    corr = _correlation_diagnostics(halo_summary, comparison_df, cohort_params)
    _write_figures(comparison_df, all_fits, df, cohort_params, figures)
    report = _build_report(
        comparison_df=comparison_df,
        model_summary=model_summary,
        param_summary=param_summary,
        class_summary=class_summary,
        cohort_params=cohort_params,
        rc_policies=rc_policies,
        corr=corr,
        sparc_csv=sparc_csv,
        inverse_design_run=inverse_design_run,
        input_run=input_run,
    )
    (reports / "tau_law_report.md").write_text(report, encoding="utf-8")

    return TauLawRunResult(
        model_summary=model_summary,
        comparison_by_galaxy=comparison_df,
        parameter_summary=param_summary,
        boundary_flags=boundary_df,
        class_summary=class_summary,
        fit_details=all_fits,
        cohort_params=cohort_params,
    )


def _boundary_row(galaxy_id: str, fr: ModelFitResult) -> dict[str, Any]:
    p = fr.params
    d_lo, d_hi = ML_STANDARD.disk_bounds
    return {
        "galaxy_id": galaxy_id,
        "model": fr.model,
        "any_boundary_hit": any(
            [
                _param_at_bound(p.get("upsilon_disk", np.nan), d_lo, d_hi),
                _param_at_bound(p.get("upsilon_bulge", np.nan), *ML_STANDARD.bulge_bounds),
                _param_at_bound(p.get("r_c", np.nan), 0.05, 50.0),
            ],
        ),
        "upsilon_disk_at_bound": _param_at_bound(p.get("upsilon_disk", np.nan), d_lo, d_hi),
        "upsilon_bulge_at_bound": _param_at_bound(
            p.get("upsilon_bulge", np.nan), *ML_STANDARD.bulge_bounds,
        ),
        "r_c_at_bound": _param_at_bound(p.get("r_c", np.nan), 0.05, 50.0),
    }


def _build_model_summary(
    comparison_df: pd.DataFrame,
    all_fits: list[ModelFitResult],
    zone_rows: list[dict[str, Any]],
) -> pd.DataFrame:
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
            for fr in all_fits
        ],
    )
    zone_df = pd.DataFrame(zone_rows) if zone_rows else pd.DataFrame()
    rows: list[dict[str, Any]] = []
    models = list(BASELINE_MODELS) + list(TAU_LAW_MODELS)
    for model in models:
        sub = fit_df[fit_df["model"] == model]
        if model == "inverse_tdf_baryon_feature_beta":
            if "bic_inverse_tdf_baryon_feature_beta" not in comparison_df.columns:
                continue
            sub_bic = comparison_df["bic_inverse_tdf_baryon_feature_beta"]
            wins = int((comparison_df["best_model_by_bic"] == model).sum())
            med_bic = float(sub_bic.median())
            med_chi2 = float("nan")
            med_red = float("nan")
            med_n = float("nan")
        elif sub.empty:
            continue
        else:
            wins = int((comparison_df["best_model_by_bic"] == model).sum())
            med_bic = float(sub["bic"].median())
            med_chi2 = float(sub["chi2"].median())
            med_red = float(sub["reduced_chi2"].median())
            med_n = float(sub["n_params"].median())

        row: dict[str, Any] = {
            "model": model,
            "galaxies_fitted": len(comparison_df),
            "bic_win_count": wins,
            "median_bic": med_bic,
            "median_reduced_chi2": med_red,
            "median_chi2": med_chi2,
            "median_n_params": med_n,
        }
        if not zone_df.empty and model in zone_df["model"].values:
            zs = zone_df[zone_df["model"] == model]
            for z in ("inner", "middle", "outer"):
                col = f"median_abs_residual_{z}"
                if col in zs.columns:
                    row[col] = float(zs[col].median())
        if model in TAU_LAW_MODELS:
            row["tau_law_vs_old_tdf_median_delta_bic"] = float(
                comparison_df["delta_bic_best_tau_law_vs_old_tdf"].median(),
            )
            row["tau_law_vs_step6b_median_delta_bic"] = float(
                comparison_df["delta_bic_best_tau_law_vs_step6b"].median(),
            )
            row["tau_law_vs_nfw_median_delta_bic"] = float(
                comparison_df["delta_bic_best_tau_law_vs_nfw"].median(),
            )
            row["tau_law_vs_pseudo_median_delta_bic"] = float(
                comparison_df["delta_bic_best_tau_law_vs_pseudo"].median(),
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _build_class_summary(comparison_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for cls in CLASS_ORDER:
        sub = comparison_df[comparison_df["galaxy_class"] == cls]
        if sub.empty:
            continue
        rows.append(
            {
                "galaxy_class": cls,
                "n_galaxies": len(sub),
                "bic_win_tau_law_A": int((sub["best_model_by_bic"] == "tau_law_A").sum()),
                "bic_win_tau_law_B": int((sub["best_model_by_bic"] == "tau_law_B").sum()),
                "bic_win_tau_law_C": int((sub["best_model_by_bic"] == "tau_law_C").sum()),
                "bic_win_tau_law_D": int((sub["best_model_by_bic"] == "tau_law_D").sum()),
                "bic_win_pseudo_isothermal": int(
                    (sub["best_model_by_bic"] == "pseudo_isothermal").sum(),
                ),
                "median_delta_bic_best_tau_law_vs_old_tdf": float(
                    sub["delta_bic_best_tau_law_vs_old_tdf"].median(),
                ),
                "median_delta_bic_best_tau_law_vs_nfw": float(
                    sub["delta_bic_best_tau_law_vs_nfw"].median(),
                ),
                "fraction_best_tau_law_beats_old_tdf": float(sub["best_tau_law_beats_old_tdf"].mean()),
                "fraction_best_tau_law_beats_pseudo": float(sub["best_tau_law_beats_pseudo"].mean()),
            },
        )
    return pd.DataFrame(rows)


def _correlation_diagnostics(
    halo_summary: pd.DataFrame,
    comparison_df: pd.DataFrame,
    cohort_params: dict[str, dict[str, float]],
) -> dict[str, float]:
    merged = halo_summary.merge(
        comparison_df[["galaxy_id", "bic_old_tdf_baseline"]],
        on="galaxy_id",
        how="inner",
    )
    out: dict[str, float] = {}
    if len(merged) >= 8:
        rho, _ = stats.spearmanr(merged["median_beta_eff"], merged["bic_old_tdf_baseline"])
        out["spearman_6a_beta_eff_vs_old_tdf_bic"] = float(rho)
    return out


def _write_figures(
    comparison_df: pd.DataFrame,
    all_fits: list[ModelFitResult],
    sparc_df: pd.DataFrame,
    cohort_params: dict[str, dict[str, float]],
    figures_dir: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if comparison_df.empty:
        return

    tau_models = list(TAU_LAW_MODELS) + ["old_tdf_baseline", "pseudo_isothermal"]
    wins = [int((comparison_df["best_model_by_bic"] == m).sum()) for m in tau_models]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(len(wins)), wins, color="steelblue")
    ax.set_xticks(range(len(wins)))
    ax.set_xticklabels([m.replace("tau_law_", "L") for m in tau_models], rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("BIC wins")
    ax.set_title("BIC wins — τ laws vs baselines")
    fig.tight_layout()
    fig.savefig(figures_dir / "bic_wins_tau_laws.png", dpi=150)
    plt.close(fig)

    for col, fname, title in [
        ("delta_bic_best_tau_law_vs_old_tdf", "tau_law_vs_old_tdf_delta_bic.png", "Best τ law − old TDF"),
        (
            "delta_bic_best_tau_law_vs_pseudo",
            "tau_law_vs_pseudo_isothermal_delta_bic.png",
            "Best τ law − pseudo-isothermal",
        ),
    ]:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(comparison_df[col].dropna(), bins=25, color="C1", alpha=0.85)
        ax.axvline(0, color="k", ls="--")
        ax.set_xlabel("ΔBIC")
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(figures_dir / fname, dpi=150)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    for cls in CLASS_ORDER:
        sub = comparison_df[comparison_df["galaxy_class"] == cls]
        ax.scatter(
            sub["delta_bic_best_tau_law_vs_old_tdf"],
            sub["delta_bic_best_tau_law_vs_nfw"],
            alpha=0.5,
            s=18,
            label=cls,
        )
    ax.axhline(0, color="k", lw=0.5)
    ax.axvline(0, color="k", lw=0.5)
    ax.legend(fontsize=8)
    ax.set_xlabel("ΔBIC vs old TDF")
    ax.set_ylabel("ΔBIC vs NFW")
    ax.set_title("Best τ law by galaxy class")
    fig.tight_layout()
    fig.savefig(figures_dir / "tau_law_by_galaxy_class.png", dpi=150)
    plt.close(fig)

    sample_pts: list[dict[str, Any]] = []
    for fr in all_fits:
        if not fr.model.startswith("tau_law_") or fr.v_pred is None:
            continue
        law = fr.model.replace("tau_law_", "")  # type: ignore[arg-type]
        dim = law_global_dim(law)  # type: ignore[arg-type]
        theta_arr = np.array(
            [float(fr.params.get(f"theta_{i}", np.nan)) for i in range(dim)],
            dtype=float,
        )
        if not np.all(np.isfinite(theta_arr)):
            continue
        gf_rows = sparc_df[sparc_df["galaxy_id"] == fr.galaxy_id].sort_values("r_kpc")
        r = gf_rows["r_kpc"].to_numpy(dtype=float)
        v2_b, a_b, sig = baryon_kinematics(
            r,
            gf_rows["v_gas"].to_numpy(dtype=float),
            gf_rows["v_disk"].to_numpy(dtype=float),
            gf_rows["v_bulge"].to_numpy(dtype=float),
            float(fr.params.get("upsilon_disk", 0.5)),
            float(fr.params.get("upsilon_bulge", 0.0) or 0.0),
        )
        be = beta_eff_from_law(law, a_b, sig, theta_arr, A0_TDF_DEFAULT)  # type: ignore[arg-type]
        for i in range(min(len(a_b), 50)):
            sample_pts.append(
                {
                    "log_ab": float(np.log10(max(float(a_b[i]), ACCEL_EPS))),
                    "beta_eff": float(be[i]),
                },
            )
    if sample_pts:
        sdf = pd.DataFrame(sample_pts)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(sdf["log_ab"], sdf["beta_eff"], alpha=0.08, s=6, c="C0")
        ax.set_xlabel("log10(a_b)")
        ax.set_ylabel("β_eff (law)")
        ax.set_title("Baryon-constrained β_eff vs a_b (sample points)")
        fig.tight_layout()
        fig.savefig(figures_dir / "beta_eff_law_vs_ab.png", dpi=150)
        plt.close(fig)

    best_ids = comparison_df.nsmallest(3, "delta_bic_best_tau_law_vs_old_tdf")["galaxy_id"].tolist()
    fig, axes = plt.subplots(1, max(1, len(best_ids)), figsize=(12, 4))
    if len(best_ids) == 1:
        axes = [axes]
    for ax, gid in zip(axes, best_ids):
        gdf = sparc_df[sparc_df["galaxy_id"] == gid].sort_values("r_kpc")
        r = gdf["r_kpc"].to_numpy()
        ax.errorbar(r, gdf["v_obs"], yerr=gdf["v_err"], fmt="o", ms=3)
        blaw = comparison_df.loc[comparison_df["galaxy_id"] == gid, "best_tau_law"].iloc[0]
        for fr in all_fits:
            if fr.galaxy_id == gid and fr.v_pred is not None and fr.model in (blaw, "old_tdf_baseline"):
                ax.plot(r, fr.v_pred, "-", label=fr.model[:14])
        ax.set_title(str(gid), fontsize=8)
        ax.legend(fontsize=6)
    fig.suptitle("Example τ-law rotation fits")
    fig.tight_layout()
    fig.savefig(figures_dir / "example_tau_law_rotation_fits.png", dpi=150)
    plt.close(fig)

    zdf = pd.DataFrame(
        [
            {
                "galaxy_id": fr.galaxy_id,
                "model": fr.model,
                **_zone_residuals(
                    sparc_df[sparc_df["galaxy_id"] == fr.galaxy_id].sort_values("r_kpc")["r_kpc"].to_numpy(),
                    sparc_df[sparc_df["galaxy_id"] == fr.galaxy_id].sort_values("r_kpc")["v_obs"].to_numpy(),
                    sparc_df[sparc_df["galaxy_id"] == fr.galaxy_id].sort_values("r_kpc")["v_err"].to_numpy(),
                    fr.v_pred,
                ),
            }
            for fr in all_fits
            if fr.v_pred is not None and fr.model in TAU_LAW_MODELS
        ],
    )
    if not zdf.empty:
        fig, ax = plt.subplots(figsize=(9, 5))
        data = [
            zdf.loc[zdf["model"] == m, "median_abs_residual_inner"].dropna().to_numpy()
            for m in TAU_LAW_MODELS
        ]
        ax.boxplot(data, tick_labels=[m.replace("tau_law_", "") for m in TAU_LAW_MODELS])
        ax.set_ylabel("median |weighted residual| inner")
        ax.set_title("Inner residuals by τ law")
        fig.tight_layout()
        fig.savefig(figures_dir / "inner_outer_residual_tau_laws.png", dpi=150)
        plt.close(fig)


def _build_report(
    *,
    comparison_df: pd.DataFrame,
    model_summary: pd.DataFrame,
    param_summary: pd.DataFrame,
    class_summary: pd.DataFrame,
    cohort_params: dict[str, dict[str, float]],
    rc_policies: dict[str, RcPolicy],
    corr: dict[str, float],
    sparc_csv: Path,
    inverse_design_run: Path,
    input_run: Path | None,
) -> str:
    n = len(comparison_df)
    best_law_wins = model_summary[model_summary["model"].isin(TAU_LAW_MODELS)].sort_values(
        "bic_win_count", ascending=False,
    )
    best_law_name = str(best_law_wins.iloc[0]["model"]) if len(best_law_wins) else "tau_law_A"
    beats_old = float(comparison_df["best_tau_law_beats_old_tdf"].mean()) if n else 0.0
    beats_6b = float(comparison_df["best_tau_law_beats_step6b"].mean()) if n else 0.0
    beats_pseudo = float(comparison_df["best_tau_law_beats_pseudo"].mean()) if n else 0.0
    med_d_old = float(comparison_df["delta_bic_best_tau_law_vs_old_tdf"].median()) if n else float("nan")
    med_d_6b = float(comparison_df["delta_bic_best_tau_law_vs_step6b"].median()) if n else float("nan")

    synthesis_ok = beats_old > 0.45 and med_d_old < 0 and beats_pseudo < 0.35

    lines = [
        "# SPARC baryon-constrained τ coupling law report (Step 6C)",
        "",
        f"## ⚠️ {BANNER_TAU_LAW}",
        "",
        f"## ⚠️ {BANNER_CALIBRATION}",
        "",
        f"**SPARC data:** `{sparc_csv}`",
        f"**Step 6A:** `{inverse_design_run}`",
        f"**Calibration ref:** `{input_run or 'n/a'}`",
        f"**Galaxies:** {n}",
        "",
        "## Model summary",
        "",
        model_summary.to_string(index=False),
        "",
        "## Global fitted parameters",
        "",
    ]
    for m, p in cohort_params.items():
        lines.append(f"- **{m}** ({rc_policies.get(m, 'global_r_c')}): {p}")
    lines.extend(
        [
            "",
            "## Parameter accounting",
            "",
            param_summary.to_string(index=False),
            "",
            "## Class summary",
            "",
            class_summary.to_string(index=False),
            "",
            "## Key questions",
            "",
            f"### Which β_eff law works best?",
            "",
            f"Among Laws A–D, **{best_law_name}** has the most BIC wins in this run. "
            f"r_c policies per law: {rc_policies}.",
            "",
            "### Does baryon-constrained β reduce per-galaxy β/M freedom?",
            "",
            "Yes in design: only Υ (and optionally one r_c) are fit per galaxy; "
            "β_eff shape is set by global law parameters (2–4) shared across the sample.",
            "",
            "### Does the new law improve over old TDF?",
            "",
            f"Best τ law beats old TDF on **{beats_old:.0%}**; median ΔBIC = **{med_d_old:.2f}**.",
            "",
            "### Does it improve over Step 6B Variant C after BIC penalties?",
            "",
            f"Beats Step 6B baryon-feature β on **{beats_6b:.0%}**; median ΔBIC = **{med_d_6b:.2f}** "
            "(negative ⇒ τ law better).",
            "",
            "### Does it reduce inner residuals?",
            "",
            "See `median_abs_residual_inner` in model summary; compare τ laws to old TDF.",
            "",
            "### Dwarf / low-acceleration performance?",
            "",
            "See class summary table for per-class ΔBIC and win counts.",
            "",
            "### Compete with pseudo-isothermal?",
            "",
            f"Beats pseudo-isothermal on **{beats_pseudo:.0%}** of galaxies only.",
            "",
            "### Ready for final SPARC synthesis?",
            "",
            (
                "**Candidate for synthesis mention** as the preferred formula-revision path, "
                "still rotation-only and not observational validation."
                if synthesis_ok
                else "**Not yet** — continue formula refinement before updating Step 7 synthesis."
            ),
            "",
            "### Paper formula (candidate, not final physics)",
            "",
            "```",
            "β_eff = β₀ / [(1 + (a_b/a₀)^p) (1 + (Σ_proxy/Σ₀)^q)]   [Law D example]",
            "a_τ = β_eff · √(a_b a₀) · R_core(r),   R_core = r/√(r²+r_c²)",
            "v² = v_baryon² + r · a_τ",
            "```",
            "",
            "## Limitations",
            "",
            "- Global law fit is approximate (not full hierarchical Bayesian).",
            "- Does not validate TDF observationally or replace dark matter.",
            "",
        ],
    )
    return "\n".join(lines)

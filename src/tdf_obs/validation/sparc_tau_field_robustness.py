"""
SPARC Step 6E — τ field-solver robustness and parameter-stability audit.

Audits Step 6D field-solved τ fits for parameter freedom, M/L sensitivity,
and boundary effects. Analysis only; not observational validation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import least_squares

from tdf_obs.fitting.metrics import bic
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
from tdf_obs.validation.sparc_ml_robustness import ML_REGIMES, MLRegimeConfig
from tdf_obs.validation.sparc_real_calibration import (
    A0_TDF_DEFAULT,
    V_ERR_FLOOR,
    _prepare_galaxy_frame,
    galaxy_has_bulge,
    validate_sparc_input_schema,
)
from tdf_obs.validation.sparc_tau_field_solver import (
    BANNER_CALIBRATION,
    TAU_FIELD_MODELS,
    FieldDiagnostics,
    GalaxyFrame,
    MuKind,
    _metrics,
    _param_at_bound,
    _zone_residuals,
    fit_tau_field_model,
    n_params_tau_field,
    predict_tau_field,
)

BANNER_TAU_FIELD_ROBUSTNESS = (
    "SPARC TAU FIELD ROBUSTNESS AUDIT — NOT FULL OBSERVATIONAL VALIDATION"
)

BEST_TAU_FIELD_KIND: MuKind = "A"
BEST_TAU_FIELD_MODEL = "tau_field_A"

LAMBDA_BOUNDS = (0.01, 5.0)
P_BOUNDS = (0.2, 3.0)

PARAMETER_STABILITY_COLUMNS: tuple[str, ...] = (
    "model",
    "parameter",
    "n_galaxies",
    "median",
    "q25",
    "q75",
    "min",
    "max",
    "iqr",
    "boundary_hit_fraction",
    "spearman_vmax",
    "spearman_galaxy_class",
    "spearman_low_acceleration_fraction",
    "spearman_beta_over_M",
    "spearman_upsilon_disk",
)

GLOBAL_LAMBDA_COLUMNS: tuple[str, ...] = (
    "galaxy_id",
    "galaxy_class",
    "lambda_b_free",
    "bic_free_lambda",
    "lambda_b_global",
    "bic_global_lambda",
    "delta_bic_global_vs_free",
    "acceptable_delta_lt_2",
    "acceptable_delta_lt_6",
    "acceptable_delta_lt_10",
)

ML_ROBUSTNESS_COLUMNS: tuple[str, ...] = (
    "ml_regime",
    "ml_regime_label",
    "model",
    "galaxies_fitted",
    "bic_win_count",
    "median_bic",
    "median_delta_bic_vs_old_tdf",
    "median_delta_bic_vs_pseudo",
    "fraction_beats_old_tdf",
    "fraction_beats_pseudo",
)

BOUNDARY_FILTER_COLUMNS: tuple[str, ...] = (
    "filter_name",
    "n_galaxies",
    "bic_win_tau_field_A",
    "bic_win_tau_field_E",
    "bic_win_old_tdf_baseline",
    "bic_win_pseudo_isothermal",
    "bic_win_nfw",
    "median_delta_bic_tau_field_A_vs_old_tdf",
    "median_delta_bic_tau_field_E_vs_old_tdf",
    "median_delta_bic_tau_field_A_vs_pseudo",
    "median_delta_bic_tau_field_E_vs_pseudo",
    "fraction_tau_field_A_beats_old_tdf",
    "fraction_tau_field_A_beats_pseudo",
)

RESIDUAL_ROBUSTNESS_COLUMNS: tuple[str, ...] = (
    "comparison",
    "n_galaxies",
    "median_delta_inner",
    "median_delta_middle",
    "median_delta_outer",
    "n_inner_improve_outer_worse",
    "n_all_zones_improve",
    "n_pseudo_dominates_all_zones",
)

FilterName = Literal[
    "all_galaxies",
    "exclude_severe_boundary_pressure",
    "exclude_tau_field_boundary_hits",
    "clean_subset",
]


@dataclass
class TauFieldRobustnessResult:
    parameter_stability: pd.DataFrame
    global_parameter_test: pd.DataFrame
    ml_robustness: pd.DataFrame
    boundary_filtered_summary: pd.DataFrame
    residual_robustness: pd.DataFrame
    per_galaxy_parameters: pd.DataFrame = field(default_factory=pd.DataFrame)


def _ml_prior_from_regime(regime: MLRegimeConfig) -> tuple[MLPriorConfig, float | None, float | None]:
    if regime.fixed_upsilon:
        ud, ub = regime.fixed_disk, regime.fixed_bulge
        eps = 1e-4
        return (
            MLPriorConfig((ud - eps, ud + eps), (ub - eps, ub + eps)),
            ud,
            ub,
        )
    return MLPriorConfig(regime.disk_bounds, regime.bulge_bounds), None, None


def fit_tau_field_variant(
    gf: GalaxyFrame,
    kind: MuKind,
    *,
    ml_prior: MLPriorConfig = ML_STANDARD,
    fix_upsilon_disk: float | None = None,
    fix_upsilon_bulge: float | None = None,
    fix_lambda: float | None = None,
    a0: float = A0_TDF_DEFAULT,
):
    """Fit τ-field with optional fixed Υ or fixed λ_b."""
    d_lo, d_hi = ml_prior.disk_bounds
    b_lo, b_hi = ml_prior.bulge_bounds
    model_name = f"tau_field_{kind}"

    opt_lb: list[float] = []
    opt_ub: list[float] = []
    opt_x0: list[float] = []

    if fix_upsilon_disk is None:
        opt_lb.append(d_lo)
        opt_ub.append(d_hi)
        opt_x0.append(0.5)
        if gf.has_bulge:
            opt_lb.append(b_lo)
            opt_ub.append(b_hi)
            opt_x0.append(0.7)
    if fix_lambda is None:
        opt_lb.append(LAMBDA_BOUNDS[0])
        opt_ub.append(LAMBDA_BOUNDS[1])
        opt_x0.append(0.3)
    if kind in ("D", "E"):
        opt_lb.append(P_BOUNDS[0])
        opt_ub.append(P_BOUNDS[1])
        opt_x0.append(1.0)
    if kind == "E":
        rc_hi = min(50.0, max(gf.rmax * 2.0, 1.0))
        opt_lb.append(0.05)
        opt_ub.append(rc_hi)
        opt_x0.append(1.5)

    lb = np.array(opt_lb, dtype=float)
    ub = np.array(opt_ub, dtype=float)
    x0 = np.clip(np.array(opt_x0, dtype=float), lb, ub)

    n_ml = 0 if fix_upsilon_disk is not None else (2 if gf.has_bulge else 1)
    n_lam = 0 if fix_lambda is not None else 1
    n_extra = (1 if kind in ("D", "E") else 0) + (1 if kind == "E" else 0)
    n_par = n_ml + n_lam + n_extra

    def parse_p(p: np.ndarray) -> tuple[float, float, float, float, float]:
        idx = 0
        if fix_upsilon_disk is not None:
            ud = float(fix_upsilon_disk)
            ubv = float(fix_upsilon_bulge or 0.0)
        else:
            ud = float(p[idx])
            idx += 1
            if gf.has_bulge:
                ubv = float(p[idx])
                idx += 1
            else:
                ubv = 0.0
        if fix_lambda is not None:
            lam = float(fix_lambda)
        else:
            lam = float(p[idx])
            idx += 1
        pp, rc = 1.0, 1.5
        if kind in ("D", "E"):
            pp = float(p[idx])
            idx += 1
        if kind == "E":
            rc = float(p[idx])
        return ud, ubv, lam, pp, rc

    def residuals(p: np.ndarray) -> np.ndarray:
        ud, ubv, lam, pp, rc = parse_p(p)
        v, _, _, _, _ = predict_tau_field(
            gf.r, gf.v_gas, gf.v_disk, gf.v_bulge, ud, ubv, kind, lam, 1.0, a0, pp, rc,
        )
        if not np.all(np.isfinite(v)):
            return np.full_like(gf.v_obs, 1e6)
        return (v - gf.v_obs) / gf.v_err

    try:
        res = least_squares(residuals, x0, bounds=(lb, ub), max_nfev=3000)
        p_opt = res.x
        success = bool(res.success)
    except Exception:  # noqa: BLE001
        p_opt = x0
        success = False

    ud, ubv, lam, pp, rc = parse_p(p_opt)
    v_pred, _, _, _, diag = predict_tau_field(
        gf.r, gf.v_gas, gf.v_disk, gf.v_bulge, ud, ubv, kind, lam, 1.0, a0, pp, rc,
    )
    if not np.all(np.isfinite(v_pred)):
        v_pred = np.where(np.isfinite(v_pred), v_pred, gf.v_obs)

    c2, chi2_red, rmse, aic_v, bic_v = _metrics(gf.v_obs, v_pred, gf.v_err, n_par)
    from tdf_obs.validation.sparc_real_calibration import ModelFitResult

    params = {
        "upsilon_disk": ud,
        "upsilon_bulge": ubv if gf.has_bulge else np.nan,
        "lambda_b": lam,
        "gamma_tau": 1.0,
        "p": pp,
        "r_c": rc,
    }
    fr = ModelFitResult(
        galaxy_id=gf.galaxy_id,
        model=model_name,  # type: ignore[assignment]
        n_points=len(gf.r),
        n_params=n_par,
        chi2=c2,
        reduced_chi2=chi2_red,
        rmse=rmse,
        aic=aic_v,
        bic=bic_v,
        success=success,
        failure_reason="",
        params=params,
        v_pred=v_pred,
    )
    return fr, diag


def load_galaxy_frames(
    sparc_df: pd.DataFrame,
    *,
    max_galaxies: int | None = None,
) -> list[GalaxyFrame]:
    props_df = build_galaxy_properties(sparc_df)
    props_map = props_df.set_index("galaxy_id").to_dict("index")
    galaxies: list[GalaxyFrame] = []
    gids = sorted(sparc_df["galaxy_id"].unique())
    if max_galaxies is not None:
        gids = gids[:max_galaxies]
    for gid in gids:
        gdf = sparc_df[sparc_df["galaxy_id"] == gid].sort_values("r_kpc")
        prep = _prepare_galaxy_frame(gdf, 5)
        if prep is None:
            continue
        props = props_map.get(gid, {})
        galaxies.append(
            GalaxyFrame(
                galaxy_id=str(gid),
                galaxy_class=str(
                    props.get(
                        "galaxy_class",
                        classify_galaxy_by_vmax(float(props.get("vmax_obs", 0))),
                    ),
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
    return galaxies


def _spearman_safe(x: pd.Series, y: pd.Series) -> float:
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 5:
        return float("nan")
    rho, _ = stats.spearmanr(x[m], y[m])
    return float(rho)


def _class_numeric(classes: pd.Series) -> pd.Series:
    mapping = {c: i for i, c in enumerate(CLASS_ORDER)}
    return classes.map(lambda c: mapping.get(str(c), -1))


def compute_parameter_stability(
    galaxies: list[GalaxyFrame],
    *,
    step5_by_galaxy: pd.DataFrame | None = None,
    props_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Refit τ-field A–E (standard M/L) and summarize parameter distributions."""
    per_galaxy: list[dict[str, Any]] = []
    for gf in galaxies:
        for kind in ("A", "B", "C", "D", "E"):
            fr, _, _ = fit_tau_field_model(gf, kind)  # type: ignore[arg-type]
            p = fr.params
            per_galaxy.append(
                {
                    "galaxy_id": gf.galaxy_id,
                    "galaxy_class": gf.galaxy_class,
                    "model": fr.model,
                    "lambda_b": p.get("lambda_b"),
                    "gamma_tau": p.get("gamma_tau"),
                    "p_screen": p.get("p"),
                    "r_c": p.get("r_c"),
                    "upsilon_disk": p.get("upsilon_disk"),
                    "upsilon_bulge": p.get("upsilon_bulge"),
                    "bic": fr.bic,
                    "lambda_b_at_bound": _param_at_bound(
                        float(p.get("lambda_b", np.nan)), *LAMBDA_BOUNDS,
                    ),
                    "p_at_bound": _param_at_bound(float(p.get("p", 1.0)), *P_BOUNDS)
                    if kind in ("D", "E")
                    else False,
                    "upsilon_disk_at_bound": _param_at_bound(
                        float(p.get("upsilon_disk", np.nan)), *ML_STANDARD.disk_bounds,
                    ),
                },
            )

    pg = pd.DataFrame(per_galaxy)
    if props_df is None:
        raise ValueError("props_df required for parameter stability correlations")
    meta = props_df[["galaxy_id", "vmax_obs"]].copy()
    meta["galaxy_id"] = meta["galaxy_id"].astype(str)
    pg = pg.merge(meta, on="galaxy_id", how="left")
    pg["galaxy_class_num"] = _class_numeric(pg["galaxy_class"])
    pg["low_acceleration_fraction"] = np.nan
    pg["beta_over_M"] = np.nan

    if step5_by_galaxy is not None:
        s5_cols = ["galaxy_id", "beta_over_M", "upsilon_disk"]
        if "low_acceleration_fraction" in step5_by_galaxy.columns:
            s5_cols.append("low_acceleration_fraction")
        s5 = step5_by_galaxy[s5_cols].copy()
        s5["galaxy_id"] = s5["galaxy_id"].astype(str)
        s5 = s5.rename(columns={"upsilon_disk": "upsilon_disk_step5"})
        pg = pg.drop(columns=["low_acceleration_fraction", "beta_over_M"], errors="ignore")
        pg = pg.merge(s5, on="galaxy_id", how="left")

    summary_rows: list[dict[str, Any]] = []
    param_map = {
        "tau_field_A": [("lambda_b", "lambda_b")],
        "tau_field_B": [("lambda_b", "lambda_b")],
        "tau_field_C": [("lambda_b", "lambda_b")],
        "tau_field_D": [("lambda_b", "lambda_b"), ("p_screen", "p")],
        "tau_field_E": [("lambda_b", "lambda_b"), ("p_screen", "p"), ("r_c", "r_c")],
    }
    for model, specs in param_map.items():
        sub = pg[pg["model"] == model]
        for col, label in specs:
            vals = sub[col].astype(float)
            finite = vals[np.isfinite(vals)]
            if len(finite) == 0:
                continue
            bound_col = f"{col}_at_bound" if col != "p_screen" else "p_at_bound"
            bfrac = float(sub[bound_col].mean()) if bound_col in sub.columns else float("nan")
            summary_rows.append(
                {
                    "model": model,
                    "parameter": label,
                    "n_galaxies": int(len(finite)),
                    "median": float(np.median(finite)),
                    "q25": float(np.quantile(finite, 0.25)),
                    "q75": float(np.quantile(finite, 0.75)),
                    "min": float(np.min(finite)),
                    "max": float(np.max(finite)),
                    "iqr": float(np.quantile(finite, 0.75) - np.quantile(finite, 0.25)),
                    "boundary_hit_fraction": bfrac,
                    "spearman_vmax": _spearman_safe(vals, sub["vmax_obs"]),
                    "spearman_galaxy_class": _spearman_safe(vals, sub["galaxy_class_num"]),
                    "spearman_low_acceleration_fraction": _spearman_safe(
                        vals, sub["low_acceleration_fraction"],
                    ),
                    "spearman_beta_over_M": _spearman_safe(vals, sub.get("beta_over_M", pd.Series())),
                    "spearman_upsilon_disk": _spearman_safe(vals, sub["upsilon_disk"]),
                },
            )
        summary_rows.append(
            {
                "model": model,
                "parameter": "gamma_tau",
                "n_galaxies": len(sub),
                "median": 1.0,
                "q25": 1.0,
                "q75": 1.0,
                "min": 1.0,
                "max": 1.0,
                "iqr": 0.0,
                "boundary_hit_fraction": 0.0,
                "spearman_vmax": float("nan"),
                "spearman_galaxy_class": float("nan"),
                "spearman_low_acceleration_fraction": float("nan"),
                "spearman_beta_over_M": float("nan"),
                "spearman_upsilon_disk": float("nan"),
            },
        )

    return pd.DataFrame(summary_rows), pg


def run_global_lambda_test(
    galaxies: list[GalaxyFrame],
    *,
    kind: MuKind = BEST_TAU_FIELD_KIND,
    per_galaxy_lambdas: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Fix λ_b to cohort median; refit Υ only; compare ΔBIC."""
    if per_galaxy_lambdas is not None:
        sub = per_galaxy_lambdas[
            per_galaxy_lambdas["model"] == f"tau_field_{kind}"
        ]
        lam_med = float(sub["lambda_b"].median())
    else:
        lams = []
        for gf in galaxies:
            fr, _, _ = fit_tau_field_model(gf, kind)
            lams.append(float(fr.params["lambda_b"]))
        lam_med = float(np.median(lams))

    rows: list[dict[str, Any]] = []
    for gf in galaxies:
        fr_free, _, _ = fit_tau_field_model(gf, kind)
        fr_glob, _ = fit_tau_field_variant(gf, kind, fix_lambda=lam_med)
        d_bic = fr_glob.bic - fr_free.bic
        rows.append(
            {
                "galaxy_id": gf.galaxy_id,
                "galaxy_class": gf.galaxy_class,
                "lambda_b_free": float(fr_free.params["lambda_b"]),
                "bic_free_lambda": fr_free.bic,
                "lambda_b_global": lam_med,
                "bic_global_lambda": fr_glob.bic,
                "delta_bic_global_vs_free": d_bic,
                "acceptable_delta_lt_2": d_bic < 2.0,
                "acceptable_delta_lt_6": d_bic < 6.0,
                "acceptable_delta_lt_10": d_bic < 10.0,
            },
        )
    return pd.DataFrame(rows)


def run_ml_robustness(galaxies: list[GalaxyFrame]) -> pd.DataFrame:
    """Refit selected models under four M/L regimes."""
    models_to_fit = [
        ("tau_field_A", "tau_field", "A"),
        ("tau_field_E", "tau_field", "E"),
        ("old_tdf_baseline", "cored", "tdf_kessence"),
        ("nfw", "cored", "nfw"),
        ("pseudo_isothermal", "cored", "pseudo_isothermal"),
    ]
    rows: list[dict[str, Any]] = []

    for regime_id, regime in ML_REGIMES.items():
        prior, fix_ud, fix_ub = _ml_prior_from_regime(regime)
        per_galaxy_bics: dict[str, dict[str, float]] = {
            m[0]: {} for m in models_to_fit
        }

        for gf in galaxies:
            for label, kind, spec in models_to_fit:
                if kind == "tau_field":
                    if fix_ud is not None:
                        fr, _ = fit_tau_field_variant(
                            gf,
                            spec,  # type: ignore[arg-type]
                            ml_prior=prior,
                            fix_upsilon_disk=fix_ud,
                            fix_upsilon_bulge=fix_ub,
                        )
                    else:
                        fr, _ = fit_tau_field_variant(gf, spec, ml_prior=prior)  # type: ignore[arg-type]
                else:
                    fr = fit_model_cored_baseline(
                        gf.r,
                        gf.v_obs,
                        gf.v_err,
                        gf.v_gas,
                        gf.v_disk,
                        gf.v_bulge,
                        gf.has_bulge,
                        spec,  # type: ignore[arg-type]
                        prior,
                    )
                    if label == "old_tdf_baseline":
                        fr.model = "old_tdf_baseline"  # type: ignore[assignment]
                per_galaxy_bics[label][gf.galaxy_id] = fr.bic

        comp = pd.DataFrame(
            {
                "galaxy_id": [g.galaxy_id for g in galaxies],
                **{f"bic_{k}": [per_galaxy_bics[k].get(g.galaxy_id, np.inf) for g in galaxies] for k, _, _ in models_to_fit},
            },
        )
        comp["best_model"] = comp[[f"bic_{k}" for k, _, _ in models_to_fit]].idxmin(axis=1).str.replace("bic_", "")

        for label, _, _ in models_to_fit:
            d_old = comp[f"bic_{label}"] - comp["bic_old_tdf_baseline"]
            d_pseudo = comp[f"bic_{label}"] - comp["bic_pseudo_isothermal"]
            rows.append(
                {
                    "ml_regime": regime_id,
                    "ml_regime_label": regime.label,
                    "model": label,
                    "galaxies_fitted": len(comp),
                    "bic_win_count": int((comp["best_model"] == label).sum()),
                    "median_bic": float(comp[f"bic_{label}"].median()),
                    "median_delta_bic_vs_old_tdf": float(d_old.median()),
                    "median_delta_bic_vs_pseudo": float(d_pseudo.median()),
                    "fraction_beats_old_tdf": float((comp[f"bic_{label}"] < comp["bic_old_tdf_baseline"]).mean()),
                    "fraction_beats_pseudo": float((comp[f"bic_{label}"] < comp["bic_pseudo_isothermal"]).mean()),
                },
            )

    return pd.DataFrame(rows)


def _build_filter_masks(
    comparison: pd.DataFrame,
    boundary_flags: pd.DataFrame,
    *,
    step5_by_galaxy: pd.DataFrame | None = None,
) -> dict[FilterName, pd.Series]:
    gids = comparison["galaxy_id"].astype(str)
    n = len(gids)
    all_mask = pd.Series(True, index=range(n))

    tf_models = list(TAU_FIELD_MODELS)
    bf = boundary_flags[boundary_flags["model"].isin(tf_models)].copy()
    tau_hit = bf.groupby("galaxy_id")["any_boundary_hit"].any()
    tau_hit_map = gids.map(lambda g: bool(tau_hit.get(g, False)))

    # Severe: ≥2 parameter bounds on τ-field A, or NFW+Υ at bound, or old-TDF β+Υ at bound.
    a_row = boundary_flags[boundary_flags["model"] == "tau_field_A"].set_index("galaxy_id")
    nfw_row = boundary_flags[boundary_flags["model"] == "nfw"].set_index("galaxy_id")

    def _severe(g: str) -> bool:
        if g in a_row.index:
            ar = a_row.loc[g]
            n_bound = int(bool(ar.get("upsilon_disk_at_bound", False)))
            n_bound += int(bool(ar.get("upsilon_bulge_at_bound", False)))
            n_bound += int(bool(ar.get("lambda_b_at_bound", False)))
            if n_bound >= 2:
                return True
        if g in nfw_row.index:
            nr = nfw_row.loc[g]
            if bool(nr.get("upsilon_disk_at_bound", False)) and (
                bool(nr.get("v200_at_bound", False)) if "v200_at_bound" in nr else False
            ):
                return True
        if step5_by_galaxy is not None:
            s5 = step5_by_galaxy[step5_by_galaxy["galaxy_id"].astype(str) == g]
            if len(s5) and bool(s5.iloc[0].get("beta_over_M_at_bound", False)):
                if g in a_row.index and bool(a_row.loc[g].get("upsilon_disk_at_bound", False)):
                    return True
        return False

    severe = gids.map(_severe)

    def _clean(g: str) -> bool:
        if g not in a_row.index:
            return False
        ar = a_row.loc[g]
        return not bool(ar.get("any_boundary_hit", True))

    clean = gids.map(_clean)

    return {
        "all_galaxies": all_mask,
        "exclude_severe_boundary_pressure": ~severe,
        "exclude_tau_field_boundary_hits": ~tau_hit_map,
        "clean_subset": clean,
    }


def boundary_filtered_summary(
    comparison: pd.DataFrame,
    boundary_flags: pd.DataFrame,
    *,
    step5_by_galaxy: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """BIC summaries on filtered galaxy subsets using Step 6D comparison table."""
    masks = _build_filter_masks(comparison, boundary_flags, step5_by_galaxy=step5_by_galaxy)
    models = ["tau_field_A", "tau_field_E", "old_tdf_baseline", "pseudo_isothermal", "nfw"]
    rows: list[dict[str, Any]] = []

    for fname, mask in masks.items():
        sub = comparison.loc[mask.values].copy()
        if sub.empty:
            continue
        best = sub[
            [f"bic_{m}" for m in models]
        ].idxmin(axis=1).str.replace("bic_", "", regex=False)
        row: dict[str, Any] = {
            "filter_name": fname,
            "n_galaxies": len(sub),
        }
        for m in models:
            row[f"bic_win_{m}"] = int((best == m).sum())
        row["median_delta_bic_tau_field_A_vs_old_tdf"] = float(
            (sub["bic_tau_field_A"] - sub["bic_old_tdf_baseline"]).median(),
        )
        row["median_delta_bic_tau_field_E_vs_old_tdf"] = float(
            (sub["bic_tau_field_E"] - sub["bic_old_tdf_baseline"]).median(),
        )
        row["median_delta_bic_tau_field_A_vs_pseudo"] = float(
            (sub["bic_tau_field_A"] - sub["bic_pseudo_isothermal"]).median(),
        )
        row["median_delta_bic_tau_field_E_vs_pseudo"] = float(
            (sub["bic_tau_field_E"] - sub["bic_pseudo_isothermal"]).median(),
        )
        row["fraction_tau_field_A_beats_old_tdf"] = float(
            (sub["bic_tau_field_A"] < sub["bic_old_tdf_baseline"]).mean(),
        )
        row["fraction_tau_field_A_beats_pseudo"] = float(
            (sub["bic_tau_field_A"] < sub["bic_pseudo_isothermal"]).mean(),
        )
        rows.append(row)

    class_rows: list[dict[str, Any]] = []
    for fname, mask in masks.items():
        sub = comparison.loc[mask.values]
        for cls in CLASS_ORDER:
            csub = sub[sub["galaxy_class"] == cls]
            if csub.empty:
                continue
            class_rows.append(
                {
                    "filter_name": fname,
                    "galaxy_class": cls,
                    "n_galaxies": len(csub),
                    "median_delta_bic_tau_field_A_vs_old_tdf": float(
                        (csub["bic_tau_field_A"] - csub["bic_old_tdf_baseline"]).median(),
                    ),
                    "fraction_tau_field_A_beats_old_tdf": float(
                        (csub["bic_tau_field_A"] < csub["bic_old_tdf_baseline"]).mean(),
                    ),
                },
            )

    summary = pd.DataFrame(rows)
    summary.attrs["class_breakdown"] = pd.DataFrame(class_rows)
    return summary


def residual_robustness(
    galaxies: list[GalaxyFrame],
    comparison: pd.DataFrame,
) -> pd.DataFrame:
    """Zone residual comparisons via lightweight refit."""
    comparisons_spec = [
        ("tau_field_A_vs_old_tdf", "tau_field_A", "old_tdf_baseline"),
        ("tau_field_E_vs_old_tdf", "tau_field_E", "old_tdf_baseline"),
        ("tau_field_A_vs_pseudo", "tau_field_A", "pseudo_isothermal"),
        ("tau_field_E_vs_pseudo", "tau_field_E", "pseudo_isothermal"),
    ]
    zone_detail: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []

    for comp_name, m_a, m_b in comparisons_spec:
        inner_d: list[float] = []
        mid_d: list[float] = []
        outer_d: list[float] = []
        n_inner_better_outer_worse = 0
        n_all_improve = 0
        n_b_dominates = 0

        for gf in galaxies:
            if m_a.startswith("tau_field"):
                kind = m_a.split("_")[-1]  # A, E
                fr_a, _, _ = fit_tau_field_model(gf, kind)  # type: ignore[arg-type]
            else:
                continue
            if m_b == "old_tdf_baseline":
                fr_b = fit_model_cored_baseline(
                    gf.r, gf.v_obs, gf.v_err, gf.v_gas, gf.v_disk, gf.v_bulge,
                    gf.has_bulge, "tdf_kessence",
                )
                fr_b.model = "old_tdf_baseline"  # type: ignore[assignment]
            else:
                fr_b = fit_model_cored_baseline(
                    gf.r, gf.v_obs, gf.v_err, gf.v_gas, gf.v_disk, gf.v_bulge,
                    gf.has_bulge, "pseudo_isothermal",
                )

            za = _zone_residuals(gf.r, gf.v_obs, gf.v_err, fr_a.v_pred)
            zb = _zone_residuals(gf.r, gf.v_obs, gf.v_err, fr_b.v_pred)
            di = za["median_abs_residual_inner"] - zb["median_abs_residual_inner"]
            dm = za["median_abs_residual_middle"] - zb["median_abs_residual_middle"]
            do = za["median_abs_residual_outer"] - zb["median_abs_residual_outer"]
            inner_d.append(di)
            mid_d.append(dm)
            outer_d.append(do)
            if di < 0 and do > 0:
                n_inner_better_outer_worse += 1
            if di < 0 and dm < 0 and do < 0:
                n_all_improve += 1
            if (
                zb["median_abs_residual_inner"] < za["median_abs_residual_inner"]
                and zb["median_abs_residual_middle"] < za["median_abs_residual_middle"]
                and zb["median_abs_residual_outer"] < za["median_abs_residual_outer"]
            ):
                n_b_dominates += 1
            zone_detail.append(
                {
                    "galaxy_id": gf.galaxy_id,
                    "comparison": comp_name,
                    "delta_inner": di,
                    "delta_middle": dm,
                    "delta_outer": do,
                },
            )

        rows.append(
            {
                "comparison": comp_name,
                "n_galaxies": len(inner_d),
                "median_delta_inner": float(np.median(inner_d)),
                "median_delta_middle": float(np.median(mid_d)),
                "median_delta_outer": float(np.median(outer_d)),
                "n_inner_improve_outer_worse": n_inner_better_outer_worse,
                "n_all_zones_improve": n_all_improve,
                "n_pseudo_dominates_all_zones": n_b_dominates
                if "pseudo" in comp_name
                else 0,
            },
        )

    return pd.DataFrame(rows)


def mu_law_comparison_table(comparison: pd.DataFrame) -> pd.DataFrame:
    """Compare μ-laws A–E using Step 6D BIC columns."""
    rows: list[dict[str, Any]] = []
    for m in TAU_FIELD_MODELS:
        kind = m.split("_")[-1]
        n_par = n_params_tau_field(kind, False)  # type: ignore[arg-type]
        rows.append(
            {
                "model": m,
                "n_params_disk_only": n_par,
                "bic_win_count": int((comparison["best_model_by_bic"] == m).sum())
                if "best_model_by_bic" in comparison.columns
                else int((comparison[[f"bic_{x}" for x in TAU_FIELD_MODELS]].idxmin(axis=1) == f"bic_{m}").sum()),
                "median_bic": float(comparison[f"bic_{m}"].median()),
            },
        )
    df = pd.DataFrame(rows)
    for cls in CLASS_ORDER:
        sub = comparison[comparison["galaxy_class"] == cls]
        if sub.empty:
            continue
        best = sub[[f"bic_{m}" for m in TAU_FIELD_MODELS]].idxmin(axis=1)
        winner = best.str.replace("bic_", "")
        for m in TAU_FIELD_MODELS:
            df.loc[df["model"] == m, f"win_fraction_{cls}"] = float((winner == m).mean())
    return df


def run_tau_field_robustness(
    field_run: Path,
    sparc_csv: Path,
    output_dir: Path,
    *,
    calibration_run: Path | None = None,
    step5_run: Path | None = None,
    max_galaxies: int | None = None,
) -> TauFieldRobustnessResult:
    output_dir = Path(output_dir)
    tables = output_dir / "tables"
    reports = output_dir / "reports"
    figures = output_dir / "figures"
    for d in (tables, reports, figures):
        d.mkdir(parents=True, exist_ok=True)

    field_run = Path(field_run)
    comparison = pd.read_csv(field_run / "tables" / "tau_field_comparison_by_galaxy.csv")
    boundary_flags = pd.read_csv(field_run / "tables" / "tau_field_boundary_flags.csv")
    model_summary = pd.read_csv(field_run / "tables" / "tau_field_model_summary.csv")
    class_summary = pd.read_csv(field_run / "tables" / "tau_field_class_summary.csv")

    if "best_model_by_bic" not in comparison.columns:
        bcols = [c for c in comparison.columns if c.startswith("bic_")]
        comparison["best_model_by_bic"] = comparison[bcols].idxmin(axis=1).str.replace("bic_", "")

    sparc_df = pd.read_csv(sparc_csv)
    validate_sparc_input_schema(sparc_df)
    galaxies = load_galaxy_frames(sparc_df, max_galaxies=max_galaxies)
    props_df = build_galaxy_properties(sparc_df)

    step5_df = None
    if step5_run:
        step5_path = Path(step5_run) / "tables" / "tdf_parameter_by_galaxy.csv"
        if step5_path.is_file():
            step5_df = pd.read_csv(step5_path)

    param_stab, per_gal = compute_parameter_stability(galaxies, step5_by_galaxy=step5_df, props_df=props_df)
    global_test = run_global_lambda_test(galaxies, per_galaxy_lambdas=per_gal)
    ml_rob = run_ml_robustness(galaxies)
    bound_filt = boundary_filtered_summary(comparison, boundary_flags, step5_by_galaxy=step5_df)
    resid = residual_robustness(galaxies, comparison)
    mu_cmp = mu_law_comparison_table(comparison)

    param_stab.to_csv(tables / "tau_field_parameter_stability.csv", index=False)
    global_test.to_csv(tables / "tau_field_global_parameter_test.csv", index=False)
    ml_rob.to_csv(tables / "tau_field_ml_robustness.csv", index=False)
    bound_filt.to_csv(tables / "tau_field_boundary_filtered_summary.csv", index=False)
    resid.to_csv(tables / "tau_field_residual_robustness.csv", index=False)
    mu_cmp.to_csv(tables / "tau_field_mu_law_comparison.csv", index=False)

    _write_figures(
        per_gal,
        global_test,
        ml_rob,
        bound_filt,
        comparison,
        mu_cmp,
        model_summary,
        figures,
    )
    report = _write_report(
        param_stab,
        global_test,
        ml_rob,
        bound_filt,
        resid,
        mu_cmp,
        model_summary,
        class_summary,
        field_run,
    )
    (reports / "tau_field_robustness_report.md").write_text(report, encoding="utf-8")

    return TauFieldRobustnessResult(
        parameter_stability=param_stab,
        global_parameter_test=global_test,
        ml_robustness=ml_rob,
        boundary_filtered_summary=bound_filt,
        residual_robustness=resid,
        per_galaxy_parameters=per_gal,
    )


def _write_figures(
    per_gal: pd.DataFrame,
    global_test: pd.DataFrame,
    ml_rob: pd.DataFrame,
    bound_filt: pd.DataFrame,
    comparison: pd.DataFrame,
    mu_cmp: pd.DataFrame,
    model_summary: pd.DataFrame,
    figures_dir: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sub_a = per_gal[per_gal["model"] == "tau_field_A"]
    if not sub_a.empty:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(sub_a["lambda_b"].dropna(), bins=30, color="steelblue", edgecolor="white")
        ax.set_xlabel("λ_b")
        ax.set_ylabel("count")
        ax.set_title("λ_b distribution (τ-field A)")
        fig.tight_layout()
        fig.savefig(figures_dir / "lambda_distribution.png", dpi=150)
        plt.close(fig)

        plot_df = sub_a[["galaxy_id", "lambda_b", "vmax_obs"]].dropna(subset=["vmax_obs"])
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(plot_df["vmax_obs"], plot_df["lambda_b"], alpha=0.5, s=20)
        ax.set_xlabel("v_max (km/s)")
        ax.set_ylabel("λ_b")
        ax.set_title("λ_b vs v_max")
        fig.tight_layout()
        fig.savefig(figures_dir / "lambda_vs_vmax.png", dpi=150)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(global_test["delta_bic_global_vs_free"].dropna(), bins=30, color="coral", edgecolor="white")
    ax.axvline(2, color="k", ls="--", lw=0.8, label="ΔBIC=2")
    ax.axvline(6, color="gray", ls=":", lw=0.8, label="ΔBIC=6")
    ax.set_xlabel("BIC(global λ) − BIC(free λ)")
    ax.legend(fontsize=8)
    ax.set_title("Global λ_b test (τ-field A)")
    fig.tight_layout()
    fig.savefig(figures_dir / "global_lambda_delta_bic.png", dpi=150)
    plt.close(fig)

    std = ml_rob[ml_rob["ml_regime"] == "C_standard_prior"]
    if not std.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        models = std["model"].tolist()
        wins = std["bic_win_count"].tolist()
        ax.bar(range(len(models)), wins, color="teal")
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=25, ha="right", fontsize=8)
        ax.set_ylabel("BIC wins")
        ax.set_title("M/L robustness — standard prior")
        fig.tight_layout()
        fig.savefig(figures_dir / "ml_robustness_bic_wins.png", dpi=150)
        plt.close(fig)

    if not bound_filt.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        x = np.arange(len(bound_filt))
        w = 0.35
        ax.bar(x - w / 2, bound_filt["median_delta_bic_tau_field_A_vs_old_tdf"], w, label="τ-field A − old TDF")
        ax.bar(x + w / 2, bound_filt["median_delta_bic_tau_field_A_vs_pseudo"], w, label="τ-field A − pseudo")
        ax.set_xticks(x)
        ax.set_xticklabels(bound_filt["filter_name"], rotation=20, ha="right", fontsize=7)
        ax.axhline(0, color="k", lw=0.5)
        ax.legend(fontsize=7)
        ax.set_title("Boundary-filtered median ΔBIC")
        fig.tight_layout()
        fig.savefig(figures_dir / "boundary_filtered_tau_field.png", dpi=150)
        plt.close(fig)

    if not mu_cmp.empty:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(mu_cmp["model"], mu_cmp["median_bic"], color="purple", alpha=0.7)
        ax.set_ylabel("median BIC")
        ax.set_title("μ-law comparison (A–E)")
        ax.tick_params(axis="x", rotation=25)
        fig.tight_layout()
        fig.savefig(figures_dir / "mu_law_comparison.png", dpi=150)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    comps = ["tau_field_A_vs_old_tdf", "tau_field_E_vs_old_tdf", "tau_field_A_vs_pseudo", "tau_field_E_vs_pseudo"]
    resid_path = figures_dir.parent / "tables" / "tau_field_residual_robustness.csv"
    if resid_path.is_file():
        resid = pd.read_csv(resid_path)
        sub = resid[resid["comparison"].isin(comps)]
        x = np.arange(len(sub))
        w = 0.25
        for i, zone in enumerate(["inner", "middle", "outer"]):
            ax.bar(x + (i - 1) * w, sub[f"median_delta_{zone}"], w, label=zone)
        ax.set_xticks(x)
        ax.set_xticklabels(sub["comparison"], rotation=15, ha="right", fontsize=7)
        ax.axhline(0, color="k", lw=0.5)
        ax.legend(fontsize=7)
        ax.set_ylabel("Δ median |residual| (A−ref)")
        ax.set_title("Zone residual comparison")
        fig.tight_layout()
        fig.savefig(figures_dir / "residual_zone_comparison.png", dpi=150)
        plt.close(fig)


def _df_to_md(df: pd.DataFrame, max_rows: int = 20) -> str:
    sub = df.head(max_rows)
    cols = list(sub.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for _, row in sub.iterrows():
        cells = [str(row[c])[:40] for c in cols]
        lines.append("| " + " | ".join(cells) + " |")
    if len(df) > max_rows:
        lines.append(f"\n*(showing {max_rows} of {len(df)} rows)*")
    return "\n".join(lines)


def _write_report(
    param_stab: pd.DataFrame,
    global_test: pd.DataFrame,
    ml_rob: pd.DataFrame,
    bound_filt: pd.DataFrame,
    resid: pd.DataFrame,
    mu_cmp: pd.DataFrame,
    model_summary: pd.DataFrame,
    class_summary: pd.DataFrame,
    field_run: Path,
) -> str:
    n = len(global_test)
    frac2 = float(global_test["acceptable_delta_lt_2"].mean()) if n else 0.0
    frac6 = float(global_test["acceptable_delta_lt_6"].mean()) if n else 0.0
    frac10 = float(global_test["acceptable_delta_lt_10"].mean()) if n else 0.0
    lam_med = float(global_test["lambda_b_global"].iloc[0]) if n else float("nan")

    std_ml = ml_rob[ml_rob["ml_regime"] == "C_standard_prior"]
    tf_a_std = std_ml[std_ml["model"] == "tau_field_A"]
    med_d_old = float(tf_a_std["median_delta_bic_vs_old_tdf"].iloc[0]) if len(tf_a_std) else float("nan")
    med_d_pseudo = float(tf_a_std["median_delta_bic_vs_pseudo"].iloc[0]) if len(tf_a_std) else float("nan")

    all_f = bound_filt[bound_filt["filter_name"] == "all_galaxies"]
    clean_f = bound_filt[bound_filt["filter_name"] == "clean_subset"]
    med_clean_old = float(clean_f["median_delta_bic_tau_field_A_vs_old_tdf"].iloc[0]) if len(clean_f) else float("nan")
    n_clean = int(clean_f["n_galaxies"].iloc[0]) if len(clean_f) else 0

    tf_a = model_summary[model_summary["model"] == "tau_field_A"]
    tf_e = model_summary[model_summary["model"] == "tau_field_E"]
    inner_a = float(tf_a["median_abs_residual_inner"].iloc[0]) if len(tf_a) else float("nan")
    inner_e = float(tf_e["median_abs_residual_inner"].iloc[0]) if len(tf_e) else float("nan")
    inner_old = float(
        model_summary[model_summary["model"] == "old_tdf_baseline"]["median_abs_residual_inner"].iloc[0],
    ) if len(model_summary) else float("nan")

    robust_old = med_d_old < 0 and float(tf_a_std["fraction_beats_old_tdf"].iloc[0]) > 0.5 if len(tf_a_std) else False
    claim_level = 2 if robust_old else 1

    lines = [
        "# SPARC τ field robustness report (Step 6E)",
        "",
        f"## ⚠️ {BANNER_TAU_FIELD_ROBUSTNESS}",
        "",
        f"## ⚠️ {BANNER_CALIBRATION}",
        "",
        f"**Step 6D input:** `{field_run}`",
        f"**Galaxies audited:** {n}",
        "",
        "## Parameter stability (λ_b, p, r_c)",
        "",
        _df_to_md(param_stab),
        "",
        "## Global λ_b test (τ-field A)",
        "",
        f"- Global median λ_b: **{lam_med:.4f}**",
        f"- Fraction ΔBIC(global) < 2: **{frac2:.1%}**",
        f"- Fraction ΔBIC(global) < 6: **{frac6:.1%}**",
        f"- Fraction ΔBIC(global) < 10: **{frac10:.1%}**",
        "",
        "## M/L robustness (standard prior)",
        "",
        _df_to_md(
            std_ml[["model", "bic_win_count", "median_bic", "median_delta_bic_vs_old_tdf", "fraction_beats_old_tdf"]],
        ),
        "",
        "## Boundary-filtered summary",
        "",
        _df_to_md(bound_filt),
        "",
        "## Residual zone robustness",
        "",
        _df_to_md(resid),
        "",
        "## μ-law comparison",
        "",
        _df_to_md(mu_cmp),
        "",
        "## Key questions",
        "",
        "### Is τ-field improvement over old TDF robust?",
        f"Under standard M/L: median ΔBIC(τ-field A − old TDF) = **{med_d_old:.2f}**; "
        f"clean subset (n={n_clean}) median = **{med_clean_old:.2f}**.",
        "",
        "### Is τ_field_A winning mostly from lower parameter count?",
        "τ-field A has fewer parameters than D/E; BIC wins vs E trade residual gains for penalty. "
        "See μ-law table and inner residuals below.",
        "",
        "### Is τ_field_E physically better in residuals despite BIC?",
        f"Median inner |residual|: A={inner_a:.3f}, E={inner_e:.3f}, old TDF={inner_old:.3f}.",
        "",
        "### Can one global λ explain most galaxies?",
        f"**{frac6:.1%}** within ΔBIC<6 of per-galaxy λ_b.",
        "",
        "### Does τ-field survive boundary filtering?",
        "See boundary-filtered table; clean-subset metrics should be compared to all-galaxies.",
        "",
        "### Ready for v0.21.0 paper text?",
        "Field-solved τ may be cited as **exploratory numerical diagnostic** (claim level **"
        f"{claim_level}**). Not observational validation.",
        "",
        "### Allowed claim level",
        f"**{claim_level}** — phenomenological calibration diagnostic only.",
        "",
        "## Limitations",
        "",
        "- Re-fitting subsets is CPU-intensive; M/L fixed-canonical uses Υ_disk=0.5, Υ_bulge=0.7.",
        "- Global λ test fixes λ only; does not marginalize over μ-law choice.",
        "- Does not validate TDF observationally.",
    ]
    return "\n".join(lines) + "\n"

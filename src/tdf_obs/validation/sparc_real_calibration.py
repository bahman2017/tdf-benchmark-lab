"""
Phase 8A — Real SPARC rotation calibration and model comparison.

Compares baryon-only, NFW, MOND (simple μ), and TDF K-essence on parsed SPARC data.
Not observational validation; not a dark-matter replacement claim.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from tdf_obs.fitting.metrics import aic, bic, chi_square, mse, reduced_chi_square
from tdf_obs.models.dark_matter import v2_nfw_halo_only
from tdf_obs.validation.disk_kessence_rotation import (
    integrated_disk_source,
    mu_interpolation,
    solve_disk_sigma_prime,
)

BENCHMARK_MODE = "real_sparc_calibration"
BANNER_SPARC_CALIBRATION = (
    "REAL SPARC CALIBRATION BENCHMARK — NOT FULL OBSERVATIONAL VALIDATION"
)
BANNER_SPARC_CORRECTED_MOND = (
    "REAL SPARC CALIBRATION BENCHMARK — CORRECTED MOND BASELINE — "
    "NOT FULL OBSERVATIONAL VALIDATION"
)

MOND_DV_TOL = 0.5  # km/s
MOND_RATIO_TOL = 1.01
MOND_CHI2_IDENTICAL_TOL = 1e-6

BANNER_CALIBRATION = (
    "These results are calibration diagnostics for a phenomenological TDF model. "
    "They do not constitute observational validation of TDF unless all required "
    "multi-channel constraints are passed."
)

ModelName = Literal["baryon_only", "nfw", "mond", "corrected_mond", "tdf_kessence"]
MuModel = Literal["deep_mond", "simple"]

V_ERR_FLOOR = 1.0  # km/s
A0_MOND_DEFAULT = 3700.0  # (km/s)^2/kpc — standard MOND scale
A0_TDF_DEFAULT = 3700.0
BIC_COMPETITIVE_DELTA = 2.0

REQUIRED_INPUT_COLUMNS: tuple[str, ...] = (
    "galaxy_id",
    "r_kpc",
    "v_obs",
    "v_err",
    "v_gas",
    "v_disk",
    "v_bulge",
    "dataset_mode",
    "real_observational_data",
)

SUMMARY_COLUMNS: tuple[str, ...] = (
    "galaxy_id",
    "model",
    "n_points",
    "n_params",
    "chi2",
    "reduced_chi2",
    "rmse",
    "aic",
    "bic",
    "success",
    "failure_reason",
    "upsilon_disk",
    "upsilon_bulge",
    "v200",
    "r_s",
    "a0",
    "beta_over_M",
    "mu_model",
    "mond_vs_baryon_median_dv",
    "mond_vs_baryon_median_boost",
    "mond_active_flag",
)

COMPARISON_COLUMNS: tuple[str, ...] = (
    "galaxy_id",
    "n_points",
    "has_bulge",
    "best_model_by_bic",
    "bic_baryon_only",
    "bic_nfw",
    "bic_mond",
    "bic_tdf_kessence",
    "delta_bic_tdf_vs_baryon",
    "delta_bic_tdf_vs_nfw",
    "delta_bic_tdf_vs_mond",
    "tdf_beats_baryon",
    "tdf_beats_nfw",
    "tdf_beats_mond",
    "tdf_bic_competitive",
    "chi2_improve_tdf_vs_baryon",
    "warnings",
)

CORRECTED_COMPARISON_COLUMNS: tuple[str, ...] = (
    "galaxy_id",
    "n_points",
    "has_bulge",
    "best_model_by_bic",
    "bic_baryon_only",
    "bic_nfw",
    "bic_corrected_mond",
    "bic_tdf_kessence",
    "delta_bic_tdf_vs_baryon",
    "delta_bic_tdf_vs_nfw",
    "delta_bic_tdf_vs_corrected_mond",
    "tdf_beats_baryon",
    "tdf_beats_nfw",
    "tdf_beats_corrected_mond",
    "tdf_bic_competitive",
    "chi2_improve_tdf_vs_baryon",
    "mond_vs_baryon_median_dv",
    "mond_vs_baryon_median_boost",
    "mond_active_flag",
    "warnings",
)


class SparcCalibrationSchemaError(ValueError):
    """Input or output schema validation failed."""


@dataclass
class ModelFitResult:
    galaxy_id: str
    model: ModelName
    n_points: int
    n_params: int
    chi2: float
    reduced_chi2: float
    rmse: float
    aic: float
    bic: float
    success: bool
    failure_reason: str = ""
    params: dict[str, float] = field(default_factory=dict)
    v_pred: np.ndarray | None = None


@dataclass
class GalaxyComparisonRow:
    galaxy_id: str
    n_points: int
    has_bulge: bool
    best_model_by_bic: str
    bic_baryon_only: float
    bic_nfw: float
    bic_mond: float
    bic_tdf_kessence: float
    delta_bic_tdf_vs_baryon: float
    delta_bic_tdf_vs_nfw: float
    delta_bic_tdf_vs_mond: float
    tdf_beats_baryon: bool
    tdf_beats_nfw: bool
    tdf_beats_mond: bool
    tdf_bic_competitive: bool
    chi2_improve_tdf_vs_baryon: float
    warnings: str = ""
    mond_vs_baryon_median_dv: float = np.nan
    mond_vs_baryon_median_boost: float = np.nan
    mond_active_flag: bool = False


@dataclass
class SparcCalibrationRunStats:
    input_path: Path
    output_dir: Path
    mode: str
    galaxies_attempted: int = 0
    galaxies_fitted: int = 0
    galaxies_skipped: int = 0
    galaxies_failed: int = 0
    total_points: int = 0
    fit_ml: bool = True
    max_galaxies: int | None = None
    skipped_galaxies: list[str] = field(default_factory=list)
    aggregate: dict[str, Any] = field(default_factory=dict)
    corrected_mond: bool = False
    mond_active_galaxy_count: int = 0
    output_suffix: str = ""
    run_id: str = ""
    report_path: Path | None = None


def validate_sparc_input_schema(df: pd.DataFrame) -> None:
    """Validate processed SPARC CSV before calibration."""
    missing = [c for c in REQUIRED_INPUT_COLUMNS if c not in df.columns]
    if missing:
        raise SparcCalibrationSchemaError(f"missing columns: {missing}")
    if len(df) == 0:
        raise SparcCalibrationSchemaError("empty input table")
    if not df["real_observational_data"].all():
        raise SparcCalibrationSchemaError("real_observational_data must be true")
    if (df["dataset_mode"] != "real_sparc").any():
        raise SparcCalibrationSchemaError("dataset_mode must be real_sparc")
    if (df["r_kpc"] <= 0).any():
        raise SparcCalibrationSchemaError("r_kpc must be > 0")
    if (df["v_obs"] < 0).any():
        raise SparcCalibrationSchemaError("v_obs must be >= 0")


def _sq_sign_safe(v: np.ndarray) -> np.ndarray:
    return np.abs(v) * np.abs(v)


def compute_v_baryon(
    v_gas: np.ndarray,
    v_disk: np.ndarray,
    v_bulge: np.ndarray | None,
    upsilon_disk: float,
    upsilon_bulge: float,
) -> np.ndarray:
    """
    v_baryon² = v_gas² + Υ_disk v_disk² + Υ_bulge v_bulge² (sign-safe |v|v).
    """
    v_gas = np.asarray(v_gas, dtype=float)
    v_disk = np.asarray(v_disk, dtype=float)
    if v_bulge is None:
        v_bulge = np.zeros_like(v_gas)
    else:
        v_bulge = np.asarray(v_bulge, dtype=float)
    v2 = (
        _sq_sign_safe(v_gas)
        + float(upsilon_disk) * _sq_sign_safe(v_disk)
        + float(upsilon_bulge) * _sq_sign_safe(v_bulge)
    )
    return np.sqrt(np.maximum(v2, 0.0))


def galaxy_has_bulge(v_bulge: np.ndarray, threshold: float = 1.0) -> bool:
    return bool(np.any(np.abs(np.asarray(v_bulge, dtype=float)) > threshold))


def _mu_mond(x: np.ndarray) -> np.ndarray:
    x = np.maximum(np.asarray(x, dtype=float), 0.0)
    return x / (1.0 + x)


def mond_g_baryon_analytic(g_b: np.ndarray, a0: float = A0_MOND_DEFAULT) -> np.ndarray:
    """
    Analytic deep-MOND solution for μ(x)=x/(1+x):

    g_mond = 0.5 * (g_b + sqrt(g_b² + 4 g_b a0))

    Units: g in (km/s)²/kpc.
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
    a0: float = A0_MOND_DEFAULT,
    *,
    return_components: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """v_mond from analytic g_mond; enforces v_mond >= v_baryon and finite values."""
    r = np.asarray(r, dtype=float)
    v_b = compute_v_baryon(v_gas, v_disk, v_bulge, upsilon_disk, upsilon_bulge)
    g_b = np.maximum(v_b**2 / np.maximum(r, 1e-3), 0.0)
    g_m = mond_g_baryon_analytic(g_b, a0)
    g_m = np.maximum(g_m, g_b)
    v_m = np.sqrt(np.maximum(r * g_m, 0.0))
    v_m = np.maximum(v_m, v_b)
    v_m = np.where(np.isfinite(v_m), v_m, v_b)
    if return_components:
        return v_m, v_b, g_b
    return v_m


def check_mond_activity(
    r: np.ndarray,
    v_gas: np.ndarray,
    v_disk: np.ndarray,
    v_bulge: np.ndarray,
    upsilon_disk: float,
    upsilon_bulge: float,
    a0: float = A0_MOND_DEFAULT,
) -> dict[str, float | bool]:
    """MOND vs baryon activity at given Υ (post-fit diagnostic)."""
    v_m, v_b, g_b = v_mond_analytic(
        r, v_gas, v_disk, v_bulge, upsilon_disk, upsilon_bulge, a0, return_components=True,
    )
    dv = np.abs(v_m - v_b)
    median_dv = float(np.median(dv))
    low = g_b < a0
    if np.any(low):
        boost = float(np.median(v_m[low] / np.maximum(v_b[low], 1e-12)))
    else:
        boost = float("nan")
    active = median_dv >= MOND_DV_TOL
    if np.any(low) and np.isfinite(boost):
        active = active and boost >= MOND_RATIO_TOL
    return {
        "mond_vs_baryon_median_dv": median_dv,
        "mond_vs_baryon_median_boost": boost,
        "mond_active_flag": bool(active),
    }


def solve_mond_g_obs(g_b: np.ndarray, a0: float) -> np.ndarray:
    """
    Solve g_obs * μ(g_obs/a0) = g_b with μ(x)=x/(1+x) at each radius.

    Returns g_obs [ (km/s)^2/kpc ].
    """
    g_b = np.maximum(np.asarray(g_b, dtype=float), 0.0)
    a0 = max(float(a0), 1e-30)
    out = np.zeros_like(g_b)
    for i, gb in enumerate(g_b):
        if gb <= 0.0:
            out[i] = 0.0
            continue
        if gb >= 0.5 * a0:
            out[i] = gb
            continue
        lo = max(gb, 1e-12)
        hi = gb + 100.0 * a0 + 1.0
        g = lo
        for _ in range(80):
            mu = g / (1.0 + g / a0)
            f = g * mu - gb
            df = mu + g * (1.0 / a0) / (1.0 + g / a0) ** 2
            if abs(df) < 1e-30:
                break
            g_new = max(g - f / df, lo)
            if abs(g_new - g) < 1e-10 * max(g, 1.0):
                g = g_new
                break
            g = g_new
        out[i] = g
    return out


def v_mond_simple(
    r: np.ndarray,
    v_gas: np.ndarray,
    v_disk: np.ndarray,
    v_bulge: np.ndarray,
    upsilon_disk: float,
    upsilon_bulge: float,
    a0: float,
    *,
    analytic: bool = False,
) -> np.ndarray:
    """MOND circular velocity; use ``analytic=True`` for corrected simple-μ solution."""
    if analytic:
        return v_mond_analytic(r, v_gas, v_disk, v_bulge, upsilon_disk, upsilon_bulge, a0)
    r = np.asarray(r, dtype=float)
    v_b = compute_v_baryon(v_gas, v_disk, v_bulge, upsilon_disk, upsilon_bulge)
    g_b = v_b**2 / np.maximum(r, 1e-3)
    g_obs = solve_mond_g_obs(g_b, a0)
    v2 = np.maximum(r * g_obs, 0.0)
    return np.sqrt(v2)


def baryonic_source_proxy_g_b(r: np.ndarray, v_baryon: np.ndarray) -> np.ndarray:
    """g_b = v_baryon²/r as TDF conformal source proxy (calibration only)."""
    r = np.asarray(r, dtype=float)
    v_baryon = np.asarray(v_baryon, dtype=float)
    return np.maximum(v_baryon**2 / np.maximum(r, 1e-3), 0.0)


def v_tdf_kessence_disk_proxy(
    r: np.ndarray,
    v_gas: np.ndarray,
    v_disk: np.ndarray,
    v_bulge: np.ndarray,
    upsilon_disk: float,
    upsilon_bulge: float,
    beta_over_M: float,
    a0: float,
    mu_model: MuModel = "deep_mond",
) -> np.ndarray:
    """
    TDF K-essence real-data calibration proxy (Phase 7B cylindrical equation).

    R μ(|σ'|/a0) σ' = ∫₀^R (β/M) g_b(R') R' dR';  v² = v_baryon² + R a_τ.
    coupling = β/M. Not a full 3D disk Poisson solve.
    """
    r = np.asarray(r, dtype=float)
    v_b = compute_v_baryon(v_gas, v_disk, v_bulge, upsilon_disk, upsilon_bulge)
    g_b = baryonic_source_proxy_g_b(r, v_b)
    s_disk = float(beta_over_M) * g_b
    i_r = integrated_disk_source(r, s_disk)
    sigma_p = solve_disk_sigma_prime(r, i_r, max(float(a0), 1e-30), mu_model)
    coupling = float(beta_over_M)
    a_tau = coupling * sigma_p
    v2 = np.maximum(v_b**2 + r * a_tau, 0.0)
    return np.sqrt(v2)


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
    """
    v²_total = v_baryon² + v200² f(r/r_s) with f = [ln(1+x)-x/(1+x)]/x.

    Uses ``v2_nfw_halo_only`` (stable) plus fitted baryons with M/L.
    """
    r = np.asarray(r, dtype=float)
    v_b = compute_v_baryon(v_gas, v_disk, v_bulge, upsilon_disk, upsilon_bulge)
    v200 = max(float(v200), 0.0)
    r_s = max(float(r_s), 1e-3)
    v2_halo = v2_nfw_halo_only(r, v200**2, r_s)
    return np.sqrt(np.maximum(v_b**2 + v2_halo, 0.0))


def _metrics(
    v_obs: np.ndarray,
    v_pred: np.ndarray,
    v_err: np.ndarray,
    n_params: int,
) -> tuple[float, float, float, float, float, float]:
    v_err = np.maximum(np.asarray(v_err, dtype=float), V_ERR_FLOOR)
    c2 = chi_square(v_obs, v_pred, v_err)
    n = len(v_obs)
    rmse = float(np.sqrt(mse(v_obs, v_pred)))
    return (
        c2,
        reduced_chi_square(c2, n, n_params),
        rmse,
        aic(c2, n_params),
        bic(c2, n, n_params),
    )


def _fit_model(
    r: np.ndarray,
    v_obs: np.ndarray,
    v_err: np.ndarray,
    v_gas: np.ndarray,
    v_disk: np.ndarray,
    v_bulge: np.ndarray,
    has_bulge: bool,
    model: ModelName,
    *,
    a0_mond: float = A0_MOND_DEFAULT,
    a0_tdf: float = A0_TDF_DEFAULT,
    mu_model: MuModel = "deep_mond",
    fit_ml: bool = True,
) -> ModelFitResult:
    galaxy_stub = "fit"
    n = len(r)
    v_err = np.maximum(v_err, V_ERR_FLOOR)

    if model == "baryon_only":

        def predict(p: np.ndarray) -> np.ndarray:
            ud = p[0]
            ub = p[1] if has_bulge else 0.0
            return compute_v_baryon(v_gas, v_disk, v_bulge, ud, ub)

        x0 = np.array([0.5, 0.7] if has_bulge else [0.5])
        lb = np.array([0.05, 0.05] if has_bulge else [0.05])
        ub = np.array([3.0, 3.0] if has_bulge else [3.0])
        n_par = 2 if has_bulge else 1
        if not has_bulge:
            predict = lambda p: compute_v_baryon(v_gas, v_disk, v_bulge, p[0], 0.0)
    elif model == "nfw":

        def predict(p: np.ndarray) -> np.ndarray:
            if has_bulge:
                ud, ub, v200, rs = p
            else:
                ud, v200, rs = p
                ub = 0.0
            return v_nfw_total(r, v_gas, v_disk, v_bulge, ud, ub, v200, rs)

        v2000 = float(np.sqrt(max(np.mean(v_obs**2), 1.0)))
        x0 = np.array([0.5, 0.7, v2000, max(r[-1] / 3.0, 0.5)] if has_bulge else [0.5, v2000, max(r[-1] / 3.0, 0.5)])
        lb = np.array([0.05, 0.05, 1.0, 0.1] if has_bulge else [0.05, 1.0, 0.1])
        ub = np.array([3.0, 3.0, 500.0, 100.0] if has_bulge else [3.0, 500.0, 100.0])
        n_par = 4 if has_bulge else 3
    elif model in ("mond", "corrected_mond"):
        use_analytic = model == "corrected_mond"

        def predict(p: np.ndarray) -> np.ndarray:
            if has_bulge:
                ud, ub = p
            else:
                ud = p[0]
                ub = 0.0
            return v_mond_simple(
                r, v_gas, v_disk, v_bulge, ud, ub, a0_mond, analytic=use_analytic,
            )

        x0 = np.array([0.5, 0.7] if has_bulge else [0.5])
        lb = np.array([0.05, 0.05] if has_bulge else [0.05])
        ub = np.array([3.0, 3.0] if has_bulge else [3.0])
        n_par = 2 if has_bulge else 1
    elif model == "tdf_kessence":

        def predict(p: np.ndarray) -> np.ndarray:
            if has_bulge:
                ud, ub, beta = p
            else:
                ud, beta = p
                ub = 0.0
            return v_tdf_kessence_disk_proxy(
                r, v_gas, v_disk, v_bulge, ud, ub, beta, a0_tdf, mu_model,
            )

        x0 = np.array([0.5, 0.7, 1.0] if has_bulge else [0.5, 1.0])
        lb = np.array([0.05, 0.05, 1e-4] if has_bulge else [0.05, 1e-4])
        ub = np.array([3.0, 3.0, 50.0] if has_bulge else [3.0, 50.0])
        n_par = 3 if has_bulge else 2
    else:
        raise ValueError(f"unknown model: {model}")

    def residuals(p: np.ndarray) -> np.ndarray:
        vp = predict(p)
        return (vp - v_obs) / v_err

    if not fit_ml:
        p_opt = x0
        success = True
        reason = ""
    else:
        try:
            res = least_squares(
                residuals,
                x0,
                bounds=(lb, ub),
                max_nfev=4000,
                ftol=1e-8,
                xtol=1e-8,
            )
            p_opt = res.x
            success = bool(res.success and res.cost < 1e12)
            reason = "" if success else str(res.message)
        except Exception as exc:  # noqa: BLE001
            p_opt = x0
            success = False
            reason = str(exc)

    v_pred = predict(p_opt)
    if not np.all(np.isfinite(v_pred)):
        success = False
        reason = reason or "non-finite model velocities"
        v_pred = np.where(np.isfinite(v_pred), v_pred, v_obs)

    c2, chi2_red, rmse, aic_v, bic_v = _metrics(v_obs, v_pred, v_err, n_par)
    params: dict[str, float] = {"upsilon_disk": float(p_opt[0])}
    if has_bulge:
        params["upsilon_bulge"] = float(p_opt[1])
    if model == "nfw":
        idx = 2 if has_bulge else 1
        params["v200"] = float(p_opt[idx])
        params["r_s"] = float(p_opt[idx + 1])
    if model in ("mond", "corrected_mond"):
        params["a0"] = a0_mond
    if model == "tdf_kessence":
        params["beta_over_M"] = float(p_opt[-1])
        params["a0"] = a0_tdf
        params["mu_model"] = float("nan")  # categorical stored separately

    return ModelFitResult(
        galaxy_id=galaxy_stub,
        model=model,
        n_points=n,
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


def _prepare_galaxy_frame(gdf: pd.DataFrame, min_points: int) -> pd.DataFrame | None:
    gdf = gdf.sort_values("r_kpc").reset_index(drop=True)
    if len(gdf) < min_points:
        return None
    for col in ("v_obs", "v_err", "v_gas", "v_disk", "v_bulge"):
        gdf[col] = pd.to_numeric(gdf[col], errors="coerce")
    gdf = gdf.dropna(subset=["r_kpc", "v_obs", "v_err"])
    gdf = gdf[gdf["r_kpc"] > 0]
    gdf = gdf[gdf["v_obs"] >= 0]
    gdf = gdf[gdf["v_err"] > 0]
    if len(gdf) < min_points:
        return None
    return gdf


def fit_galaxy_all_models(
    galaxy_id: str,
    gdf: pd.DataFrame,
    *,
    min_points: int = 5,
    fit_ml: bool = True,
    a0_mond: float = A0_MOND_DEFAULT,
    a0_tdf: float = A0_TDF_DEFAULT,
    mu_model: MuModel = "deep_mond",
    use_corrected_mond: bool = False,
) -> tuple[list[ModelFitResult], GalaxyComparisonRow | None]:
    gdf = _prepare_galaxy_frame(gdf, min_points)
    if gdf is None:
        return [], None

    r = gdf["r_kpc"].to_numpy()
    v_obs = gdf["v_obs"].to_numpy()
    v_err = np.maximum(gdf["v_err"].to_numpy(), V_ERR_FLOOR)
    v_gas = gdf["v_gas"].to_numpy()
    v_disk = gdf["v_disk"].to_numpy()
    v_bulge = gdf["v_bulge"].to_numpy()
    has_bulge = galaxy_has_bulge(v_bulge)

    mond_model: ModelName = "corrected_mond" if use_corrected_mond else "mond"
    model_list: tuple[ModelName, ...] = (
        "baryon_only",
        mond_model,
        "nfw",
        "tdf_kessence",
    )

    results: list[ModelFitResult] = []
    for model in model_list:
        fr = _fit_model(
            r, v_obs, v_err, v_gas, v_disk, v_bulge, has_bulge, model,
            a0_mond=a0_mond, a0_tdf=a0_tdf, mu_model=mu_model, fit_ml=fit_ml,
        )
        fr.galaxy_id = galaxy_id
        results.append(fr)

    by_model = {r.model: r for r in results}
    bics = {m: by_model[m].bic for m in by_model}
    best = min(bics, key=bics.get)  # type: ignore[arg-type]

    bic_tdf = bics["tdf_kessence"]
    bic_b = bics["baryon_only"]
    bic_n = bics["nfw"]
    bic_m = bics[mond_model]

    mond_activity = check_mond_activity(
        r, v_gas, v_disk, v_bulge,
        by_model[mond_model].params.get("upsilon_disk", 0.5),
        by_model[mond_model].params.get("upsilon_bulge", 0.0),
        a0_mond,
    )

    warnings_list: list[str] = []
    if by_model["tdf_kessence"].chi2 < by_model["baryon_only"].chi2 and bic_tdf > bic_b:
        warnings_list.append("TDF lower chi2 but worse BIC vs baryon (extra parameters)")
    if (
        abs(by_model[mond_model].chi2 - by_model["baryon_only"].chi2)
        < MOND_CHI2_IDENTICAL_TOL
    ):
        warnings_list.append("MOND chi2 identical to baryon-only (check solver/units)")
    if use_corrected_mond and not mond_activity["mond_active_flag"]:
        warnings_list.append("corrected MOND inactive vs baryon at fitted Υ")

    row = GalaxyComparisonRow(
        galaxy_id=galaxy_id,
        n_points=len(r),
        has_bulge=has_bulge,
        best_model_by_bic=best,
        bic_baryon_only=bic_b,
        bic_nfw=bic_n,
        bic_mond=bic_m,
        bic_tdf_kessence=bic_tdf,
        delta_bic_tdf_vs_baryon=bic_tdf - bic_b,
        delta_bic_tdf_vs_nfw=bic_tdf - bic_n,
        delta_bic_tdf_vs_mond=bic_tdf - bic_m,
        tdf_beats_baryon=bic_tdf < bic_b,
        tdf_beats_nfw=bic_tdf < bic_n,
        tdf_beats_mond=bic_tdf < bic_m,
        tdf_bic_competitive=(bic_tdf - min(bic_b, bic_n, bic_m)) < BIC_COMPETITIVE_DELTA,
        chi2_improve_tdf_vs_baryon=by_model["baryon_only"].chi2 - by_model["tdf_kessence"].chi2,
        warnings="; ".join(warnings_list),
        mond_vs_baryon_median_dv=float(mond_activity["mond_vs_baryon_median_dv"]),
        mond_vs_baryon_median_boost=float(mond_activity["mond_vs_baryon_median_boost"]),
        mond_active_flag=bool(mond_activity["mond_active_flag"]),
    )
    return results, row


def _fit_row_to_dict(fr: ModelFitResult) -> dict[str, Any]:
    row: dict[str, Any] = {
        "galaxy_id": fr.galaxy_id,
        "model": fr.model,
        "n_points": fr.n_points,
        "n_params": fr.n_params,
        "chi2": fr.chi2,
        "reduced_chi2": fr.reduced_chi2,
        "rmse": fr.rmse,
        "aic": fr.aic,
        "bic": fr.bic,
        "success": fr.success,
        "failure_reason": fr.failure_reason,
        "upsilon_disk": fr.params.get("upsilon_disk", np.nan),
        "upsilon_bulge": fr.params.get("upsilon_bulge", np.nan),
        "v200": fr.params.get("v200", np.nan),
        "r_s": fr.params.get("r_s", np.nan),
        "a0": fr.params.get("a0", np.nan),
        "beta_over_M": fr.params.get("beta_over_M", np.nan),
        "mu_model": "deep_mond" if fr.model == "tdf_kessence" else np.nan,
        "mond_vs_baryon_median_dv": (
            fr.params.get("mond_vs_baryon_median_dv", np.nan)
            if fr.model in ("mond", "corrected_mond")
            else np.nan
        ),
        "mond_vs_baryon_median_boost": (
            fr.params.get("mond_vs_baryon_median_boost", np.nan)
            if fr.model in ("mond", "corrected_mond")
            else np.nan
        ),
        "mond_active_flag": (
            fr.params.get("mond_active_flag", np.nan)
            if fr.model in ("mond", "corrected_mond")
            else np.nan
        ),
    }
    return row


def _comparison_row_to_dict(row: GalaxyComparisonRow, *, corrected_mond: bool) -> dict[str, Any]:
    d = row.__dict__.copy()
    if corrected_mond:
        d["bic_corrected_mond"] = d.pop("bic_mond")
        d["delta_bic_tdf_vs_corrected_mond"] = d.pop("delta_bic_tdf_vs_mond")
        d["tdf_beats_corrected_mond"] = d.pop("tdf_beats_mond")
    return d


def _aggregate_stats(
    summary_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
    *,
    corrected_mond: bool = False,
) -> dict[str, Any]:
    agg: dict[str, Any] = {}
    if summary_df.empty or "model" not in summary_df.columns:
        return agg
    models = (
        ("baryon_only", "nfw", "corrected_mond", "tdf_kessence")
        if corrected_mond
        else ("baryon_only", "nfw", "mond", "tdf_kessence")
    )
    for model in models:
        sub = summary_df[(summary_df["model"] == model) & summary_df["success"]]
        if len(sub):
            agg[f"median_reduced_chi2_{model}"] = float(sub["reduced_chi2"].median())
            agg[f"median_bic_{model}"] = float(sub["bic"].median())
    if len(comparison_df):
        agg["bic_win_baryon_only"] = int((comparison_df["best_model_by_bic"] == "baryon_only").sum())
        agg["bic_win_nfw"] = int((comparison_df["best_model_by_bic"] == "nfw").sum())
        mond_key = "corrected_mond" if corrected_mond else "mond"
        agg[f"bic_win_{mond_key}"] = int((comparison_df["best_model_by_bic"] == mond_key).sum())
        agg["bic_win_tdf_kessence"] = int((comparison_df["best_model_by_bic"] == "tdf_kessence").sum())
        agg["tdf_beats_baryon_count"] = int(comparison_df["tdf_beats_baryon"].sum())
        agg["tdf_beats_nfw_count"] = int(comparison_df["tdf_beats_nfw"].sum())
        beats_mond_col = (
            "tdf_beats_corrected_mond" if corrected_mond else "tdf_beats_mond"
        )
        if beats_mond_col in comparison_df.columns:
            agg["tdf_beats_mond_count"] = int(comparison_df[beats_mond_col].sum())
        agg["tdf_bic_competitive_count"] = int(comparison_df["tdf_bic_competitive"].sum())
        if "mond_active_flag" in comparison_df.columns:
            agg["mond_active_galaxy_count"] = int(comparison_df["mond_active_flag"].sum())
    return agg


def _plot_bic_comparison(
    comparison_df: pd.DataFrame,
    path: Path,
    *,
    corrected_mond: bool = False,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mond_label = "corrected_mond" if corrected_mond else "mond"
    models = ["baryon_only", "nfw", mond_label, "tdf_kessence"]
    wins = [int((comparison_df["best_model_by_bic"] == m).sum()) for m in models]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(models, wins, color=["C0", "C1", "C2", "C3"])
    ax.set_ylabel("Galaxies with lowest BIC")
    ax.set_title("SPARC BIC wins by model (not full validation)")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_residual_distribution(
    summary_df: pd.DataFrame,
    path: Path,
    *,
    corrected_mond: bool = False,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    mond_label = "corrected_mond" if corrected_mond else "mond"
    for model, color in zip(
        ("baryon_only", "nfw", mond_label, "tdf_kessence"),
        ("C0", "C1", "C2", "C3"),
    ):
        sub = summary_df[(summary_df["model"] == model) & summary_df["success"]]
        if len(sub):
            ax.hist(sub["reduced_chi2"], bins=15, alpha=0.5, label=model, color=color)
    ax.set_xlabel("Reduced χ²")
    ax.set_ylabel("Galaxy count")
    ax.legend()
    ax.set_title("Reduced χ² distribution by model")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_tdf_bic_delta(
    comparison_df: pd.DataFrame,
    path: Path,
    *,
    column: str = "delta_bic_tdf_vs_nfw",
    title: str = "TDF vs NFW by BIC",
    xlabel: str = "ΔBIC (TDF − NFW)",
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(comparison_df[column], bins=20, alpha=0.7, color="C3")
    ax.axvline(0.0, color="k", ls="--", lw=1)
    ax.axvline(BIC_COMPETITIVE_DELTA, color="gray", ls=":", lw=1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Galaxy count")
    ax.set_title(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_example_fits(
    df: pd.DataFrame,
    fit_cache: dict[str, list[ModelFitResult]],
    examples_dir: Path,
    max_examples: int = 6,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    examples_dir.mkdir(parents=True, exist_ok=True)
    gids = list(fit_cache.keys())[:max_examples]
    for gid in gids:
        gdf = df[df["galaxy_id"] == gid].sort_values("r_kpc")
        r = gdf["r_kpc"].to_numpy()
        v_obs = gdf["v_obs"].to_numpy()
        v_err = gdf["v_err"].to_numpy()
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.errorbar(r, v_obs, yerr=v_err, fmt="o", ms=3, label="SPARC v_obs", alpha=0.7)
        styles = {
            "baryon_only": ("--", "C0"),
            "nfw": ("-", "C1"),
            "mond": ("-", "C2"),
            "corrected_mond": ("-", "C2"),
            "tdf_kessence": ("-", "C3"),
        }
        for fr in fit_cache[gid]:
            if fr.v_pred is not None and fr.success:
                ls, c = styles.get(fr.model, ("-", "k"))
                ax.plot(r, fr.v_pred, ls=ls, color=c, label=fr.model, lw=1.5)
        ax.set_xlabel("r [kpc]")
        ax.set_ylabel("v [km/s]")
        ax.set_title(f"{gid} — real SPARC (calibration proxy)")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(examples_dir / f"{gid}_rotation_fit.png", dpi=150)
        plt.close(fig)


def _build_report(
    stats: SparcCalibrationRunStats,
    agg: dict[str, Any],
    *,
    corrected_mond: bool = False,
) -> str:
    banner = BANNER_SPARC_CORRECTED_MOND if corrected_mond else BANNER_SPARC_CALIBRATION
    title = (
        "# SPARC real calibration report — corrected MOND baseline (Phase 8A.2)"
        if corrected_mond
        else "# SPARC real calibration report (Phase 8A)"
    )
    lines = [
        title,
        "",
        f"## ⚠️ {banner}",
        "",
        f"## ⚠️ {BANNER_CALIBRATION}",
        "",
        "## Dataset",
        "",
        "- `dataset_mode` = **real_sparc**",
        "- `real_observational_data` = **true**",
        f"- **Input:** `{stats.input_path}`",
        f"- **Mode:** `{stats.mode}`",
        "",
        "## Run summary",
        "",
        f"- Galaxies attempted: {stats.galaxies_attempted}",
        f"- Successfully fitted: {stats.galaxies_fitted}",
        f"- Skipped (< min points): {stats.galaxies_skipped}",
        f"- Failed: {stats.galaxies_failed}",
        f"- Total rotation points: {stats.total_points}",
        "",
        "## Aggregate model comparison",
        "",
        "| Metric | Value |",
        "| --- | --- |",
    ]
    for key, val in sorted(agg.items()):
        lines.append(f"| {key} | {val} |")

    if corrected_mond:
        n_active = agg.get("mond_active_galaxy_count", "—")
        lines.extend(
            [
                "",
                "## Corrected MOND baseline",
                "",
                "Simple MOND with μ(x)=x/(1+x) and fixed a₀ = 3700 (km/s)²/kpc.",
                "",
                "Analytic solution (g = v²/r in (km/s)²/kpc):",
                "",
                "`g_mond = 0.5 (g_b + sqrt(g_b² + 4 g_b a₀))`",
                "",
                "`v_mond² = r g_mond` with v_mond ≥ v_baryon.",
                "",
                f"- Galaxies with **mond_active_flag** = true: **{n_active}**",
                "",
                "Corrected MOND rerun provides a **fairer SPARC rotation-only calibration "
                "comparison** than the prior iterative solver that returned g_obs ≈ g_b "
                "for most radii.",
            ],
        )

    lines.extend(
        [
            "",
            "## Models compared",
            "",
            "1. **baryon_only** — Υ_disk, Υ_bulge (M/L nuisance parameters)",
            "2. **NFW** — baryons + halo with v200, r_s (stable f(r/r_s) form)",
        ],
    )
    if corrected_mond:
        lines.append(
            "3. **corrected_mond** — analytic simple μ, fixed a₀ ≈ 3700 (km/s)²/kpc",
        )
    else:
        lines.append(
            "3. **MOND simple** — μ(x)=x/(1+x), fixed a₀ ≈ 3700 (km/s)²/kpc",
        )
    lines.extend(
        [
            "4. **TDF K-essence** — cylindrical source proxy from g_b(R); deep μ; not full 3D solve",
            "",
            "## Scientific interpretation",
            "",
            "Lower BIC indicates a better penalized fit on rotation curves only. "
            "This does **not** validate TDF observationally, prove dark-matter replacement, "
            "or establish lensing/cosmological consistency.",
        ],
    )
    if corrected_mond:
        lines.extend(
            [
                "",
                "TDF vs NFW / corrected MOND: see aggregate BIC wins above. "
                "TDF may remain competitive or not depending on these corrected baselines; "
                "either outcome is a calibration diagnostic only.",
                "",
                "## Parameter bounds (warning)",
                "",
                "Υ_disk, Υ_bulge ∈ [0.05, 3]; NFW v200 ∈ [1, 500] km/s, r_s ∈ [0.1, 100] kpc; "
                "TDF β/M ∈ [1e-4, 50]. Fits hitting bounds should be interpreted cautiously "
                "(see Phase 8A.1 parameter audit).",
            ],
        )

    lines.extend(
        [
            "",
            "## Limitations",
            "",
            "- Rotation curves only; no lensing or distance-systematics propagation.",
            "- No full 3D disk Poisson solver; TDF uses Phase 7B calibration proxy.",
            "- M/L treated as free nuisance parameters per galaxy.",
            "- Does not prove TDF replaces dark matter.",
            "",
            "## Disclaimer",
            "",
            f"- {BANNER_CALIBRATION}",
            "",
        ],
    )
    return "\n".join(lines)


def run_sparc_real_calibration(
    input_path: Path,
    output_dir: Path,
    *,
    max_galaxies: int | None = 20,
    quality_min_points: int = 5,
    fit_ml: bool = True,
    mode: str = "real_sparc",
    a0_mond: float = A0_MOND_DEFAULT,
    a0_tdf: float = A0_TDF_DEFAULT,
    mu_model: MuModel = "deep_mond",
    max_example_plots: int = 6,
    corrected_mond: bool = False,
    file_suffix: str | None = None,
    run_id: str = "",
) -> tuple[pd.DataFrame, pd.DataFrame, SparcCalibrationRunStats]:
    """Run SPARC calibration; write tables, report, and figures."""
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    tables = output_dir / "tables"
    reports = output_dir / "reports"
    figures = output_dir / "figures"
    examples = figures / "sparc_examples"
    for d in (tables, reports, figures, examples):
        d.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    validate_sparc_input_schema(df)

    if file_suffix is None:
        suffix = "_corrected_mond" if corrected_mond else ""
    else:
        suffix = file_suffix
    stats = SparcCalibrationRunStats(
        input_path=input_path,
        output_dir=output_dir,
        mode=mode,
        fit_ml=fit_ml,
        max_galaxies=max_galaxies,
        total_points=len(df),
        corrected_mond=corrected_mond,
        output_suffix=suffix,
        run_id=run_id,
    )

    galaxy_ids = sorted(df["galaxy_id"].unique())
    if max_galaxies is not None:
        galaxy_ids = galaxy_ids[: int(max_galaxies)]

    stats.galaxies_attempted = len(galaxy_ids)
    summary_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    fit_cache: dict[str, list[ModelFitResult]] = {}

    for gid in galaxy_ids:
        gdf = df[df["galaxy_id"] == gid]
        if len(gdf) < quality_min_points:
            stats.galaxies_skipped += 1
            stats.skipped_galaxies.append(f"{gid}: < {quality_min_points} points")
            continue
        try:
            fits, comp = fit_galaxy_all_models(
                gid,
                gdf,
                min_points=quality_min_points,
                fit_ml=fit_ml,
                a0_mond=a0_mond,
                a0_tdf=a0_tdf,
                mu_model=mu_model,
                use_corrected_mond=corrected_mond,
            )
            if comp is None:
                stats.galaxies_skipped += 1
                continue
            fit_cache[gid] = fits
            mond_model = "corrected_mond" if corrected_mond else "mond"
            for fr in fits:
                row_dict = _fit_row_to_dict(fr)
                if fr.model == mond_model:
                    row_dict["mond_vs_baryon_median_dv"] = comp.mond_vs_baryon_median_dv
                    row_dict["mond_vs_baryon_median_boost"] = comp.mond_vs_baryon_median_boost
                    row_dict["mond_active_flag"] = comp.mond_active_flag
                summary_rows.append(row_dict)
            comparison_rows.append(
                _comparison_row_to_dict(comp, corrected_mond=corrected_mond),
            )
            stats.galaxies_fitted += 1
        except Exception as exc:  # noqa: BLE001
            stats.galaxies_failed += 1
            stats.skipped_galaxies.append(f"{gid}: {exc}")

    summary_df = pd.DataFrame(summary_rows)
    comparison_df = pd.DataFrame(comparison_rows)

    if len(summary_df):
        for col in SUMMARY_COLUMNS:
            if col not in summary_df.columns:
                summary_df[col] = np.nan
        summary_df = summary_df[list(SUMMARY_COLUMNS)]

    comp_cols = CORRECTED_COMPARISON_COLUMNS if corrected_mond else COMPARISON_COLUMNS
    if len(comparison_df):
        for col in comp_cols:
            if col not in comparison_df.columns:
                comparison_df[col] = np.nan
        comparison_df = comparison_df[list(comp_cols)]

    stats.aggregate = _aggregate_stats(
        summary_df, comparison_df, corrected_mond=corrected_mond,
    )
    stats.mond_active_galaxy_count = int(
        stats.aggregate.get("mond_active_galaxy_count", 0),
    )

    summary_path = tables / f"sparc_real_calibration_summary{suffix}.csv"
    comparison_path = tables / f"sparc_model_comparison_by_galaxy{suffix}.csv"
    summary_df.to_csv(summary_path, index=False)
    comparison_df.to_csv(comparison_path, index=False)

    if len(comparison_df):
        _plot_bic_comparison(
            comparison_df,
            figures / f"sparc_model_bic_comparison{suffix}.png",
            corrected_mond=corrected_mond,
        )
        _plot_residual_distribution(
            summary_df,
            figures / f"sparc_residual_distribution{suffix}.png",
            corrected_mond=corrected_mond,
        )
        _plot_tdf_bic_delta(
            comparison_df,
            figures / f"sparc_tdf_vs_nfw_bic_delta{suffix}.png",
        )
        mond_delta_col = (
            "delta_bic_tdf_vs_corrected_mond"
            if corrected_mond
            else "delta_bic_tdf_vs_mond"
        )
        if mond_delta_col in comparison_df.columns:
            _plot_tdf_bic_delta(
                comparison_df,
                figures / f"sparc_tdf_vs_mond_bic_delta{suffix}.png",
                column=mond_delta_col,
                title="TDF vs corrected MOND by BIC" if corrected_mond else "TDF vs MOND by BIC",
                xlabel="ΔBIC (TDF − MOND)",
            )
    if fit_cache:
        _plot_example_fits(df, fit_cache, examples, max_examples=max_example_plots)
        # combined example panel
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        n_ex = min(max_example_plots, len(fit_cache))
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        axes_flat = axes.flatten()
        for ax, gid in zip(axes_flat, list(fit_cache.keys())[:n_ex]):
            gdf = df[df["galaxy_id"] == gid].sort_values("r_kpc")
            r = gdf["r_kpc"].to_numpy()
            ax.errorbar(r, gdf["v_obs"], yerr=gdf["v_err"], fmt="o", ms=2, alpha=0.6)
            for fr in fit_cache[gid]:
                if fr.v_pred is not None:
                    ax.plot(r, fr.v_pred, lw=1.2, label=fr.model)
            ax.set_title(gid, fontsize=8)
            ax.grid(True, alpha=0.3)
        for ax in axes_flat[n_ex:]:
            ax.axis("off")
        fig.suptitle("Example SPARC rotation fits (calibration only)")
        fig.tight_layout()
        fig.savefig(figures / f"sparc_example_rotation_fits{suffix}.png", dpi=150)
        plt.close(fig)

    report_text = _build_report(stats, stats.aggregate, corrected_mond=corrected_mond)
    report_name = f"sparc_real_calibration_report{suffix}.md"
    report_path = reports / report_name
    report_path.write_text(report_text, encoding="utf-8")
    stats.report_path = report_path

    return summary_df, comparison_df, stats

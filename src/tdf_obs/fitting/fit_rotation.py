"""Single-galaxy rotation fitting: baryon-only, TDF simple, NFW simple baselines."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import curve_fit

from tdf_obs.fitting.metrics import (
    aic,
    bic,
    chi_square,
    mse,
    percent_improvement,
    reduced_chi_square,
)
from tdf_obs.io.schemas import RotationCurveData
from tdf_obs.models.dark_matter import v2_nfw_simple, v_nfw_simple
from tdf_obs.models.rotation import baryon_only_model, v2_tdf_simple, v_tdf_simple

N_PARAMS_BARYON = 0
N_PARAMS_TDF = 2
N_PARAMS_NFW = 2


@dataclass
class RotationFitResult:
    galaxy_id: str
    # TDF parameters
    tdf_B: float
    tdf_r0: float
    # NFW simple parameters
    nfw_Vh2: float
    nfw_rs: float
    # MSE
    mse_baryon: float
    mse_tdf: float
    mse_nfw: float
    # Chi-square
    chi2_baryon: float
    chi2_tdf: float
    chi2_nfw: float
    chi2_red_baryon: float
    chi2_red_tdf: float
    chi2_red_nfw: float
    # Information criteria (from chi2; n_params as documented)
    aic_baryon: float
    aic_tdf: float
    aic_nfw: float
    bic_baryon: float
    bic_tdf: float
    bic_nfw: float
    # Model selection
    best_model_by_bic: str
    tdf_vs_baryon_improvement_percent: float
    tdf_vs_nfw_improvement_percent: float
    tdf_beats_baryon_by_bic: bool
    tdf_beats_nfw_by_bic: bool
    # Success flags
    success_tdf: bool
    success_nfw: bool
    success: bool
    data_mode: str
    warnings: list[str] = field(default_factory=list)

    # Backward-compatible aliases
    @property
    def best_B(self) -> float:
        return self.tdf_B

    @property
    def best_r0(self) -> float:
        return self.tdf_r0

    @property
    def improvement_percent(self) -> float:
        return self.tdf_vs_baryon_improvement_percent


def _fit_tdf_params(
    r: np.ndarray,
    v_obs: np.ndarray,
    v_err: np.ndarray,
    v_baryon: np.ndarray,
) -> tuple[float, float, list[str]]:
    """Fit (B, r0) in v^2 space for the TDF simple model."""
    warnings: list[str] = []
    v2_obs = v_obs**2
    v2_err = np.maximum(2.0 * v_obs * v_err, 1.0)

    def model_v2(r_fit: np.ndarray, B: float, r0: float) -> np.ndarray:
        return v2_tdf_simple(r_fit, v_baryon, B, r0)

    r_outer = r[-1]
    excess = max(v2_obs[-1] - v_baryon[-1] ** 2, 1.0)
    p0 = (excess * (r_outer + 3.0) / r_outer, 3.0)
    bounds = ([0.0, 0.05], [1e5, 100.0])

    try:
        popt, _ = curve_fit(
            model_v2,
            r,
            v2_obs,
            p0=p0,
            bounds=bounds,
            sigma=v2_err,
            absolute_sigma=True,
            maxfev=20_000,
        )
        return float(popt[0]), float(popt[1]), warnings
    except Exception as exc:
        warnings.append(f"TDF curve_fit failed: {exc}")
        return float("nan"), float("nan"), warnings


def _fit_nfw_params(
    r: np.ndarray,
    v_obs: np.ndarray,
    v_err: np.ndarray,
    v_baryon: np.ndarray,
) -> tuple[float, float, list[str]]:
    """Fit (Vh2, r_s) in v^2 space for the simplified NFW-like model."""
    warnings: list[str] = []
    v2_obs = v_obs**2
    v2_err = np.maximum(2.0 * v_obs * v_err, 1.0)

    def model_v2(r_fit: np.ndarray, Vh2: float, rs: float) -> np.ndarray:
        return v2_nfw_simple(r_fit, v_baryon, Vh2, rs)

    r_outer = r[-1]
    excess = max(v2_obs[-1] - v_baryon[-1] ** 2, 1.0)
    p0 = (excess * 2.0, max(r_outer / 3.0, 0.5))
    bounds = ([0.0, 0.05], [1e6, 500.0])

    try:
        popt, _ = curve_fit(
            model_v2,
            r,
            v2_obs,
            p0=p0,
            bounds=bounds,
            sigma=v2_err,
            absolute_sigma=True,
            maxfev=20_000,
        )
        return float(popt[0]), float(popt[1]), warnings
    except Exception as exc:
        warnings.append(f"NFW curve_fit failed: {exc}")
        return float("nan"), float("nan"), warnings


def _model_metrics(
    v_obs: np.ndarray,
    v_pred: np.ndarray,
    v_err: np.ndarray,
    n_params: int,
) -> tuple[float, float, float, float, float, float]:
    """Return mse, chi2, chi2_red, aic, bic."""
    c2 = chi_square(v_obs, v_pred, v_err)
    n = len(v_obs)
    return (
        mse(v_obs, v_pred),
        c2,
        reduced_chi_square(c2, n, n_params),
        aic(c2, n_params),
        bic(c2, n, n_params),
    )


def fit_single_galaxy_rotation(data: RotationCurveData) -> RotationFitResult:
    """
    Fit and compare baryon-only (0 params), TDF simple (2), NFW simple (2).

    Model selection by BIC uses chi² as the likelihood proxy; lower MSE alone
    is not sufficient when parameter counts differ.
    """
    r = np.asarray(data.r_kpc, dtype=float)
    v_obs = np.asarray(data.v_obs, dtype=float)
    v_err = np.maximum(np.asarray(data.v_err, dtype=float), 1e-3)
    v_baryon = np.asarray(data.v_baryon, dtype=float)
    n = len(v_obs)

    warnings: list[str] = []

    v_bary = baryon_only_model(v_baryon)
    mse_b, chi2_b, chi2_red_b, aic_b, bic_b = _model_metrics(
        v_obs, v_bary, v_err, N_PARAMS_BARYON
    )

    B, r0, tdf_w = _fit_tdf_params(r, v_obs, v_err, v_baryon)
    warnings.extend(tdf_w)
    success_tdf = np.isfinite(B) and np.isfinite(r0)
    if success_tdf:
        v_tdf = v_tdf_simple(r, v_baryon, B, r0)
    else:
        warnings.append("TDF fit did not converge; TDF metrics use baryon-only prediction.")
        v_tdf = v_bary
        B, r0 = float("nan"), float("nan")

    mse_t, chi2_t, chi2_red_t, aic_t, bic_t = _model_metrics(
        v_obs, v_tdf, v_err, N_PARAMS_TDF
    )

    Vh2, rs, nfw_w = _fit_nfw_params(r, v_obs, v_err, v_baryon)
    warnings.extend(nfw_w)
    success_nfw = np.isfinite(Vh2) and np.isfinite(rs)
    if success_nfw:
        v_nfw = v_nfw_simple(r, v_baryon, Vh2, rs)
    else:
        warnings.append("NFW fit did not converge; NFW metrics use baryon-only prediction.")
        v_nfw = v_bary
        Vh2, rs = float("nan"), float("nan")

    mse_n, chi2_n, chi2_red_n, aic_n, bic_n = _model_metrics(
        v_obs, v_nfw, v_err, N_PARAMS_NFW
    )

    bic_scores = {
        "baryon_only": bic_b,
        "tdf_simple": bic_t,
        "nfw_simple": bic_n,
    }
    best_model_by_bic = min(bic_scores, key=bic_scores.get)  # type: ignore[arg-type]

    tdf_beats_baryon = bic_t < bic_b
    tdf_beats_nfw = bic_t < bic_n

    if mse_t < mse_b and not tdf_beats_baryon:
        warnings.append(
            "TDF has lower MSE than baryon-only but higher BIC (extra parameters); "
            "prefer BIC for model comparison.",
        )
    if mse_t < mse_n and not tdf_beats_nfw:
        warnings.append(
            "TDF has lower MSE than NFW but higher BIC; prefer BIC when parameter counts differ.",
        )

    data_mode = str(
        data.metadata.get("dataset_mode", data.metadata.get("data_mode", "unknown")),
    )

    return RotationFitResult(
        galaxy_id=data.galaxy_id,
        tdf_B=B,
        tdf_r0=r0,
        nfw_Vh2=Vh2,
        nfw_rs=rs,
        mse_baryon=mse_b,
        mse_tdf=mse_t,
        mse_nfw=mse_n,
        chi2_baryon=chi2_b,
        chi2_tdf=chi2_t,
        chi2_nfw=chi2_n,
        chi2_red_baryon=chi2_red_b,
        chi2_red_tdf=chi2_red_t,
        chi2_red_nfw=chi2_red_n,
        aic_baryon=aic_b,
        aic_tdf=aic_t,
        aic_nfw=aic_n,
        bic_baryon=bic_b,
        bic_tdf=bic_t,
        bic_nfw=bic_n,
        best_model_by_bic=best_model_by_bic,
        tdf_vs_baryon_improvement_percent=percent_improvement(mse_b, mse_t),
        tdf_vs_nfw_improvement_percent=percent_improvement(mse_n, mse_t),
        tdf_beats_baryon_by_bic=tdf_beats_baryon,
        tdf_beats_nfw_by_bic=tdf_beats_nfw,
        success_tdf=success_tdf,
        success_nfw=success_nfw,
        success=success_tdf or success_nfw,
        data_mode=data_mode,
        warnings=warnings,
    )

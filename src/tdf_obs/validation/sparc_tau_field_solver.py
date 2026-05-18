"""
SPARC Step 6D — radial τ field-equation solver (numerical diagnostic).

Solves cumulative form:
  R · μ(|σ'|/a₀) · σ' = ∫₀ᴿ S_b(R') R' dR',  S_b = λ_b Σ_proxy
Numerical field diagnostic only; not observational validation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy.optimize import brentq, least_squares

from tdf_obs.fitting.metrics import aic, bic, chi_square, mse, reduced_chi_square
from tdf_obs.validation.sparc_baryon_constrained_tau_law import (
    INNER_FRAC,
    OUTER_FRAC,
    fit_galaxy_baselines,
)
from tdf_obs.validation.sparc_cored_halo_baseline import ML_STANDARD, MLPriorConfig
from tdf_obs.validation.sparc_galaxy_class_analysis import (
    CLASS_ORDER,
    build_galaxy_properties,
    classify_galaxy_by_vmax,
)
from tdf_obs.validation.sparc_inverse_tau_response import R_core
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

BANNER_TAU_FIELD = (
    "SPARC TAU FIELD-EQUATION SOLVER — NUMERICAL FIELD DIAGNOSTIC, "
    "NOT FULL OBSERVATIONAL VALIDATION"
)

BANNER_CALIBRATION = (
    "These results are calibration diagnostics for a phenomenological TDF model. "
    "They do not constitute observational validation of TDF unless all required "
    "multi-channel constraints are passed."
)

R_MIN_KPC = 0.02
ACCEL_EPS = 1e-6
SIGMA_EPS = 1e-6

MuKind = Literal["A", "B", "C", "D", "E"]
TAU_FIELD_MODELS: tuple[str, ...] = (
    "tau_field_A",
    "tau_field_B",
    "tau_field_C",
    "tau_field_D",
    "tau_field_E",
)

BASELINE_LOADED: tuple[str, ...] = (
    "baryon_only",
    "corrected_mond",
    "nfw",
    "pseudo_isothermal",
    "old_tdf_baseline",
    "inverse_tdf_baryon_feature_beta",
    "best_step6c_tau_law",
)

ALL_MODELS: tuple[str, ...] = BASELINE_LOADED + TAU_FIELD_MODELS

PROFILE_COLUMNS: tuple[str, ...] = (
    "galaxy_id",
    "model",
    "r_kpc",
    "sigma_proxy",
    "cumulative_flux",
    "sigma_prime",
    "mu_eff",
    "a_tau",
    "v_baryon",
    "v_total",
    "v_obs",
)

SUMMARY_COLUMNS: tuple[str, ...] = (
    "model",
    "galaxies_fitted",
    "bic_win_count",
    "median_bic",
    "median_reduced_chi2",
    "median_n_params",
    "median_abs_residual_inner",
    "median_abs_residual_middle",
    "median_abs_residual_outer",
    "field_unstable_fraction",
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
class FieldDiagnostics:
    stable: bool
    monotonic_sigma: bool
    n_oscillations: int
    max_sigma_prime: float
    notes: str


@dataclass
class TauFieldRunResult:
    model_summary: pd.DataFrame
    comparison_by_galaxy: pd.DataFrame
    parameter_summary: pd.DataFrame
    boundary_flags: pd.DataFrame
    profiles: pd.DataFrame
    class_summary: pd.DataFrame
    fit_details: list[ModelFitResult] = field(default_factory=list)


def sigma_proxy_raw(
    r: np.ndarray,
    v_disk: np.ndarray,
    v_bulge: np.ndarray,
) -> np.ndarray:
    from tdf_obs.validation.sparc_real_calibration import _sq_sign_safe

    r_safe = np.maximum(np.asarray(r, dtype=float), R_MIN_KPC)
    raw = _sq_sign_safe(v_disk) + _sq_sign_safe(v_bulge)
    return np.maximum(raw, SIGMA_EPS) / r_safe


def mu_tau(
    y: float,
    kind: MuKind,
    *,
    r_kpc: float = 1.0,
    p: float = 1.0,
    r_c: float = 1.0,
) -> float:
    """μ(|σ'|/a₀); positive and finite."""
    y = max(float(y), 0.0)
    if kind == "A":
        return 1.0
    if kind == "B":
        return y / (1.0 + y) if y > 0 else 0.0
    if kind == "C":
        return y / np.sqrt(1.0 + y * y) if y > 0 else 0.0
    if kind == "D":
        pp = max(float(p), 0.05)
        return (y**pp) / (1.0 + y**pp) if y > 0 else 0.0
    pp = max(float(p), 0.05)
    core = float(R_core(np.array([r_kpc]), r_c)[0])
    screened = (y**pp) / (1.0 + y**pp) if y > 0 else 0.0
    return screened * core


def _mu_times_sigma(s: float, target: float, kind: MuKind, a0: float, r_kpc: float, p: float, r_c: float) -> float:
    if s <= 0:
        return -target
    y = s / max(float(a0), ACCEL_EPS)
    return mu_tau(y, kind, r_kpc=r_kpc, p=p, r_c=r_c) * s - target


def solve_sigma_prime_at_radius(
    target: float,
    kind: MuKind,
    a0: float,
    r_kpc: float,
    p: float = 1.0,
    r_c: float = 1.0,
) -> float:
    """Solve μ(y)·σ' = target for σ' ≥ 0."""
    target = float(target)
    if not np.isfinite(target) or target <= ACCEL_EPS:
        return 0.0
    if kind == "A":
        return max(target, 0.0)
    a0 = max(float(a0), ACCEL_EPS)
    hi = max(target * 50.0, a0 * 200.0, 1.0)
    try:
        return float(brentq(
            lambda s: _mu_times_sigma(s, target, kind, a0, r_kpc, p, r_c),
            0.0,
            hi,
            xtol=1e-8,
            maxiter=200,
        ))
    except ValueError:
        return max(target, 0.0)


def cumulative_flux(r: np.ndarray, s_b: np.ndarray) -> np.ndarray:
    """G(R) = ∫₀ᴿ S_b(R') R' dR'."""
    r = np.asarray(r, dtype=float)
    s_b = np.asarray(s_b, dtype=float)
    integrand = s_b * r
    g = np.zeros_like(r)
    for i in range(1, len(r)):
        g[i] = float(np.trapz(integrand[: i + 1], r[: i + 1]))
    return g


def solve_sigma_prime_profile(
    r: np.ndarray,
    sigma_proxy: np.ndarray,
    lambda_b: float,
    kind: MuKind,
    a0: float,
    p: float = 1.0,
    r_c: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (sigma_prime, cumulative_flux G)."""
    r = np.asarray(r, dtype=float)
    s_b = float(lambda_b) * np.maximum(sigma_proxy, 0.0)
    g = cumulative_flux(r, s_b)
    sigma = np.zeros_like(r)
    for i, rk in enumerate(r):
        rk = max(float(rk), R_MIN_KPC)
        target = g[i] / rk
        sigma[i] = solve_sigma_prime_at_radius(target, kind, a0, rk, p, r_c)
    return sigma, g


def field_diagnostics(sigma_prime: np.ndarray) -> FieldDiagnostics:
    s = np.asarray(sigma_prime, dtype=float)
    if not np.all(np.isfinite(s)):
        return FieldDiagnostics(False, False, 0, float("nan"), "non-finite sigma_prime")
    ds = np.diff(s)
    n_osc = int(np.sum(np.diff(np.sign(ds)) != 0)) if len(ds) > 1 else 0
    mono = bool(np.all(ds >= -1e-6) or np.all(ds <= 1e-6))
    stable = n_osc <= max(3, len(s) // 4)
    return FieldDiagnostics(
        stable=stable,
        monotonic_sigma=mono,
        n_oscillations=n_osc,
        max_sigma_prime=float(np.max(s)) if len(s) else 0.0,
        notes="ok" if stable else "oscillatory or non-monotonic field",
    )


def predict_tau_field(
    r: np.ndarray,
    v_gas: np.ndarray,
    v_disk: np.ndarray,
    v_bulge: np.ndarray,
    upsilon_disk: float,
    upsilon_bulge: float,
    kind: MuKind,
    lambda_b: float,
    gamma_tau: float,
    a0: float,
    p: float = 1.0,
    r_c: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, FieldDiagnostics]:
    """Return v_total, sigma_prime, sigma_proxy, a_tau."""
    sig = sigma_proxy_raw(r, v_disk, v_bulge)
    sigma_p, g = solve_sigma_prime_profile(r, sig, lambda_b, kind, a0, p, r_c)
    a_tau = float(gamma_tau) * sigma_p
    v2_b = v2_baryon_user(v_gas, v_disk, v_bulge, upsilon_disk, upsilon_bulge)
    v2 = v2_b + np.maximum(r, R_MIN_KPC) * np.maximum(a_tau, 0.0)
    v_tot = np.sqrt(np.maximum(v2, 0.0))
    diag = field_diagnostics(sigma_p)
    return v_tot, sigma_p, sig, a_tau, diag


def n_params_tau_field(kind: MuKind, has_bulge: bool) -> int:
    """Υ + λ_b + (p) + (r_c) + fixed γ_τ=1."""
    n = (2 if has_bulge else 1) + 1
    if kind in ("D", "E"):
        n += 1
    if kind == "E":
        n += 1
    return n


def _metrics(v_obs, v_pred, v_err, n_params) -> tuple[float, float, float, float, float]:
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


def _zone_residuals(r, v_obs, v_err, v_pred) -> dict[str, float]:
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


def fit_tau_field_model(
    gf: GalaxyFrame,
    kind: MuKind,
    ml_prior: MLPriorConfig = ML_STANDARD,
    a0: float = A0_TDF_DEFAULT,
) -> tuple[ModelFitResult, FieldDiagnostics, pd.DataFrame]:
    """Fit one τ-field model for a galaxy."""
    d_lo, d_hi = ml_prior.disk_bounds
    b_lo, b_hi = ml_prior.bulge_bounds
    model_name = f"tau_field_{kind}"

    if gf.has_bulge:
        ml_lb = np.array([d_lo, b_lo])
        ml_ub = np.array([d_hi, b_hi])
        ml_x0 = np.array([0.5, 0.7])
    else:
        ml_lb = np.array([d_lo])
        ml_ub = np.array([d_hi])
        ml_x0 = np.array([0.5])

    extra_lb = [0.01]
    extra_ub = [5.0]
    extra_x0 = [0.3]
    if kind in ("D", "E"):
        extra_lb.append(0.2)
        extra_ub.append(3.0)
        extra_x0.append(1.0)
    if kind == "E":
        extra_lb.append(0.05)
        extra_ub.append(min(50.0, max(gf.rmax * 2.0, 1.0)))
        extra_x0.append(1.5)

    lb = np.concatenate([ml_lb, extra_lb])
    ub = np.concatenate([ml_ub, extra_ub])
    x0 = np.clip(np.concatenate([ml_x0, extra_x0]), lb, ub)
    n_par = n_params_tau_field(kind, gf.has_bulge)

    def parse_p(p: np.ndarray) -> tuple[float, float, float, float, float]:
        if gf.has_bulge:
            ud, ubv, lam = float(p[0]), float(p[1]), float(p[2])
            idx = 3
        else:
            ud, ubv, lam = float(p[0]), 0.0, float(p[1])
            idx = 2
        pp, rc = 1.0, 1.5
        if kind in ("D", "E"):
            pp = float(p[idx])
            idx += 1
        if kind == "E":
            rc = float(p[idx])
        return ud, ubv, lam, pp, rc

    def predict(p: np.ndarray) -> np.ndarray:
        ud, ubv, lam, pp, rc = parse_p(p)
        v, _, _, _, _ = predict_tau_field(
            gf.r, gf.v_gas, gf.v_disk, gf.v_bulge, ud, ubv, kind, lam, 1.0, a0, pp, rc,
        )
        return v

    def residuals(p: np.ndarray) -> np.ndarray:
        pred = predict(p)
        if not np.all(np.isfinite(pred)):
            return np.full_like(gf.v_obs, 1e6)
        return (pred - gf.v_obs) / gf.v_err

    try:
        res = least_squares(residuals, x0, bounds=(lb, ub), max_nfev=4000)
        p_opt = res.x
        success = bool(res.success)
        reason = "" if success else str(res.message)
    except Exception as exc:  # noqa: BLE001
        p_opt = x0
        success = False
        reason = str(exc)

    ud, ubv, lam, pp, rc = parse_p(p_opt)
    v_pred, sigma_p, sig, a_tau, diag = predict_tau_field(
        gf.r, gf.v_gas, gf.v_disk, gf.v_bulge, ud, ubv, kind, lam, 1.0, a0, pp, rc,
    )
    if not np.all(np.isfinite(v_pred)):
        success = False
        reason = reason or "non-finite velocity"
        v_pred = np.where(np.isfinite(v_pred), v_pred, gf.v_obs)

    c2, chi2_red, rmse, aic_v, bic_v = _metrics(gf.v_obs, v_pred, gf.v_err, n_par)
    g = cumulative_flux(gf.r, lam * sig)
    mu_eff = np.zeros_like(gf.r)
    for i, rk in enumerate(gf.r):
        y = sigma_p[i] / max(a0, ACCEL_EPS)
        mu_eff[i] = mu_tau(y, kind, r_kpc=float(rk), p=pp, r_c=rc)

    prof = pd.DataFrame(
        {
            "galaxy_id": gf.galaxy_id,
            "model": model_name,
            "r_kpc": gf.r,
            "sigma_proxy": sig,
            "cumulative_flux": g,
            "sigma_prime": sigma_p,
            "mu_eff": mu_eff,
            "a_tau": a_tau,
            "v_baryon": np.sqrt(np.maximum(
                v2_baryon_user(gf.v_gas, gf.v_disk, gf.v_bulge, ud, ubv), 0.0,
            )),
            "v_total": v_pred,
            "v_obs": gf.v_obs,
        },
    )

    params: dict[str, float] = {
        "upsilon_disk": ud,
        "upsilon_bulge": ubv if gf.has_bulge else np.nan,
        "lambda_b": lam,
        "gamma_tau": 1.0,
        "a0": a0,
        "p": pp,
        "r_c": rc,
        "field_stable": float(diag.stable),
        "n_oscillations": float(diag.n_oscillations),
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
        failure_reason=reason,
        params=params,
        v_pred=v_pred,
    )
    return fr, diag, prof


def _load_prior_bics(
    step6b_path: Path | None,
    step6c_path: Path | None,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    b6 = pd.read_csv(step6b_path) if step6b_path and step6b_path.is_file() else None
    c6 = pd.read_csv(step6c_path) if step6c_path and step6c_path.is_file() else None
    return b6, c6


def _best_step6c_bic(row6c: pd.Series) -> tuple[str, float]:
    laws = ["tau_law_A", "tau_law_B", "tau_law_C", "tau_law_D"]
    bics = {law: float(row6c.get(f"bic_{law}", np.inf)) for law in laws}
    best = min(bics, key=bics.get)  # type: ignore[arg-type]
    return best, bics[best]


def build_parameter_summary() -> pd.DataFrame:
    rows = [
        {
            "model": "tau_field_A",
            "n_params_formula": "n_Υ + λ_b",
            "mu_description": "μ=1 (canonical)",
        },
        {
            "model": "tau_field_B",
            "n_params_formula": "n_Υ + λ_b",
            "mu_description": "μ=y/(1+y) deep-MOND-like",
        },
        {
            "model": "tau_field_C",
            "n_params_formula": "n_Υ + λ_b",
            "mu_description": "μ=y/√(1+y²)",
        },
        {
            "model": "tau_field_D",
            "n_params_formula": "n_Υ + λ_b + p",
            "mu_description": "μ=y^p/(1+y^p)",
        },
        {
            "model": "tau_field_E",
            "n_params_formula": "n_Υ + λ_b + p + r_c",
            "mu_description": "screened × core factor",
        },
    ]
    return pd.DataFrame(rows)


def run_tau_field_solver(
    sparc_csv: Path,
    output_dir: Path,
    *,
    inverse_design_run: Path | None = None,
    inverse_response_run: Path | None = None,
    tau_law_run: Path | None = None,
    input_run: Path | None = None,
    max_galaxies: int | None = None,
) -> TauFieldRunResult:
    output_dir = Path(output_dir)
    tables = output_dir / "tables"
    reports = output_dir / "reports"
    figures = output_dir / "figures"
    for d in (tables, reports, figures):
        d.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(sparc_csv)
    validate_sparc_input_schema(df)
    props_df = build_galaxy_properties(df)
    props_map = props_df.set_index("galaxy_id").to_dict("index")

    step6b = None
    step6c = None
    if inverse_response_run:
        step6b, _ = _load_prior_bics(
            inverse_response_run / "tables" / "inverse_tau_comparison_by_galaxy.csv",
            None,
        )
    if tau_law_run:
        _, step6c = _load_prior_bics(
            None,
            tau_law_run / "tables" / "tau_law_comparison_by_galaxy.csv",
        )

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
    profiles_list: list[pd.DataFrame] = []
    comparisons: list[dict[str, Any]] = []
    boundary_rows: list[dict[str, Any]] = []
    zone_rows: list[dict[str, Any]] = []
    unstable_count: dict[str, int] = {m: 0 for m in TAU_FIELD_MODELS}

    for gf in galaxies:
        comp: dict[str, Any] = {
            "galaxy_id": gf.galaxy_id,
            "galaxy_class": gf.galaxy_class,
            "n_points": len(gf.r),
            "has_bulge": gf.has_bulge,
        }
        base_fits = fit_galaxy_baselines(gf)
        all_fits.extend(base_fits)
        for fr in base_fits:
            comp[f"bic_{fr.model}"] = fr.bic
            comp[f"reduced_chi2_{fr.model}"] = fr.reduced_chi2
            boundary_rows.append(_boundary_row(gf.galaxy_id, fr))
            if fr.v_pred is not None:
                zone_rows.append(
                    {"galaxy_id": gf.galaxy_id, "model": fr.model, **_zone_residuals(
                        gf.r, gf.v_obs, gf.v_err, fr.v_pred,
                    )},
                )

        if step6b is not None and gf.galaxy_id in step6b["galaxy_id"].values:
            r6 = step6b[step6b["galaxy_id"] == gf.galaxy_id].iloc[0]
            comp["bic_inverse_tdf_baryon_feature_beta"] = float(
                r6["bic_inverse_tdf_baryon_feature_beta"],
            )
        else:
            comp["bic_inverse_tdf_baryon_feature_beta"] = float("nan")

        if step6c is not None and gf.galaxy_id in step6c["galaxy_id"].values:
            r6c = step6c[step6c["galaxy_id"] == gf.galaxy_id].iloc[0]
            best6c, bic6c = _best_step6c_bic(r6c)
            comp["best_step6c_tau_law"] = best6c
            comp["bic_best_step6c_tau_law"] = bic6c
        else:
            comp["best_step6c_tau_law"] = ""
            comp["bic_best_step6c_tau_law"] = float("nan")

        field_bics: dict[str, float] = {}
        for kind in ("A", "B", "C", "D", "E"):
            fr, diag, prof = fit_tau_field_model(gf, kind)  # type: ignore[arg-type]
            all_fits.append(fr)
            profiles_list.append(prof)
            mname = fr.model
            field_bics[mname] = fr.bic
            comp[f"bic_{mname}"] = fr.bic
            comp[f"reduced_chi2_{mname}"] = fr.reduced_chi2
            comp[f"field_stable_{mname}"] = diag.stable
            comp[f"monotonic_sigma_{mname}"] = diag.monotonic_sigma
            comp[f"n_oscillations_{mname}"] = diag.n_oscillations
            if not diag.stable:
                unstable_count[mname] += 1
            boundary_rows.append(_boundary_row(gf.galaxy_id, fr))
            if fr.v_pred is not None:
                zone_rows.append(
                    {"galaxy_id": gf.galaxy_id, "model": mname, **_zone_residuals(
                        gf.r, gf.v_obs, gf.v_err, fr.v_pred,
                    )},
                )

        best_field = min(field_bics, key=field_bics.get)  # type: ignore[arg-type]
        comp["best_tau_field"] = best_field
        comp["bic_best_tau_field"] = field_bics[best_field]
        comp["delta_bic_best_tau_field_vs_old_tdf"] = (
            comp["bic_best_tau_field"] - comp["bic_old_tdf_baseline"]
        )
        comp["delta_bic_best_tau_field_vs_step6b"] = (
            comp["bic_best_tau_field"] - comp.get("bic_inverse_tdf_baryon_feature_beta", np.nan)
        )
        comp["delta_bic_best_tau_field_vs_step6c"] = (
            comp["bic_best_tau_field"] - comp.get("bic_best_step6c_tau_law", np.nan)
        )
        comp["delta_bic_best_tau_field_vs_nfw"] = comp["bic_best_tau_field"] - comp["bic_nfw"]
        comp["delta_bic_best_tau_field_vs_pseudo"] = (
            comp["bic_best_tau_field"] - comp["bic_pseudo_isothermal"]
        )
        comp["best_tau_field_beats_old_tdf"] = comp["bic_best_tau_field"] < comp["bic_old_tdf_baseline"]
        comp["best_tau_field_beats_pseudo"] = (
            comp["bic_best_tau_field"] < comp["bic_pseudo_isothermal"]
        )

        all_bics = {k[4:]: float(v) for k, v in comp.items() if k.startswith("bic_") and np.isfinite(v)}
        comp["best_model_by_bic"] = min(all_bics, key=all_bics.get)  # type: ignore[arg-type]
        comparisons.append(comp)

    comparison_df = pd.DataFrame(comparisons)
    profiles = pd.concat(profiles_list, ignore_index=True) if profiles_list else pd.DataFrame(columns=PROFILE_COLUMNS)
    model_summary = _build_model_summary(comparison_df, all_fits, zone_rows, unstable_count)
    param_summary = build_parameter_summary()
    boundary_df = pd.DataFrame(boundary_rows)
    class_summary = _build_class_summary(comparison_df)

    model_summary.to_csv(tables / "tau_field_model_summary.csv", index=False)
    comparison_df.to_csv(tables / "tau_field_comparison_by_galaxy.csv", index=False)
    param_summary.to_csv(tables / "tau_field_parameter_summary.csv", index=False)
    boundary_df.to_csv(tables / "tau_field_boundary_flags.csv", index=False)
    profiles.to_csv(tables / "tau_field_profiles.csv", index=False)
    class_summary.to_csv(tables / "tau_field_class_summary.csv", index=False)

    _write_figures(comparison_df, all_fits, df, profiles, figures)
    report = _build_report(
        comparison_df, model_summary, param_summary, class_summary, unstable_count,
        sparc_csv, input_run,
    )
    (reports / "tau_field_solver_report.md").write_text(report, encoding="utf-8")

    return TauFieldRunResult(
        model_summary=model_summary,
        comparison_by_galaxy=comparison_df,
        parameter_summary=param_summary,
        boundary_flags=boundary_df,
        profiles=profiles,
        class_summary=class_summary,
        fit_details=all_fits,
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
                _param_at_bound(p.get("lambda_b", np.nan), 0.01, 5.0),
                _param_at_bound(p.get("p", np.nan), 0.2, 3.0),
                _param_at_bound(p.get("r_c", np.nan), 0.05, 50.0),
            ],
        ),
        "upsilon_disk_at_bound": _param_at_bound(p.get("upsilon_disk", np.nan), d_lo, d_hi),
        "upsilon_bulge_at_bound": _param_at_bound(
            p.get("upsilon_bulge", np.nan), *ML_STANDARD.bulge_bounds,
        ),
        "lambda_b_at_bound": _param_at_bound(p.get("lambda_b", np.nan), 0.01, 5.0),
        "p_at_bound": _param_at_bound(p.get("p", np.nan), 0.2, 3.0),
        "r_c_at_bound": _param_at_bound(p.get("r_c", np.nan), 0.05, 50.0),
    }


def _build_model_summary(
    comparison_df: pd.DataFrame,
    all_fits: list[ModelFitResult],
    zone_rows: list[dict[str, Any]],
    unstable_count: dict[str, int],
) -> pd.DataFrame:
    fit_df = pd.DataFrame(
        [
            {
                "model": fr.model,
                "bic": fr.bic,
                "chi2": fr.chi2,
                "reduced_chi2": fr.reduced_chi2,
                "n_params": fr.n_params,
            }
            for fr in all_fits
        ],
    )
    zone_df = pd.DataFrame(zone_rows) if zone_rows else pd.DataFrame()
    n_gal = len(comparison_df)
    rows: list[dict[str, Any]] = []
    for model in ALL_MODELS:
        if model == "inverse_tdf_baryon_feature_beta":
            if "bic_inverse_tdf_baryon_feature_beta" not in comparison_df.columns:
                continue
            sub_bic = comparison_df["bic_inverse_tdf_baryon_feature_beta"]
            wins = int((comparison_df["best_model_by_bic"] == model).sum())
            med_bic = float(sub_bic.median())
            med_red = float("nan")
            med_n = float("nan")
        elif model == "best_step6c_tau_law":
            if "bic_best_step6c_tau_law" not in comparison_df.columns:
                continue
            wins = 0
            med_bic = float(comparison_df["bic_best_step6c_tau_law"].median())
            med_red = float("nan")
            med_n = float("nan")
        else:
            sub = fit_df[fit_df["model"] == model]
            if sub.empty:
                continue
            wins = int((comparison_df["best_model_by_bic"] == model).sum())
            med_bic = float(sub["bic"].median())
            med_red = float(sub["reduced_chi2"].median())
            med_n = float(sub["n_params"].median())

        row: dict[str, Any] = {
            "model": model,
            "galaxies_fitted": n_gal,
            "bic_win_count": wins,
            "median_bic": med_bic,
            "median_reduced_chi2": med_red,
            "median_n_params": med_n,
        }
        if not zone_df.empty and model in zone_df["model"].values:
            zs = zone_df[zone_df["model"] == model]
            for z in ("inner", "middle", "outer"):
                col = f"median_abs_residual_{z}"
                if col in zs.columns:
                    row[col] = float(zs[col].median())
        if model in TAU_FIELD_MODELS:
            row["field_unstable_fraction"] = unstable_count.get(model, 0) / max(n_gal, 1)
            row["tau_field_vs_old_tdf_median_delta_bic"] = float(
                comparison_df["delta_bic_best_tau_field_vs_old_tdf"].median(),
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
                "fraction_best_tau_field_beats_old_tdf": float(sub["best_tau_field_beats_old_tdf"].mean()),
                "fraction_best_tau_field_beats_pseudo": float(sub["best_tau_field_beats_pseudo"].mean()),
                "median_delta_bic_vs_old_tdf": float(sub["delta_bic_best_tau_field_vs_old_tdf"].median()),
                "median_delta_bic_vs_nfw": float(sub["delta_bic_best_tau_field_vs_nfw"].median()),
            },
        )
    return pd.DataFrame(rows)


def _write_figures(
    comparison_df: pd.DataFrame,
    all_fits: list[ModelFitResult],
    sparc_df: pd.DataFrame,
    profiles: pd.DataFrame,
    figures_dir: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if comparison_df.empty:
        return

    models_plot = list(TAU_FIELD_MODELS) + ["old_tdf_baseline", "pseudo_isothermal"]
    wins = [int((comparison_df["best_model_by_bic"] == m).sum()) for m in models_plot]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(len(wins)), wins, color="steelblue")
    ax.set_xticks(range(len(wins)))
    ax.set_xticklabels([m.replace("tau_field_", "F") for m in models_plot], rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("BIC wins")
    ax.set_title("BIC wins — τ field solver vs baselines")
    fig.tight_layout()
    fig.savefig(figures_dir / "bic_wins_tau_field.png", dpi=150)
    plt.close(fig)

    for col, fname, title in [
        ("delta_bic_best_tau_field_vs_old_tdf", "tau_field_vs_old_tdf_delta_bic.png", "Best τ-field − old TDF"),
        ("delta_bic_best_tau_field_vs_pseudo", "tau_field_vs_pseudo_isothermal_delta_bic.png", "Best τ-field − pseudo"),
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
            sub["delta_bic_best_tau_field_vs_old_tdf"],
            sub["delta_bic_best_tau_field_vs_nfw"],
            alpha=0.5,
            s=18,
            label=cls,
        )
    ax.axhline(0, color="k", lw=0.5)
    ax.axvline(0, color="k", lw=0.5)
    ax.legend(fontsize=8)
    ax.set_xlabel("ΔBIC vs old TDF")
    ax.set_ylabel("ΔBIC vs NFW")
    fig.tight_layout()
    fig.savefig(figures_dir / "tau_field_by_galaxy_class.png", dpi=150)
    plt.close(fig)

    if not profiles.empty:
        sample_gid = profiles["galaxy_id"].iloc[0]
        subp = profiles[(profiles["galaxy_id"] == sample_gid) & (profiles["model"].str.startswith("tau_field"))]
        if len(subp):
            fig, ax = plt.subplots(figsize=(8, 5))
            for m in subp["model"].unique():
                s = subp[subp["model"] == m]
                ax.plot(s["r_kpc"], s["sigma_prime"], "-", label=m.replace("tau_field_", ""), alpha=0.8)
            ax.set_xlabel("r [kpc]")
            ax.set_ylabel("σ' [km/s²/kpc proxy]")
            ax.legend(fontsize=7)
            ax.set_title(f"σ' profiles — {sample_gid}")
            fig.tight_layout()
            fig.savefig(figures_dir / "sigma_prime_profiles.png", dpi=150)
            plt.close(fig)

    best_ids = comparison_df.nsmallest(3, "delta_bic_best_tau_field_vs_old_tdf")["galaxy_id"].tolist()
    fig, axes = plt.subplots(1, max(1, len(best_ids)), figsize=(12, 4))
    if len(best_ids) == 1:
        axes = [axes]
    for ax, gid in zip(axes, best_ids):
        gdf = sparc_df[sparc_df["galaxy_id"] == gid].sort_values("r_kpc")
        r = gdf["r_kpc"].to_numpy()
        ax.errorbar(r, gdf["v_obs"], yerr=gdf["v_err"], fmt="o", ms=3)
        blaw = comparison_df.loc[comparison_df["galaxy_id"] == gid, "best_tau_field"].iloc[0]
        for fr in all_fits:
            if fr.galaxy_id == gid and fr.v_pred is not None and fr.model in (blaw, "old_tdf_baseline"):
                ax.plot(r, fr.v_pred, "-", label=fr.model[:16])
        ax.set_title(str(gid), fontsize=8)
        ax.legend(fontsize=6)
    fig.suptitle("Example τ-field rotation fits")
    fig.tight_layout()
    fig.savefig(figures_dir / "example_tau_field_rotation_fits.png", dpi=150)
    plt.close(fig)

    zdf = pd.DataFrame(
        [
            {
                "model": fr.model,
                **_zone_residuals(
                    sparc_df[sparc_df["galaxy_id"] == fr.galaxy_id].sort_values("r_kpc")["r_kpc"].to_numpy(),
                    sparc_df[sparc_df["galaxy_id"] == fr.galaxy_id].sort_values("r_kpc")["v_obs"].to_numpy(),
                    sparc_df[sparc_df["galaxy_id"] == fr.galaxy_id].sort_values("r_kpc")["v_err"].to_numpy(),
                    fr.v_pred,
                ),
            }
            for fr in all_fits
            if fr.v_pred is not None and fr.model in TAU_FIELD_MODELS
        ],
    )
    if not zdf.empty:
        fig, ax = plt.subplots(figsize=(9, 5))
        data = [
            zdf.loc[zdf["model"] == m, "median_abs_residual_inner"].dropna().to_numpy()
            for m in TAU_FIELD_MODELS
        ]
        ax.boxplot(data, tick_labels=[m.replace("tau_field_", "") for m in TAU_FIELD_MODELS])
        ax.set_ylabel("inner |weighted residual|")
        fig.tight_layout()
        fig.savefig(figures_dir / "inner_outer_residual_tau_field.png", dpi=150)
        plt.close(fig)


def _build_report(
    comparison_df: pd.DataFrame,
    model_summary: pd.DataFrame,
    param_summary: pd.DataFrame,
    class_summary: pd.DataFrame,
    unstable_count: dict[str, int],
    sparc_csv: Path,
    input_run: Path | None,
) -> str:
    n = len(comparison_df)
    beats_old = float(comparison_df["best_tau_field_beats_old_tdf"].mean()) if n else 0.0
    beats_pseudo = float(comparison_df["best_tau_field_beats_pseudo"].mean()) if n else 0.0
    med_d = float(comparison_df["delta_bic_best_tau_field_vs_old_tdf"].median()) if n else float("nan")

    tf = model_summary[model_summary["model"].str.startswith("tau_field_")]
    best_field = str(tf.sort_values("bic_win_count", ascending=False).iloc[0]["model"]) if len(tf) else "tau_field_B"

    lines = [
        "# SPARC τ field-equation solver report (Step 6D)",
        "",
        f"## ⚠️ {BANNER_TAU_FIELD}",
        "",
        f"## ⚠️ {BANNER_CALIBRATION}",
        "",
        f"**SPARC data:** `{sparc_csv}`",
        f"**Calibration ref:** `{input_run or 'n/a'}`",
        f"**Galaxies:** {n}",
        "",
        "## Model summary",
        "",
        model_summary.to_string(index=False),
        "",
        "## Parameter summary",
        "",
        param_summary.to_string(index=False),
        "",
        "## Field stability",
        "",
    ]
    for m, c in unstable_count.items():
        lines.append(f"- **{m}:** {c}/{n} galaxies flagged unstable/oscillatory")
    lines.extend(
        [
            "",
            "## Key questions",
            "",
            f"### Does solving the τ field equation improve over old TDF?",
            "",
            f"Best τ-field variant beats old TDF on **{beats_old:.0%}**; median ΔBIC = **{med_d:.2f}**.",
            "",
            "### Does it improve over Step 6B/6C algebraic laws?",
            "",
            f"Median ΔBIC vs Step 6B Variant C: **{comparison_df['delta_bic_best_tau_field_vs_step6b'].median():.2f}**; "
            f"vs Step 6C best law: **{comparison_df['delta_bic_best_tau_field_vs_step6c'].median():.2f}**.",
            "",
            "### Compete with pseudo-isothermal?",
            "",
            f"Beats pseudo on **{beats_pseudo:.0%}** of galaxies.",
            "",
            "### Ready for final synthesis?",
            "",
            "Field-solved τ is a **numerical diagnostic** only. "
            + (
                "Not ready as a replacement for synthesis until BIC and stability improve."
                if med_d > 0 or beats_pseudo < 0.4
                else "May be mentioned as an exploratory field formulation alongside algebraic laws."
            ),
            "",
            "### Paper formula (candidate)",
            "",
            "```",
            "(1/R) d/dR [ R μ(|σ'|/a₀) σ' ] = λ_b Σ_proxy(R)",
            "a_τ = γ_τ σ',   v² = v_baryon² + R a_τ",
            "```",
            "",
            f"Best-performing μ variant in this run: **{best_field}**.",
            "",
            "## Limitations",
            "",
            "- Cylindrical/radial proxy; not full covariant τ dynamics.",
            "- Per-radius root solve; not a global boundary-value solver.",
            "- Does not validate TDF observationally.",
            "",
        ],
    )
    return "\n".join(lines)

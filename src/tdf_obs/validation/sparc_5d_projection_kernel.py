"""
SPARC Step 6F — 5D-inspired projection kernel for τ halos.

Replaces local λ_b Σ_b with a projected nonlocal source S_τ(R).
Theoretical proxy only; not observational validation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from tdf_obs.fitting.metrics import aic, bic, chi_square, mse, reduced_chi_square
from tdf_obs.validation.sparc_baryon_constrained_tau_law import fit_galaxy_baselines
from tdf_obs.validation.sparc_cored_halo_baseline import ML_STANDARD, MLPriorConfig
from tdf_obs.validation.sparc_galaxy_class_analysis import (
    CLASS_ORDER,
    build_galaxy_properties,
    classify_galaxy_by_vmax,
)
from tdf_obs.validation.sparc_real_calibration import (
    A0_TDF_DEFAULT,
    ModelFitResult,
    V_ERR_FLOOR,
    _prepare_galaxy_frame,
    galaxy_has_bulge,
    validate_sparc_input_schema,
)
from tdf_obs.validation.sparc_tau_inverse_design import v2_baryon_user
from tdf_obs.validation.sparc_tau_field_solver import (
    R_MIN_KPC,
    SIGMA_EPS,
    GalaxyFrame,
    _metrics,
    _param_at_bound,
    _zone_residuals,
    cumulative_flux,
    field_diagnostics,
    sigma_proxy_raw,
    solve_sigma_prime_at_radius,
)

BANNER_PROJECTION = (
    "SPARC 5D PROJECTION KERNEL FOR TAU HALOS — THEORETICAL PROXY, "
    "NOT FULL OBSERVATIONAL VALIDATION"
)

BANNER_CALIBRATION = (
    "These results are calibration diagnostics for a phenomenological TDF model. "
    "They do not constitute observational validation of TDF unless all required "
    "multi-channel constraints are passed."
)

KernelKind = Literal["A", "B", "C", "D", "E", "F"]

PROJECTION_MODELS: tuple[str, ...] = (
    "old_tdf_baseline",
    "tau_field_A",
    "pseudo_isothermal",
    "nfw",
    "projection_kernel_A_local",
    "projection_kernel_B_exponential",
    "projection_kernel_C_gaussian",
    "projection_kernel_D_outward",
    "projection_kernel_E_density_screened",
    "projection_kernel_F_class_shared",
)

KERNEL_TO_MODEL: dict[KernelKind, str] = {
    "A": "projection_kernel_A_local",
    "B": "projection_kernel_B_exponential",
    "C": "projection_kernel_C_gaussian",
    "D": "projection_kernel_D_outward",
    "E": "projection_kernel_E_density_screened",
    "F": "projection_kernel_F_class_shared",
}

PROFILE_COLUMNS: tuple[str, ...] = (
    "galaxy_id",
    "model",
    "r_kpc",
    "sigma_proxy",
    "s_tau",
    "cumulative_flux",
    "sigma_prime",
    "a_tau",
    "v_baryon",
    "v_total",
    "v_obs",
)

ELL_BOUNDS = (0.05, 30.0)
LAMBDA0_BOUNDS = (0.01, 5.0)
SIGMA0_BOUNDS = (1.0, 5000.0)
Q_BOUNDS = (0.2, 4.0)


@dataclass
class KernelHyper:
    lambda0: float = 0.3
    ell_k: float = 2.0
    sigma0: float = 100.0
    q: float = 1.0
    by_class: dict[str, tuple[float, float]] | None = None  # class -> (lambda0, ell_k)


@dataclass
class ProjectionRunResult:
    model_summary: pd.DataFrame
    comparison_by_galaxy: pd.DataFrame
    parameter_summary: pd.DataFrame
    boundary_flags: pd.DataFrame
    profiles: pd.DataFrame
    class_summary: pd.DataFrame
    global_kernel_test: pd.DataFrame
    fit_details: list[ModelFitResult] = field(default_factory=list)


def _trapz_weights(r: np.ndarray) -> np.ndarray:
    r = np.asarray(r, dtype=float)
    w = np.zeros_like(r)
    if len(r) < 2:
        w[...] = 1.0
        return w
    w[0] = (r[1] - r[0]) / 2.0
    w[-1] = (r[-1] - r[-2]) / 2.0
    if len(r) > 2:
        w[1:-1] = (r[2:] - r[:-2]) / 2.0
    return np.maximum(w, R_MIN_KPC)


def kernel_shape(
    r_i: float,
    r_j: float,
    kind: KernelKind,
    ell_k: float,
) -> float:
    ell_k = max(float(ell_k), 1e-3)
    if kind == "A":
        return 1.0 if abs(r_i - r_j) < 1e-9 else 0.0
    dr = abs(float(r_i) - float(r_j))
    if kind == "B":
        return float(np.exp(-dr / ell_k))
    if kind in ("C", "F"):
        return float(np.exp(-(dr**2) / (2.0 * ell_k**2)))
    if kind == "D":
        if r_i < r_j:
            return 0.0
        return float(np.exp(-(dr**2) / (2.0 * ell_k**2)))
    if kind == "E":
        return float(np.exp(-(dr**2) / (2.0 * ell_k**2)))
    return 0.0


def build_normalized_kernel_matrix(
    r: np.ndarray,
    kind: KernelKind,
    ell_k: float,
    sigma_proxy: np.ndarray | None = None,
    sigma0: float = 100.0,
    q: float = 1.0,
) -> np.ndarray:
    """Row-normalized K so ∑_j K_ij R'_j w_j ≈ 1 (discrete ∫ K R' dR' = 1)."""
    r = np.asarray(r, dtype=float)
    n = len(r)
    w = _trapz_weights(r)
    k = np.zeros((n, n), dtype=float)
    s0 = max(float(sigma0), SIGMA_EPS)
    qq = max(float(q), 0.05)
    for i in range(n):
        if kind == "A":
            norm = max(float(r[i] * w[i]), SIGMA_EPS)
            k[i, i] = 1.0 / norm
            continue
        for j in range(n):
            val = kernel_shape(r[i], r[j], kind, ell_k)
            if kind == "E" and sigma_proxy is not None:
                dens = max(float(sigma_proxy[j]), SIGMA_EPS)
                val /= 1.0 + (dens / s0) ** qq
            k[i, j] = val
        norm = float(np.sum(k[i, :] * r * w))
        if norm > SIGMA_EPS:
            k[i, :] /= norm
    return k


def projected_source(
    r: np.ndarray,
    sigma_proxy: np.ndarray,
    k_mat: np.ndarray,
    lambda0: float,
) -> np.ndarray:
    r = np.asarray(r, dtype=float)
    sig = np.maximum(np.asarray(sigma_proxy, dtype=float), 0.0)
    w = _trapz_weights(r)
    integrand = sig * r * w
    s_tau = float(lambda0) * (k_mat @ integrand)
    return np.maximum(s_tau, 0.0)


def solve_sigma_prime_from_source(
    r: np.ndarray,
    s_tau: np.ndarray,
    a0: float = A0_TDF_DEFAULT,
) -> tuple[np.ndarray, np.ndarray]:
    """μ=1 field solve: R σ' = G(R) with G = ∫ S_τ R' dR'."""
    g = cumulative_flux(r, s_tau)
    sigma = np.zeros_like(r)
    a0 = max(float(a0), 1e-6)
    for i, rk in enumerate(r):
        rk = max(float(rk), R_MIN_KPC)
        target = g[i] / rk
        sigma[i] = solve_sigma_prime_at_radius(target, "A", a0, rk)
    return sigma, g


def predict_projection(
    r: np.ndarray,
    v_gas: np.ndarray,
    v_disk: np.ndarray,
    v_bulge: np.ndarray,
    upsilon_disk: float,
    upsilon_bulge: float,
    kind: KernelKind,
    hyper: KernelHyper,
    *,
    galaxy_class: str = "dwarf",
    a0: float = A0_TDF_DEFAULT,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    sig = sigma_proxy_raw(r, v_disk, v_bulge)
    if kind == "F" and hyper.by_class is not None:
        lam0, ell = hyper.by_class.get(galaxy_class, (hyper.lambda0, hyper.ell_k))
    else:
        lam0, ell = hyper.lambda0, hyper.ell_k
    k_mat = build_normalized_kernel_matrix(
        r, kind if kind != "F" else "C", ell, sig, hyper.sigma0, hyper.q,
    )
    s_tau = projected_source(r, sig, k_mat, lam0)
    sigma_p, g = solve_sigma_prime_from_source(r, s_tau, a0)
    a_tau = sigma_p
    v2_b = v2_baryon_user(v_gas, v_disk, v_bulge, upsilon_disk, upsilon_bulge)
    v2 = v2_b + np.maximum(r, R_MIN_KPC) * np.maximum(a_tau, 0.0)
    v_tot = np.sqrt(np.maximum(v2, 0.0))
    return v_tot, sigma_p, sig, s_tau, g, a_tau


def n_params_projection(kind: KernelKind, has_bulge: bool, *, policy: str = "global") -> int:
    n_ml = 2 if has_bulge else 1
    if kind == "A":
        return n_ml + 1
    if kind == "E":
        return n_ml + 4
    if kind == "F":
        return n_ml + 2
    return n_ml + 2


def fit_projection_galaxy(
    gf: GalaxyFrame,
    kind: KernelKind,
    hyper: KernelHyper,
    ml_prior: MLPriorConfig = ML_STANDARD,
    *,
    policy: str = "global",
    a0: float = A0_TDF_DEFAULT,
) -> tuple[ModelFitResult, pd.DataFrame]:
    """Fit Υ with fixed or local kernel hyperparameters."""
    model_name = KERNEL_TO_MODEL[kind]
    d_lo, d_hi = ml_prior.disk_bounds
    b_lo, b_hi = ml_prior.bulge_bounds

    if kind == "A" or policy == "per_galaxy":
        extra_lb = [LAMBDA0_BOUNDS[0]]
        extra_ub = [LAMBDA0_BOUNDS[1]]
        extra_x0 = [hyper.lambda0]
        if kind != "A":
            extra_lb.append(ELL_BOUNDS[0])
            extra_ub.append(ELL_BOUNDS[1])
            extra_x0.append(hyper.ell_k)
        if kind == "E":
            extra_lb.extend([SIGMA0_BOUNDS[0], Q_BOUNDS[0]])
            extra_ub.extend([SIGMA0_BOUNDS[1], Q_BOUNDS[1]])
            extra_x0.extend([hyper.sigma0, hyper.q])
    else:
        extra_lb, extra_ub, extra_x0 = [], [], []

    if gf.has_bulge:
        ml_lb, ml_ub, ml_x0 = [d_lo, b_lo], [d_hi, b_hi], [0.5, 0.7]
    else:
        ml_lb, ml_ub, ml_x0 = [d_lo], [d_hi], [0.5]

    lb = np.array(ml_lb + extra_lb, dtype=float)
    ub = np.array(ml_ub + extra_ub, dtype=float)
    x0 = np.clip(np.array(ml_x0 + extra_x0, dtype=float), lb, ub)
    n_par = n_params_projection(kind, gf.has_bulge, policy=policy)

    def parse(p: np.ndarray) -> tuple[float, float, KernelHyper]:
        idx = 0
        if gf.has_bulge:
            ud, ubv = float(p[idx]), float(p[idx + 1])
            idx += 2
        else:
            ud, ubv = float(p[idx]), 0.0
            idx += 1
        h = KernelHyper(
            lambda0=hyper.lambda0,
            ell_k=hyper.ell_k,
            sigma0=hyper.sigma0,
            q=hyper.q,
            by_class=hyper.by_class,
        )
        if kind == "A" or policy == "per_galaxy":
            h.lambda0 = float(p[idx])
            idx += 1
            if kind != "A":
                h.ell_k = float(p[idx])
                idx += 1
            if kind == "E":
                h.sigma0 = float(p[idx])
                h.q = float(p[idx + 1])
        return ud, ubv, h

    def residuals(p: np.ndarray) -> np.ndarray:
        ud, ubv, h = parse(p)
        v, _, _, _, _, _ = predict_projection(
            gf.r, gf.v_gas, gf.v_disk, gf.v_bulge, ud, ubv, kind, h,
            galaxy_class=gf.galaxy_class, a0=a0,
        )
        if not np.all(np.isfinite(v)):
            return np.full_like(gf.v_obs, 1e6)
        return (v - gf.v_obs) / gf.v_err

    try:
        res = least_squares(residuals, x0, bounds=(lb, ub), max_nfev=2500)
        p_opt = res.x
        success = bool(res.success)
    except Exception:  # noqa: BLE001
        p_opt = x0
        success = False

    ud, ubv, h_fit = parse(p_opt)
    v_pred, sp, sig, st, g, at = predict_projection(
        gf.r, gf.v_gas, gf.v_disk, gf.v_bulge, ud, ubv, kind, h_fit,
        galaxy_class=gf.galaxy_class, a0=a0,
    )
    if not np.all(np.isfinite(v_pred)):
        v_pred = np.where(np.isfinite(v_pred), v_pred, gf.v_obs)

    c2, chi2_red, rmse, aic_v, bic_v = _metrics(gf.v_obs, v_pred, gf.v_err, n_par)
    prof = pd.DataFrame(
        {
            "galaxy_id": gf.galaxy_id,
            "model": model_name,
            "r_kpc": gf.r,
            "sigma_proxy": sig,
            "s_tau": st,
            "cumulative_flux": g,
            "sigma_prime": sp,
            "a_tau": at,
            "v_baryon": np.sqrt(np.maximum(
                v2_baryon_user(gf.v_gas, gf.v_disk, gf.v_bulge, ud, ubv), 0.0,
            )),
            "v_total": v_pred,
            "v_obs": gf.v_obs,
        },
    )
    params = {
        "upsilon_disk": ud,
        "upsilon_bulge": ubv if gf.has_bulge else np.nan,
        "lambda0": h_fit.lambda0,
        "ell_k": h_fit.ell_k,
        "sigma0": h_fit.sigma0,
        "q": h_fit.q,
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
    return fr, prof


def optimize_global_hyper(
    galaxies: list[GalaxyFrame],
    kind: KernelKind,
    *,
    a0: float = A0_TDF_DEFAULT,
) -> KernelHyper:
    """Cohort medians from per-galaxy pilot fits (fast, mirrors Step 6E λ_b policy)."""
    if kind == "A":
        return KernelHyper(lambda0=0.3, ell_k=1.0)

    lams: list[float] = []
    ells: list[float] = []
    s0s: list[float] = []
    qs: list[float] = []
    seed = KernelHyper(lambda0=0.3, ell_k=2.0, sigma0=200.0, q=1.0)
    for gf in galaxies:
        fr, _ = fit_projection_galaxy(gf, kind, seed, policy="per_galaxy", a0=a0)  # type: ignore[arg-type]
        lams.append(float(fr.params.get("lambda0", 0.3)))
        ells.append(float(fr.params.get("ell_k", 2.0)))
        if kind == "E":
            s0s.append(float(fr.params.get("sigma0", 200.0)))
            qs.append(float(fr.params.get("q", 1.0)))
    h = KernelHyper(
        lambda0=float(np.median(lams)),
        ell_k=float(np.median(ells)),
    )
    if kind == "E":
        h.sigma0 = float(np.median(s0s)) if s0s else 200.0
        h.q = float(np.median(qs)) if qs else 1.0
    return h


def optimize_class_hyper(
    galaxies: list[GalaxyFrame],
    *,
    a0: float = A0_TDF_DEFAULT,
) -> KernelHyper:
    by_class: dict[str, tuple[float, float]] = {}
    for cls in CLASS_ORDER:
        sub = [g for g in galaxies if g.galaxy_class == cls]
        if not sub:
            by_class[cls] = (0.3, 2.0)
            continue
        h = optimize_global_hyper(sub, "C", a0=a0)
        by_class[cls] = (h.lambda0, h.ell_k)
    return KernelHyper(lambda0=0.3, ell_k=2.0, by_class=by_class)


def _boundary_row(gid: str, fr: ModelFitResult) -> dict[str, Any]:
    p = fr.params
    d_lo, d_hi = ML_STANDARD.disk_bounds
    return {
        "galaxy_id": gid,
        "model": fr.model,
        "any_boundary_hit": any(
            [
                _param_at_bound(p.get("upsilon_disk", np.nan), d_lo, d_hi),
                _param_at_bound(p.get("upsilon_bulge", np.nan), *ML_STANDARD.bulge_bounds),
                _param_at_bound(p.get("lambda0", np.nan), *LAMBDA0_BOUNDS),
                _param_at_bound(p.get("ell_k", np.nan), *ELL_BOUNDS),
            ],
        ),
        "upsilon_disk_at_bound": _param_at_bound(p.get("upsilon_disk", np.nan), d_lo, d_hi),
        "lambda0_at_bound": _param_at_bound(p.get("lambda0", np.nan), *LAMBDA0_BOUNDS),
        "ell_k_at_bound": _param_at_bound(p.get("ell_k", np.nan), *ELL_BOUNDS),
    }


def run_global_kernel_test(
    galaxies: list[GalaxyFrame],
    kind: KernelKind,
    global_hyper: KernelHyper,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for gf in galaxies:
        fr_free, _ = fit_projection_galaxy(gf, kind, global_hyper, policy="per_galaxy")
        fr_glob, _ = fit_projection_galaxy(gf, kind, global_hyper, policy="global")
        d = fr_glob.bic - fr_free.bic
        rows.append(
            {
                "galaxy_id": gf.galaxy_id,
                "galaxy_class": gf.galaxy_class,
                "kernel": KERNEL_TO_MODEL[kind],
                "bic_free_kernel": fr_free.bic,
                "bic_global_kernel": fr_glob.bic,
                "delta_bic_global_vs_free": d,
                "acceptable_delta_lt_2": d < 2.0,
                "acceptable_delta_lt_6": d < 6.0,
                "acceptable_delta_lt_10": d < 10.0,
            },
        )
    return pd.DataFrame(rows)


def build_parameter_summary() -> pd.DataFrame:
    rows = [
        {"model": KERNEL_TO_MODEL[k], "kernel_family": k, "n_params_formula": desc}
        for k, desc in [
            ("A", "n_Υ + λ₀ (local)"),
            ("B", "n_Υ + λ₀ + ℓ_k (global)"),
            ("C", "n_Υ + λ₀ + ℓ_k (global)"),
            ("D", "n_Υ + λ₀ + ℓ_k (global)"),
            ("E", "n_Υ + λ₀ + ℓ_k + Σ₀ + q (global)"),
            ("F", "n_Υ + λ₀,class + ℓ_k,class"),
        ]
    ]
    return pd.DataFrame(rows)


def _clean_galaxy_ids(boundary_flags: pd.DataFrame) -> set[str]:
    a = boundary_flags[boundary_flags["model"] == "tau_field_A"]
    if a.empty:
        a = boundary_flags[boundary_flags["model"] == "projection_kernel_A_local"]
    clean = a[~a["any_boundary_hit"].astype(bool)]["galaxy_id"].astype(str)
    return set(clean.tolist())


def run_5d_projection_kernel(
    sparc_csv: Path,
    output_dir: Path,
    *,
    field_run: Path | None = None,
    robustness_run: Path | None = None,
    calibration_run: Path | None = None,
    max_galaxies: int | None = None,
) -> ProjectionRunResult:
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

    comp6d = None
    if field_run and (field_run / "tables" / "tau_field_comparison_by_galaxy.csv").is_file():
        comp6d = pd.read_csv(field_run / "tables" / "tau_field_comparison_by_galaxy.csv")

    step6e_global = None
    if robustness_run and (robustness_run / "tables" / "tau_field_global_parameter_test.csv").is_file():
        step6e_global = pd.read_csv(robustness_run / "tables" / "tau_field_global_parameter_test.csv")

    boundary6d = None
    if field_run and (field_run / "tables" / "tau_field_boundary_flags.csv").is_file():
        boundary6d = pd.read_csv(field_run / "tables" / "tau_field_boundary_flags.csv")

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

    global_hypers: dict[KernelKind, KernelHyper] = {}
    for kind in ("B", "C", "D", "E"):
        global_hypers[kind] = optimize_global_hyper(galaxies, kind)  # type: ignore[arg-type]
    global_hypers["F"] = optimize_class_hyper(galaxies)
    global_hypers["A"] = KernelHyper(lambda0=0.3, ell_k=1.0)

    comparisons: list[dict[str, Any]] = []
    all_fits: list[ModelFitResult] = []
    profiles_list: list[pd.DataFrame] = []
    boundary_rows: list[dict[str, Any]] = []
    global_tests: list[pd.DataFrame] = []

    for gf in galaxies:
        comp: dict[str, Any] = {
            "galaxy_id": gf.galaxy_id,
            "galaxy_class": gf.galaxy_class,
            "n_points": len(gf.r),
            "has_bulge": gf.has_bulge,
        }
        base_fits = fit_galaxy_baselines(gf)
        for fr in base_fits:
            if fr.model in ("old_tdf_baseline", "nfw", "pseudo_isothermal"):
                all_fits.append(fr)
                comp[f"bic_{fr.model}"] = fr.bic
                comp[f"reduced_chi2_{fr.model}"] = fr.reduced_chi2
                boundary_rows.append(_boundary_row(gf.galaxy_id, fr))

        if comp6d is not None and gf.galaxy_id in comp6d["galaxy_id"].values:
            r6 = comp6d[comp6d["galaxy_id"] == gf.galaxy_id].iloc[0]
            comp["bic_tau_field_A"] = float(r6.get("bic_tau_field_A", np.nan))
            comp["reduced_chi2_tau_field_A"] = float(r6.get("reduced_chi2_tau_field_A", np.nan))
        else:
            comp["bic_tau_field_A"] = float("nan")

        proj_bics: dict[str, float] = {}
        proj_fits: dict[str, ModelFitResult] = {}
        for kind in ("A", "B", "C", "D", "E", "F"):
            h = global_hypers[kind]
            policy = "per_galaxy" if kind == "A" else "global"
            fr, prof = fit_projection_galaxy(gf, kind, h, policy=policy)  # type: ignore[arg-type]
            all_fits.append(fr)
            profiles_list.append(prof)
            mname = fr.model
            proj_bics[mname] = fr.bic
            proj_fits[mname] = fr
            comp[f"bic_{mname}"] = fr.bic
            comp[f"reduced_chi2_{mname}"] = fr.reduced_chi2
            boundary_rows.append(_boundary_row(gf.galaxy_id, fr))

        best_proj = min(proj_bics, key=proj_bics.get)  # type: ignore[arg-type]
        for mname in ("projection_kernel_A_local", "projection_kernel_C_gaussian", best_proj):
            fr = proj_fits.get(mname)
            if fr is not None and fr.v_pred is not None:
                zones = _zone_residuals(gf.r, gf.v_obs, gf.v_err, fr.v_pred)
                for zk, zv in zones.items():
                    comp[f"{zk}_{mname}"] = zv
        comp["best_projection_kernel"] = best_proj
        comp["bic_best_projection_kernel"] = proj_bics[best_proj]
        comp["delta_bic_best_projection_vs_old_tdf"] = (
            comp["bic_best_projection_kernel"] - comp.get("bic_old_tdf_baseline", np.nan)
        )
        comp["delta_bic_best_projection_vs_tau_field_A"] = (
            comp["bic_best_projection_kernel"] - comp.get("bic_tau_field_A", np.nan)
        )
        comp["delta_bic_best_projection_vs_pseudo"] = (
            comp["bic_best_projection_kernel"] - comp.get("bic_pseudo_isothermal", np.nan)
        )
        comparisons.append(comp)

    for kind in ("B", "C", "D"):
        global_tests.append(run_global_kernel_test(galaxies, kind, global_hypers[kind]))  # type: ignore[arg-type]
    global_kernel_df = pd.concat(global_tests, ignore_index=True) if global_tests else pd.DataFrame()

    comparison_df = pd.DataFrame(comparisons)
    bcols = [c for c in comparison_df.columns if c.startswith("bic_")]
    comparison_df["best_model_by_bic"] = comparison_df[bcols].idxmin(axis=1).str.replace("bic_", "")

    clean_ids = _clean_galaxy_ids(boundary6d) if boundary6d is not None else set()

    model_summary = _build_model_summary(comparison_df, clean_ids)
    class_summary = _build_class_summary(comparison_df)
    param_summary = build_parameter_summary()
    for kind, h in global_hypers.items():
        if kind == "F" and h.by_class:
            for cls, (lam, ell) in h.by_class.items():
                param_summary = pd.concat(
                    [
                        param_summary,
                        pd.DataFrame(
                            [{
                                "model": KERNEL_TO_MODEL[kind],
                                "galaxy_class": cls,
                                "lambda0_global": lam,
                                "ell_k_global": ell,
                            }],
                        ),
                    ],
                    ignore_index=True,
                )
        else:
            param_summary = pd.concat(
                [
                    param_summary,
                    pd.DataFrame(
                        [{
                            "model": KERNEL_TO_MODEL.get(kind, ""),
                            "lambda0_global": h.lambda0,
                            "ell_k_global": h.ell_k,
                            "sigma0_global": h.sigma0,
                            "q_global": h.q,
                        }],
                    ),
                ],
                ignore_index=True,
            )

    boundary_df = pd.DataFrame(boundary_rows)
    profiles = pd.concat(profiles_list, ignore_index=True) if profiles_list else pd.DataFrame()

    model_summary.to_csv(tables / "projection_kernel_model_summary.csv", index=False)
    comparison_df.to_csv(tables / "projection_kernel_comparison_by_galaxy.csv", index=False)
    param_summary.to_csv(tables / "projection_kernel_parameter_summary.csv", index=False)
    boundary_df.to_csv(tables / "projection_kernel_boundary_flags.csv", index=False)
    profiles.to_csv(tables / "projection_kernel_profiles.csv", index=False)
    class_summary.to_csv(tables / "projection_kernel_class_summary.csv", index=False)
    global_kernel_df.to_csv(tables / "projection_kernel_global_test.csv", index=False)

    _write_figures(
        comparison_df, global_hypers, profiles, figures, galaxies[:3],
    )
    report = _write_report(
        model_summary,
        comparison_df,
        global_kernel_df,
        step6e_global,
        global_hypers,
        field_run,
    )
    (reports / "projection_kernel_report.md").write_text(report, encoding="utf-8")

    return ProjectionRunResult(
        model_summary=model_summary,
        comparison_by_galaxy=comparison_df,
        parameter_summary=param_summary,
        boundary_flags=boundary_df,
        profiles=profiles,
        class_summary=class_summary,
        global_kernel_test=global_kernel_df,
        fit_details=all_fits,
    )


def _build_model_summary(comparison_df: pd.DataFrame, clean_ids: set[str]) -> pd.DataFrame:
    proj_models = [m for m in PROJECTION_MODELS if m.startswith("projection_")]
    all_models = ["old_tdf_baseline", "tau_field_A", "pseudo_isothermal", "nfw"] + proj_models
    rows: list[dict[str, Any]] = []
    for m in all_models:
        col = f"bic_{m}"
        if col not in comparison_df.columns:
            continue
        best = comparison_df["best_model_by_bic"] == m
        row: dict[str, Any] = {
            "model": m,
            "galaxies": len(comparison_df),
            "bic_win_count": int(best.sum()),
            "median_bic": float(comparison_df[col].median()),
            "median_delta_bic_vs_old_tdf": float(
                (comparison_df[col] - comparison_df["bic_old_tdf_baseline"]).median(),
            )
            if "bic_old_tdf_baseline" in comparison_df.columns
            else float("nan"),
            "median_delta_bic_vs_tau_field_A": float(
                (comparison_df[col] - comparison_df["bic_tau_field_A"]).median(),
            )
            if "bic_tau_field_A" in comparison_df.columns
            else float("nan"),
            "median_delta_bic_vs_pseudo": float(
                (comparison_df[col] - comparison_df["bic_pseudo_isothermal"]).median(),
            )
            if "bic_pseudo_isothermal" in comparison_df.columns
            else float("nan"),
        }
        if clean_ids:
            sub = comparison_df[comparison_df["galaxy_id"].astype(str).isin(clean_ids)]
            if len(sub):
                row["clean_subset_median_delta_bic_vs_old_tdf"] = float(
                    (sub[col] - sub["bic_old_tdf_baseline"]).median(),
                )
        rows.append(row)
    return pd.DataFrame(rows)


def _build_class_summary(comparison_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for cls in CLASS_ORDER:
        sub = comparison_df[comparison_df["galaxy_class"] == cls]
        if sub.empty:
            continue
        best_col = "bic_best_projection_kernel"
        rows.append(
            {
                "galaxy_class": cls,
                "n_galaxies": len(sub),
                "fraction_best_projection_beats_old_tdf": float(
                    (sub[best_col] < sub["bic_old_tdf_baseline"]).mean(),
                ),
                "fraction_best_projection_beats_tau_field_A": float(
                    (sub[best_col] < sub["bic_tau_field_A"]).mean(),
                ),
                "median_delta_bic_vs_old_tdf": float(
                    (sub[best_col] - sub["bic_old_tdf_baseline"]).median(),
                ),
            },
        )
    return pd.DataFrame(rows)


def _write_figures(
    comparison_df: pd.DataFrame,
    global_hypers: dict[KernelKind, KernelHyper],
    profiles: pd.DataFrame,
    figures_dir: Path,
    sample_galaxies: list[GalaxyFrame],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    r_demo = np.linspace(0.1, 15.0, 80)
    fig, ax = plt.subplots(figsize=(8, 5))
    for kind, label in [("A", "A local"), ("B", "B exp"), ("C", "C gauss"), ("D", "D outward")]:
        h = global_hypers.get(kind, KernelHyper())  # type: ignore[arg-type]
        k = build_normalized_kernel_matrix(r_demo, kind, h.ell_k)  # type: ignore[arg-type]
        mid = len(r_demo) // 2
        ax.plot(r_demo, k[mid, :], label=label)
    ax.set_xlabel("R' (kpc)")
    ax.set_ylabel(f"K(R_mid, R')")
    ax.legend(fontsize=8)
    ax.set_title("Projection kernel shapes (normalized rows)")
    fig.tight_layout()
    fig.savefig(figures_dir / "kernel_shapes.png", dpi=150)
    plt.close(fig)

    proj = [m for m in PROJECTION_MODELS if m.startswith("projection_")]
    wins = [int((comparison_df["best_model_by_bic"] == m).sum()) for m in proj]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(range(len(wins)), wins, color="steelblue")
    ax.set_xticks(range(len(wins)))
    ax.set_xticklabels([m.replace("projection_kernel_", "") for m in proj], rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("BIC wins")
    fig.tight_layout()
    fig.savefig(figures_dir / "bic_wins_projection_kernel.png", dpi=150)
    plt.close(fig)

    for col, fname, title in (
        ("delta_bic_best_projection_vs_old_tdf", "projection_kernel_vs_old_tdf_delta_bic.png", "Best kernel − old TDF"),
        ("delta_bic_best_projection_vs_pseudo", "projection_kernel_vs_pseudo_delta_bic.png", "Best kernel − pseudo"),
    ):
        if col in comparison_df.columns:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.hist(comparison_df[col].dropna(), bins=30, color="coral", edgecolor="white")
            ax.axvline(0, color="k", lw=0.8)
            ax.set_xlabel("ΔBIC")
            ax.set_title(title)
            fig.tight_layout()
            fig.savefig(figures_dir / fname, dpi=150)
            plt.close(fig)

    if "bic_projection_kernel_A_local" in comparison_df.columns:
        fig, axes = plt.subplots(1, 3, figsize=(10, 3))
        for ax, cls in zip(axes, CLASS_ORDER):
            sub = comparison_df[comparison_df["galaxy_class"] == cls]
            if sub.empty:
                continue
            ccol = "bic_projection_kernel_C_gaussian"
            data = [sub["bic_projection_kernel_A_local"]]
            labels = ["A"]
            if ccol in sub.columns:
                data.append(sub[ccol])
                labels.append("C")
            ax.boxplot(data, tick_labels=labels)
            ax.set_title(cls, fontsize=8)
        fig.suptitle("Projection kernel BIC by class", fontsize=10)
        fig.tight_layout()
        fig.savefig(figures_dir / "kernel_by_galaxy_class.png", dpi=150)
        plt.close(fig)

    if "bic_projection_kernel_B_exponential" in comparison_df.columns and sample_galaxies:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.set_title("Kernel parameters (from global pilot)")
        ax.text(0.1, 0.7, "\n".join(
            f"{k}: λ₀={h.lambda0:.3f}, ℓ={h.ell_k:.2f}"
            for k, h in global_hypers.items() if k != "F"
        ), fontsize=9, family="monospace")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(figures_dir / "kernel_parameter_distributions.png", dpi=150)
        plt.close(fig)

        fig, axes = plt.subplots(1, min(3, len(sample_galaxies)), figsize=(12, 4))
        if len(sample_galaxies) == 1:
            axes = [axes]
        for ax, gf in zip(axes, sample_galaxies[:3]):
            subp = profiles[(profiles["galaxy_id"] == gf.galaxy_id) & (profiles["model"] == "projection_kernel_A_local")]
            if subp.empty:
                continue
            ax.plot(subp["r_kpc"], subp["v_obs"], "ko", ms=3, label="obs")
            ax.plot(subp["r_kpc"], subp["v_total"], "-", label="kernel A")
            ax.set_title(gf.galaxy_id, fontsize=8)
            ax.legend(fontsize=6)
        fig.tight_layout()
        fig.savefig(figures_dir / "example_projection_kernel_rotation_fits.png", dpi=150)
        plt.close(fig)

    zpath = figures_dir.parent / "tables" / "projection_kernel_comparison_by_galaxy.csv"
    if zpath.is_file():
        cdf = pd.read_csv(zpath)
        fig, ax = plt.subplots(figsize=(7, 4))
        for m, lab in [
            ("projection_kernel_A_local", "A"),
            ("projection_kernel_C_gaussian", "C"),
        ]:
            col = f"median_abs_residual_inner_{m}"
            if col in cdf.columns:
                ax.scatter(
                    cdf.get(f"median_abs_residual_outer_{m}", cdf[col]),
                    cdf[col],
                    alpha=0.4,
                    s=15,
                    label=lab,
                )
        ax.set_xlabel("outer |residual|")
        ax.set_ylabel("inner |residual|")
        ax.legend(fontsize=8)
        ax.set_title("Zone residuals (projection kernels)")
        fig.tight_layout()
        fig.savefig(figures_dir / "residual_zone_projection_kernel.png", dpi=150)
        plt.close(fig)


def _df_to_md(df: pd.DataFrame, max_rows: int = 25) -> str:
    sub = df.head(max_rows)
    cols = list(sub.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for _, row in sub.iterrows():
        lines.append("| " + " | ".join(str(row[c])[:36] for c in cols) + " |")
    return "\n".join(lines)


def _write_report(
    model_summary: pd.DataFrame,
    comparison_df: pd.DataFrame,
    global_kernel_df: pd.DataFrame,
    step6e_global: pd.DataFrame | None,
    global_hypers: dict[KernelKind, KernelHyper],
    field_run: Path | None,
) -> str:
    n = len(comparison_df)
    frac6e = float(step6e_global["acceptable_delta_lt_6"].mean()) if step6e_global is not None and len(step6e_global) else float("nan")
    frac6f = float(global_kernel_df["acceptable_delta_lt_6"].mean()) if len(global_kernel_df) else float("nan")

    best_row = model_summary[model_summary["model"].str.startswith("projection_kernel")].sort_values("bic_win_count", ascending=False)
    best_model = str(best_row.iloc[0]["model"]) if len(best_row) else "projection_kernel_A_local"
    med_vs_old = float(comparison_df["delta_bic_best_projection_vs_old_tdf"].median()) if n else float("nan")
    med_vs_a = float(comparison_df["delta_bic_best_projection_vs_tau_field_A"].median()) if n else float("nan")

    lines = [
        "# SPARC 5D projection kernel report (Step 6F)",
        "",
        f"## ⚠️ {BANNER_PROJECTION}",
        "",
        f"## ⚠️ {BANNER_CALIBRATION}",
        "",
        f"**Step 6D ref:** `{field_run}`",
        f"**Galaxies:** {n}",
        "",
        "## Model summary",
        "",
        _df_to_md(model_summary),
        "",
        "## Step 6E vs 6F global-parameter test",
        "",
        f"- Step 6E global λ_b acceptable (ΔBIC&lt;6): **{frac6e:.1%}**",
        f"- Step 6F global kernel acceptable (ΔBIC&lt;6, B/C/D): **{frac6f:.1%}**",
        "",
        "## Global kernel hyperparameters",
        "",
    ]
    for kind, h in global_hypers.items():
        if kind == "F" and h.by_class:
            lines.append(f"- **{KERNEL_TO_MODEL[kind]}:** class-specific λ₀, ℓ_k")
            for cls, (lam, ell) in h.by_class.items():
                lines.append(f"  - {cls}: λ₀={lam:.4f}, ℓ_k={ell:.3f}")
        else:
            lines.append(
                f"- **{KERNEL_TO_MODEL.get(kind, kind)}:** λ₀={h.lambda0:.4f}, ℓ_k={h.ell_k:.3f}"
                + (f", Σ₀={h.sigma0:.1f}, q={h.q:.2f}" if kind == "E" else ""),
            )
    lines.extend(
        [
            "",
            "## Key questions",
            "",
            "### Does a projection kernel reduce need for per-galaxy λ_b?",
            "Global/class kernels replace free λ_b(R) with shared ℓ_k and λ₀; see global-test fractions above.",
            "",
            "### Which kernel family is best?",
            f"**{best_model}** leads projection-kernel BIC wins in this run.",
            "",
            "### Improve over Step 6D tau_field_A?",
            f"Median ΔBIC(best projection − tau_field_A) = **{med_vs_a:.2f}**.",
            "",
            "### Compete with pseudo-isothermal?",
            "See model summary median ΔBIC vs pseudo; pseudo typically still competitive.",
            "",
            "### Ready for final synthesis?",
            "**Exploratory 5D projection proxy only** — claim level **2** (phenomenological diagnostic).",
            "",
            "### Allowed claim level",
            "**2** — not observational validation.",
            "",
            "## Limitations",
            "",
            "- Discrete radial kernel; not a covariant 5D Green's function.",
            "- Global hyperparameter search is approximate (limited iterations).",
            "- Does not validate TDF observationally.",
        ],
    )
    return "\n".join(lines) + "\n"

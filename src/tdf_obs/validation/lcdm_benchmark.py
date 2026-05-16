"""Phase 3C — ΛCDM/GR benchmark recovery scaffolds (not observational validation)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tdf_obs.constants import SOLAR_MASS
from tdf_obs.models.black_hole import hawking_temperature, non_return_radius_tdf, schwarzschild_radius, tdf_temperature
from tdf_obs.models.redshift import z_tau
from tdf_obs.models.solar_system import epsilon_tau, passes_gr_safe_limit

BENCHMARK_MODE = "lcdm_benchmark_recovery"
BANNER_LCDM_BENCHMARK = "ΛCDM BENCHMARK RECOVERY — NOT REAL OBSERVATIONAL DATA"

# Representative solar-system cases (configurable assumed epsilon_tau, not fitted from data).
DEFAULT_SOLAR_SYSTEM_CASES: tuple[dict[str, Any], ...] = (
    {
        "case_name": "earth_orbit",
        "regime": "Earth orbit",
        "phi_b": -6.2637e7,
        "phi_tau_assumed": 1e-10,
        "max_allowed_epsilon": 1e-8,
    },
    {
        "case_name": "mercury_orbit",
        "regime": "Mercury orbit",
        "phi_b": -8.5e7,
        "phi_tau_assumed": 5e-11,
        "max_allowed_epsilon": 1e-8,
    },
    {
        "case_name": "light_bending_sun",
        "regime": "light-bending near Sun",
        "phi_b": -1.0e8,
        "phi_tau_assumed": 1e-12,
        "max_allowed_epsilon": 1e-9,
    },
    {
        "case_name": "gps_weak_field_clock",
        "regime": "GPS-like weak-field clock regime",
        "phi_b": -6.0e7,
        "phi_tau_assumed": 1e-11,
        "max_allowed_epsilon": 1e-8,
    },
)

DEFAULT_RC_OVER_RS_VALUES: tuple[float, ...] = (0.0, 1e-6, 1e-4, 1e-2, 0.1, 0.5, 0.9)

# Synthetic delta_tau values (z_tau = K_tau * delta_tau / c^2); not fitted from data.
DEFAULT_REDSHIFT_CASES: tuple[dict[str, Any], ...] = (
    {
        "case_name": "negligible",
        "delta_tau": 1.0,
        "K_tau": 1.0,
        "max_allowed_z_tau": 1e-15,
        "category": "negligible",
    },
    {
        "case_name": "small_allowed",
        "delta_tau": 1e8,
        "K_tau": 1.0,
        "max_allowed_z_tau": 1e-6,
        "category": "small_allowed",
    },
    {
        "case_name": "borderline",
        "delta_tau": 8.9e10,
        "K_tau": 1.0,
        "max_allowed_z_tau": 1e-6,
        "category": "borderline",
    },
    {
        "case_name": "too_large",
        "delta_tau": 1e12,
        "K_tau": 1.0,
        "max_allowed_z_tau": 1e-8,
        "category": "too_large",
    },
)


@dataclass
class BenchmarkRow:
    benchmark: str
    case_name: str
    status: str
    details: str
    extra: dict[str, Any]


def run_solar_system_gr_benchmark(
    cases: tuple[dict[str, Any], ...] | None = None,
) -> list[BenchmarkRow]:
    """ε_τ = Φ_τ/Φ_b must stay below configured caps (GR-safe scaffold)."""
    cases = cases or DEFAULT_SOLAR_SYSTEM_CASES
    rows: list[BenchmarkRow] = []
    for case in cases:
        eps = epsilon_tau(case["phi_tau_assumed"], case["phi_b"])
        passed = passes_gr_safe_limit(eps, case["max_allowed_epsilon"])
        rows.append(
            BenchmarkRow(
                benchmark="solar_system_gr_safe",
                case_name=case["case_name"],
                status="pass" if passed else "fail",
                details=case["regime"],
                extra={
                    "epsilon_tau": eps,
                    "max_allowed_epsilon": case["max_allowed_epsilon"],
                    "phi_b": case["phi_b"],
                    "phi_tau_assumed": case["phi_tau_assumed"],
                },
            ),
        )
    return rows


def run_black_hole_gr_limit_benchmark(
    *,
    mass_solar: float = 10.0,
    rc_over_rs_values: tuple[float, ...] | None = None,
    rtol_gr: float = 1e-4,
) -> list[BenchmarkRow]:
    """
    Check TDF BH ansatz → GR when rc/rs → 0.

    r_nr_ratio = r_nr / r_s = sqrt(1 - (rc/rs)^2)
    T_ratio = T_TDF / T_H = sqrt(1 - (rc/rs)^2)
    """
    M = mass_solar * SOLAR_MASS
    rs = schwarzschild_radius(M)
    T_H = hawking_temperature(M)
    rc_values = rc_over_rs_values or DEFAULT_RC_OVER_RS_VALUES
    rows: list[BenchmarkRow] = []

    for rc_over_rs in rc_values:
        rc_over_rs = float(rc_over_rs)
        expected_ratio = float(np.sqrt(max(0.0, 1.0 - rc_over_rs**2)))
        rc = rc_over_rs * rs

        if rc_over_rs >= 1.0:
            rows.append(
                BenchmarkRow(
                    benchmark="black_hole_gr_limit",
                    case_name=f"rc_over_rs_{rc_over_rs:g}",
                    status="fail",
                    details="rc/rs >= 1 unphysical for this ansatz",
                    extra={"rc_over_rs": rc_over_rs, "expected_ratio": 0.0},
                ),
            )
            continue

        r_nr = non_return_radius_tdf(M, rc)
        r_nr_ratio = r_nr / rs if np.isfinite(r_nr) else float("nan")
        T_TDF = tdf_temperature(M, rc)
        T_ratio = T_TDF / T_H if T_H > 0 else float("nan")

        if rc_over_rs == 0.0:
            passed = np.isclose(r_nr_ratio, 1.0, rtol=rtol_gr) and np.isclose(T_ratio, 1.0, rtol=rtol_gr)
        elif rc_over_rs <= 0.1:
            passed = (
                np.isfinite(r_nr_ratio)
                and np.isfinite(T_ratio)
                and abs(r_nr_ratio - expected_ratio) / max(expected_ratio, 1e-12) < 0.05
                and abs(T_ratio - expected_ratio) / max(expected_ratio, 1e-12) < 0.05
            )
        else:
            passed = (
                np.isfinite(r_nr_ratio)
                and np.isfinite(T_ratio)
                and r_nr_ratio < 0.99
                and T_ratio < 0.99
            )

        rows.append(
            BenchmarkRow(
                benchmark="black_hole_gr_limit",
                case_name=f"rc_over_rs_{rc_over_rs:g}",
                status="pass" if passed else "fail",
                details=f"expected_ratio={expected_ratio:.6g}",
                extra={
                    "rc_over_rs": rc_over_rs,
                    "r_nr_ratio": r_nr_ratio,
                    "T_ratio": T_ratio,
                    "expected_ratio": expected_ratio,
                },
            ),
        )
    return rows


def run_redshift_sanity_benchmark(
    cases: tuple[dict[str, Any], ...] | None = None,
) -> list[BenchmarkRow]:
    """z_τ = K_τ Δτ̄_l / c² must stay within configurable tolerance."""
    cases = cases or DEFAULT_REDSHIFT_CASES
    rows: list[BenchmarkRow] = []
    for case in cases:
        z = float(z_tau(case["delta_tau"], case["K_tau"]))
        max_z = case["max_allowed_z_tau"]
        passed = abs(z) <= max_z
        warning = ""
        if case["category"] == "borderline" and passed:
            warning = "borderline: within tolerance but near limit"
        if case["category"] == "too_large":
            warning = "expected fail: tau correction too large for scaffold"

        rows.append(
            BenchmarkRow(
                benchmark="redshift_sanity",
                case_name=case["case_name"],
                status="pass" if passed else "fail",
                details=warning or case["category"],
                extra={
                    "z_tau": z,
                    "delta_tau": case["delta_tau"],
                    "K_tau": case["K_tau"],
                    "max_allowed_z_tau": max_z,
                    "category": case["category"],
                },
            ),
        )
    return rows


def _load_nfw_surrogate_status(project_root: Path) -> str:
    summary = project_root / "outputs" / "tables" / "nfw_surrogate_fit_summary.csv"
    report = project_root / "outputs" / "reports" / "nfw_surrogate_report.md"
    if not summary.is_file():
        return "_NFW surrogate Phase 3B outputs not found; run `python scripts/run_nfw_surrogate.py` first._"
    df = pd.read_csv(summary)
    n = len(df)
    if "mimic_success" in df.columns:
        mimic = int(df["mimic_success"].sum())
    elif "tdf_mimics_teacher" in df.columns:
        mimic = int(df["tdf_mimics_teacher"].sum())
    else:
        mimic = 0
    lines = [
        f"NFW surrogate summary available: {n} benchmark case(s) in `{summary.name}`.",
        f"TDF mimic success (Phase 4A): {mimic}/{n}.",
    ]
    if report.is_file():
        lines.append(f"See `{report.name}` for details.")
    lines.append("_Surrogate success does not imply observational validation._")
    return "\n".join(lines)


def _rows_to_dataframe(rows: list[BenchmarkRow]) -> pd.DataFrame:
    records = []
    for row in rows:
        rec = {
            "benchmark_mode": BENCHMARK_MODE,
            "is_real_observational_data": False,
            "benchmark": row.benchmark,
            "case_name": row.case_name,
            "status": row.status,
            "details": row.details,
        }
        for k, v in row.extra.items():
            rec[k] = v
        records.append(rec)
    return pd.DataFrame(records)


def _build_report(
    df: pd.DataFrame,
    nfw_status: str,
    project_root: Path,
) -> str:
    n_pass = int((df["status"] == "pass").sum())
    n_fail = int((df["status"] == "fail").sum())

    lines = [
        "# ΛCDM benchmark recovery report (Phase 3C)",
        "",
        f"## ⚠️ {BANNER_LCDM_BENCHMARK}",
        "",
        "> **This is not observational validation.** Configurable scaffold cases only. "
        "> No real SPARC, solar-system ephemeris, lensing, or cosmological data were used.",
        "",
        "## Purpose",
        "",
        "Extend benchmark recovery beyond rotation (Phase 3B NFW surrogate) with:",
        "",
        "1. Solar-system GR-safe checks on ε_τ = Φ_τ/Φ_b",
        "2. Black-hole ansatz GR-limit recovery when r_c/r_s → 0",
        "3. Redshift sanity on z_τ = K_τ Δτ̄_l / c²",
        "",
        "## NFW rotation surrogate status (Phase 3B)",
        "",
        nfw_status,
        "",
        "## Summary",
        "",
        f"- **Total cases:** {len(df)}",
        f"- **Pass:** {n_pass}",
        f"- **Fail:** {n_fail} _(expected for intentional `too_large` redshift case)_",
        "",
    ]

    for bench in ("solar_system_gr_safe", "black_hole_gr_limit", "redshift_sanity"):
        sub = df[df["benchmark"] == bench]
        lines.append(f"## {bench.replace('_', ' ').title()}")
        lines.append("")
        lines.append("| Case | Status | Key values |")
        lines.append("| --- | --- | --- |")
        for _, r in sub.iterrows():
            if bench == "solar_system_gr_safe":
                key = f"ε_τ={r.get('epsilon_tau', float('nan')):.2e}, max={r.get('max_allowed_epsilon', float('nan')):.2e}"
            elif bench == "black_hole_gr_limit":
                key = f"r_nr/rs={r.get('r_nr_ratio', float('nan')):.4f}, T/T_H={r.get('T_ratio', float('nan')):.4f}"
            else:
                key = f"z_τ={r.get('z_tau', float('nan')):.2e}, max={r.get('max_allowed_z_tau', float('nan')):.2e}"
            lines.append(f"| {r['case_name']} | {r['status']} | {key} |")
        lines.append("")

    lines.extend(
        [
            "## Failure modes",
            "",
            "- Assumed Φ_τ in solar-system cases are placeholders until a dynamical TDF metric is coupled.",
            "- Black-hole formulas are phenomenological ansatz, not derived from a full metric.",
            "- Redshift scaffold uses synthetic Δτ̄_l; no line-of-sight or velocity degeneracy model yet.",
            "- Passing here does not constrain lensing, CMB, or structure formation.",
            "",
            "## Next steps",
            "",
            "1. Ingest real SPARC rotation data (`prepare_sparc_rotation.py`).",
            "2. Implement lensing consistency with shared τ parameters (Phase 4).",
            "3. Replace assumed solar-system ε_τ with predictions from a weak-field TDF metric.",
            "4. Tie redshift tests to independent kinematic baselines without double-counting.",
            "",
            "## Disclaimer",
            "",
            "- Does **not** validate TDF against ΛCDM or observations.",
            "- Phase 3B NFW surrogate success is a **shape** benchmark only.",
            "",
        ],
    )
    return "\n".join(lines)


def run_lcdm_benchmark_pipeline(
    outputs_root: Path | None = None,
    project_root: Path | None = None,
) -> pd.DataFrame:
    """Run all Phase 3C benchmarks and write CSV + markdown report."""
    root = project_root or Path(__file__).resolve().parents[3]
    outputs = outputs_root or (root / "outputs")
    tables_dir = outputs / "tables"
    reports_dir = outputs / "reports"
    for d in (tables_dir, reports_dir):
        d.mkdir(parents=True, exist_ok=True)

    rows: list[BenchmarkRow] = []
    rows.extend(run_solar_system_gr_benchmark())
    rows.extend(run_black_hole_gr_limit_benchmark())
    rows.extend(run_redshift_sanity_benchmark())

    df = _rows_to_dataframe(rows)
    df.to_csv(tables_dir / "lcdm_benchmark_summary.csv", index=False)

    nfw_status = _load_nfw_surrogate_status(root)
    (reports_dir / "lcdm_benchmark_report.md").write_text(
        _build_report(df, nfw_status, root),
        encoding="utf-8",
    )
    return df

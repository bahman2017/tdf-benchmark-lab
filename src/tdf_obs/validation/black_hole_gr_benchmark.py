"""Phase 4C — Black-hole exterior GR-limit benchmark (not observational validation)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from tdf_obs.constants import SOLAR_MASS
from tdf_obs.models.black_hole import (
    hawking_temperature,
    non_return_radius_tdf,
    remnant_mass,
    schwarzschild_radius,
    tdf_temperature,
)

BENCHMARK_MODE = "black_hole_gr_limit_benchmark"
BANNER_BH_GR_LIMIT = "BLACK-HOLE GR-LIMIT BENCHMARK — NOT REAL OBSERVATIONAL DATA"

DEFAULT_Q_VALUES: tuple[float, ...] = (
    0.0,
    1e-8,
    1e-6,
    1e-4,
    1e-2,
    0.05,
    0.1,
    0.25,
    0.5,
    0.75,
    0.9,
    0.99,
    1.0,
)

REQUIRED_SUMMARY_COLUMNS: tuple[str, ...] = (
    "q",
    "rc_over_rs",
    "r_nr_ratio",
    "temperature_ratio",
    "expected_ratio",
    "deviation_from_gr_percent",
    "status",
    "M_rem_kg",
    "mass_solar",
)

GR_LIKE_MAX_DEVIATION_PERCENT = 0.1
MILDLY_MODIFIED_MAX_DEVIATION_PERCENT = 5.0


@dataclass
class BlackHoleGrBenchmarkRow:
    q: float
    rc_over_rs: float
    mass_solar: float
    rs_m: float
    rc_m: float
    r_nr_ratio: float
    temperature_ratio: float
    expected_ratio: float
    deviation_from_gr_percent: float
    status: str
    M_rem_kg: float
    T_H_K: float
    T_TDF_K: float


def expected_gr_ratio(q: float) -> float:
    """GR exterior ansatz sqrt(1 - q^2) when q is below unity."""
    q = float(q)
    if q >= 1.0:
        return float("nan")
    return float(np.sqrt(max(0.0, 1.0 - q**2)))


def classify_bh_gr_limit(q: float, ratio: float) -> str:
    """Classify recovery vs Hawking limit (ratio tends to 1 as q tends to 0)."""
    q = float(q)
    if q >= 1.0:
        return "no_horizon"
    if not np.isfinite(ratio):
        return "no_horizon"
    deviation = 100.0 * abs(1.0 - ratio)
    if deviation < GR_LIKE_MAX_DEVIATION_PERCENT:
        return "GR-like"
    if deviation < MILDLY_MODIFIED_MAX_DEVIATION_PERCENT:
        return "mildly_modified"
    return "strongly_modified"


def _deviation_from_gr_percent(ratio: float) -> float:
    if not np.isfinite(ratio):
        return float("nan")
    return float(100.0 * abs(1.0 - ratio))


def run_black_hole_gr_limit_benchmark(
    q_values: Sequence[float] | None = None,
    *,
    mass_solar: float = 10.0,
) -> list[BlackHoleGrBenchmarkRow]:
    """
    Evaluate TDF BH phenomenology vs GR/Hawking exterior limit across q = rc/rs.

    Uses existing model functions (no equation changes).
    """
    q_list = tuple(q_values) if q_values is not None else DEFAULT_Q_VALUES
    M = float(mass_solar) * SOLAR_MASS
    rs = schwarzschild_radius(M)
    T_H = hawking_temperature(M)
    rows: list[BlackHoleGrBenchmarkRow] = []

    for q in q_list:
        q = float(q)
        if q >= 1.0:
            rows.append(
                BlackHoleGrBenchmarkRow(
                    q=q,
                    rc_over_rs=q,
                    mass_solar=mass_solar,
                    rs_m=rs,
                    rc_m=q * rs,
                    r_nr_ratio=float("nan"),
                    temperature_ratio=float("nan"),
                    expected_ratio=float("nan"),
                    deviation_from_gr_percent=float("nan"),
                    status="no_horizon",
                    M_rem_kg=remnant_mass(q * rs) if q > 0 else 0.0,
                    T_H_K=T_H,
                    T_TDF_K=float("nan"),
                ),
            )
            continue

        rc = q * rs
        r_nr = non_return_radius_tdf(M, rc)
        r_nr_ratio = r_nr / rs if np.isfinite(r_nr) else float("nan")
        T_TDF = tdf_temperature(M, rc)
        temperature_ratio = T_TDF / T_H if T_H > 0 else float("nan")
        exp_ratio = expected_gr_ratio(q)

        # Use mean of r_nr and T ratios for classification when both finite.
        if np.isfinite(r_nr_ratio) and np.isfinite(temperature_ratio):
            ratio_for_class = 0.5 * (r_nr_ratio + temperature_ratio)
        elif np.isfinite(temperature_ratio):
            ratio_for_class = temperature_ratio
        else:
            ratio_for_class = r_nr_ratio

        deviation = _deviation_from_gr_percent(ratio_for_class)
        status = classify_bh_gr_limit(q, ratio_for_class)

        rows.append(
            BlackHoleGrBenchmarkRow(
                q=q,
                rc_over_rs=q,
                mass_solar=mass_solar,
                rs_m=rs,
                rc_m=rc,
                r_nr_ratio=r_nr_ratio,
                temperature_ratio=temperature_ratio,
                expected_ratio=exp_ratio,
                deviation_from_gr_percent=deviation,
                status=status,
                M_rem_kg=remnant_mass(rc),
                T_H_K=T_H,
                T_TDF_K=T_TDF,
            ),
        )

    return rows


def summarize_black_hole_benchmark(
    results: Sequence[BlackHoleGrBenchmarkRow],
) -> dict[str, Any]:
    """Aggregate status counts and GR-recovery metrics."""
    status_counts: dict[str, int] = {}
    for row in results:
        status_counts[row.status] = status_counts.get(row.status, 0) + 1

    gr_like = [r for r in results if r.status == "GR-like"]
    finite_q = [r.q for r in results if r.q < 1.0 and np.isfinite(r.temperature_ratio)]
    max_q_gr_like = max((r.q for r in gr_like), default=float("nan"))

    return {
        "n_cases": len(results),
        "status_counts": status_counts,
        "n_gr_like": status_counts.get("GR-like", 0),
        "n_mildly_modified": status_counts.get("mildly_modified", 0),
        "n_strongly_modified": status_counts.get("strongly_modified", 0),
        "n_no_horizon": status_counts.get("no_horizon", 0),
        "max_q_gr_like": max_q_gr_like,
        "min_q": min(finite_q) if finite_q else float("nan"),
        "max_q_finite": max(finite_q) if finite_q else float("nan"),
    }


def _row_to_dict(row: BlackHoleGrBenchmarkRow) -> dict[str, Any]:
    return {
        "q": row.q,
        "rc_over_rs": row.rc_over_rs,
        "r_nr_ratio": row.r_nr_ratio,
        "temperature_ratio": row.temperature_ratio,
        "expected_ratio": row.expected_ratio,
        "deviation_from_gr_percent": row.deviation_from_gr_percent,
        "status": row.status,
        "M_rem_kg": row.M_rem_kg,
        "mass_solar": row.mass_solar,
        "rs_m": row.rs_m,
        "rc_m": row.rc_m,
        "T_H_K": row.T_H_K,
        "T_TDF_K": row.T_TDF_K,
        "benchmark_mode": BENCHMARK_MODE,
        "is_real_observational_data": False,
    }


def _build_report(
    results: list[BlackHoleGrBenchmarkRow],
    summary: dict[str, Any],
    *,
    mass_solar: float,
) -> str:
    lines = [
        "# Black-hole exterior GR-limit benchmark report (Phase 4C)",
        "",
        f"## ⚠️ {BANNER_BH_GR_LIMIT}",
        "",
        "## Purpose",
        "",
        "This checks whether the **phenomenological** TDF black-hole ansatz recovers "
        "GR/Hawking behavior in the exterior limit **q = r_c/r_s ≪ 1**.",
        "",
        "> **Not observational validation.** No ringdown, shadow, or evaporation data are fitted.",
        "",
        "## Formulas (TDF v0.8.1 ansatz level)",
        "",
        "```text",
        "r_s = 2 G M / c^2",
        "r_nr_TDF = sqrt(r_s^2 - r_c^2)",
        "T_TDF = T_H * sqrt(1 - r_c^2 / r_s^2)",
        "M_rem = c^2 r_c / (2 G)",
        "q = r_c / r_s",
        "r_nr_TDF / r_s = sqrt(1 - q^2)",
        "T_TDF / T_H = sqrt(1 - q^2)",
        "```",
        "",
        f"**Benchmark mass:** {mass_solar} M☉ (configurable; not from observation).",
        "",
        "## Summary",
        "",
        f"- **Cases:** {summary['n_cases']}",
        f"- **GR-like:** {summary['n_gr_like']}",
        f"- **Mildly modified:** {summary['n_mildly_modified']}",
        f"- **Strongly modified:** {summary['n_strongly_modified']}",
        f"- **No horizon (q ≥ 1):** {summary['n_no_horizon']}",
        f"- **Largest q still GR-like:** {summary['max_q_gr_like']}",
        "",
        "## Results table",
        "",
        "| q | r_nr/r_s | T/T_H | expected | dev. from GR % | status |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in results:
        rnr = f"{row.r_nr_ratio:.6g}" if np.isfinite(row.r_nr_ratio) else "—"
        tr = f"{row.temperature_ratio:.6g}" if np.isfinite(row.temperature_ratio) else "—"
        exp = f"{row.expected_ratio:.6g}" if np.isfinite(row.expected_ratio) else "—"
        dev = f"{row.deviation_from_gr_percent:.4g}" if np.isfinite(row.deviation_from_gr_percent) else "—"
        lines.append(f"| {row.q:g} | {rnr} | {tr} | {exp} | {dev} | {row.status} |")

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- **q → 0** recovers Hawking/GR exterior ratios (r_nr/r_s → 1, T/T_H → 1).",
            "- **Large q** produces strong departures from the q=0 limit and explores remnant scale M_rem.",
            "- This is **not** a full nonlinear black-hole solution; Kerr spin, backreaction, and real data are out of scope.",
            "",
            "## Failure modes",
            "",
            "- Formulas are **phenomenological ansatz-level** only.",
            "- Real ringdown, shadow, and evaporation constraints are **not** fitted.",
            "- **Kerr spin** is not included.",
            "- **Backreaction** is not included.",
            "- q ≥ 1 is flagged `no_horizon` for this exterior ansatz.",
            "",
            "## Disclaimer",
            "",
            "- **NOT REAL OBSERVATIONAL DATA**",
            "- Passing GR-limit recovery in this table does **not** validate TDF.",
            "",
        ],
    )
    return "\n".join(lines)


def run_black_hole_gr_benchmark_pipeline(
    outputs_root: Path | None = None,
    *,
    q_values: Sequence[float] | None = None,
    mass_solar: float = 10.0,
) -> pd.DataFrame:
    """Run Phase 4C benchmark and write CSV + markdown report."""
    root = Path(__file__).resolve().parents[3]
    outputs = outputs_root or (root / "outputs")
    tables_dir = outputs / "tables"
    reports_dir = outputs / "reports"
    for d in (tables_dir, reports_dir):
        d.mkdir(parents=True, exist_ok=True)

    results = run_black_hole_gr_limit_benchmark(q_values, mass_solar=mass_solar)
    summary = summarize_black_hole_benchmark(results)
    df = pd.DataFrame([_row_to_dict(r) for r in results])
    df.to_csv(tables_dir / "black_hole_gr_benchmark_summary.csv", index=False)
    (reports_dir / "black_hole_gr_benchmark_report.md").write_text(
        _build_report(results, summary, mass_solar=mass_solar),
        encoding="utf-8",
    )
    return df

"""
SPARC Step 1 — boundary-filtered model comparison (analysis only).

Compares TDF / NFW / corrected MOND BIC outcomes with and without galaxies
where NFW or TDF fits hit parameter bounds. Not observational validation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from tdf_obs.validation.sparc_real_calibration import BIC_COMPETITIVE_DELTA

BANNER_BOUNDARY_FILTERED = (
    "SPARC BOUNDARY-FILTERED ANALYSIS — NOT FULL OBSERVATIONAL VALIDATION"
)

BANNER_CALIBRATION = (
    "These results are calibration diagnostics for a phenomenological TDF model. "
    "They do not constitute observational validation of TDF unless all required "
    "multi-channel constraints are passed."
)

FilterName = Literal[
    "all_galaxies",
    "exclude_any_nfw_or_tdf_boundary_hit",
    "exclude_severe_boundary_pressure",
]

BOUND_REL_TOL = 0.02
BOUND_ABS_TOL = 0.08

FILTER_COMPARISON_COLUMNS: tuple[str, ...] = (
    "filter_name",
    "n_galaxies",
    "bic_win_baryon_only",
    "bic_win_corrected_mond",
    "bic_win_nfw",
    "bic_win_tdf_kessence",
    "median_delta_bic_tdf_minus_nfw",
    "median_delta_bic_tdf_minus_corrected_mond",
    "tdf_competitive_count",
    "tdf_strong_win_vs_nfw",
    "nfw_strong_win_vs_tdf",
    "boundary_filtering_available",
)

GALAXY_FLAG_COLUMNS: tuple[str, ...] = (
    "galaxy_id",
    "boundary_data_source",
    "any_nfw_boundary_hit",
    "any_tdf_boundary_hit",
    "nfw_boundary_hit_count",
    "tdf_boundary_hit_count",
    "nfw_v200_at_bound",
    "nfw_r_s_at_bound",
    "tdf_beta_over_M_at_bound",
    "severe_boundary_pressure",
    "in_exclude_any_nfw_or_tdf_boundary_hit",
    "in_exclude_severe_boundary_pressure",
)


@dataclass
class BoundaryFilteredRunResult:
    comparison_by_filter: pd.DataFrame
    galaxy_flags: pd.DataFrame
    filtered_comparison: pd.DataFrame
    boundary_filtering_available: bool
    boundary_data_source: str
    input_run: Path
    audit_run: Path | None


def _at_bound(value: float, lo: float, hi: float) -> bool:
    eps_lo = max(BOUND_ABS_TOL, BOUND_REL_TOL * max(abs(lo), 1e-6))
    eps_hi = max(BOUND_ABS_TOL, BOUND_REL_TOL * max(abs(hi), 1e-6))
    return value <= lo + eps_lo or value >= hi - eps_hi


def _param_at_bound(
    value: float | None,
    lo: float,
    hi: float,
) -> bool:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return False
    return _at_bound(float(value), lo, hi)


def load_comparison_table(path: Path) -> pd.DataFrame:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"comparison table not found: {path}")
    df = pd.read_csv(path)
    required = {
        "galaxy_id",
        "best_model_by_bic",
        "bic_baryon_only",
        "bic_nfw",
        "bic_tdf_kessence",
        "delta_bic_tdf_vs_nfw",
    }
    mond_col = "bic_corrected_mond" if "bic_corrected_mond" in df.columns else "bic_mond"
    delta_mond_col = (
        "delta_bic_tdf_vs_corrected_mond"
        if "delta_bic_tdf_vs_corrected_mond" in df.columns
        else "delta_bic_tdf_vs_mond"
    )
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"comparison table missing columns: {missing}")
    if mond_col not in df.columns:
        raise ValueError("comparison table missing corrected MOND BIC column")
    df = df.copy()
    df["_bic_mond"] = df[mond_col]
    df["_delta_bic_tdf_vs_mond"] = df[delta_mond_col]
    return df


def _boundary_row_from_audit(row: pd.Series) -> dict[str, Any]:
    return {
        "any_boundary_hit": bool(row["any_boundary_hit"]),
        "boundary_hit_count": int(row["boundary_hit_count"]),
        "v200_at_bound": bool(row.get("v200_at_bound", False)),
        "r_s_at_bound": bool(row.get("r_s_at_bound", False)),
        "beta_over_M_at_bound": bool(row.get("beta_over_M_at_bound", False)),
    }


def _boundary_row_from_summary(row: pd.Series, model: str) -> dict[str, Any]:
    hits: list[str] = []
    if model in ("nfw", "tdf_kessence", "baryon_only", "mond", "corrected_mond"):
        if _param_at_bound(row.get("upsilon_disk"), 0.05, 3.0):
            hits.append("upsilon_disk")
        if _param_at_bound(row.get("upsilon_bulge"), 0.05, 3.0):
            hits.append("upsilon_bulge")
    v200_at = r_s_at = beta_at = False
    if model == "nfw":
        v200_at = _param_at_bound(row.get("v200"), 1.0, 500.0)
        r_s_at = _param_at_bound(row.get("r_s"), 0.1, 100.0)
        if v200_at:
            hits.append("v200")
        if r_s_at:
            hits.append("r_s")
    if model == "tdf_kessence":
        beta_at = _param_at_bound(row.get("beta_over_M"), 1e-4, 50.0)
        if beta_at:
            hits.append("beta_over_M")
    return {
        "any_boundary_hit": len(hits) > 0,
        "boundary_hit_count": len(hits),
        "v200_at_bound": v200_at,
        "r_s_at_bound": r_s_at,
        "beta_over_M_at_bound": beta_at,
    }


def build_galaxy_boundary_flags(
    comparison_df: pd.DataFrame,
    boundary_flags_path: Path | None,
    summary_path: Path | None,
) -> tuple[pd.DataFrame, bool, str]:
    """
    Per-galaxy boundary metadata for NFW and TDF.

    Returns (galaxy_flags_df, filtering_available, source_label).
    """
    galaxy_ids = comparison_df["galaxy_id"].astype(str).tolist()
    source = "unavailable"
    available = False

    nfw_info: dict[str, dict[str, Any]] = {}
    tdf_info: dict[str, dict[str, Any]] = {}

    if boundary_flags_path is not None and Path(boundary_flags_path).is_file():
        bdf = pd.read_csv(boundary_flags_path)
        source = f"audit:{boundary_flags_path}"
        available = True
        for gid in galaxy_ids:
            nfw_row = bdf[(bdf["galaxy_id"] == gid) & (bdf["model"] == "nfw")]
            tdf_row = bdf[(bdf["galaxy_id"] == gid) & (bdf["model"] == "tdf_kessence")]
            if len(nfw_row):
                nfw_info[gid] = _boundary_row_from_audit(nfw_row.iloc[0])
            if len(tdf_row):
                tdf_info[gid] = _boundary_row_from_audit(tdf_row.iloc[0])
    elif summary_path is not None and Path(summary_path).is_file():
        sdf = pd.read_csv(summary_path)
        source = f"inferred_from_summary:{summary_path}"
        available = True
        for gid in galaxy_ids:
            nfw_row = sdf[(sdf["galaxy_id"] == gid) & (sdf["model"] == "nfw")]
            tdf_row = sdf[(sdf["galaxy_id"] == gid) & (sdf["model"] == "tdf_kessence")]
            if len(nfw_row):
                nfw_info[gid] = _boundary_row_from_summary(nfw_row.iloc[0], "nfw")
            if len(tdf_row):
                tdf_info[gid] = _boundary_row_from_summary(tdf_row.iloc[0], "tdf_kessence")

    rows: list[dict[str, Any]] = []
    for gid in galaxy_ids:
        nfw = nfw_info.get(
            gid,
            {
                "any_boundary_hit": False,
                "boundary_hit_count": 0,
                "v200_at_bound": False,
                "r_s_at_bound": False,
                "beta_over_M_at_bound": False,
            },
        )
        tdf = tdf_info.get(
            gid,
            {
                "any_boundary_hit": False,
                "boundary_hit_count": 0,
                "v200_at_bound": False,
                "r_s_at_bound": False,
                "beta_over_M_at_bound": False,
            },
        )
        any_hit = bool(nfw["any_boundary_hit"] or tdf["any_boundary_hit"])
        severe = (
            int(nfw["boundary_hit_count"]) >= 2
            or int(tdf["boundary_hit_count"]) >= 2
            or bool(nfw["v200_at_bound"])
            or bool(nfw["r_s_at_bound"])
            or bool(tdf["beta_over_M_at_bound"])
        )
        rows.append(
            {
                "galaxy_id": gid,
                "boundary_data_source": source,
                "any_nfw_boundary_hit": bool(nfw["any_boundary_hit"]),
                "any_tdf_boundary_hit": bool(tdf["any_boundary_hit"]),
                "nfw_boundary_hit_count": int(nfw["boundary_hit_count"]),
                "tdf_boundary_hit_count": int(tdf["boundary_hit_count"]),
                "nfw_v200_at_bound": bool(nfw["v200_at_bound"]),
                "nfw_r_s_at_bound": bool(nfw["r_s_at_bound"]),
                "tdf_beta_over_M_at_bound": bool(tdf["beta_over_M_at_bound"]),
                "severe_boundary_pressure": severe,
                "in_exclude_any_nfw_or_tdf_boundary_hit": not any_hit,
                "in_exclude_severe_boundary_pressure": not severe,
            },
        )

    return pd.DataFrame(rows), available, source


def _apply_filter(
    comparison_df: pd.DataFrame,
    galaxy_flags: pd.DataFrame,
    filter_name: FilterName,
) -> pd.DataFrame:
    merged = comparison_df.merge(galaxy_flags, on="galaxy_id", how="left")
    if filter_name == "all_galaxies":
        return merged
    if filter_name == "exclude_any_nfw_or_tdf_boundary_hit":
        col = "in_exclude_any_nfw_or_tdf_boundary_hit"
        return merged[merged[col].fillna(True)]
    if filter_name == "exclude_severe_boundary_pressure":
        col = "in_exclude_severe_boundary_pressure"
        return merged[merged[col].fillna(True)]
    raise ValueError(filter_name)


def compute_filter_statistics(
    sub: pd.DataFrame,
    filter_name: FilterName,
    boundary_filtering_available: bool,
) -> dict[str, Any]:
    if sub.empty:
        return {
            "filter_name": filter_name,
            "n_galaxies": 0,
            "bic_win_baryon_only": 0,
            "bic_win_corrected_mond": 0,
            "bic_win_nfw": 0,
            "bic_win_tdf_kessence": 0,
            "median_delta_bic_tdf_minus_nfw": np.nan,
            "median_delta_bic_tdf_minus_corrected_mond": np.nan,
            "tdf_competitive_count": 0,
            "tdf_strong_win_vs_nfw": 0,
            "nfw_strong_win_vs_tdf": 0,
            "boundary_filtering_available": boundary_filtering_available,
        }

    delta_nfw = sub["delta_bic_tdf_vs_nfw"].astype(float)
    delta_mond = sub["_delta_bic_tdf_vs_mond"].astype(float)
    bic_tdf = sub["bic_tdf_kessence"].astype(float)
    bic_b = sub["bic_baryon_only"].astype(float)
    bic_n = sub["bic_nfw"].astype(float)
    bic_m = sub["_bic_mond"].astype(float)
    best_other = np.minimum(np.minimum(bic_b, bic_n), bic_m)

    return {
        "filter_name": filter_name,
        "n_galaxies": len(sub),
        "bic_win_baryon_only": int((sub["best_model_by_bic"] == "baryon_only").sum()),
        "bic_win_corrected_mond": int(
            (sub["best_model_by_bic"].isin(["corrected_mond", "mond"])).sum(),
        ),
        "bic_win_nfw": int((sub["best_model_by_bic"] == "nfw").sum()),
        "bic_win_tdf_kessence": int((sub["best_model_by_bic"] == "tdf_kessence").sum()),
        "median_delta_bic_tdf_minus_nfw": float(delta_nfw.median()),
        "median_delta_bic_tdf_minus_corrected_mond": float(delta_mond.median()),
        "tdf_competitive_count": int((bic_tdf - best_other < BIC_COMPETITIVE_DELTA).sum()),
        "tdf_strong_win_vs_nfw": int((delta_nfw < -6).sum()),
        "nfw_strong_win_vs_tdf": int((delta_nfw > 6).sum()),
        "boundary_filtering_available": boundary_filtering_available,
    }


def run_boundary_filtered_analysis(
    input_run: Path,
    output_dir: Path,
    *,
    audit_run: Path | None = None,
    boundary_flags_path: Path | None = None,
    overwrite_run: bool = False,
) -> BoundaryFilteredRunResult:
    """
    Run Step 1 analysis; write tables, figures, and report under output_dir.
    """
    input_run = Path(input_run)
    output_dir = Path(output_dir)
    tables = output_dir / "tables"
    reports = output_dir / "reports"
    figures = output_dir / "figures"
    metadata = output_dir / "metadata"
    for d in (tables, reports, figures, metadata):
        d.mkdir(parents=True, exist_ok=True)

    comparison_path = input_run / "tables" / "sparc_model_comparison_by_galaxy.csv"
    summary_path = input_run / "tables" / "sparc_real_calibration_summary.csv"
    comparison_df = load_comparison_table(comparison_path)

    if boundary_flags_path is None and audit_run is not None:
        candidate = audit_run / "tables" / "sparc_parameter_boundary_flags.csv"
        if candidate.is_file():
            boundary_flags_path = candidate
    galaxy_flags, bnd_available, bnd_source = build_galaxy_boundary_flags(
        comparison_df,
        boundary_flags_path,
        summary_path if summary_path.is_file() else None,
    )

    filter_names: list[FilterName] = [
        "all_galaxies",
        "exclude_any_nfw_or_tdf_boundary_hit",
        "exclude_severe_boundary_pressure",
    ]
    stats_rows = [
        compute_filter_statistics(
            _apply_filter(comparison_df, galaxy_flags, fn),
            fn,
            bnd_available,
        )
        for fn in filter_names
    ]
    comparison_by_filter = pd.DataFrame(stats_rows)

    comparison_by_filter.to_csv(
        tables / "boundary_filtered_model_comparison.csv",
        index=False,
    )
    galaxy_flags.to_csv(tables / "boundary_filter_galaxy_flags.csv", index=False)

    _write_figures(comparison_df, galaxy_flags, comparison_by_filter, figures)
    report = _build_report(
        input_run=input_run,
        audit_run=audit_run,
        comparison_by_filter=comparison_by_filter,
        galaxy_flags=galaxy_flags,
        boundary_filtering_available=bnd_available,
        boundary_data_source=bnd_source,
        boundary_flags_path=boundary_flags_path,
    )
    (reports / "boundary_filtered_analysis_report.md").write_text(
        report,
        encoding="utf-8",
    )

    return BoundaryFilteredRunResult(
        comparison_by_filter=comparison_by_filter,
        galaxy_flags=galaxy_flags,
        filtered_comparison=comparison_df,
        boundary_filtering_available=bnd_available,
        boundary_data_source=bnd_source,
        input_run=input_run,
        audit_run=audit_run,
    )


def _write_figures(
    comparison_df: pd.DataFrame,
    galaxy_flags: pd.DataFrame,
    comparison_by_filter: pd.DataFrame,
    figures_dir: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures_dir.mkdir(parents=True, exist_ok=True)
    merged = comparison_df.merge(galaxy_flags, on="galaxy_id", how="left")

    # ΔBIC TDF − NFW: all vs filtered
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(
        merged["delta_bic_tdf_vs_nfw"],
        bins=25,
        alpha=0.5,
        label="all galaxies",
    )
    if "in_exclude_any_nfw_or_tdf_boundary_hit" in merged.columns:
        clean = merged[merged["in_exclude_any_nfw_or_tdf_boundary_hit"].fillna(False)]
        if len(clean):
            ax.hist(
                clean["delta_bic_tdf_vs_nfw"],
                bins=25,
                alpha=0.5,
                label="exclude any NFW/TDF boundary hit",
            )
    ax.axvline(0, color="k", ls="--")
    ax.axvline(-6, color="gray", ls=":")
    ax.axvline(6, color="gray", ls=":")
    ax.set_xlabel("ΔBIC (TDF − NFW)")
    ax.set_ylabel("Galaxy count")
    ax.legend()
    ax.set_title("TDF vs NFW — all vs boundary-filtered")
    fig.tight_layout()
    fig.savefig(figures_dir / "delta_bic_tdf_nfw_all_vs_filtered.png", dpi=150)
    plt.close(fig)

    # BIC wins bar chart
    fig, ax = plt.subplots(figsize=(9, 5))
    filters = comparison_by_filter["filter_name"].astype(str)
    x = np.arange(len(filters))
    width = 0.18
    for i, (col, label) in enumerate(
        [
            ("bic_win_baryon_only", "baryon"),
            ("bic_win_corrected_mond", "corr. MOND"),
            ("bic_win_nfw", "NFW"),
            ("bic_win_tdf_kessence", "TDF"),
        ],
    ):
        ax.bar(x + (i - 1.5) * width, comparison_by_filter[col], width, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(filters, rotation=15, ha="right")
    ax.set_ylabel("BIC wins")
    ax.legend()
    ax.set_title("BIC wins by filter")
    fig.tight_layout()
    fig.savefig(figures_dir / "bic_wins_all_vs_filtered.png", dpi=150)
    plt.close(fig)

    # Boundary hit counts
    if galaxy_flags["boundary_data_source"].iloc[0] != "unavailable":
        fig, ax = plt.subplots(figsize=(6, 4))
        n_any = int(
            (galaxy_flags["any_nfw_boundary_hit"] | galaxy_flags["any_tdf_boundary_hit"]).sum(),
        )
        n_severe = int(galaxy_flags["severe_boundary_pressure"].sum())
        n_total = len(galaxy_flags)
        ax.bar(
            ["any NFW/TDF hit", "severe pressure", "clean (no any hit)"],
            [n_any, n_severe, n_total - n_any],
            color=["C1", "C3", "C2"],
        )
        ax.set_ylabel("Galaxy count")
        ax.set_title("Boundary pressure (NFW/TDF)")
        fig.tight_layout()
        fig.savefig(figures_dir / "boundary_hit_counts_by_model.png", dpi=150)
        plt.close(fig)


def _interpretation_text(comparison_by_filter: pd.DataFrame) -> str:
    all_row = comparison_by_filter[
        comparison_by_filter["filter_name"] == "all_galaxies"
    ].iloc[0]
    filt_row = comparison_by_filter[
        comparison_by_filter["filter_name"] == "exclude_any_nfw_or_tdf_boundary_hit"
    ]
    if filt_row.empty:
        return (
            "Boundary filtering data unavailable; only the all-galaxy subset is reported."
        )
    filt_row = filt_row.iloc[0]
    n_all = int(all_row["n_galaxies"])
    n_f = int(filt_row["n_galaxies"])
    med_all = float(all_row["median_delta_bic_tdf_minus_nfw"])
    med_f = float(filt_row["median_delta_bic_tdf_minus_nfw"])
    tdf_wins_all = int(all_row["bic_win_tdf_kessence"])
    tdf_wins_f = int(filt_row["bic_win_tdf_kessence"])
    nfw_wins_all = int(all_row["bic_win_nfw"])
    nfw_wins_f = int(filt_row["bic_win_nfw"])

    parts = [
        f"All galaxies (n={n_all}): median ΔBIC(TDF−NFW)={med_all:.2f}, "
        f"TDF BIC wins={tdf_wins_all}, NFW BIC wins={nfw_wins_all}.",
        f"After excluding any NFW/TDF boundary hit (n={n_f}): "
        f"median ΔBIC={med_f:.2f}, TDF wins={tdf_wins_f}, NFW wins={nfw_wins_f}.",
    ]
    if med_f > med_all + 1.0 and tdf_wins_f < tdf_wins_all:
        parts.append(
            "**TDF competitiveness weakens** after boundary filtering "
            "(higher median ΔBIC vs NFW, fewer TDF BIC wins)."
        )
    elif med_f < med_all - 1.0 and tdf_wins_f >= tdf_wins_all:
        parts.append(
            "**TDF competitiveness strengthens** after boundary filtering."
        )
    else:
        parts.append(
            "**TDF remains competitive** with NFW after boundary filtering; "
            "changes are modest on median ΔBIC and BIC win counts."
        )
    parts.append(
        "This is a rotation-curve calibration diagnostic only — not observational "
        "validation and not evidence that TDF replaces dark matter.",
    )
    return "\n\n".join(parts)


def _build_report(
    *,
    input_run: Path,
    audit_run: Path | None,
    comparison_by_filter: pd.DataFrame,
    galaxy_flags: pd.DataFrame,
    boundary_filtering_available: bool,
    boundary_data_source: str,
    boundary_flags_path: Path | None,
) -> str:
    lines = [
        "# SPARC boundary-filtered analysis (Step 1)",
        "",
        f"## ⚠️ {BANNER_BOUNDARY_FILTERED}",
        "",
        f"## ⚠️ {BANNER_CALIBRATION}",
        "",
        "## Input",
        "",
        f"- **Calibration run:** `{input_run}`",
        f"- **Audit run:** `{audit_run}`" if audit_run else "- **Audit run:** (not specified)",
        f"- **Boundary flags:** `{boundary_flags_path}`"
        if boundary_flags_path
        else "- **Boundary flags:** inferred or unavailable",
        f"- **Boundary data source:** {boundary_data_source}",
        f"- **Boundary filtering available:** {boundary_filtering_available}",
        "",
        "## Filters",
        "",
        "1. **all_galaxies** — no exclusion",
        "2. **exclude_any_nfw_or_tdf_boundary_hit** — drop galaxies with any NFW or TDF parameter at bound",
        "3. **exclude_severe_boundary_pressure** — drop if NFW or TDF `boundary_hit_count` ≥ 2, "
        "or NFW `v200`/`r_s` at bound, or TDF `beta_over_M` at bound",
        "",
        "## Model comparison by filter",
        "",
        "| Filter | N | BIC baryon | BIC corr. MOND | BIC NFW | BIC TDF | "
        "Med ΔBIC TDF−NFW | Med ΔBIC TDF−MOND | TDF competitive | TDF strong win | NFW strong win |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for _, row in comparison_by_filter.iterrows():
        lines.append(
            f"| {row['filter_name']} | {row['n_galaxies']} | "
            f"{row['bic_win_baryon_only']} | {row['bic_win_corrected_mond']} | "
            f"{row['bic_win_nfw']} | {row['bic_win_tdf_kessence']} | "
            f"{row['median_delta_bic_tdf_minus_nfw']:.2f} | "
            f"{row['median_delta_bic_tdf_minus_corrected_mond']:.2f} | "
            f"{row['tdf_competitive_count']} | {row['tdf_strong_win_vs_nfw']} | "
            f"{row['nfw_strong_win_vs_tdf']} |",
        )

    n_severe = int(galaxy_flags["severe_boundary_pressure"].sum())
    n_any = int(
        (galaxy_flags["any_nfw_boundary_hit"] | galaxy_flags["any_tdf_boundary_hit"]).sum(),
    )
    lines.extend(
        [
            "",
            "## Galaxy boundary summary",
            "",
            f"- Galaxies with any NFW/TDF boundary hit: **{n_any}** / {len(galaxy_flags)}",
            f"- Galaxies with severe boundary pressure: **{n_severe}** / {len(galaxy_flags)}",
            "",
            "## Interpretation",
            "",
            _interpretation_text(comparison_by_filter),
            "",
            "## Limitations",
            "",
            "- Analysis only; no new fitting.",
            "- Boundary flags from Phase 8A.1 audit refit (or coarse inference from summary).",
            "- Rotation curves only; no lensing or cosmology.",
            "- Does not prove TDF replaces dark matter or constitute full SPARC validation.",
            "",
        ],
    )
    return "\n".join(lines)

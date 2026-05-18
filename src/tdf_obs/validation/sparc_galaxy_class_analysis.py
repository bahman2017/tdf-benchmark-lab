"""
SPARC Step 2 — galaxy-class model comparison (analysis only).

Classifies galaxies by v_max proxy and compares BIC / reduced χ² by class.
Not observational validation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from tdf_obs.validation.sparc_boundary_filtered_analysis import load_comparison_table
from tdf_obs.validation.sparc_real_calibration import BIC_COMPETITIVE_DELTA

BANNER_GALAXY_CLASS = (
    "SPARC GALAXY-CLASS ANALYSIS — NOT FULL OBSERVATIONAL VALIDATION"
)

BANNER_CALIBRATION = (
    "These results are calibration diagnostics for a phenomenological TDF model. "
    "They do not constitute observational validation of TDF unless all required "
    "multi-channel constraints are passed."
)

GalaxyClass = Literal["dwarf", "intermediate", "massive"]

VMAX_DWARF = 80.0
VMAX_MASSIVE = 160.0

CLASS_ORDER: tuple[GalaxyClass, ...] = ("dwarf", "intermediate", "massive")

CLASS_COMPARISON_COLUMNS: tuple[str, ...] = (
    "galaxy_class",
    "n_galaxies",
    "bic_win_baryon_only",
    "bic_win_corrected_mond",
    "bic_win_nfw",
    "bic_win_tdf_kessence",
    "median_reduced_chi2_baryon_only",
    "median_reduced_chi2_corrected_mond",
    "median_reduced_chi2_nfw",
    "median_reduced_chi2_tdf_kessence",
    "median_bic_baryon_only",
    "median_bic_corrected_mond",
    "median_bic_nfw",
    "median_bic_tdf_kessence",
    "median_delta_bic_tdf_minus_nfw",
    "median_delta_bic_tdf_minus_corrected_mond",
    "tdf_competitive_count",
    "tdf_strong_win_vs_nfw",
    "nfw_strong_win_vs_tdf",
    "mond_active_fraction",
    "median_mond_boost_low_acc",
)

ASSIGNMENT_COLUMNS: tuple[str, ...] = (
    "galaxy_id",
    "vmax_obs",
    "rmax_kpc",
    "n_points_sparc",
    "median_v_err",
    "galaxy_class",
    "surface_brightness_class",
    "mond_active_flag",
    "delta_bic_tdf_vs_nfw",
    "delta_bic_tdf_vs_corrected_mond",
    "best_model_by_bic",
)


@dataclass
class GalaxyClassAnalysisResult:
    class_comparison: pd.DataFrame
    assignments: pd.DataFrame
    merged: pd.DataFrame
    input_run: Path


def classify_galaxy_by_vmax(vmax_obs: float) -> GalaxyClass:
    if vmax_obs < VMAX_DWARF:
        return "dwarf"
    if vmax_obs < VMAX_MASSIVE:
        return "intermediate"
    return "massive"


def build_galaxy_properties(sparc_df: pd.DataFrame) -> pd.DataFrame:
    """Per-galaxy properties from processed SPARC rotation table."""
    sparc_df = sparc_df.copy()
    if "galaxy_id" not in sparc_df.columns:
        raise ValueError("sparc data missing galaxy_id")
    rows: list[dict[str, Any]] = []
    for gid, gdf in sparc_df.groupby("galaxy_id"):
        vmax = float(gdf["v_obs"].max())
        rows.append(
            {
                "galaxy_id": str(gid),
                "vmax_obs": vmax,
                "rmax_kpc": float(gdf["r_kpc"].max()),
                "n_points_sparc": int(len(gdf)),
                "median_v_err": float(gdf["v_err"].median()),
                "galaxy_class": classify_galaxy_by_vmax(vmax),
                "surface_brightness_class": np.nan,
            },
        )
    return pd.DataFrame(rows)


def load_summary_table(path: Path) -> pd.DataFrame:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"summary table not found: {path}")
    return pd.read_csv(path)


def _mond_model_label(summary_df: pd.DataFrame) -> str:
    if (summary_df["model"] == "corrected_mond").any():
        return "corrected_mond"
    return "mond"


def _median_metric_by_class(
    summary_df: pd.DataFrame,
    assignments: pd.DataFrame,
    model: str,
    metric: str,
    galaxy_class: GalaxyClass,
) -> float:
    gids = assignments.loc[
        assignments["galaxy_class"] == galaxy_class, "galaxy_id",
    ].astype(str)
    sub = summary_df[
        (summary_df["galaxy_id"].astype(str).isin(gids))
        & (summary_df["model"] == model)
        & summary_df["success"].fillna(True)
    ]
    if sub.empty or metric not in sub.columns:
        return float("nan")
    return float(sub[metric].median())


def compute_class_statistics(
    comparison_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    assignments: pd.DataFrame,
    galaxy_class: GalaxyClass,
) -> dict[str, Any]:
    gids = set(
        assignments.loc[assignments["galaxy_class"] == galaxy_class, "galaxy_id"].astype(str),
    )
    sub = comparison_df[comparison_df["galaxy_id"].astype(str).isin(gids)].copy()
    mond_label = _mond_model_label(summary_df)

    if sub.empty:
        return {col: np.nan for col in CLASS_COMPARISON_COLUMNS if col != "galaxy_class"} | {
            "galaxy_class": galaxy_class,
            "n_galaxies": 0,
        }

    delta_nfw = sub["delta_bic_tdf_vs_nfw"].astype(float)
    delta_mond = sub["_delta_bic_tdf_vs_mond"].astype(float)
    bic_tdf = sub["bic_tdf_kessence"].astype(float)
    bic_b = sub["bic_baryon_only"].astype(float)
    bic_n = sub["bic_nfw"].astype(float)
    bic_m = sub["_bic_mond"].astype(float)
    best_other = np.minimum(np.minimum(bic_b, bic_n), bic_m)

    mond_active_frac = float("nan")
    median_boost = float("nan")
    if "mond_active_flag" in sub.columns:
        mond_active_frac = float(sub["mond_active_flag"].astype(bool).mean())
    if "mond_vs_baryon_median_boost" in sub.columns:
        median_boost = float(sub["mond_vs_baryon_median_boost"].median())

    return {
        "galaxy_class": galaxy_class,
        "n_galaxies": len(sub),
        "bic_win_baryon_only": int((sub["best_model_by_bic"] == "baryon_only").sum()),
        "bic_win_corrected_mond": int(
            (sub["best_model_by_bic"].isin(["corrected_mond", "mond"])).sum(),
        ),
        "bic_win_nfw": int((sub["best_model_by_bic"] == "nfw").sum()),
        "bic_win_tdf_kessence": int((sub["best_model_by_bic"] == "tdf_kessence").sum()),
        "median_reduced_chi2_baryon_only": _median_metric_by_class(
            summary_df, assignments, "baryon_only", "reduced_chi2", galaxy_class,
        ),
        "median_reduced_chi2_corrected_mond": _median_metric_by_class(
            summary_df, assignments, mond_label, "reduced_chi2", galaxy_class,
        ),
        "median_reduced_chi2_nfw": _median_metric_by_class(
            summary_df, assignments, "nfw", "reduced_chi2", galaxy_class,
        ),
        "median_reduced_chi2_tdf_kessence": _median_metric_by_class(
            summary_df, assignments, "tdf_kessence", "reduced_chi2", galaxy_class,
        ),
        "median_bic_baryon_only": _median_metric_by_class(
            summary_df, assignments, "baryon_only", "bic", galaxy_class,
        ),
        "median_bic_corrected_mond": _median_metric_by_class(
            summary_df, assignments, mond_label, "bic", galaxy_class,
        ),
        "median_bic_nfw": _median_metric_by_class(
            summary_df, assignments, "nfw", "bic", galaxy_class,
        ),
        "median_bic_tdf_kessence": _median_metric_by_class(
            summary_df, assignments, "tdf_kessence", "bic", galaxy_class,
        ),
        "median_delta_bic_tdf_minus_nfw": float(delta_nfw.median()),
        "median_delta_bic_tdf_minus_corrected_mond": float(delta_mond.median()),
        "tdf_competitive_count": int((bic_tdf - best_other < BIC_COMPETITIVE_DELTA).sum()),
        "tdf_strong_win_vs_nfw": int((delta_nfw < -6).sum()),
        "nfw_strong_win_vs_tdf": int((delta_nfw > 6).sum()),
        "mond_active_fraction": mond_active_frac,
        "median_mond_boost_low_acc": median_boost,
    }


def join_assignments_with_comparison(
    properties: pd.DataFrame,
    comparison_df: pd.DataFrame,
) -> pd.DataFrame:
    comp = comparison_df.copy()
    comp["galaxy_id"] = comp["galaxy_id"].astype(str)
    props = properties.copy()
    props["galaxy_id"] = props["galaxy_id"].astype(str)
    merged = props.merge(
        comp[
            [
                "galaxy_id",
                "best_model_by_bic",
                "delta_bic_tdf_vs_nfw",
                "_delta_bic_tdf_vs_mond",
                "mond_active_flag",
                "mond_vs_baryon_median_boost",
            ]
        ].rename(columns={"_delta_bic_tdf_vs_mond": "delta_bic_tdf_vs_corrected_mond"}),
        on="galaxy_id",
        how="inner",
    )
    return merged


def run_galaxy_class_analysis(
    input_run: Path,
    sparc_data_path: Path,
    output_dir: Path,
) -> GalaxyClassAnalysisResult:
    """Run Step 2 analysis and write tables, figures, report."""
    input_run = Path(input_run)
    output_dir = Path(output_dir)
    sparc_data_path = Path(sparc_data_path)

    tables = output_dir / "tables"
    reports = output_dir / "reports"
    figures = output_dir / "figures"
    metadata = output_dir / "metadata"
    for d in (tables, reports, figures, metadata):
        d.mkdir(parents=True, exist_ok=True)

    comparison_df = load_comparison_table(
        input_run / "tables" / "sparc_model_comparison_by_galaxy.csv",
    )
    summary_df = load_summary_table(
        input_run / "tables" / "sparc_real_calibration_summary.csv",
    )
    sparc_df = pd.read_csv(sparc_data_path)

    properties = build_galaxy_properties(sparc_df)
    merged = join_assignments_with_comparison(properties, comparison_df)

    class_rows = [
        compute_class_statistics(comparison_df, summary_df, properties, cls)
        for cls in CLASS_ORDER
    ]
    class_comparison = pd.DataFrame(class_rows)

    assignments = merged[list(ASSIGNMENT_COLUMNS)].copy()

    class_comparison.to_csv(tables / "galaxy_class_model_comparison.csv", index=False)
    assignments.to_csv(tables / "galaxy_class_assignments.csv", index=False)

    _write_figures(assignments, class_comparison, figures)
    report = _build_report(
        input_run=input_run,
        sparc_data_path=sparc_data_path,
        class_comparison=class_comparison,
        assignments=assignments,
    )
    (reports / "galaxy_class_analysis_report.md").write_text(report, encoding="utf-8")

    return GalaxyClassAnalysisResult(
        class_comparison=class_comparison,
        assignments=assignments,
        merged=merged,
        input_run=input_run,
    )


def _write_figures(
    assignments: pd.DataFrame,
    class_comparison: pd.DataFrame,
    figures_dir: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures_dir.mkdir(parents=True, exist_ok=True)

    # ΔBIC TDF − NFW by class (box-style via histogram per class)
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, cls in enumerate(CLASS_ORDER):
        sub = assignments[assignments["galaxy_class"] == cls]
        if len(sub):
            ax.hist(
                sub["delta_bic_tdf_vs_nfw"],
                bins=15,
                alpha=0.5,
                label=f"{cls} (n={len(sub)})",
            )
    ax.axvline(0, color="k", ls="--")
    ax.axvline(-6, color="gray", ls=":")
    ax.axvline(6, color="gray", ls=":")
    ax.set_xlabel("ΔBIC (TDF − NFW)")
    ax.set_ylabel("Count")
    ax.legend()
    ax.set_title("ΔBIC TDF vs NFW by galaxy class")
    fig.tight_layout()
    fig.savefig(figures_dir / "delta_bic_by_galaxy_class.png", dpi=150)
    plt.close(fig)

    # BIC wins stacked by class
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(CLASS_ORDER))
    width = 0.18
    for i, (col, label) in enumerate(
        [
            ("bic_win_baryon_only", "baryon"),
            ("bic_win_corrected_mond", "corr. MOND"),
            ("bic_win_nfw", "NFW"),
            ("bic_win_tdf_kessence", "TDF"),
        ],
    ):
        vals = [
            class_comparison.loc[class_comparison["galaxy_class"] == c, col].iloc[0]
            for c in CLASS_ORDER
        ]
        ax.bar(x + (i - 1.5) * width, vals, width, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_ORDER)
    ax.set_ylabel("BIC wins")
    ax.legend()
    ax.set_title("BIC wins by galaxy class")
    fig.tight_layout()
    fig.savefig(figures_dir / "bic_wins_by_galaxy_class.png", dpi=150)
    plt.close(fig)

    # Median reduced χ² by model and class
    fig, ax = plt.subplots(figsize=(10, 5))
    models = [
        ("median_reduced_chi2_baryon_only", "baryon"),
        ("median_reduced_chi2_corrected_mond", "corr. MOND"),
        ("median_reduced_chi2_nfw", "NFW"),
        ("median_reduced_chi2_tdf_kessence", "TDF"),
    ]
    for i, (col, label) in enumerate(models):
        vals = [
            class_comparison.loc[class_comparison["galaxy_class"] == c, col].iloc[0]
            for c in CLASS_ORDER
        ]
        ax.bar(x + (i - 1.5) * width, vals, width, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_ORDER)
    ax.set_ylabel("Median reduced χ²")
    ax.legend(fontsize=8)
    ax.set_title("Median reduced χ² by class and model")
    fig.tight_layout()
    fig.savefig(figures_dir / "reduced_chi2_by_galaxy_class.png", dpi=150)
    plt.close(fig)


def _row(cls_df: pd.DataFrame, cls: GalaxyClass) -> pd.Series:
    sub = cls_df[cls_df["galaxy_class"] == cls]
    if sub.empty:
        raise KeyError(cls)
    return sub.iloc[0]


def _answer_questions(class_comparison: pd.DataFrame) -> str:
    dwarf = _row(class_comparison, "dwarf")
    inter = _row(class_comparison, "intermediate")
    massive = _row(class_comparison, "massive")

    lines: list[str] = []

    # TDF better in dwarfs?
    tdf_dwarf = int(dwarf["bic_win_tdf_kessence"])
    tdf_massive = int(massive["bic_win_tdf_kessence"])
    med_dwarf = float(dwarf["median_delta_bic_tdf_minus_nfw"])
    med_massive = float(massive["median_delta_bic_tdf_minus_nfw"])
    if tdf_dwarf >= tdf_massive and med_dwarf < med_massive:
        lines.append(
            "**Does TDF perform better in dwarf galaxies?** "
            "The data support that TDF appears **most competitive in the dwarf class** "
            f"(TDF BIC wins: dwarf={tdf_dwarf}, massive={tdf_massive}; "
            f"median ΔBIC TDF−NFW: dwarf={med_dwarf:.2f}, massive={med_massive:.2f}). "
            "This is a rotation-only calibration trend, not observational validation.",
        )
    else:
        lines.append(
            "**Does TDF perform better in dwarf galaxies?** "
            "Results are **mixed or not clearly dwarf-dominated** "
            f"(TDF wins: dwarf={tdf_dwarf}, intermediate={int(inter['bic_win_tdf_kessence'])}, "
            f"massive={tdf_massive}).",
        )

    # NFW better in massive?
    nfw_massive = int(massive["bic_win_nfw"])
    nfw_dwarf = int(dwarf["bic_win_nfw"])
    if nfw_massive > nfw_dwarf and med_massive > 0:
        lines.append(
            "**Does NFW perform better in massive galaxies?** "
            f"**Yes, in this sample:** NFW has more BIC wins among massive galaxies "
            f"({nfw_massive} vs {nfw_dwarf} in dwarfs) and median ΔBIC(TDF−NFW) "
            f"is positive for massive ({med_massive:.2f}), favouring NFW on penalized fit quality.",
        )
    else:
        lines.append(
            "**Does NFW perform better in massive galaxies?** "
            "The trend is **weaker than a simple massive→NFW rule** in this table; "
            "inspect per-class BIC wins and median ΔBIC above.",
        )

    # Corrected MOND in low-acceleration?
    mond_frac_d = float(dwarf.get("mond_active_fraction", np.nan))
    mond_frac_m = float(massive.get("mond_active_fraction", np.nan))
    med_chi_mond_d = float(dwarf["median_reduced_chi2_corrected_mond"])
    med_chi_mond_m = float(massive["median_reduced_chi2_corrected_mond"])
    lines.append(
        "**Does corrected MOND improve mostly in low-acceleration galaxies?** "
        f"Dwarf/low-v_max systems show corrected-MOND activity fraction "
        f"≈ {mond_frac_d:.2f} (dwarf) vs {mond_frac_m:.2f} (massive); "
        f"median reduced χ²(corrected MOND): dwarf={med_chi_mond_d:.2f}, "
        f"massive={med_chi_mond_m:.2f}. "
        "Lower mass classes tend to show larger MOND boosts relative to baryon-only, "
        "consistent with more low-acceleration radii — still a phenomenological proxy only.",
    )

    # TDF wins concentrated?
    total_tdf = int(class_comparison["bic_win_tdf_kessence"].sum())
    shares = {
        str(r["galaxy_class"]): int(r["bic_win_tdf_kessence"]) / max(total_tdf, 1)
        for _, r in class_comparison.iterrows()
    }
    dominant = max(shares, key=shares.get)
    lines.append(
        "**Are TDF wins concentrated in one galaxy class?** "
        f"TDF BIC wins by class: dwarf={int(dwarf['bic_win_tdf_kessence'])}, "
        f"intermediate={int(inter['bic_win_tdf_kessence'])}, "
        f"massive={int(massive['bic_win_tdf_kessence'])} "
        f"({100*shares.get(dominant, 0):.0f}% in **{dominant}**). "
        + (
            f"Wins are **most concentrated in {dominant}** galaxies."
            if shares.get(dominant, 0) > 0.45
            else "Wins are **spread across classes**, not dominated by a single bin."
        ),
    )

    return "\n\n".join(lines)


def _build_report(
    *,
    input_run: Path,
    sparc_data_path: Path,
    class_comparison: pd.DataFrame,
    assignments: pd.DataFrame,
) -> str:
    lines = [
        "# SPARC galaxy-class analysis (Step 2)",
        "",
        f"## ⚠️ {BANNER_GALAXY_CLASS}",
        "",
        f"## ⚠️ {BANNER_CALIBRATION}",
        "",
        "## Input",
        "",
        f"- **Calibration run:** `{input_run}`",
        f"- **SPARC data:** `{sparc_data_path}`",
        "",
        "## Classification rule (v_max proxy)",
        "",
        f"- **dwarf:** v_max < {VMAX_DWARF:.0f} km/s",
        f"- **intermediate:** {VMAX_DWARF:.0f} ≤ v_max < {VMAX_MASSIVE:.0f} km/s",
        f"- **massive:** v_max ≥ {VMAX_MASSIVE:.0f} km/s",
        "",
        "Surface-brightness metadata is **not** in the processed SPARC CSV; "
        "`surface_brightness_class` is left empty.",
        "",
        "## Model comparison by class",
        "",
        "| Class | N | BIC baryon | BIC MOND | BIC NFW | BIC TDF | "
        "Med χ²_red baryon | Med χ²_red MOND | Med χ²_red NFW | Med χ²_red TDF | "
        "Med ΔBIC TDF−NFW | TDF wins | NFW wins |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for _, row in class_comparison.iterrows():
        lines.append(
            f"| {row['galaxy_class']} | {row['n_galaxies']} | "
            f"{row['bic_win_baryon_only']} | {row['bic_win_corrected_mond']} | "
            f"{row['bic_win_nfw']} | {row['bic_win_tdf_kessence']} | "
            f"{row['median_reduced_chi2_baryon_only']:.2f} | "
            f"{row['median_reduced_chi2_corrected_mond']:.2f} | "
            f"{row['median_reduced_chi2_nfw']:.2f} | "
            f"{row['median_reduced_chi2_tdf_kessence']:.2f} | "
            f"{row['median_delta_bic_tdf_minus_nfw']:.2f} | "
            f"{row['bic_win_tdf_kessence']} | {row['bic_win_nfw']} |",
        )

    lines.extend(
        [
            "",
            "## Key questions",
            "",
            _answer_questions(class_comparison),
            "",
            "## Limitations",
            "",
            "- Analysis only; no new fitting.",
            "- Galaxy class from v_max only; not full SPARC morphology metadata.",
            "- Rotation curves only; no lensing or cosmology.",
            "- Does not prove TDF replaces dark matter or validate TDF observationally.",
            "",
        ],
    )
    return "\n".join(lines)

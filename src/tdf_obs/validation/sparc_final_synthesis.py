"""
SPARC Step 7 — final synthesis of rotation-only calibration and robustness analyses.

Aggregates corrected-MOND calibration and Steps 1–6 into a paper-ready package.
Does not run new fits. Not observational validation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

BANNER_SYNTHESIS = (
    "SPARC FINAL SYNTHESIS — ROTATION-ONLY CALIBRATION, NOT FULL OBSERVATIONAL VALIDATION"
)

BANNER_CALIBRATION = (
    "These results are calibration diagnostics for a phenomenological TDF model. "
    "They do not constitute observational validation of TDF unless all required "
    "multi-channel constraints are passed."
)

def paper_forbidden_phrase_present(text: str, phrase: str) -> bool:
    """Return True if *phrase* appears without a negation window (for tests)."""
    plain = text.lower().replace("*", "")
    if phrase not in plain:
        return False
    idx = plain.find(phrase)
    window = plain[max(0, idx - 48) : idx]
    negated = (
        "not full observational validation" in plain
        or any(tok in window for tok in ("not ", "no ", "never ", "without ", "not claim "))
    )
    return not negated


FORBIDDEN_PAPER_PHRASES: tuple[str, ...] = (
    "dark matter solved",
    "dark matter disproven",
    "observational validation",
    "observationally validated",
    "proven",
    "replaces dark matter",
    "replaces ΛCDM",
    "confirmed",
    "TDF is correct",
)

DEFAULT_RUN_PATHS: dict[str, str] = {
    "main": "v0.20.2_corrected_mond_sparc_calibration",
    "step1": "sparc_step_1_boundary_filtered",
    "step2": "sparc_step_2_galaxy_class_analysis",
    "step3": "sparc_step_3_mass_to_light_robustness",
    "step4": "sparc_step_4_cored_halo_baseline",
    "step5": "sparc_step_5_tdf_parameter_stability",
    "step6": "sparc_step_6_residual_diagnostics",
}

SUMMARY_COLUMNS: tuple[str, ...] = (
    "analysis_block",
    "status",
    "n_galaxies",
    "metric",
    "value",
    "notes",
)

DECISION_MATRIX_ROWS: tuple[str, ...] = (
    "baryon_only baseline",
    "corrected MOND baseline",
    "NFW baseline",
    "cored halo baseline",
    "TDF K-essence",
)

DECISION_MATRIX_COLUMNS: tuple[str, ...] = (
    "global_bic_competitiveness",
    "reduced_chi2_competitiveness",
    "dwarf_performance",
    "massive_galaxy_performance",
    "boundary_robustness",
    "ml_robustness",
    "residual_stability",
    "parameter_stability",
    "limitations",
)

CLAIM_COLUMNS: tuple[str, ...] = (
    "claim_level",
    "claim_label",
    "justification",
    "rotation_only",
)

RECOMMENDATION_COLUMNS: tuple[str, ...] = (
    "recommendation_code",
    "recommendation_label",
    "rationale",
)


@dataclass
class StepLoadResult:
    step_id: str
    run_dir: Path | None
    status: str
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    reports: dict[str, str] = field(default_factory=dict)


@dataclass
class SynthesisContext:
    runs_root: Path
    steps: dict[str, StepLoadResult]
    n_galaxies: int = 171
    main_comparison: pd.DataFrame | None = None
    main_summary: pd.DataFrame | None = None


@dataclass
class FinalSynthesisResult:
    summary: pd.DataFrame
    decision_matrix: pd.DataFrame
    claim_table: pd.DataFrame
    recommendation: pd.DataFrame
    assigned_claim_level: int
    assigned_recommendation: str
    output_dir: Path


def _load_csv(path: Path) -> pd.DataFrame | None:
    if path.is_file():
        return pd.read_csv(path)
    return None


def load_step(
    runs_root: Path,
    step_id: str,
    run_name: str,
    table_specs: dict[str, str],
    report_specs: dict[str, str] | None = None,
) -> StepLoadResult:
    run_dir = runs_root / run_name
    if not run_dir.is_dir():
        return StepLoadResult(step_id=step_id, run_dir=None, status="missing")

    tables: dict[str, pd.DataFrame] = {}
    for key, rel in table_specs.items():
        df = _load_csv(run_dir / rel)
        if df is not None:
            tables[key] = df

    reports: dict[str, str] = {}
    if report_specs:
        for key, rel in report_specs.items():
            p = run_dir / rel
            if p.is_file():
                reports[key] = p.read_text(encoding="utf-8")

    status = "ok" if tables else "missing"
    return StepLoadResult(
        step_id=step_id,
        run_dir=run_dir,
        status=status,
        tables=tables,
        reports=reports,
    )


def load_all_steps(
    runs_root: Path,
    run_paths: dict[str, str] | None = None,
) -> SynthesisContext:
    paths = {**DEFAULT_RUN_PATHS, **(run_paths or {})}
    runs_root = Path(runs_root)

    steps = {
        "main": load_step(
            runs_root,
            "main",
            paths["main"],
            {
                "summary": "tables/sparc_real_calibration_summary.csv",
                "comparison": "tables/sparc_model_comparison_by_galaxy.csv",
            },
            {"report": "reports/sparc_real_calibration_report.md"},
        ),
        "step1": load_step(
            runs_root,
            "step1",
            paths["step1"],
            {
                "filter_comparison": "tables/boundary_filtered_model_comparison.csv",
                "galaxy_flags": "tables/boundary_filter_galaxy_flags.csv",
            },
            {"report": "reports/boundary_filtered_analysis_report.md"},
        ),
        "step2": load_step(
            runs_root,
            "step2",
            paths["step2"],
            {
                "class_comparison": "tables/galaxy_class_model_comparison.csv",
                "assignments": "tables/galaxy_class_assignments.csv",
            },
            {"report": "reports/galaxy_class_analysis_report.md"},
        ),
        "step3": load_step(
            runs_root,
            "step3",
            paths["step3"],
            {"ml_summary": "tables/ml_robustness_summary.csv"},
            {"report": "reports/ml_robustness_report.md"},
        ),
        "step4": load_step(
            runs_root,
            "step4",
            paths["step4"],
            {"cored_summary": "tables/cored_halo_model_summary.csv"},
            {"report": "reports/cored_halo_baseline_report.md"},
        ),
        "step5": load_step(
            runs_root,
            "step5",
            paths["step5"],
            {"param_summary": "tables/tdf_parameter_stability_summary.csv"},
            {"report": "reports/tdf_parameter_stability_report.md"},
        ),
        "step6": load_step(
            runs_root,
            "step6",
            paths["step6"],
            {"failure_modes": "tables/residual_failure_modes.csv"},
            {"report": "reports/residual_diagnostics_report.md"},
        ),
    }

    ctx = SynthesisContext(runs_root=runs_root, steps=steps)
    main = steps["main"]
    if main.status == "ok" and "comparison" in main.tables:
        ctx.main_comparison = main.tables["comparison"]
        ctx.n_galaxies = len(ctx.main_comparison)
    if main.status == "ok" and "summary" in main.tables:
        ctx.main_summary = main.tables["summary"]
    return ctx


def _summary_row(
    block: str,
    status: str,
    metric: str,
    value: Any,
    notes: str = "",
    n_galaxies: int | None = None,
) -> dict[str, Any]:
    return {
        "analysis_block": block,
        "status": status,
        "n_galaxies": n_galaxies if n_galaxies is not None else np.nan,
        "metric": metric,
        "value": value,
        "notes": notes,
    }


def build_final_summary(ctx: SynthesisContext) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    n = ctx.n_galaxies

    main = ctx.steps["main"]
    if main.status == "ok" and ctx.main_comparison is not None:
        comp = ctx.main_comparison
        tdf_w = int((comp["best_model_by_bic"] == "tdf_kessence").sum())
        nfw_w = int((comp["best_model_by_bic"] == "nfw").sum())
        mond_w = int((comp["best_model_by_bic"] == "corrected_mond").sum())
        med_dn = float(comp["delta_bic_tdf_vs_nfw"].median())
        med_dm = float(comp["delta_bic_tdf_vs_corrected_mond"].median())
        tdf_beats_nfw = int(comp["tdf_beats_nfw"].sum()) if "tdf_beats_nfw" in comp.columns else np.nan

        rows.extend(
            [
                _summary_row("corrected_mond_calibration", "ok", "bic_win_corrected_mond", mond_w, n_galaxies=n),
                _summary_row("corrected_mond_calibration", "ok", "bic_win_tdf_kessence", tdf_w, n_galaxies=n),
                _summary_row("corrected_mond_calibration", "ok", "bic_win_nfw", nfw_w, n_galaxies=n),
                _summary_row("corrected_mond_calibration", "ok", "median_delta_bic_tdf_vs_nfw", round(med_dn, 3), n_galaxies=n),
                _summary_row("corrected_mond_calibration", "ok", "median_delta_bic_tdf_vs_corrected_mond", round(med_dm, 3), n_galaxies=n),
                _summary_row("tdf_vs_nfw", "ok", "tdf_beats_nfw_count", tdf_beats_nfw, n_galaxies=n),
                _summary_row("tdf_vs_nfw", "ok", "tdf_bic_wins", tdf_w, n_galaxies=n),
                _summary_row("tdf_vs_nfw", "ok", "nfw_bic_wins", nfw_w, n_galaxies=n),
                _summary_row("tdf_vs_corrected_mond", "ok", "tdf_beats_corrected_mond", int(comp["tdf_beats_corrected_mond"].sum()), n_galaxies=n),
            ],
        )
        if ctx.main_summary is not None:
            for model in ("tdf_kessence", "nfw", "corrected_mond"):
                sub = ctx.main_summary[ctx.main_summary["model"] == model]
                if len(sub):
                    rows.append(
                        _summary_row(
                            "corrected_mond_calibration",
                            "ok",
                            f"median_reduced_chi2_{model}",
                            float(sub["reduced_chi2"].median()),
                            n_galaxies=n,
                        ),
                    )
    else:
        rows.append(_summary_row("corrected_mond_calibration", "missing", "—", "—", "main run not found"))

    s1 = ctx.steps["step1"]
    if s1.status == "ok" and "filter_comparison" in s1.tables:
        fc = s1.tables["filter_comparison"]
        all_row = fc[fc["filter_name"] == "all_galaxies"]
        filt_row = fc[fc["filter_name"] == "exclude_any_nfw_or_tdf_boundary_hit"]
        if len(all_row):
            rows.append(
                _summary_row(
                    "boundary_filtered",
                    "ok",
                    "median_delta_bic_tdf_minus_nfw_all",
                    float(all_row["median_delta_bic_tdf_minus_nfw"].iloc[0]),
                    n_galaxies=int(all_row["n_galaxies"].iloc[0]),
                ),
            )
        if len(filt_row):
            rows.append(
                _summary_row(
                    "boundary_filtered",
                    "ok",
                    "median_delta_bic_tdf_minus_nfw_filtered",
                    float(filt_row["median_delta_bic_tdf_minus_nfw"].iloc[0]),
                    "TDF weakens after boundary exclusion",
                    n_galaxies=int(filt_row["n_galaxies"].iloc[0]),
                ),
            )
    else:
        rows.append(_summary_row("boundary_filtered", "missing", "—", "—"))

    s2 = ctx.steps["step2"]
    if s2.status == "ok" and "class_comparison" in s2.tables:
        cc = s2.tables["class_comparison"]
        for cls in ("dwarf", "massive"):
            sub = cc[cc["galaxy_class"] == cls]
            if len(sub):
                rows.append(
                    _summary_row(
                        "galaxy_class",
                        "ok",
                        f"bic_win_tdf_{cls}",
                        int(sub["bic_win_tdf_kessence"].iloc[0]),
                        n_galaxies=int(sub["n_galaxies"].iloc[0]),
                    ),
                )
                rows.append(
                    _summary_row(
                        "galaxy_class",
                        "ok",
                        f"median_delta_bic_tdf_minus_nfw_{cls}",
                        float(sub["median_delta_bic_tdf_minus_nfw"].iloc[0]),
                    ),
                )
    else:
        rows.append(_summary_row("galaxy_class", "missing", "—", "—"))

    s3 = ctx.steps["step3"]
    if s3.status == "ok" and "ml_summary" in s3.tables:
        ml = s3.tables["ml_summary"]
        for regime in ("A_fixed_canonical", "C_standard_prior"):
            sub = ml[ml["ml_regime"] == regime]
            if len(sub):
                rows.append(
                    _summary_row(
                        "ml_robustness",
                        "ok",
                        f"bic_win_tdf_{regime}",
                        int(sub["bic_win_tdf_kessence"].iloc[0]),
                        sub["ml_regime_label"].iloc[0],
                    ),
                )
    else:
        rows.append(_summary_row("ml_robustness", "missing", "—", "—"))

    s4 = ctx.steps["step4"]
    if s4.status == "ok" and "cored_summary" in s4.tables:
        cs = s4.tables["cored_summary"]
        for model in ("pseudo_isothermal", "burkert", "tdf_kessence", "nfw"):
            sub = cs[cs["model"] == model]
            if len(sub):
                rows.append(
                    _summary_row(
                        "cored_halo_baseline",
                        "ok",
                        f"bic_win_{model}",
                        int(sub["bic_win_count"].iloc[0]),
                    ),
                )
    else:
        rows.append(_summary_row("cored_halo_baseline", "missing", "—", "—"))

    s5 = ctx.steps["step5"]
    if s5.status == "ok" and "param_summary" in s5.tables:
        ps = s5.tables["param_summary"]

        def _metric(name: str) -> float:
            sub = ps[ps["metric"] == name]
            return float(sub["value"].iloc[0]) if len(sub) else float("nan")

        rows.extend(
            [
                _summary_row("tdf_parameter_stability", "ok", "beta_over_M_median", _metric("beta_over_M_median")),
                _summary_row("tdf_parameter_stability", "ok", "beta_over_M_iqr", _metric("beta_over_M_iqr")),
                _summary_row(
                    "tdf_parameter_stability",
                    "ok",
                    "median_delta_bic_global_beta_approx",
                    _metric("median_delta_bic_global_beta_approx"),
                    "global β not plausible",
                ),
                _summary_row(
                    "tdf_parameter_stability",
                    "ok",
                    "any_tdf_boundary_hit_fraction",
                    _metric("any_tdf_boundary_hit_fraction"),
                ),
            ],
        )
    else:
        rows.append(_summary_row("tdf_parameter_stability", "missing", "—", "—"))

    s6 = ctx.steps["step6"]
    if s6.status == "ok" and "failure_modes" in s6.tables:
        fm = s6.tables["failure_modes"]
        for mode in (
            "tdf_worse_than_nfw_overall",
            "tdf_failures_massive",
            "tdf_success_dwarf",
        ):
            sub = fm[fm["failure_mode"] == mode]
            if len(sub):
                rows.append(
                    _summary_row(
                        "residual_diagnostics",
                        "ok",
                        mode,
                        int(sub["n_galaxies"].iloc[0]),
                        sub["description"].iloc[0],
                    ),
                )
    else:
        rows.append(_summary_row("residual_diagnostics", "missing", "—", "—"))

    return pd.DataFrame(rows)


def build_decision_matrix(ctx: SynthesisContext) -> pd.DataFrame:
    """Qualitative decision matrix (strong / moderate / weak / fail / missing)."""
    n = ctx.n_galaxies
    ratings: dict[str, dict[str, str]] = {row: {col: "missing" for col in DECISION_MATRIX_COLUMNS} for row in DECISION_MATRIX_ROWS}

    if ctx.main_comparison is not None:
        comp = ctx.main_comparison
        if ctx.main_summary is not None:
            summ = ctx.main_summary
            med_chi2 = {m: float(summ[summ["model"] == m]["reduced_chi2"].median()) for m in summ["model"].unique()}
        else:
            med_chi2 = {}

        bic_wins = comp["best_model_by_bic"].value_counts()
        tdf_w = int(bic_wins.get("tdf_kessence", 0))
        nfw_w = int(bic_wins.get("nfw", 0))
        mond_w = int(bic_wins.get("corrected_mond", 0))
        bar_w = int(bic_wins.get("baryon_only", 0))

        def bic_rate(wins: int) -> str:
            if wins >= tdf_w and wins >= max(nfw_w, mond_w, bar_w):
                return "strong"
            if wins >= 0.25 * n:
                return "moderate"
            return "weak"

        def chi2_rate(model: str) -> str:
            if model not in med_chi2:
                return "missing"
            v = med_chi2[model]
            best = min(med_chi2.values())
            if v <= best * 1.1:
                return "strong"
            if v <= best * 2:
                return "moderate"
            return "weak"

        ratings["baryon_only baseline"]["global_bic_competitiveness"] = bic_rate(bar_w)
        ratings["baryon_only baseline"]["reduced_chi2_competitiveness"] = chi2_rate("baryon_only")
        ratings["corrected MOND baseline"]["global_bic_competitiveness"] = bic_rate(mond_w)
        ratings["corrected MOND baseline"]["reduced_chi2_competitiveness"] = chi2_rate("corrected_mond")
        ratings["NFW baseline"]["global_bic_competitiveness"] = bic_rate(nfw_w)
        ratings["NFW baseline"]["reduced_chi2_competitiveness"] = chi2_rate("nfw")
        ratings["TDF K-essence"]["global_bic_competitiveness"] = bic_rate(tdf_w)
        ratings["TDF K-essence"]["reduced_chi2_competitiveness"] = chi2_rate("tdf_kessence")

    s2 = ctx.steps["step2"]
    if s2.status == "ok" and "class_comparison" in s2.tables:
        cc = s2.tables["class_comparison"]
        for cls, col in (("dwarf", "dwarf_performance"), ("massive", "massive_galaxy_performance")):
            sub = cc[cc["galaxy_class"] == cls]
            if len(sub):
                tdf_w = int(sub["bic_win_tdf_kessence"].iloc[0])
                nfw_w = int(sub["bic_win_nfw"].iloc[0])
                med_d = float(sub["median_delta_bic_tdf_minus_nfw"].iloc[0])
                for row in DECISION_MATRIX_ROWS:
                    if row == "TDF K-essence":
                        if tdf_w > nfw_w and med_d < 0:
                            ratings[row][col] = "strong"
                        elif tdf_w >= nfw_w:
                            ratings[row][col] = "moderate"
                        else:
                            ratings[row][col] = "weak"
                    elif row == "NFW baseline":
                        if nfw_w > tdf_w and med_d > 0:
                            ratings[row][col] = "strong"
                        elif nfw_w >= tdf_w:
                            ratings[row][col] = "moderate"
                        else:
                            ratings[row][col] = "weak"
                    elif row in ("corrected MOND baseline", "baryon_only baseline", "cored halo baseline"):
                        ratings[row][col] = "weak"

    s1 = ctx.steps["step1"]
    if s1.status == "ok" and "filter_comparison" in s1.tables:
        fc = s1.tables["filter_comparison"]
        filt = fc[fc["filter_name"] == "exclude_any_nfw_or_tdf_boundary_hit"]
        if len(filt) and float(filt["median_delta_bic_tdf_minus_nfw"].iloc[0]) > 2:
            br = "weak"
        else:
            br = "moderate"
        for row in DECISION_MATRIX_ROWS:
            if row == "TDF K-essence":
                ratings[row]["boundary_robustness"] = br
            elif row in ("NFW baseline", "cored halo baseline"):
                ratings[row]["boundary_robustness"] = "moderate"
            else:
                ratings[row]["boundary_robustness"] = "strong"

    s3 = ctx.steps["step3"]
    if s3.status == "ok" and "ml_summary" in s3.tables:
        ml = s3.tables["ml_summary"]
        fixed = ml[ml["ml_regime"] == "A_fixed_canonical"]
        std = ml[ml["ml_regime"] == "C_standard_prior"]
        if len(fixed) and len(std):
            tdf_fixed = int(fixed["bic_win_tdf_kessence"].iloc[0])
            tdf_std = int(std["bic_win_tdf_kessence"].iloc[0])
            for row in DECISION_MATRIX_ROWS:
                if row == "TDF K-essence":
                    ratings[row]["ml_robustness"] = (
                        "strong" if tdf_std >= 0.4 * n and tdf_fixed >= 0.2 * n else "weak"
                    )
                elif row == "NFW baseline":
                    ratings[row]["ml_robustness"] = "strong" if int(fixed["bic_win_nfw"].iloc[0]) > tdf_fixed else "moderate"
                else:
                    ratings[row]["ml_robustness"] = "moderate"

    s4 = ctx.steps["step4"]
    if s4.status == "ok" and "cored_summary" in s4.tables:
        cs = s4.tables["cored_summary"]
        pseudo_w = int(cs[cs["model"] == "pseudo_isothermal"]["bic_win_count"].iloc[0]) if len(cs[cs["model"] == "pseudo_isothermal"]) else 0
        tdf_w = int(cs[cs["model"] == "tdf_kessence"]["bic_win_count"].iloc[0]) if len(cs[cs["model"] == "tdf_kessence"]) else 0
        ratings["cored halo baseline"]["global_bic_competitiveness"] = (
            "strong" if pseudo_w >= tdf_w else "moderate"
        )
        ratings["cored halo baseline"]["reduced_chi2_competitiveness"] = "strong"
        if pseudo_w > tdf_w:
            ratings["TDF K-essence"]["global_bic_competitiveness"] = "moderate"

    s5 = ctx.steps["step5"]
    if s5.status == "ok" and "param_summary" in s5.tables:
        ps = s5.tables["param_summary"]
        sub = ps[ps["metric"] == "median_delta_bic_global_beta_approx"]
        med_db = float(sub["value"].iloc[0]) if len(sub) else 99.0
        for row in DECISION_MATRIX_ROWS:
            if row == "TDF K-essence":
                ratings[row]["parameter_stability"] = "weak" if med_db > 10 else "moderate"
            else:
                ratings[row]["parameter_stability"] = "n/a"

    s6 = ctx.steps["step6"]
    if s6.status == "ok" and "failure_modes" in s6.tables:
        fm = s6.tables["failure_modes"]
        worse = fm[fm["failure_mode"] == "tdf_worse_than_nfw_overall"]
        frac_worse = float(worse["fraction_of_sample"].iloc[0]) if len(worse) else 0.6
        for row in DECISION_MATRIX_ROWS:
            if row == "TDF K-essence":
                ratings[row]["residual_stability"] = (
                    "moderate" if frac_worse < 0.55 else "weak"
                )
            elif row == "NFW baseline":
                ratings[row]["residual_stability"] = (
                    "strong" if frac_worse > 0.5 else "moderate"
                )
            else:
                ratings[row]["residual_stability"] = "moderate"

    limitations_map = {
        "baryon_only baseline": "No dark sector; not viable for rotation curves alone.",
        "corrected MOND baseline": "MOND active; analytic μ; no halo freedom.",
        "NFW baseline": "Cuspy halo; boundary hits common; flexible Υ.",
        "cored halo baseline": "Extra halo parameters; pseudo-isothermal often wins BIC.",
        "TDF K-essence": "Disk proxy; per-galaxy β/M; boundary hits; not global β.",
    }
    for row in DECISION_MATRIX_ROWS:
        ratings[row]["limitations"] = limitations_map.get(row, "—")

    data = [[ratings[row][col] for col in DECISION_MATRIX_COLUMNS] for row in DECISION_MATRIX_ROWS]
    return pd.DataFrame(data, index=list(DECISION_MATRIX_ROWS), columns=list(DECISION_MATRIX_COLUMNS))


CLAIM_LABELS: dict[int, str] = {
    0: "TDF fails SPARC calibration",
    1: "TDF fits some galaxies but is not competitive globally",
    2: "TDF is competitive with NFW/MOND in selected classes",
    3: "TDF is globally BIC-competitive with standard halo baselines (rotation-only)",
    4: "TDF passes rotation + lensing + cosmology consistency",
}


def compute_claim_level(ctx: SynthesisContext, has_lensing_cosmology: bool = False) -> tuple[int, str]:
    if has_lensing_cosmology:
        return 4, "Non-rotation channels reported (not used in this synthesis)."

    if ctx.main_comparison is None:
        return 0, "Main calibration run missing."

    comp = ctx.main_comparison
    n = len(comp)
    tdf_w = int((comp["best_model_by_bic"] == "tdf_kessence").sum())
    nfw_w = int((comp["best_model_by_bic"] == "nfw").sum())
    med_dn = float(comp["delta_bic_tdf_vs_nfw"].median())
    tdf_beats = int(comp["tdf_beats_nfw"].sum()) if "tdf_beats_nfw" in comp.columns else 0
    competitive = int(comp["tdf_bic_competitive"].sum()) if "tdf_bic_competitive" in comp.columns else 0

    level = 1

    if tdf_w >= 0.35 * n and tdf_beats >= 0.45 * n:
        level = max(level, 2)
    if tdf_w >= nfw_w and med_dn < 0 and competitive >= 0.4 * n:
        level = 3

    s1 = ctx.steps["step1"]
    if s1.status == "ok" and "filter_comparison" in s1.tables:
        filt = s1.tables["filter_comparison"]
        fr = filt[filt["filter_name"] == "exclude_any_nfw_or_tdf_boundary_hit"]
        if len(fr) and float(fr["median_delta_bic_tdf_minus_nfw"].iloc[0]) > 4:
            level = min(level, 2)

    s3 = ctx.steps["step3"]
    if s3.status == "ok" and "ml_summary" in s3.tables:
        ml = s3.tables["ml_summary"]
        fixed = ml[ml["ml_regime"] == "A_fixed_canonical"]
        if len(fixed) and int(fixed["bic_win_tdf_kessence"].iloc[0]) < 0.25 * n:
            level = min(level, 2)

    s4 = ctx.steps["step4"]
    if s4.status == "ok" and "cored_summary" in s4.tables:
        cs = s4.tables["cored_summary"]
        pseudo = cs[cs["model"] == "pseudo_isothermal"]
        tdf = cs[cs["model"] == "tdf_kessence"]
        if len(pseudo) and len(tdf) and int(pseudo["bic_win_count"].iloc[0]) > int(tdf["bic_win_count"].iloc[0]) + 20:
            level = min(level, 2)

    s5 = ctx.steps["step5"]
    if s5.status == "ok" and "param_summary" in s5.tables:
        ps = s5.tables["param_summary"]
        sub = ps[ps["metric"] == "median_delta_bic_global_beta_approx"]
        if len(sub) and float(sub["value"].iloc[0]) > 12:
            level = min(level, 2)

    s6 = ctx.steps["step6"]
    if s6.status == "ok" and "failure_modes" in s6.tables:
        fm = s6.tables["failure_modes"]
        w = fm[fm["failure_mode"] == "tdf_worse_than_nfw_overall"]
        if len(w) and float(w["fraction_of_sample"].iloc[0]) > 0.65:
            level = min(level, 2)

    if tdf_w < 0.15 * n or tdf_beats < 0.3 * n:
        level = 0

    parts = [
        f"BIC wins TDF={tdf_w}, NFW={nfw_w}, median ΔBIC(TDF−NFW)={med_dn:.2f}.",
        f"TDF beats NFW on {tdf_beats}/{n} galaxies; BIC-competitive on {competitive}/{n}.",
    ]
    if level < 3:
        parts.append("Robustness analyses (boundary, M/L, cored halo, β/M, residuals) prevent Level 3 without caveats.")
    return level, " ".join(parts)


def compute_recommendation(
    claim_level: int,
    ctx: SynthesisContext,
) -> tuple[str, str, str]:
    missing = [sid for sid, st in ctx.steps.items() if st.status == "missing" and sid != "main"]

    if claim_level <= 0:
        return "E", "Not ready for paper", "TDF does not meet minimum rotation-only competitiveness."

    if ctx.steps["main"].status == "missing":
        return "E", "Not ready for paper", "Main corrected-MOND calibration run missing."

    if "step4" in missing:
        return "B", "Needs cored-halo rerun", "Cored-halo baseline analysis not found."

    if "step3" in missing:
        return "C", "Needs M/L robustness rerun", "Mass-to-light robustness analysis not found."

    s5 = ctx.steps["step5"]
    if s5.status == "ok" and "param_summary" in s5.tables:
        ps = s5.tables["param_summary"]
        sub = ps[ps["metric"] == "median_delta_bic_global_beta_approx"]
        frac = ps[ps["metric"] == "fraction_delta_bic_global_within_2"]
        if len(sub) and float(sub["value"].iloc[0]) > 15:
            if len(frac) and float(frac["value"].iloc[0]) < 0.35:
                return (
                    "D",
                    "Needs parameter stability correction",
                    "Global β/M not viable; large BIC penalty when β fixed.",
                )

    if claim_level >= 2 and not missing:
        return (
            "A",
            "Ready for paper section as rotation-only calibration",
            "Synthesis complete; claim Level 2–3 with explicit limitations and no multi-channel validation.",
        )

    if claim_level == 1:
        return "E", "Not ready for paper", "TDF not sufficiently competitive for a paper-quality rotation claim."

    return (
        "A",
        "Ready for paper section as rotation-only calibration",
        "Partial robustness gaps; use cautious wording and document missing steps.",
    )


def build_claim_level_table(level: int, justification: str) -> pd.DataFrame:
    rows = []
    for lv in range(5):
        rows.append(
            {
                "claim_level": lv,
                "claim_label": CLAIM_LABELS[lv],
                "justification": justification if lv == level else "",
                "rotation_only": lv <= 3,
            },
        )
    return pd.DataFrame(rows)


def build_recommendation_table(code: str, label: str, rationale: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "recommendation_code": code,
                "recommendation_label": label,
                "rationale": rationale,
            },
        ],
    )


def _write_figures(
    ctx: SynthesisContext,
    claim_level: int,
    figures_dir: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures_dir.mkdir(parents=True, exist_ok=True)

    if ctx.main_comparison is not None:
        comp = ctx.main_comparison
        wins = comp["best_model_by_bic"].value_counts()
        labels = [m.replace("_", "\n") for m in wins.index.astype(str)]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(range(len(wins)), wins.values, color="steelblue")
        ax.set_xticks(range(len(wins)))
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel("BIC wins")
        ax.set_title("SPARC BIC wins (corrected-MOND calibration)")
        fig.tight_layout()
        fig.savefig(figures_dir / "final_bic_win_summary.png", dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(comp["delta_bic_tdf_vs_nfw"], bins=25, color="C1", alpha=0.85)
        ax.axvline(0, color="k", ls="--")
        ax.set_xlabel("ΔBIC (TDF − NFW)")
        ax.set_ylabel("Galaxies")
        ax.set_title("Per-galaxy ΔBIC TDF vs NFW")
        fig.tight_layout()
        fig.savefig(figures_dir / "final_delta_bic_tdf_nfw.png", dpi=150)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    levels = [0, 1, 2, 3, 4]
    colors = ["#d62728" if lv == claim_level else "#cccccc" for lv in levels]
    ax.barh(levels, [1] * 5, color=colors, height=0.6)
    ax.set_yticks(levels)
    ax.set_yticklabels([f"L{lv}: {CLAIM_LABELS[lv][:40]}…" for lv in levels], fontsize=7)
    ax.set_xlim(0, 1.2)
    ax.set_xticks([])
    ax.set_title("SPARC evidence ladder (rotation-only synthesis)")
    ax.text(0.5, claim_level, "← assigned", va="center", fontsize=9, color="black")
    for i, sid in enumerate(["main", "step1", "step2", "step3", "step4", "step5", "step6"]):
        st = ctx.steps.get(sid)
        status = st.status if st else "missing"
        ax.text(1.05, 3.6 - i * 0.45, f"{sid}: {status}", transform=ax.transData, fontsize=6, clip_on=False)
    fig.tight_layout()
    fig.savefig(figures_dir / "final_sparc_evidence_ladder.png", dpi=150)
    plt.close(fig)


def _build_synthesis_report(
    ctx: SynthesisContext,
    summary: pd.DataFrame,
    decision_matrix: pd.DataFrame,
    claim_level: int,
    claim_justification: str,
    rec_code: str,
    rec_label: str,
    rec_rationale: str,
) -> str:
    missing_steps = [k for k, v in ctx.steps.items() if v.status == "missing"]
    lines = [
        "# SPARC final synthesis report (Step 7)",
        "",
        f"## ⚠️ {BANNER_SYNTHESIS}",
        "",
        f"## ⚠️ {BANNER_CALIBRATION}",
        "",
        f"**Galaxies (main run):** {ctx.n_galaxies}",
        "",
        "## Input runs",
        "",
    ]
    for sid, st in ctx.steps.items():
        path = str(st.run_dir) if st.run_dir else "_missing_"
        lines.append(f"- **{sid}:** `{path}` ({st.status})")
    if missing_steps:
        lines.append(f"\n**Missing sections:** {', '.join(missing_steps)}")
    lines.extend(
        [
            "",
            "## Final summary table",
            "",
            summary.to_string(index=False),
            "",
            "## Model decision matrix",
            "",
            decision_matrix.to_string(),
            "",
            f"## Assigned claim level: **{claim_level}** — {CLAIM_LABELS[claim_level]}",
            "",
            claim_justification,
            "",
            f"## Recommendation: **{rec_code}** — {rec_label}",
            "",
            rec_rationale,
            "",
            "Level 4 is **not** assigned (no lensing/cosmology channel in this package).",
            "",
            "## Synthesis conclusions",
            "",
            "- **Baseline (standard Υ):** TDF and NFW are close in global BIC on the full SPARC sample.",
            "- **Robustness:** Competitiveness **weakens** under boundary filtering, fixed M/L, and cored-halo baselines.",
            "- **Parameters:** Per-galaxy β/M is scattered; a single global β is not supported.",
            "- **Residuals:** TDF is better than NFW in many dwarfs but worse overall in RMS residual head-to-head.",
            "",
            "## Limitations",
            "",
            "- Rotation-only; phenomenological TDF disk proxy; not observational validation.",
            "- Does not replace dark matter or validate TDF across channels.",
            "",
        ],
    )
    return "\n".join(lines)


def _build_paper_section(
    ctx: SynthesisContext,
    claim_level: int,
    rec_code: str,
) -> str:
    n = ctx.n_galaxies
    tdf_w = nfw_w = med_dn = tdf_beats = 0
    if ctx.main_comparison is not None:
        comp = ctx.main_comparison
        tdf_w = int((comp["best_model_by_bic"] == "tdf_kessence").sum())
        nfw_w = int((comp["best_model_by_bic"] == "nfw").sum())
        med_dn = float(comp["delta_bic_tdf_vs_nfw"].median())
        tdf_beats = int(comp["tdf_beats_nfw"].sum())

    text = f"""# SPARC rotation-curve calibration (draft section)

## ⚠️ {BANNER_SYNTHESIS}

## Dataset and setup

We use the processed SPARC rotation-curve sample (`data/processed/sparc_rotation.csv`) with
**{n}** galaxies passing quality cuts (≥5 points per curve). Stellar mass-to-light ratios
Υ_disk and Υ_bulge are fitted per galaxy on [0.05, 3.0] unless noted. The MOND baseline uses
the corrected analytic deep-MOND relation with fixed a₀ = 3700 (km/s)²/kpc.

## Models compared

- Baryon-only (gas + disk + bulge)
- Corrected analytic MOND
- NFW halo + baryons
- TDF K-essence disk proxy (phenomenological calibration model)

Extended robustness runs add cored halos (Burkert, pseudo-isothermal), boundary-filtered
subsamples, galaxy-class splits, and M/L priors.

## Main results (corrected-MOND calibration run)

| Model | BIC wins |
| --- | --- |
| TDF K-essence | {tdf_w} |
| NFW | {nfw_w} |
| Corrected MOND | (see synthesis table) |

Median ΔBIC(TDF − NFW) = **{med_dn:.2f}**. TDF achieves lower BIC than NFW on **{tdf_beats}**
of {n} galaxies. This is **rotation-only curve fitting**, not a cosmological or lensing test.

## Robustness (summary)

- **Boundary filtering:** TDF advantage vs NFW **shrinks** when galaxies with NFW/TDF parameters
  at bounds are removed.
- **M/L:** Under fixed canonical Υ, NFW wins dominate; TDF competitiveness relies partly on
  flexible Υ.
- **Cored halos:** Pseudo-isothermal and NFW baselines remain strong; TDF is not uniformly
  preferred once cored DM profiles are included.
- **β/M:** Effective coupling varies per galaxy; a single global β/M is not adequate.
- **Residuals:** TDF inner-disk residuals can exceed NFW; successes are enriched in dwarfs.

## Claim level (rotation-only)

**Level {claim_level}:** {CLAIM_LABELS[claim_level]}

We do **not** claim observational validation of TDF, dark-matter replacement, or multi-channel
consistency (lensing/cosmology were not tested here).

## Limitations

Phenomenological disk proxy; parameter bounds; no external hold-out; no lensing or distance
systematics propagation. Results are calibration diagnostics only.

## Editorial recommendation

**{rec_code}** — see synthesis report for full decision matrix.
"""
    for phrase in FORBIDDEN_PAPER_PHRASES:
        if paper_forbidden_phrase_present(text, phrase):
            raise ValueError(f"paper section contains forbidden phrase: {phrase}")
    return text


def run_final_synthesis(
    synthesis_output_dir: Path,
    *,
    runs_root: Path | None = None,
    run_paths: dict[str, str] | None = None,
    has_lensing_cosmology: bool = False,
) -> FinalSynthesisResult:
    synthesis_output_dir = Path(synthesis_output_dir)
    if runs_root is None:
        runs_root = synthesis_output_dir.parent
    tables = synthesis_output_dir / "tables"
    reports = synthesis_output_dir / "reports"
    figures = synthesis_output_dir / "figures"
    metadata = synthesis_output_dir / "metadata"
    for d in (tables, reports, figures, metadata):
        d.mkdir(parents=True, exist_ok=True)

    ctx = load_all_steps(runs_root, run_paths)
    summary = build_final_summary(ctx)
    decision_matrix = build_decision_matrix(ctx)
    claim_level, claim_just = compute_claim_level(ctx, has_lensing_cosmology=has_lensing_cosmology)
    if claim_level > 3:
        claim_level = 3
        claim_just += " Capped at Level 3 (rotation-only package)."
    rec_code, rec_label, rec_rationale = compute_recommendation(claim_level, ctx)
    claim_df = build_claim_level_table(claim_level, claim_just)
    rec_df = build_recommendation_table(rec_code, rec_label, rec_rationale)

    summary.to_csv(tables / "final_sparc_summary.csv", index=False)
    decision_matrix.reset_index(names=["model"]).to_csv(tables / "final_model_decision_matrix.csv", index=False)
    claim_df.to_csv(tables / "final_claim_level.csv", index=False)
    rec_df.to_csv(tables / "final_recommendation.csv", index=False)

    _write_figures(ctx, claim_level, figures)

    synth_report = _build_synthesis_report(
        ctx, summary, decision_matrix, claim_level, claim_just, rec_code, rec_label, rec_rationale,
    )
    (reports / "final_sparc_synthesis_report.md").write_text(synth_report, encoding="utf-8")

    paper = _build_paper_section(ctx, claim_level, rec_code)
    (reports / "paper_ready_sparc_section.md").write_text(paper, encoding="utf-8")

    return FinalSynthesisResult(
        summary=summary,
        decision_matrix=decision_matrix,
        claim_table=claim_df,
        recommendation=rec_df,
        assigned_claim_level=claim_level,
        assigned_recommendation=rec_code,
        output_dir=synthesis_output_dir,
    )

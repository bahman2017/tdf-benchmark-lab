"""SPARC Step 7 — final synthesis tests."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from tdf_obs.utils.run_outputs import make_versioned_output_dir
from tdf_obs.validation.sparc_final_synthesis import (
    BANNER_SYNTHESIS,
    CLAIM_COLUMNS,
    DECISION_MATRIX_COLUMNS,
    DECISION_MATRIX_ROWS,
    FORBIDDEN_PAPER_PHRASES,
    SUMMARY_COLUMNS,
    build_final_summary,
    load_all_steps,
    paper_forbidden_phrase_present,
    run_final_synthesis,
)

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"


@pytest.fixture
def minimal_runs(tmp_path: Path) -> Path:
    """Minimal fake step outputs under tmp_path/runs/."""
    runs = tmp_path / "runs"
    main = runs / "main_cal"
    (main / "tables").mkdir(parents=True)
    comp = pd.DataFrame(
        {
            "galaxy_id": ["g1", "g2", "g3"],
            "best_model_by_bic": ["tdf_kessence", "nfw", "tdf_kessence"],
            "bic_baryon_only": [100.0, 110.0, 120.0],
            "bic_nfw": [50.0, 40.0, 55.0],
            "bic_corrected_mond": [80.0, 70.0, 85.0],
            "bic_tdf_kessence": [45.0, 55.0, 48.0],
            "delta_bic_tdf_vs_nfw": [-5.0, 15.0, -7.0],
            "delta_bic_tdf_vs_corrected_mond": [-35.0, -15.0, -37.0],
            "tdf_beats_nfw": [True, False, True],
            "tdf_beats_corrected_mond": [True, True, True],
            "tdf_bic_competitive": [True, False, True],
        },
    )
    summ = pd.DataFrame(
        {
            "galaxy_id": ["g1", "g1", "g2", "g2", "g3", "g3"],
            "model": ["tdf_kessence", "nfw"] * 3,
            "reduced_chi2": [1.0, 0.9, 2.0, 1.5, 1.1, 1.0],
            "bic": [45, 50, 55, 40, 48, 52],
        },
    )
    comp.to_csv(main / "tables" / "sparc_model_comparison_by_galaxy.csv", index=False)
    summ.to_csv(main / "tables" / "sparc_real_calibration_summary.csv", index=False)

    return tmp_path


def test_loads_available_step_tables(minimal_runs: Path) -> None:
    ctx = load_all_steps(minimal_runs / "runs", {"main": "main_cal", "step1": "no_such_dir"})
    assert ctx.steps["main"].status == "ok"
    assert ctx.steps["step1"].status == "missing"
    summary = build_final_summary(ctx)
    assert len(summary) > 0


def test_missing_step_folders_do_not_crash(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    runs.mkdir()
    layout = make_versioned_output_dir(tmp_path, "syn_test")
    result = run_final_synthesis(
        layout.run_dir,
        runs_root=tmp_path / "runs",
        run_paths={"main": "absent"},
    )
    assert (layout.reports / "final_sparc_synthesis_report.md").is_file()
    assert result.assigned_claim_level == 0


def test_claim_level_never_exceeds_three_without_lensing(minimal_runs: Path) -> None:
    out = minimal_runs / "syn_out"
    result = run_final_synthesis(
        out,
        runs_root=minimal_runs / "runs",
        run_paths={"main": "main_cal"},
    )
    assert result.assigned_claim_level <= 3

    capped = run_final_synthesis(
        minimal_runs / "syn_out_lensing",
        runs_root=minimal_runs / "runs",
        run_paths={"main": "main_cal"},
        has_lensing_cosmology=True,
    )
    assert capped.assigned_claim_level <= 3
    assert bool(
        capped.claim_table.loc[capped.claim_table["claim_level"] == 4, "rotation_only"].iloc[0]
    ) is False


def test_pipeline_outputs(minimal_runs: Path) -> None:
    layout = make_versioned_output_dir(minimal_runs, "final_syn")
    result = run_final_synthesis(
        layout.run_dir,
        runs_root=minimal_runs / "runs",
        run_paths={"main": "main_cal"},
    )
    for col in SUMMARY_COLUMNS:
        assert col in result.summary.columns
    dm = pd.read_csv(layout.tables / "final_model_decision_matrix.csv")
    assert list(dm.columns)[1 : 1 + len(DECISION_MATRIX_COLUMNS)] == list(DECISION_MATRIX_COLUMNS)
    assert len(dm) == len(DECISION_MATRIX_ROWS)
    claim = pd.read_csv(layout.tables / "final_claim_level.csv")
    for col in CLAIM_COLUMNS:
        assert col in claim.columns
    report = (layout.reports / "final_sparc_synthesis_report.md").read_text()
    assert "NOT FULL OBSERVATIONAL VALIDATION" in report
    assert BANNER_SYNTHESIS in report
    paper = (layout.reports / "paper_ready_sparc_section.md").read_text()
    for phrase in FORBIDDEN_PAPER_PHRASES:
        assert not paper_forbidden_phrase_present(paper, phrase)

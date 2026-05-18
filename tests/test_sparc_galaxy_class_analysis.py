"""SPARC Step 2 — galaxy-class analysis tests."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from tdf_obs.utils.run_outputs import make_versioned_output_dir
from tdf_obs.validation.sparc_galaxy_class_analysis import (
    BANNER_GALAXY_CLASS,
    ASSIGNMENT_COLUMNS,
    CLASS_COMPARISON_COLUMNS,
    CLASS_ORDER,
    build_galaxy_properties,
    classify_galaxy_by_vmax,
    compute_class_statistics,
    load_comparison_table,
    run_galaxy_class_analysis,
)

ROOT = Path(__file__).resolve().parents[1]
INPUT_RUN = ROOT / "outputs" / "runs" / "v0.20.2_corrected_mond_sparc_calibration"
SPARC_CSV = ROOT / "data" / "processed" / "sparc_rotation.csv"


def test_classify_galaxy_by_vmax() -> None:
    assert classify_galaxy_by_vmax(50.0) == "dwarf"
    assert classify_galaxy_by_vmax(80.0) == "intermediate"
    assert classify_galaxy_by_vmax(200.0) == "massive"


def test_all_galaxies_assigned_one_class() -> None:
    if not SPARC_CSV.is_file():
        pytest.skip("sparc_rotation.csv missing")
    props = build_galaxy_properties(pd.read_csv(SPARC_CSV))
    assert props["galaxy_class"].isin(CLASS_ORDER).all()
    assert len(props) == props["galaxy_id"].nunique()


@pytest.fixture
def mini_tables(tmp_path: Path) -> Path:
    if not (INPUT_RUN / "tables" / "sparc_model_comparison_by_galaxy.csv").is_file():
        pytest.skip("input run missing")
    comp = load_comparison_table(
        INPUT_RUN / "tables" / "sparc_model_comparison_by_galaxy.csv",
    )
    summary = pd.read_csv(INPUT_RUN / "tables" / "sparc_real_calibration_summary.csv")
    run = tmp_path / "input_run"
    (run / "tables").mkdir(parents=True)
    comp.head(30).to_csv(run / "tables" / "sparc_model_comparison_by_galaxy.csv", index=False)
    summary[summary["galaxy_id"].isin(comp.head(30)["galaxy_id"])].to_csv(
        run / "tables" / "sparc_real_calibration_summary.csv",
        index=False,
    )
    return run


def test_class_comparison_columns(mini_tables: Path) -> None:
    if not SPARC_CSV.is_file():
        pytest.skip("sparc csv missing")
    comp = load_comparison_table(
        mini_tables / "tables" / "sparc_model_comparison_by_galaxy.csv",
    )
    summary = pd.read_csv(mini_tables / "tables" / "sparc_real_calibration_summary.csv")
    props = build_galaxy_properties(
        pd.read_csv(SPARC_CSV)[
            pd.read_csv(SPARC_CSV)["galaxy_id"].isin(comp["galaxy_id"])
        ],
    )
    row = compute_class_statistics(comp, summary, props, "dwarf")
    for col in CLASS_COMPARISON_COLUMNS:
        assert col in row


def test_pipeline_tmp_path_only(mini_tables: Path) -> None:
    if not SPARC_CSV.is_file():
        pytest.skip("sparc csv missing")
    sparc_sub = pd.read_csv(SPARC_CSV)
    sparc_sub = sparc_sub[sparc_sub["galaxy_id"].isin(
        pd.read_csv(mini_tables / "tables" / "sparc_model_comparison_by_galaxy.csv")[
            "galaxy_id"
        ],
    )]
    tmp_sparc = mini_tables / "sparc_sub.csv"
    sparc_sub.to_csv(tmp_sparc, index=False)
    layout = make_versioned_output_dir(mini_tables, "test_galaxy_class")
    result = run_galaxy_class_analysis(mini_tables, tmp_sparc, layout.run_dir)
    assert len(result.class_comparison) == 3
    out = layout.tables / "galaxy_class_model_comparison.csv"
    assert out.is_file()
    df = pd.read_csv(out)
    for col in CLASS_COMPARISON_COLUMNS:
        assert col in df.columns
    assign = pd.read_csv(layout.tables / "galaxy_class_assignments.csv")
    for col in ASSIGNMENT_COLUMNS:
        assert col in assign.columns
    report = (layout.reports / "galaxy_class_analysis_report.md").read_text()
    assert "NOT FULL OBSERVATIONAL VALIDATION" in report
    assert BANNER_GALAXY_CLASS in report

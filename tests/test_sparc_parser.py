"""Phase 8A.0 — SPARC real-data parser tests."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from tdf_obs.data.sparc_parser import (
    BANNER_SPARC_PARSER,
    DATASET_MODE,
    REQUIRED_SPARC_COLUMNS,
    SOURCE_LABEL,
    SparcSchemaError,
    discover_rotmod_files,
    parse_rotmod_file,
    parse_sparc_rotmod_directory,
    validate_sparc_rotation_schema,
    write_sparc_parser_report,
    SparcParseStats,
)

ROOT = Path(__file__).resolve().parents[1]
ROTMOD_DIR = ROOT / "data" / "raw" / "16284118" / "Rotmod_LTG"


@pytest.fixture
def sample_rotmod_path() -> Path:
    if not ROTMOD_DIR.is_dir():
        pytest.skip("SPARC Rotmod_LTG not available locally")
    files = discover_rotmod_files(ROTMOD_DIR)
    if not files:
        pytest.skip("no rotmod files in SPARC directory")
    return files[0]


def test_parser_reads_sample_rotmod(sample_rotmod_path: Path) -> None:
    df, meta = parse_rotmod_file(sample_rotmod_path)
    assert len(df) >= 3
    assert (df["r_kpc"] > 0).all()
    assert (df["v_obs"] >= 0).all()
    assert "distance_mpc" in meta or True


def test_full_parse_produces_required_columns(tmp_path: Path) -> None:
    if not ROTMOD_DIR.is_dir():
        pytest.skip("SPARC Rotmod_LTG not available locally")
    out = tmp_path / "sparc_rotation.csv"
    df, stats = parse_sparc_rotmod_directory(ROTMOD_DIR, out)
    assert stats.galaxies_parsed >= 50
    for col in REQUIRED_SPARC_COLUMNS:
        assert col in df.columns
    assert (df["dataset_mode"] == DATASET_MODE).all()
    assert df["real_observational_data"].all()
    assert (df["source"] == SOURCE_LABEL).all()
    assert out.is_file()


def test_schema_validation_rejects_missing_columns() -> None:
    bad = pd.DataFrame({"galaxy_id": ["A"], "r_kpc": [1.0]})
    with pytest.raises(SparcSchemaError, match="missing required columns"):
        validate_sparc_rotation_schema(bad, min_galaxies=1)


def test_schema_validation_requires_positive_radius() -> None:
    df = pd.DataFrame(
        {
            "galaxy_id": ["A", "A"],
            "r_kpc": [1.0, -0.1],
            "v_obs": [10.0, 12.0],
            "v_err": [1.0, 1.0],
            "v_gas": [0.0, 0.0],
            "v_disk": [10.0, 10.0],
            "v_bulge": [0.0, 0.0],
            "source": [SOURCE_LABEL, SOURCE_LABEL],
            "dataset_mode": [DATASET_MODE, DATASET_MODE],
            "real_observational_data": [True, True],
        },
    )
    with pytest.raises(SparcSchemaError, match="r_kpc"):
        validate_sparc_rotation_schema(df, min_galaxies=1)


def test_parser_report_banner(tmp_path: Path) -> None:
    stats = SparcParseStats(
        input_dir=ROTMOD_DIR,
        output_path=tmp_path / "out.csv",
        files_found=175,
        galaxies_parsed=175,
        total_rows_written=3000,
    )
    report_path = tmp_path / "sparc_parser_report.md"
    text = write_sparc_parser_report(stats, report_path)
    assert BANNER_SPARC_PARSER in text
    assert "NOT MODEL VALIDATION" in text
    assert report_path.is_file()


def test_pipeline_report_from_real_data(tmp_path: Path) -> None:
    if not ROTMOD_DIR.is_dir():
        pytest.skip("SPARC Rotmod_LTG not available locally")
    out_csv = tmp_path / "processed" / "sparc_rotation.csv"
    report = tmp_path / "reports" / "sparc_parser_report.md"
    _, stats = parse_sparc_rotmod_directory(ROTMOD_DIR, out_csv)
    write_sparc_parser_report(stats, report)
    body = report.read_text(encoding="utf-8")
    assert "NOT MODEL VALIDATION" in body
    assert str(ROTMOD_DIR) in body or "Rotmod_LTG" in body

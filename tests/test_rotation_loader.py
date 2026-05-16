"""Dataset labeling, metadata sidecar, and rotation CSV loading."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from tdf_obs.io.dataset_metadata import (
    BANNER_DEMO_FIXTURE,
    BANNER_SYNTHETIC,
    resolve_dataset_info,
    resolve_rotation_run_mode,
)
from tdf_obs.io.loaders import (
    RotationCsvError,
    load_rotation_csv,
    processed_rotation_available,
    validate_rotation_csv_columns,
)
from tdf_obs.pipelines.run_rotation_pipeline import run_rotation_pipeline

FIXTURES = Path(__file__).parent / "fixtures"
SAMPLE_CSV = FIXTURES / "sample_rotation.csv"


def _write_demo_metadata(path: Path) -> None:
    path.write_text(
        yaml.safe_dump(
            {
                "dataset_type": "demo_fixture",
                "source": "test_fixture",
                "description": "Small demo dataset used to test pipeline behavior.",
                "is_real_observational_data": False,
            },
        ),
        encoding="utf-8",
    )


def _write_real_metadata(path: Path, source: str = "SPARC") -> None:
    path.write_text(
        yaml.safe_dump(
            {
                "dataset_type": "real_observational",
                "source": source,
                "description": "Processed SPARC rotation curves.",
                "is_real_observational_data": True,
            },
        ),
        encoding="utf-8",
    )


def test_load_valid_processed_rotation_csv(tmp_path: Path) -> None:
    csv_path = tmp_path / "rotation.csv"
    csv_path.write_text(SAMPLE_CSV.read_text(encoding="utf-8"), encoding="utf-8")
    _write_demo_metadata(tmp_path / "rotation_metadata.yaml")
    info = resolve_dataset_info(csv_path)
    curves = load_rotation_csv(csv_path, dataset_info=info)
    assert len(curves) == 2
    assert curves[0].metadata["dataset_mode"] == "demo_fixture_calibration"
    assert curves[0].metadata["is_real_observational_data"] is False


def test_missing_required_columns_raises(tmp_path: Path) -> None:
    df = pd.DataFrame({"galaxy_id": ["g1"], "r_kpc": [1.0], "v_obs": [10.0]})
    with pytest.raises(RotationCsvError, match="missing required columns"):
        validate_rotation_csv_columns(df)

    path = tmp_path / "bad_rotation.csv"
    df.to_csv(path, index=False)
    with pytest.raises(RotationCsvError, match="missing required columns"):
        load_rotation_csv(path)


def test_missing_metadata_defaults_to_demo_fixture(tmp_path: Path) -> None:
    csv_path = tmp_path / "rotation.csv"
    csv_path.write_text(SAMPLE_CSV.read_text(encoding="utf-8"), encoding="utf-8")
    info = resolve_dataset_info(csv_path)
    assert info.dataset_mode == "demo_fixture_calibration"
    assert info.is_real_observational_data is False
    assert info.warning_banner == BANNER_DEMO_FIXTURE


def test_demo_fixture_metadata_labels_correctly(tmp_path: Path) -> None:
    csv_path = tmp_path / "rotation.csv"
    csv_path.write_text(SAMPLE_CSV.read_text(encoding="utf-8"), encoding="utf-8")
    _write_demo_metadata(tmp_path / "rotation_metadata.yaml")
    info = resolve_dataset_info(csv_path)
    assert resolve_rotation_run_mode(csv_path) == "demo_fixture_calibration"
    assert info.dataset_source == "test_fixture"
    assert info.warning_banner == BANNER_DEMO_FIXTURE


def test_real_observational_metadata_labels_correctly(tmp_path: Path) -> None:
    csv_path = tmp_path / "rotation.csv"
    csv_path.write_text(SAMPLE_CSV.read_text(encoding="utf-8"), encoding="utf-8")
    _write_real_metadata(tmp_path / "rotation_metadata.yaml")
    info = resolve_dataset_info(csv_path)
    assert info.dataset_mode == "real_data_calibration"
    assert info.is_real_observational_data is True
    assert info.dataset_source == "SPARC"
    assert info.warning_banner is None


def test_fallback_synthetic_when_no_processed_file(tmp_path: Path) -> None:
    outputs = tmp_path / "outputs"
    csv_path = tmp_path / "rotation.csv"
    assert not csv_path.exists()

    result = run_rotation_pipeline(processed_csv=csv_path, outputs_root=outputs)

    assert result.mode == "synthetic_validation"
    assert result.dataset_info.is_real_observational_data is False
    report = result.report_path.read_text(encoding="utf-8")
    assert BANNER_SYNTHETIC in report
    assert "**Real observational data** | False" in report

    summary = pd.read_csv(result.summary_csv_path)
    assert summary["dataset_mode"].iloc[0] == "synthetic_validation"
    assert summary["is_real_observational_data"].iloc[0] == False  # noqa: E712


def test_report_demo_fixture_banner(tmp_path: Path) -> None:
    csv_path = tmp_path / "rotation.csv"
    csv_path.write_text(SAMPLE_CSV.read_text(encoding="utf-8"), encoding="utf-8")
    _write_demo_metadata(tmp_path / "rotation_metadata.yaml")
    outputs = tmp_path / "outputs"

    result = run_rotation_pipeline(processed_csv=csv_path, outputs_root=outputs)

    assert result.mode == "demo_fixture_calibration"
    report = result.report_path.read_text(encoding="utf-8")
    assert BANNER_DEMO_FIXTURE in report
    assert BANNER_SYNTHETIC not in report
    assert "demo_fixture_calibration" in report

    summary = pd.read_csv(result.summary_csv_path)
    assert (summary["dataset_mode"] == "demo_fixture_calibration").all()
    assert (summary["is_real_observational_data"] == False).all()  # noqa: E712
    assert summary["dataset_source"].iloc[0] == "test_fixture"


def test_report_real_observational_no_demo_banner(tmp_path: Path) -> None:
    csv_path = tmp_path / "rotation.csv"
    csv_path.write_text(SAMPLE_CSV.read_text(encoding="utf-8"), encoding="utf-8")
    _write_real_metadata(tmp_path / "rotation_metadata.yaml")
    outputs = tmp_path / "outputs"

    result = run_rotation_pipeline(processed_csv=csv_path, outputs_root=outputs)

    assert result.mode == "real_data_calibration"
    report = result.report_path.read_text(encoding="utf-8")
    assert BANNER_DEMO_FIXTURE not in report
    assert BANNER_SYNTHETIC not in report
    assert "real_data_calibration" in report
    assert "**Real observational data** | True" in report

    summary = pd.read_csv(result.summary_csv_path)
    assert (summary["is_real_observational_data"] == True).all()  # noqa: E712


def test_synthetic_mode_when_csv_missing(tmp_path: Path) -> None:
    missing = tmp_path / "rotation.csv"
    assert not processed_rotation_available(missing)
    assert resolve_rotation_run_mode(missing) == "synthetic_validation"

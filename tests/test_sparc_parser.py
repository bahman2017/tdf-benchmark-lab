"""SPARC raw ingestion parser tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from tdf_obs.io.sparc_parser import (
    REQUIRED_PROCESSED_COLUMNS,
    compute_baryonic_velocity,
    parse_sparc_folder,
    parse_sparc_single_csv,
    prepare_sparc_rotation,
    resolve_sparc_raw_input,
    write_rotation_metadata,
)

FIXTURES = Path(__file__).parent / "fixtures"
SPARC_SINGLE = FIXTURES / "sparc_single.csv"
SPARC_FOLDER = FIXTURES / "sparc_folder"


def test_compute_baryonic_velocity_gas_disk_bulge() -> None:
    v_gas = np.array([10.0, 20.0])
    v_disk = np.array([15.0, 25.0])
    v_bulge = np.array([5.0, 0.0])
    v_b = compute_baryonic_velocity(
        v_gas, v_disk, v_bulge, upsilon_disk=0.5, upsilon_bulge=0.7
    )
    expected = np.sqrt(
        np.array(
            [
                10**2 + 0.5 * 15**2 + 0.7 * 5**2,
                20**2 + 0.5 * 25**2 + 0.0,
            ],
        ),
    )
    np.testing.assert_allclose(v_b, expected, rtol=1e-9)


def test_compute_baryonic_velocity_missing_bulge() -> None:
    v_gas = np.array([12.0])
    v_disk = np.array([8.0])
    v_b = compute_baryonic_velocity(v_gas, v_disk, v_bulge=None)
    assert v_b[0] == pytest.approx(np.sqrt(12**2 + 0.5 * 8**2))


def test_compute_baryonic_velocity_negative_component_safe() -> None:
    # Sign convention: negative v_gas still contributes positively via |v|*v
    v_gas = np.array([-20.0])
    v_disk = np.array([10.0])
    v_b = compute_baryonic_velocity(v_gas, v_disk)
    assert v_b[0] == pytest.approx(np.sqrt(20**2 + 0.5 * 10**2))


def test_parse_single_csv_fixture(tmp_path: Path) -> None:
    out = tmp_path / "rotation.csv"
    df = parse_sparc_single_csv(SPARC_SINGLE, out)
    assert out.is_file()
    for col in REQUIRED_PROCESSED_COLUMNS:
        assert col in df.columns
    assert set(df["galaxy_id"]) == {"TestGalaxy", "OtherGalaxy"}
    assert (df["v_baryon"] > 0).all()
    loaded = pd.read_csv(out)
    assert list(loaded.columns) >= list(REQUIRED_PROCESSED_COLUMNS)


def test_parse_folder_fixture(tmp_path: Path) -> None:
    out = tmp_path / "rotation.csv"
    df = parse_sparc_folder(SPARC_FOLDER, out)
    assert out.is_file()
    assert set(df["galaxy_id"]) == {"DDO154", "NGC2403"}
    assert len(df) == 9
    assert "v_gas" in df.columns
    assert df["source"].iloc[0] == "SPARC"


def test_missing_raw_data_does_not_create_processed(tmp_path: Path) -> None:
    raw = tmp_path / "data" / "raw"
    processed = tmp_path / "data" / "processed"
    raw.mkdir(parents=True)
    assert resolve_sparc_raw_input(tmp_path) is None

    with pytest.raises(FileNotFoundError, match="No SPARC raw data found"):
        prepare_sparc_rotation(tmp_path)

    assert not (processed / "rotation.csv").exists()
    assert not (processed / "rotation_metadata.yaml").exists()


def test_metadata_written_after_successful_parse(tmp_path: Path) -> None:
    raw = tmp_path / "data" / "raw" / "sparc"
    raw.mkdir(parents=True)
    (raw / "GalaxyA.csv").write_text(
        (FIXTURES / "sparc_folder" / "DDO154.csv").read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    prepare_sparc_rotation(tmp_path)

    meta_path = tmp_path / "data" / "processed" / "rotation_metadata.yaml"
    assert meta_path.is_file()
    with meta_path.open(encoding="utf-8") as f:
        meta = yaml.safe_load(f)
    assert meta["dataset_type"] == "real_observational"
    assert meta["source"] == "SPARC"
    assert meta["is_real_observational_data"] is True


def test_write_rotation_metadata_demo_mode(tmp_path: Path) -> None:
    path = write_rotation_metadata(tmp_path, source="test", real=False)
    with path.open(encoding="utf-8") as f:
        meta = yaml.safe_load(f)
    assert meta["is_real_observational_data"] is False

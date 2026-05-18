"""Tests for versioned run output utilities."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tdf_obs.utils.run_outputs import (
    assert_production_output_safe,
    is_test_like_path,
    make_versioned_output_dir,
    resolve_versioned_run_dir,
    write_run_manifest,
)


def test_is_test_like_path() -> None:
    assert is_test_like_path("/var/folders/xx/T/pytest-0/sparc.csv")
    assert not is_test_like_path("data/processed/sparc_rotation.csv")


def test_make_versioned_output_dir(tmp_path: Path) -> None:
    layout = make_versioned_output_dir(tmp_path, "test_run")
    assert layout.tables.is_dir()
    assert layout.reports.is_dir()
    assert layout.figures.is_dir()
    assert layout.metadata.is_dir()
    assert layout.run_dir == tmp_path / "runs" / "test_run"


def test_write_run_manifest(tmp_path: Path) -> None:
    layout = make_versioned_output_dir(tmp_path, "test_run")
    path = write_run_manifest(
        layout.run_dir,
        {
            "run_id": "test_run",
            "script_name": "test_script.py",
            "input_path": "/data/sparc.csv",
            "warning_banner": "TEST BANNER",
        },
    )
    data = json.loads(path.read_text())
    assert data["run_id"] == "test_run"
    assert data["warning_banner"] == "TEST BANNER"
    assert "created_at" in data

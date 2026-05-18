"""
Versioned benchmark run directories and production-output safety guards.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PRODUCTION_RUN_IDS = frozenset(
    {
        "v0.20.0_sparc_initial_calibration",
        "v0.20.1_sparc_parameter_audit",
        "v0.20.2_corrected_mond_sparc_calibration",
    },
)

_TEST_PATH_MARKERS = ("pytest", "tmp", "temporarydirectory", "/var/folders/")


@dataclass(frozen=True)
class RunOutputLayout:
    """Paths for a single versioned benchmark run."""

    run_dir: Path
    run_id: str
    tables: Path
    reports: Path
    figures: Path
    metadata: Path


def is_test_like_path(path: Path | str) -> bool:
    """True if path looks like a pytest/tmp fixture location."""
    s = str(path).lower().replace("\\", "/")
    return any(marker in s for marker in _TEST_PATH_MARKERS)


def _git_info() -> tuple[str | None, str | None]:
    try:
        branch = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
        br = branch.stdout.strip() if branch.returncode == 0 else None
        co = commit.stdout.strip() if commit.returncode == 0 else None
        return br, co
    except (OSError, subprocess.SubprocessError):
        return None, None


def make_versioned_output_dir(base_output_dir: Path, run_id: str) -> RunOutputLayout:
    """
    Create ``outputs/runs/<run_id>/{tables,reports,figures,metadata}``.

    Does not handle overwrite/timestamping — use ``resolve_versioned_run_dir``.
    """
    base_output_dir = Path(base_output_dir)
    run_dir = base_output_dir / "runs" / run_id
    tables = run_dir / "tables"
    reports = run_dir / "reports"
    figures = run_dir / "figures"
    metadata = run_dir / "metadata"
    for d in (tables, reports, figures, metadata):
        d.mkdir(parents=True, exist_ok=True)
    return RunOutputLayout(
        run_dir=run_dir,
        run_id=run_id,
        tables=tables,
        reports=reports,
        figures=figures,
        metadata=metadata,
    )


def resolve_versioned_run_dir(
    base_output_dir: Path,
    run_id: str,
    *,
    overwrite_run: bool = False,
) -> RunOutputLayout:
    """
    Resolve the run directory, timestamping if it already exists and overwrite is false.
    """
    base_output_dir = Path(base_output_dir)
    runs_root = base_output_dir / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)
    effective_id = run_id
    candidate = runs_root / effective_id
    if candidate.exists() and not overwrite_run:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        effective_id = f"{run_id}__{ts}"
        candidate = runs_root / effective_id
        print(
            f"Run folder exists; writing to timestamped path: "
            f"runs/{effective_id}/",
        )
    return make_versioned_output_dir(base_output_dir, effective_id)


def assert_production_output_safe(
    input_path: Path | str,
    base_output_dir: Path | str,
    run_id: str,
    *,
    project_outputs_root: Path | str,
    versioned_output: bool = True,
) -> None:
    """
    Block test/fixture inputs from writing into production versioned run folders.
    """
    if not is_test_like_path(input_path):
        return
    if run_id not in PRODUCTION_RUN_IDS:
        return
    if not versioned_output:
        raise ValueError(
            f"Test-like input path {input_path!r} cannot write to production "
            f"outputs with versioned_output=false. Use tmp_path as --output-dir.",
        )
    project_root = Path(project_outputs_root).resolve()
    base = Path(base_output_dir).resolve()
    try:
        base.relative_to(project_root)
    except ValueError:
        return
    raise ValueError(
        f"Test-like input path {input_path!r} cannot write to production run "
        f"'{run_id}' under {project_root}. Use a temporary --output-dir and a "
        f"test run_id (e.g. test_corrected_mond_run).",
    )


def write_run_manifest(run_dir: Path, metadata: dict[str, Any]) -> Path:
    """Write ``metadata/run_manifest.json`` and return its path."""
    run_dir = Path(run_dir)
    meta_dir = run_dir / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)
    branch, commit = _git_info()
    payload = {
        "run_id": metadata.get("run_id"),
        "created_at": metadata.get(
            "created_at",
            datetime.now(timezone.utc).isoformat(),
        ),
        "script_name": metadata.get("script_name"),
        "input_path": metadata.get("input_path"),
        "dataset_mode": metadata.get("dataset_mode"),
        "real_observational_data": metadata.get("real_observational_data"),
        "git_branch": metadata.get("git_branch", branch),
        "git_commit": metadata.get("git_commit", commit),
        "command": metadata.get("command"),
        "output_files": metadata.get("output_files", []),
        "warning_banner": metadata.get("warning_banner"),
        **{
            k: v
            for k, v in metadata.items()
            if k
            not in {
                "run_id",
                "created_at",
                "script_name",
                "input_path",
                "dataset_mode",
                "real_observational_data",
                "git_branch",
                "git_commit",
                "command",
                "output_files",
                "warning_banner",
            }
        },
    }
    path = meta_dir / "run_manifest.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def copy_run_to_legacy_outputs(layout: RunOutputLayout, legacy_root: Path) -> None:
    """Copy tables/reports/figures from a versioned run into legacy flat folders."""
    legacy_root = Path(legacy_root)
    for sub, src in (
        ("tables", layout.tables),
        ("reports", layout.reports),
        ("figures", layout.figures),
    ):
        dest = legacy_root / sub
        dest.mkdir(parents=True, exist_ok=True)
        if not src.is_dir():
            continue
        for item in src.iterdir():
            target = dest / item.name
            if item.is_dir():
                if target.exists():
                    shutil.rmtree(target)
                shutil.copytree(item, target)
            else:
                shutil.copy2(item, target)


def collect_output_files(layout: RunOutputLayout) -> list[str]:
    """Relative paths of files under the run directory."""
    files: list[str] = []
    for path in sorted(layout.run_dir.rglob("*")):
        if path.is_file():
            files.append(str(path.relative_to(layout.run_dir)))
    return files

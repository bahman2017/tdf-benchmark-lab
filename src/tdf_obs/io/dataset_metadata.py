"""Dataset mode resolution and rotation_metadata.yaml handling."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml

RotationDatasetMode = Literal[
    "synthetic_validation",
    "demo_fixture_calibration",
    "real_data_calibration",
    "nfw_surrogate_validation",
]

BANNER_SYNTHETIC = "SYNTHETIC ONLY — NO REAL DATA USED"
BANNER_DEMO_FIXTURE = "DEMO FIXTURE DATA — NOT REAL SPARC"
BANNER_NFW_SURROGATE = "ΛCDM/NFW BENCHMARK — NOT REAL OBSERVATIONAL DATA"

# Default when rotation.csv exists but rotation_metadata.yaml is absent.
DEFAULT_MISSING_METADATA: dict[str, Any] = {
    "dataset_type": "demo_fixture",
    "source": "unknown_csv_no_metadata",
    "description": (
        "Processed rotation.csv present without rotation_metadata.yaml; "
        "treated as demo/fixture data until metadata confirms real observations."
    ),
    "is_real_observational_data": False,
}


class RotationMetadataError(ValueError):
    """Invalid or inconsistent rotation dataset metadata."""


@dataclass(frozen=True)
class RotationDatasetInfo:
    """Resolved labeling for a rotation pipeline run."""

    dataset_mode: RotationDatasetMode
    dataset_type: str
    dataset_source: str
    description: str
    is_real_observational_data: bool
    csv_path: Path | None
    metadata_path: Path | None
    warning_banner: str | None

    @property
    def data_mode(self) -> str:
        """Alias used in RotationCurveData / fit result metadata."""
        return self.dataset_mode


def default_processed_rotation_path(project_root: Path | None = None) -> Path:
    if project_root is None:
        project_root = Path(__file__).resolve().parents[3]
    return project_root / "data" / "processed" / "rotation.csv"


def default_rotation_metadata_path(csv_path: Path) -> Path:
    """Sidecar metadata next to processed rotation.csv."""
    return csv_path.parent / "rotation_metadata.yaml"


def processed_rotation_available(csv_path: Path | None = None) -> bool:
    path = csv_path or default_processed_rotation_path()
    return path.is_file()


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise RotationMetadataError(f"metadata file must be a YAML mapping: {path}")
    return data


def is_confirmed_real_observational(metadata: dict[str, Any]) -> bool:
    """
    True only when metadata explicitly confirms real observational data.

    Required:
    - dataset_type == real_observational
    - is_real_observational_data is true
    - source is a non-empty string (e.g. SPARC)
    """
    if metadata.get("dataset_type") != "real_observational":
        return False
    if metadata.get("is_real_observational_data") is not True:
        return False
    source = metadata.get("source")
    if not isinstance(source, str) or not source.strip():
        return False
    return True


def resolve_dataset_info(
    csv_path: Path | None = None,
    *,
    project_root: Path | None = None,
) -> RotationDatasetInfo:
    """
    Resolve one of three dataset modes.

    - No CSV → synthetic_validation
    - CSV without metadata sidecar → demo_fixture_calibration (default)
    - CSV with metadata → demo or real per YAML content
    """
    path = Path(csv_path) if csv_path else default_processed_rotation_path(project_root)

    if not path.is_file():
        return RotationDatasetInfo(
            dataset_mode="synthetic_validation",
            dataset_type="synthetic",
            dataset_source="in_memory_generator",
            description="No processed rotation.csv; synthetic validation only.",
            is_real_observational_data=False,
            csv_path=None,
            metadata_path=None,
            warning_banner=BANNER_SYNTHETIC,
        )

    meta_path = default_rotation_metadata_path(path)
    if not meta_path.is_file():
        meta = dict(DEFAULT_MISSING_METADATA)
        return RotationDatasetInfo(
            dataset_mode="demo_fixture_calibration",
            dataset_type=str(meta["dataset_type"]),
            dataset_source=str(meta["source"]),
            description=str(meta["description"]),
            is_real_observational_data=False,
            csv_path=path.resolve(),
            metadata_path=None,
            warning_banner=BANNER_DEMO_FIXTURE,
        )

    meta = _load_yaml(meta_path)
    if is_confirmed_real_observational(meta):
        return RotationDatasetInfo(
            dataset_mode="real_data_calibration",
            dataset_type="real_observational",
            dataset_source=str(meta.get("source", "")).strip(),
            description=str(meta.get("description", "")),
            is_real_observational_data=True,
            csv_path=path.resolve(),
            metadata_path=meta_path.resolve(),
            warning_banner=None,
        )

    dataset_type = str(meta.get("dataset_type", "demo_fixture"))
    return RotationDatasetInfo(
        dataset_mode="demo_fixture_calibration",
        dataset_type=dataset_type,
        dataset_source=str(meta.get("source", "unspecified_fixture")),
        description=str(
            meta.get(
                "description",
                "Processed CSV with metadata that does not confirm real observations.",
            ),
        ),
        is_real_observational_data=False,
        csv_path=path.resolve(),
        metadata_path=meta_path.resolve(),
        warning_banner=BANNER_DEMO_FIXTURE,
    )


def resolve_rotation_run_mode(
    csv_path: Path | None = None,
    *,
    project_root: Path | None = None,
) -> RotationDatasetMode:
    """Backward-compatible mode resolver (three modes)."""
    return resolve_dataset_info(csv_path, project_root=project_root).dataset_mode

"""Load processed observational tables from CSV."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from tdf_obs.io.dataset_metadata import (
    RotationDatasetInfo,
    RotationDatasetMode,
    default_processed_rotation_path,
    default_rotation_metadata_path,
    processed_rotation_available,
    resolve_dataset_info,
    resolve_rotation_run_mode,
)
from tdf_obs.io.schemas import RotationCurveData

# SPARC-style processed rotation table (long format, one row per radius point).
REQUIRED_ROTATION_COLUMNS: tuple[str, ...] = (
    "galaxy_id",
    "r_kpc",
    "v_obs",
    "v_err",
    "v_baryon",
)

# Re-export for backward compatibility
RotationRunMode = RotationDatasetMode


class RotationCsvError(ValueError):
    """Raised when a processed rotation CSV fails schema or quality checks."""


def validate_rotation_csv_columns(df: pd.DataFrame) -> None:
    """Raise RotationCsvError if required SPARC-processed columns are missing."""
    missing = set(REQUIRED_ROTATION_COLUMNS) - set(df.columns)
    if missing:
        raise RotationCsvError(
            f"rotation CSV missing required columns: {sorted(missing)}. "
            f"Expected: {list(REQUIRED_ROTATION_COLUMNS)}",
        )


def load_rotation_csv(
    path: Path,
    dataset_info: RotationDatasetInfo | None = None,
) -> list[RotationCurveData]:
    """
    Load processed rotation curves from CSV with dataset labeling.

    Schema (header row required)::

        galaxy_id,r_kpc,v_obs,v_err,v_baryon

    Labeling comes from ``dataset_info`` or is resolved via ``rotation_metadata.yaml``.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Processed rotation CSV not found: {path}")

    if dataset_info is None:
        dataset_info = resolve_dataset_info(path)

    df = pd.read_csv(path)
    validate_rotation_csv_columns(df)

    numeric_cols = ["r_kpc", "v_obs", "v_err", "v_baryon"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if df["galaxy_id"].isna().any():
        raise RotationCsvError("galaxy_id must not be null")

    if df[numeric_cols].isna().any().any():
        raise RotationCsvError("numeric columns contain non-numeric or missing values")

    curves: list[RotationCurveData] = []
    for gid, group in df.groupby("galaxy_id", sort=False):
        group = group.sort_values("r_kpc")
        r = group["r_kpc"].to_numpy(dtype=float)
        v_obs = group["v_obs"].to_numpy(dtype=float)
        v_err = group["v_err"].to_numpy(dtype=float)
        v_baryon = group["v_baryon"].to_numpy(dtype=float)

        if len(r) < 3:
            raise RotationCsvError(
                f"galaxy_id={gid!r} has fewer than 3 radial points ({len(r)})",
            )
        if np.any(r <= 0):
            raise RotationCsvError(f"galaxy_id={gid!r}: r_kpc must be positive")
        if np.any(v_err <= 0):
            raise RotationCsvError(f"galaxy_id={gid!r}: v_err must be positive")

        curves.append(
            RotationCurveData(
                galaxy_id=str(gid),
                r_kpc=r,
                v_obs=v_obs,
                v_err=v_err,
                v_baryon=v_baryon,
                metadata={
                    "source_path": str(path.resolve()),
                    "dataset_mode": dataset_info.dataset_mode,
                    "dataset_source": dataset_info.dataset_source,
                    "is_real_observational_data": dataset_info.is_real_observational_data,
                    "dataset_type": dataset_info.dataset_type,
                },
            ),
        )

    if not curves:
        raise RotationCsvError("rotation CSV contains no galaxy rows")

    return curves

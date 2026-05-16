"""Parse SPARC-style raw rotation curves into processed pipeline schema."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

REQUIRED_PROCESSED_COLUMNS: tuple[str, ...] = (
    "galaxy_id",
    "r_kpc",
    "v_obs",
    "v_err",
    "v_baryon",
)

OPTIONAL_PROCESSED_COLUMNS: tuple[str, ...] = (
    "v_gas",
    "v_disk",
    "v_bulge",
    "upsilon_disk",
    "upsilon_bulge",
    "source",
)

# Minimum columns needed in raw SPARC inputs to derive v_baryon.
RAW_REQUIRED_FOR_BARYON: tuple[str, ...] = ("r_kpc", "v_obs", "v_err", "v_gas", "v_disk")
RAW_OPTIONAL: tuple[str, ...] = ("v_bulge", "galaxy_id")


class SparcParseError(ValueError):
    """SPARC raw data failed validation or parsing."""


def compute_baryonic_velocity(
    v_gas: np.ndarray | pd.Series,
    v_disk: np.ndarray | pd.Series,
    v_bulge: np.ndarray | pd.Series | None = None,
    *,
    upsilon_disk: float = 0.5,
    upsilon_bulge: float = 0.7,
) -> np.ndarray:
    """
    Sign-safe baryonic circular speed from SPARC velocity components [km/s].

    v_baryon^2 = |v_gas|^2 + upsilon_disk*|v_disk|^2 + upsilon_bulge*|v_bulge|^2
    (|v|*|v| for sign-safe handling of negative SPARC components)
    v_baryon = sqrt(max(v_baryon^2, 0))
    """
    v_gas = np.asarray(v_gas, dtype=float)
    v_disk = np.asarray(v_disk, dtype=float)
    if v_bulge is None:
        v_bulge = np.zeros_like(v_gas, dtype=float)
    else:
        v_bulge = np.asarray(v_bulge, dtype=float)

    if not (v_gas.shape == v_disk.shape == v_bulge.shape):
        raise SparcParseError("v_gas, v_disk, and v_bulge must have the same shape")

    # Sign-safe: use |v|*|v| so negative SPARC components still add positive v^2.
    def _sq(v: np.ndarray) -> np.ndarray:
        return np.abs(v) * np.abs(v)

    v2_baryon = (
        _sq(v_gas)
        + upsilon_disk * _sq(v_disk)
        + upsilon_bulge * _sq(v_bulge)
    )
    return np.sqrt(np.maximum(v2_baryon, 0.0))


def _validate_raw_frame(df: pd.DataFrame, context: str) -> None:
    missing = set(RAW_REQUIRED_FOR_BARYON) - set(df.columns)
    if missing:
        raise SparcParseError(
            f"{context}: missing required columns {sorted(missing)}. "
            f"Need at least: {list(RAW_REQUIRED_FOR_BARYON)}",
        )


def _frame_to_processed(
    df: pd.DataFrame,
    galaxy_id: str,
    *,
    upsilon_disk: float,
    upsilon_bulge: float,
    source: str = "SPARC",
) -> pd.DataFrame:
    _validate_raw_frame(df, f"galaxy_id={galaxy_id!r}")

    df = df.reset_index(drop=True)
    n = len(df)
    out = pd.DataFrame({"galaxy_id": [str(galaxy_id)] * n})
    for col in ("r_kpc", "v_obs", "v_err", "v_gas", "v_disk"):
        out[col] = pd.to_numeric(df[col], errors="raise").to_numpy()

    if "v_bulge" in df.columns:
        v_bulge = pd.to_numeric(df["v_bulge"], errors="raise").to_numpy()
        out["v_bulge"] = v_bulge
    else:
        v_bulge = np.zeros(n, dtype=float)

    out["v_baryon"] = compute_baryonic_velocity(
        out["v_gas"].to_numpy(),
        out["v_disk"].to_numpy(),
        v_bulge,
        upsilon_disk=upsilon_disk,
        upsilon_bulge=upsilon_bulge,
    )
    out["upsilon_disk"] = upsilon_disk
    out["upsilon_bulge"] = upsilon_bulge
    out["source"] = source

    out = out.sort_values("r_kpc").reset_index(drop=True)

    if (out["r_kpc"] <= 0).any():
        raise SparcParseError(f"galaxy_id={galaxy_id!r}: r_kpc must be positive")
    if (out["v_err"] <= 0).any():
        raise SparcParseError(f"galaxy_id={galaxy_id!r}: v_err must be positive")
    if len(out) < 3:
        raise SparcParseError(f"galaxy_id={galaxy_id!r}: fewer than 3 radial points")

    return out


def parse_sparc_single_csv(
    input_path: Path,
    output_path: Path,
    *,
    upsilon_disk: float = 0.5,
    upsilon_bulge: float = 0.7,
) -> pd.DataFrame:
    """
    Parse one combined SPARC-style CSV into processed long-format table.

    Raw columns: galaxy_id, r_kpc, v_obs, v_err, v_gas, v_disk [, v_bulge]
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    if not input_path.is_file():
        raise FileNotFoundError(f"SPARC input not found: {input_path}")

    df = pd.read_csv(input_path)
    if "galaxy_id" not in df.columns:
        raise SparcParseError("single CSV must include galaxy_id column")

    frames: list[pd.DataFrame] = []
    for gid, group in df.groupby("galaxy_id", sort=False):
        if pd.isna(gid):
            continue
        frames.append(
            _frame_to_processed(
                group.drop(columns=["galaxy_id"], errors="ignore"),
                str(gid),
                upsilon_disk=upsilon_disk,
                upsilon_bulge=upsilon_bulge,
            ),
        )

    if not frames:
        raise SparcParseError(f"no galaxies found in {input_path}")

    result = pd.concat(frames, ignore_index=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    return result


def parse_sparc_folder(
    input_dir: Path,
    output_path: Path,
    *,
    upsilon_disk: float = 0.5,
    upsilon_bulge: float = 0.7,
) -> pd.DataFrame:
    """
    Parse per-galaxy CSV files from a directory.

    Galaxy ID = file stem (e.g. ``DDO154.csv`` → ``DDO154``).
    """
    input_dir = Path(input_dir)
    output_path = Path(output_path)
    if not input_dir.is_dir():
        raise FileNotFoundError(f"SPARC input directory not found: {input_dir}")

    csv_files = sorted(input_dir.glob("*.csv"))
    if not csv_files:
        raise SparcParseError(f"no CSV files in {input_dir}")

    frames: list[pd.DataFrame] = []
    for path in csv_files:
        df = pd.read_csv(path)
        galaxy_id = path.stem
        frames.append(
            _frame_to_processed(
                df,
                galaxy_id,
                upsilon_disk=upsilon_disk,
                upsilon_bulge=upsilon_bulge,
            ),
        )

    result = pd.concat(frames, ignore_index=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    return result


def write_rotation_metadata(
    output_dir: Path,
    *,
    source: str = "SPARC",
    real: bool = True,
    description: str | None = None,
) -> Path:
    """
    Write ``rotation_metadata.yaml`` for processed rotation data.

    Only use ``real=True`` when raw SPARC data was actually ingested by the user.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    meta_path = output_dir / "rotation_metadata.yaml"

    if real:
        payload: dict[str, Any] = {
            "dataset_type": "real_observational",
            "source": source,
            "description": description or "Processed SPARC rotation curves.",
            "is_real_observational_data": True,
        }
    else:
        payload = {
            "dataset_type": "demo_fixture",
            "source": source,
            "description": description or "Non-observational processed rotation export.",
            "is_real_observational_data": False,
        }

    with meta_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False, default_flow_style=False)

    return meta_path


def resolve_sparc_raw_input(project_root: Path) -> tuple[str, Path] | None:
    """
    Locate user-provided SPARC raw data (no download).

    Returns
    -------
    (kind, path) where kind is ``single_csv`` or ``folder``, or None if missing.
    """
    root = Path(project_root)
    single = root / "data" / "raw" / "sparc_rotation_curves.csv"
    folder = root / "data" / "raw" / "sparc"

    if single.is_file():
        return ("single_csv", single)
    if folder.is_dir() and any(folder.glob("*.csv")):
        return ("folder", folder)
    return None


def prepare_sparc_rotation(
    project_root: Path,
    *,
    upsilon_disk: float = 0.5,
    upsilon_bulge: float = 0.7,
) -> pd.DataFrame:
    """
    Parse SPARC raw data into ``data/processed/rotation.csv`` and metadata.

    Raises
    ------
    FileNotFoundError
        If no raw SPARC files are present.
    """
    root = Path(project_root)
    located = resolve_sparc_raw_input(root)
    if located is None:
        raise FileNotFoundError(
            "No SPARC raw data found. Place files in data/raw/ before running this script.",
        )

    processed_dir = root / "data" / "processed"
    output_csv = processed_dir / "rotation.csv"
    kind, path = located

    if kind == "single_csv":
        result = parse_sparc_single_csv(
            path,
            output_csv,
            upsilon_disk=upsilon_disk,
            upsilon_bulge=upsilon_bulge,
        )
    else:
        result = parse_sparc_folder(
            path,
            output_csv,
            upsilon_disk=upsilon_disk,
            upsilon_bulge=upsilon_bulge,
        )

    write_rotation_metadata(
        processed_dir,
        source="SPARC",
        real=True,
        description="Processed SPARC rotation curves.",
    )
    return result

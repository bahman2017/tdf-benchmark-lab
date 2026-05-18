"""
Phase 8A.0 — Parse real SPARC Rotmod_LTG files into a unified processed CSV.

Column mapping (Lelli et al. 2016 rotmod format):
  Rad → r_kpc, Vobs → v_obs, errV → v_err,
  Vgas → v_gas, Vdisk → v_disk, Vbul → v_bulge
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

BANNER_SPARC_PARSER = "REAL SPARC DATA PARSED — NOT MODEL VALIDATION"

SOURCE_LABEL = "SPARC_Lelli2016"
DATASET_MODE = "real_sparc"

REQUIRED_SPARC_COLUMNS: tuple[str, ...] = (
    "galaxy_id",
    "r_kpc",
    "v_obs",
    "v_err",
    "v_gas",
    "v_disk",
    "v_bulge",
    "source",
    "dataset_mode",
    "real_observational_data",
)

# SPARC rotmod header tokens (case-insensitive) → output column
_ROTMOD_COLUMN_ALIASES: dict[str, str] = {
    "rad": "r_kpc",
    "r": "r_kpc",
    "radius": "r_kpc",
    "r_kpc": "r_kpc",
    "vobs": "v_obs",
    "v_obs": "v_obs",
    "vobs_kms": "v_obs",
    "errv": "v_err",
    "verr": "v_err",
    "v_err": "v_err",
    "err_v": "v_err",
    "sigma_v": "v_err",
    "vgas": "v_gas",
    "v_gas": "v_gas",
    "vdisk": "v_disk",
    "v_disk": "v_disk",
    "vbul": "v_bulge",
    "v_bulge": "v_bulge",
    "bulge": "v_bulge",
}

# Fixed order when header is missing (standard SPARC rotmod)
_STANDARD_ROTMOD_ORDER: tuple[str, ...] = (
    "r_kpc",
    "v_obs",
    "v_err",
    "v_gas",
    "v_disk",
    "v_bulge",
)

_ROTMOD_GLOB_PATTERNS = ("*_rotmod.dat", "*_rotmod.txt", "*.dat", "*.txt")


class SparcSchemaError(ValueError):
    """Processed SPARC table failed schema validation."""


@dataclass
class SparcParseStats:
    input_dir: Path
    output_path: Path
    files_found: int = 0
    galaxies_parsed: int = 0
    galaxies_skipped: int = 0
    total_rows_written: int = 0
    rows_removed: int = 0
    mrt_galaxy_count: int | None = None
    warnings: list[str] = field(default_factory=list)
    skipped_files: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_dir": str(self.input_dir),
            "output_path": str(self.output_path),
            "files_found": self.files_found,
            "galaxies_parsed": self.galaxies_parsed,
            "galaxies_skipped": self.galaxies_skipped,
            "total_rows_written": self.total_rows_written,
            "rows_removed": self.rows_removed,
            "mrt_galaxy_count": self.mrt_galaxy_count,
            "warnings": list(self.warnings),
            "skipped_files": list(self.skipped_files),
        }


def _normalize_header_token(token: str) -> str:
    return re.sub(r"[^a-z0-9_]", "", token.strip().lower())


def _map_rotmod_headers(raw_headers: Iterable[str]) -> list[str]:
    """Map rotmod header tokens to output names; unknown columns dropped."""
    mapped: list[str] = []
    for raw in raw_headers:
        key = _normalize_header_token(raw)
        if key in _ROTMOD_COLUMN_ALIASES:
            mapped.append(_ROTMOD_COLUMN_ALIASES[key])
    return mapped


def _galaxy_id_from_path(path: Path) -> str:
    stem = path.stem
    if stem.endswith("_rotmod"):
        return stem[: -len("_rotmod")]
    return stem


def _parse_distance_mpc(comment_line: str) -> float | None:
    m = re.search(r"distance\s*=\s*([\d.]+)\s*mpc", comment_line, flags=re.I)
    if m:
        return float(m.group(1))
    return None


def parse_rotmod_file(path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Parse one SPARC ``*_rotmod.dat`` file into a per-galaxy frame.

    Returns (dataframe, metadata) where metadata may include distance_mpc.
    """
    path = Path(path)
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    meta: dict[str, Any] = {"file": path.name}

    header_tokens: list[str] | None = None
    data_start = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            if i == 0:
                dist = _parse_distance_mpc(stripped)
                if dist is not None:
                    meta["distance_mpc"] = dist
            if "rad" in stripped.lower() and "vobs" in stripped.lower():
                body = stripped.lstrip("#").strip()
                header_tokens = re.split(r"\s+|\t+", body)
            continue
        data_start = i
        break

    if not lines[data_start:]:
        raise ValueError(f"no data rows in {path}")

    if header_tokens:
        col_map = _map_rotmod_headers(header_tokens)
        if "r_kpc" not in col_map:
            raise ValueError(f"could not find Rad column in header of {path}")
        usecols = list(range(min(len(col_map), len(header_tokens))))
        raw = pd.read_csv(
            path,
            sep=r"\s+",
            comment="#",
            header=None,
            usecols=usecols,
            engine="python",
        )
        if raw.shape[1] != len(col_map):
            raw = raw.iloc[:, : len(col_map)]
        raw.columns = col_map[: raw.shape[1]]
    else:
        raw = pd.read_csv(
            path,
            sep=r"\s+",
            comment="#",
            header=None,
            engine="python",
        )
        n_use = min(raw.shape[1], len(_STANDARD_ROTMOD_ORDER))
        raw = raw.iloc[:, :n_use]
        raw.columns = list(_STANDARD_ROTMOD_ORDER[:n_use])

    for col in _STANDARD_ROTMOD_ORDER:
        if col not in raw.columns:
            raw[col] = 0.0 if col.startswith("v_") else np.nan

    out = raw[list(_STANDARD_ROTMOD_ORDER)].copy()
    for col in out.columns:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    return out, meta


def _clean_galaxy_frame(
    df: pd.DataFrame,
    galaxy_id: str,
) -> tuple[pd.DataFrame, int]:
    """Filter invalid rows; return cleaned frame and count removed."""
    n_before = len(df)
    work = df.copy()
    work = work.dropna(subset=["r_kpc", "v_obs"])
    work = work[work["r_kpc"] > 0]
    work = work[work["v_obs"] >= 0]
    # errV must be positive when present
    err = pd.to_numeric(work["v_err"], errors="coerce")
    work = work[err > 0]
    if len(work) == 0:
        return work, n_before

    work["galaxy_id"] = galaxy_id
    work["source"] = SOURCE_LABEL
    work["dataset_mode"] = DATASET_MODE
    work["real_observational_data"] = True
    work = work.sort_values("r_kpc").reset_index(drop=True)
    return work, n_before - len(work)


def count_mrt_galaxies(mrt_path: Path | None) -> int | None:
    """Count non-comment data lines in SPARC_Lelli2016c.mrt (approximate)."""
    if mrt_path is None or not mrt_path.is_file():
        return None
    count = 0
    for line in mrt_path.read_text(encoding="utf-8", errors="replace").splitlines():
        s = line.strip()
        if not s or s.startswith("-") or s.startswith("="):
            continue
        if s.startswith("Byte") or s.startswith("Title") or s.startswith("Authors"):
            continue
        if s.startswith("Note") or s.startswith("Table"):
            continue
        # Galaxy name in first 11 bytes — lines with letters at start
        if re.match(r"^[A-Za-z]", s[:3]):
            count += 1
    return count


def discover_rotmod_files(input_dir: Path) -> list[Path]:
    input_dir = Path(input_dir)
    found: list[Path] = []
    seen: set[str] = set()
    for pattern in _ROTMOD_GLOB_PATTERNS:
        for path in sorted(input_dir.glob(pattern)):
            key = path.name.lower()
            if key in seen:
                continue
            seen.add(key)
            found.append(path)
    return found


def parse_sparc_rotmod_directory(
    input_dir: Path,
    output_path: Path,
    *,
    sparc_mrt_path: Path | None = None,
    mass_mrt_path: Path | None = None,
) -> tuple[pd.DataFrame, SparcParseStats]:
    """
    Parse all rotmod files in ``input_dir`` and write combined CSV.

    Parameters
    ----------
    input_dir
        Typically ``data/raw/16284118/Rotmod_LTG``.
    output_path
        e.g. ``data/processed/sparc_rotation.csv``.
  sparc_mrt_path, mass_mrt_path
        Optional metadata tables (counts only for report).
    """
    input_dir = Path(input_dir)
    output_path = Path(output_path)
    stats = SparcParseStats(input_dir=input_dir, output_path=output_path)

    if not input_dir.is_dir():
        raise FileNotFoundError(f"SPARC rotmod directory not found: {input_dir}")

    rotmod_files = discover_rotmod_files(input_dir)
    stats.files_found = len(rotmod_files)
    if not rotmod_files:
        raise FileNotFoundError(f"no rotmod .dat/.txt files in {input_dir}")

    if sparc_mrt_path is None:
        sparc_mrt_path = input_dir.parent / "SPARC_Lelli2016c.mrt"
    stats.mrt_galaxy_count = count_mrt_galaxies(
        sparc_mrt_path if sparc_mrt_path.is_file() else None,
    )
    if mass_mrt_path is None:
        mass_mrt_path = input_dir.parent / "MassModels_Lelli2016c.mrt"
    if mass_mrt_path.is_file() and stats.mrt_galaxy_count is None:
        stats.warnings.append("MassModels MRT present; SPARC table count unavailable.")

    frames: list[pd.DataFrame] = []
    rows_removed = 0

    for path in rotmod_files:
        galaxy_id = _galaxy_id_from_path(path)
        try:
            raw, _file_meta = parse_rotmod_file(path)
            cleaned, n_drop = _clean_galaxy_frame(raw, galaxy_id)
            rows_removed += n_drop
            if len(cleaned) < 3:
                stats.galaxies_skipped += 1
                stats.skipped_files.append(f"{path.name}: fewer than 3 valid points")
                continue
            frames.append(cleaned)
            stats.galaxies_parsed += 1
        except Exception as exc:  # noqa: BLE001 — collect per-file failures
            stats.galaxies_skipped += 1
            stats.skipped_files.append(f"{path.name}: {exc}")

    if not frames:
        raise SparcSchemaError(
            f"no galaxies parsed from {input_dir}; skipped={stats.skipped_files}",
        )

    result = pd.concat(frames, ignore_index=True)
    result = result[list(REQUIRED_SPARC_COLUMNS)]

    validate_sparc_rotation_schema(
        result,
        min_galaxies=50 if stats.files_found >= 50 else 1,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)

    stats.total_rows_written = len(result)
    stats.rows_removed = rows_removed
    return result, stats


def validate_sparc_rotation_schema(
    df: pd.DataFrame,
    *,
    min_galaxies: int = 1,
) -> None:
    """
    Validate processed SPARC rotation table.

    Raises
    ------
    SparcSchemaError
        If required columns, positivity, or galaxy counts fail.
    """
    missing = [c for c in REQUIRED_SPARC_COLUMNS if c not in df.columns]
    if missing:
        raise SparcSchemaError(f"missing required columns: {missing}")

    if len(df) == 0:
        raise SparcSchemaError("empty dataframe")

    if (df["r_kpc"] <= 0).any():
        raise SparcSchemaError("r_kpc must be > 0 for all rows")

    if (df["v_obs"] < 0).any():
        raise SparcSchemaError("v_obs must be >= 0")

    err = pd.to_numeric(df["v_err"], errors="coerce")
    if (err <= 0).any():
        raise SparcSchemaError("v_err must be > 0 where present")

    if df["dataset_mode"].nunique() != 1 or df["dataset_mode"].iloc[0] != DATASET_MODE:
        raise SparcSchemaError(f"dataset_mode must be {DATASET_MODE!r}")

    if not df["real_observational_data"].all():
        raise SparcSchemaError("real_observational_data must be true for all rows")

    if df["source"].iloc[0] != SOURCE_LABEL:
        raise SparcSchemaError(f"source must be {SOURCE_LABEL!r}")

    n_gal = df["galaxy_id"].nunique()
    if n_gal < min_galaxies:
        raise SparcSchemaError(
            f"expected at least {min_galaxies} galaxies, found {n_gal}",
        )

    for gid, group in df.groupby("galaxy_id"):
        if len(group) == 0:
            raise SparcSchemaError(f"galaxy {gid!r} has no rows")


def write_sparc_parser_report(
    stats: SparcParseStats,
    report_path: Path,
) -> str:
    """Write markdown parser report; return report text."""
    lines = [
        "# SPARC parser report (Phase 8A.0)",
        "",
        f"## ⚠️ {BANNER_SPARC_PARSER}",
        "",
        "## Summary",
        "",
        f"- **Input path:** `{stats.input_dir}`",
        f"- **Galaxy files found:** {stats.files_found}",
        f"- **Galaxies parsed:** {stats.galaxies_parsed}",
        f"- **Galaxies skipped:** {stats.galaxies_skipped}",
        f"- **Total rotation points:** {stats.total_rows_written}",
        f"- **Invalid rows removed:** {stats.rows_removed}",
        f"- **Output CSV:** `{stats.output_path}`",
    ]
    if stats.mrt_galaxy_count is not None:
        lines.append(f"- **SPARC_Lelli2016c.mrt catalog entries (approx.):** {stats.mrt_galaxy_count}")

    lines.extend(
        [
            "",
            "## Column mapping (rotmod → processed)",
            "",
            "| Rotmod | Processed |",
            "| --- | --- |",
            "| Rad | r_kpc |",
            "| Vobs | v_obs |",
            "| errV | v_err |",
            "| Vgas | v_gas |",
            "| Vdisk | v_disk |",
            "| Vbul | v_bulge |",
            "",
            "## Metadata columns",
            "",
            f"- `source` = `{SOURCE_LABEL}`",
            f"- `dataset_mode` = `{DATASET_MODE}`",
            f"- `real_observational_data` = `true`",
            "",
            "## Disclaimer",
            "",
            "Parsing real SPARC rotation-curve files does **not** constitute TDF model "
            "validation, SPARC fitting, or a claim that TDF fits the data.",
            "",
        ],
    )

    if stats.skipped_files:
        lines.extend(["## Skipped / warnings", ""])
        for item in stats.skipped_files[:30]:
            lines.append(f"- {item}")
        if len(stats.skipped_files) > 30:
            lines.append(f"- … and {len(stats.skipped_files) - 30} more")
    if stats.warnings:
        lines.extend(["", "## Notes", ""])
        for w in stats.warnings:
            lines.append(f"- {w}")

    text = "\n".join(lines)
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(text, encoding="utf-8")
    return text

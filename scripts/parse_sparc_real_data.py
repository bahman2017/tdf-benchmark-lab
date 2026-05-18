#!/usr/bin/env python3
"""
Phase 8A.0 — Parse real SPARC Rotmod_LTG files into processed CSV.

Does NOT run model fitting or claim observational validation.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tdf_obs.data.sparc_parser import (
    BANNER_SPARC_PARSER,
    parse_sparc_rotmod_directory,
    write_sparc_parser_report,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Parse SPARC Rotmod_LTG files into sparc_rotation.csv",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "data" / "raw" / "16284118" / "Rotmod_LTG",
        help="Directory containing *_rotmod.dat files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "processed" / "sparc_rotation.csv",
        help="Output processed CSV path",
    )
    parser.add_argument(
        "--sparc-mrt",
        type=Path,
        default=ROOT / "data" / "raw" / "16284118" / "SPARC_Lelli2016c.mrt",
        help="Optional SPARC catalog MRT for report metadata",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "outputs" / "reports" / "sparc_parser_report.md",
        help="Parser report markdown path",
    )
    args = parser.parse_args()

    print(BANNER_SPARC_PARSER)

    df, stats = parse_sparc_rotmod_directory(
        args.input,
        args.output,
        sparc_mrt_path=args.sparc_mrt if args.sparc_mrt.is_file() else None,
    )
    write_sparc_parser_report(stats, args.report)

    print(f"Input:  {args.input}")
    print(f"Files:  {stats.files_found} | Parsed: {stats.galaxies_parsed} galaxies")
    print(f"Rows:   {len(df)} | Removed: {stats.rows_removed} invalid rows")
    print(f"CSV:    {args.output}")
    print(f"Report: {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

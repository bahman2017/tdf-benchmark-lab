#!/usr/bin/env python3
"""
SPARC Step 6 — rotation-curve residual diagnostics.

NOT full observational validation.
"""

from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tdf_obs.utils.run_outputs import (
    collect_output_files,
    resolve_versioned_run_dir,
    write_run_manifest,
)
from tdf_obs.validation.sparc_residual_diagnostics import (
    BANNER_CALIBRATION,
    BANNER_RESIDUAL,
    run_residual_diagnostics,
)

DEFAULT_INPUT_RUN = "v0.20.2_corrected_mond_sparc_calibration"
DEFAULT_RUN_ID = "sparc_step_6_residual_diagnostics"


def main() -> int:
    parser = argparse.ArgumentParser(description="SPARC residual diagnostics (Step 6)")
    parser.add_argument(
        "--input-run",
        type=Path,
        default=ROOT / "outputs" / "runs" / DEFAULT_INPUT_RUN,
    )
    parser.add_argument(
        "--sparc-data",
        type=Path,
        default=ROOT / "data" / "processed" / "sparc_rotation.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    parser.add_argument("--run-id", type=str, default=DEFAULT_RUN_ID)
    parser.add_argument(
        "--versioned-output",
        type=lambda x: str(x).lower() in ("1", "true", "yes"),
        default=True,
    )
    parser.add_argument(
        "--overwrite-run",
        type=lambda x: str(x).lower() in ("1", "true", "yes"),
        default=False,
    )
    parser.add_argument("--max-galaxies", type=int, default=None)
    args = parser.parse_args()

    print(BANNER_RESIDUAL)
    print(BANNER_CALIBRATION)

    input_run = Path(args.input_run)
    if not input_run.is_dir():
        raise SystemExit(f"Input run not found: {input_run}")

    if args.versioned_output:
        layout = resolve_versioned_run_dir(
            args.output_dir,
            args.run_id,
            overwrite_run=args.overwrite_run,
        )
        run_dir = layout.run_dir
    else:
        layout = None
        run_dir = args.output_dir / "runs" / args.run_id

    result = run_residual_diagnostics(
        input_run,
        args.sparc_data,
        run_dir,
        max_galaxies=args.max_galaxies,
    )

    if layout is not None:
        manifest_path = write_run_manifest(
            layout.run_dir,
            {
                "run_id": layout.run_id,
                "script_name": Path(__file__).name,
                "input_run": str(input_run.resolve()),
                "sparc_data": str(Path(args.sparc_data).resolve()),
                "command": " ".join(shlex.quote(a) for a in sys.argv),
                "output_files": collect_output_files(layout),
                "warning_banner": BANNER_RESIDUAL,
            },
        )
        print(f"Run directory: {layout.run_dir}")
        print(f"Manifest: {manifest_path}")

    print(f"\nPoints: {len(result.by_point)} rows")
    print(f"Galaxies (per model): {len(result.by_galaxy)} rows")
    print(f"Report: {run_dir / 'reports' / 'residual_diagnostics_report.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

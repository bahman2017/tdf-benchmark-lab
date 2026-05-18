#!/usr/bin/env python3
"""
SPARC Step 6F — 5D projection kernel for τ halos.

Theoretical proxy only; not observational validation.
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
from tdf_obs.validation.sparc_5d_projection_kernel import (
    BANNER_CALIBRATION,
    BANNER_PROJECTION,
    run_5d_projection_kernel,
)

DEFAULT_FIELD_RUN = "sparc_step_6d_tau_field_equation_solver"
DEFAULT_ROBUSTNESS_RUN = "sparc_step_6e_tau_field_robustness"
DEFAULT_CALIBRATION = "v0.20.2_corrected_mond_sparc_calibration"
DEFAULT_RUN_ID = "sparc_step_6f_5d_projection_kernel"


def main() -> int:
    parser = argparse.ArgumentParser(description="SPARC 5D projection kernel (Step 6F)")
    parser.add_argument(
        "--sparc-data",
        type=Path,
        default=ROOT / "data" / "processed" / "sparc_rotation.csv",
    )
    parser.add_argument(
        "--field-run",
        type=Path,
        default=ROOT / "outputs" / "runs" / DEFAULT_FIELD_RUN,
    )
    parser.add_argument(
        "--robustness-run",
        type=Path,
        default=ROOT / "outputs" / "runs" / DEFAULT_ROBUSTNESS_RUN,
    )
    parser.add_argument(
        "--calibration-run",
        type=Path,
        default=ROOT / "outputs" / "runs" / DEFAULT_CALIBRATION,
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

    print(BANNER_PROJECTION)
    print(BANNER_CALIBRATION)

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

    result = run_5d_projection_kernel(
        args.sparc_data,
        run_dir,
        field_run=args.field_run,
        robustness_run=args.robustness_run,
        calibration_run=args.calibration_run,
        max_galaxies=args.max_galaxies,
    )

    if layout is not None:
        manifest_path = write_run_manifest(
            layout.run_dir,
            {
                "run_id": layout.run_id,
                "script_name": Path(__file__).name,
                "command": " ".join(shlex.quote(a) for a in sys.argv),
                "output_files": collect_output_files(layout),
                "warning_banner": BANNER_PROJECTION,
                "n_galaxies": int(len(result.comparison_by_galaxy)),
            },
        )
        print(f"Run directory: {layout.run_dir}")
        print(f"Manifest: {manifest_path}")

    print(f"Galaxies: {len(result.comparison_by_galaxy)}")
    print(f"Report: {run_dir / 'reports' / 'projection_kernel_report.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

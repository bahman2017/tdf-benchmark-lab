#!/usr/bin/env python3
"""
SPARC Step 6E — τ field robustness and parameter-stability audit.

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
from tdf_obs.validation.sparc_tau_field_robustness import (
    BANNER_CALIBRATION,
    BANNER_TAU_FIELD_ROBUSTNESS,
    run_tau_field_robustness,
)

DEFAULT_FIELD_RUN = "sparc_step_6d_tau_field_equation_solver"
DEFAULT_CALIBRATION = "v0.20.2_corrected_mond_sparc_calibration"
DEFAULT_STEP5 = "sparc_step_5_tdf_parameter_stability"
DEFAULT_RUN_ID = "sparc_step_6e_tau_field_robustness"


def main() -> int:
    parser = argparse.ArgumentParser(description="SPARC τ field robustness audit (Step 6E)")
    parser.add_argument(
        "--field-run",
        type=Path,
        default=ROOT / "outputs" / "runs" / DEFAULT_FIELD_RUN,
    )
    parser.add_argument(
        "--sparc-data",
        type=Path,
        default=ROOT / "data" / "processed" / "sparc_rotation.csv",
    )
    parser.add_argument(
        "--calibration-run",
        type=Path,
        default=ROOT / "outputs" / "runs" / DEFAULT_CALIBRATION,
    )
    parser.add_argument(
        "--step5-run",
        type=Path,
        default=ROOT / "outputs" / "runs" / DEFAULT_STEP5,
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

    print(BANNER_TAU_FIELD_ROBUSTNESS)
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

    result = run_tau_field_robustness(
        args.field_run,
        args.sparc_data,
        run_dir,
        calibration_run=args.calibration_run,
        step5_run=args.step5_run,
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
                "warning_banner": BANNER_TAU_FIELD_ROBUSTNESS,
                "field_run": str(args.field_run),
            },
        )
        print(f"Run directory: {layout.run_dir}")
        print(f"Manifest: {manifest_path}")

    print(f"Global λ test galaxies: {len(result.global_parameter_test)}")
    print(f"Report: {run_dir / 'reports' / 'tau_field_robustness_report.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
SPARC Step 6D — τ field-equation solver.

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
from tdf_obs.validation.sparc_tau_field_solver import (
    BANNER_CALIBRATION,
    BANNER_TAU_FIELD,
    run_tau_field_solver,
)

DEFAULT_INPUT_RUN = "v0.20.2_corrected_mond_sparc_calibration"
DEFAULT_INVERSE_DESIGN = "sparc_step_6a_tau_inverse_design"
DEFAULT_INVERSE_RESPONSE = "sparc_step_6b_inverse_tau_response"
DEFAULT_TAU_LAW = "sparc_step_6c_baryon_constrained_tau_law"
DEFAULT_RUN_ID = "sparc_step_6d_tau_field_equation_solver"


def main() -> int:
    parser = argparse.ArgumentParser(description="SPARC τ field-equation solver (Step 6D)")
    parser.add_argument("--sparc-data", type=Path, default=ROOT / "data" / "processed" / "sparc_rotation.csv")
    parser.add_argument("--input-run", type=Path, default=ROOT / "outputs" / "runs" / DEFAULT_INPUT_RUN)
    parser.add_argument(
        "--inverse-design-run",
        type=Path,
        default=ROOT / "outputs" / "runs" / DEFAULT_INVERSE_DESIGN,
    )
    parser.add_argument(
        "--inverse-response-run",
        type=Path,
        default=ROOT / "outputs" / "runs" / DEFAULT_INVERSE_RESPONSE,
    )
    parser.add_argument("--tau-law-run", type=Path, default=ROOT / "outputs" / "runs" / DEFAULT_TAU_LAW)
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

    print(BANNER_TAU_FIELD)
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

    result = run_tau_field_solver(
        args.sparc_data,
        run_dir,
        inverse_design_run=args.inverse_design_run,
        inverse_response_run=args.inverse_response_run,
        tau_law_run=args.tau_law_run,
        input_run=args.input_run,
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
                "warning_banner": BANNER_TAU_FIELD,
                "n_galaxies": int(len(result.comparison_by_galaxy)),
            },
        )
        print(f"Run directory: {layout.run_dir}")
        print(f"Manifest: {manifest_path}")

    print(f"\nGalaxies: {len(result.comparison_by_galaxy)}")
    print(f"Profile rows: {len(result.profiles)}")
    print(f"Report: {run_dir / 'reports' / 'tau_field_solver_report.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

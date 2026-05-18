#!/usr/bin/env python3
"""
SPARC Step 1 — boundary-filtered model comparison (analysis only).

NOT full observational validation. Does not modify prior benchmark runs.
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
from tdf_obs.validation.sparc_boundary_filtered_analysis import (
    BANNER_BOUNDARY_FILTERED,
    BANNER_CALIBRATION,
    run_boundary_filtered_analysis,
)

DEFAULT_INPUT_RUN = "v0.20.2_corrected_mond_sparc_calibration"
DEFAULT_AUDIT_RUN = "v0.20.1_sparc_parameter_audit"
DEFAULT_RUN_ID = "sparc_step_1_boundary_filtered"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="SPARC boundary-filtered model comparison (Step 1)",
    )
    parser.add_argument(
        "--input-run",
        type=Path,
        default=ROOT / "outputs" / "runs" / DEFAULT_INPUT_RUN,
    )
    parser.add_argument(
        "--audit-run",
        type=Path,
        default=ROOT / "outputs" / "runs" / DEFAULT_AUDIT_RUN,
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
    parser.add_argument(
        "--boundary-flags",
        type=Path,
        default=None,
        help="Override path to sparc_parameter_boundary_flags.csv",
    )
    args = parser.parse_args()

    print(BANNER_BOUNDARY_FILTERED)
    print(BANNER_CALIBRATION)

    input_run = Path(args.input_run)
    if not input_run.is_dir():
        raise SystemExit(f"Input run not found: {input_run}")

    audit_run = Path(args.audit_run) if args.audit_run else None
    if audit_run is not None and not audit_run.is_dir():
        audit_run = None

    boundary_flags = args.boundary_flags
    if boundary_flags is None:
        legacy = ROOT / "outputs" / "tables" / "sparc_parameter_boundary_flags.csv"
        if legacy.is_file():
            boundary_flags = legacy

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

    result = run_boundary_filtered_analysis(
        input_run,
        run_dir,
        audit_run=audit_run,
        boundary_flags_path=boundary_flags,
    )

    if layout is not None:
        manifest_path = write_run_manifest(
            layout.run_dir,
            {
                "run_id": layout.run_id,
                "script_name": Path(__file__).name,
                "input_run": str(input_run.resolve()),
                "audit_run": str(audit_run.resolve()) if audit_run else None,
                "boundary_data_source": result.boundary_data_source,
                "command": " ".join(shlex.quote(a) for a in sys.argv),
                "output_files": collect_output_files(layout),
                "warning_banner": BANNER_BOUNDARY_FILTERED,
            },
        )
        print(f"Run directory: {layout.run_dir}")
        print(f"Manifest: {manifest_path}")

    print(f"Boundary filtering available: {result.boundary_filtering_available}")
    print(f"Boundary source: {result.boundary_data_source}")
    print("\nComparison by filter:")
    print(result.comparison_by_filter.to_string(index=False))
    print(f"\nReport: {run_dir / 'reports' / 'boundary_filtered_analysis_report.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

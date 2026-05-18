#!/usr/bin/env python3
"""
Phase 8A.1 — SPARC parameter / MOND baseline audit (v0.20.1).

Audits existing Phase 8A calibration outputs. Does NOT modify prior benchmark tables.
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
    assert_production_output_safe,
    collect_output_files,
    copy_run_to_legacy_outputs,
    resolve_versioned_run_dir,
    write_run_manifest,
)
from tdf_obs.validation.sparc_parameter_audit import (
    BANNER_CALIBRATION,
    BANNER_SPARC_AUDIT,
    run_sparc_parameter_audit,
)

DEFAULT_RUN_ID = "v0.20.1_sparc_parameter_audit"
DEFAULT_CALIBRATION_RUN = "v0.20.0_sparc_initial_calibration"


def _default_calibration_summary() -> Path:
    versioned = (
        ROOT
        / "outputs"
        / "runs"
        / DEFAULT_CALIBRATION_RUN
        / "tables"
        / "sparc_real_calibration_summary.csv"
    )
    legacy = ROOT / "outputs" / "tables" / "sparc_real_calibration_summary.csv"
    return versioned if versioned.is_file() else legacy


def _default_comparison() -> Path:
    versioned = (
        ROOT
        / "outputs"
        / "runs"
        / DEFAULT_CALIBRATION_RUN
        / "tables"
        / "sparc_model_comparison_by_galaxy.csv"
    )
    legacy = ROOT / "outputs" / "tables" / "sparc_model_comparison_by_galaxy.csv"
    return versioned if versioned.is_file() else legacy


def main() -> int:
    parser = argparse.ArgumentParser(description="SPARC parameter audit (Phase 8A.1)")
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "data" / "processed" / "sparc_rotation.csv",
    )
    parser.add_argument(
        "--calibration-summary",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--comparison",
        type=Path,
        default=None,
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    parser.add_argument("--run-id", type=str, default=DEFAULT_RUN_ID)
    parser.add_argument(
        "--versioned-output",
        type=lambda x: str(x).lower() in ("1", "true", "yes"),
        default=True,
    )
    parser.add_argument(
        "--write-legacy-copy",
        type=lambda x: str(x).lower() in ("1", "true", "yes"),
        default=False,
    )
    parser.add_argument(
        "--overwrite-run",
        type=lambda x: str(x).lower() in ("1", "true", "yes"),
        default=False,
    )
    parser.add_argument(
        "--skip-boundary-refit",
        action="store_true",
        help="Skip per-galaxy refit for boundary audit (faster, incomplete)",
    )
    args = parser.parse_args()

    calibration_summary = args.calibration_summary or _default_calibration_summary()
    comparison = args.comparison or _default_comparison()

    assert_production_output_safe(
        args.input,
        args.output_dir,
        args.run_id,
        project_outputs_root=ROOT / "outputs",
        versioned_output=args.versioned_output,
    )

    print(BANNER_SPARC_AUDIT)
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
        run_dir = args.output_dir

    result = run_sparc_parameter_audit(
        args.input,
        calibration_summary,
        comparison,
        run_dir,
        refit_for_boundaries=not args.skip_boundary_refit,
    )

    if layout is not None:
        manifest_path = write_run_manifest(
            layout.run_dir,
            {
                "run_id": layout.run_id,
                "script_name": Path(__file__).name,
                "input_path": str(args.input.resolve()),
                "calibration_summary": str(calibration_summary.resolve()),
                "comparison_csv": str(comparison.resolve()),
                "command": " ".join(shlex.quote(a) for a in sys.argv),
                "output_files": collect_output_files(layout),
                "warning_banner": BANNER_SPARC_AUDIT,
                "recommendation": result.get("recommendation"),
            },
        )
        if args.write_legacy_copy:
            copy_run_to_legacy_outputs(layout, args.output_dir)
        print(f"Run directory: {layout.run_dir}")
        print(f"Manifest: {manifest_path}")
        report_path = layout.reports / "sparc_parameter_audit_report.md"
    else:
        report_path = run_dir / "reports" / "sparc_parameter_audit_report.md"

    print(f"Recommendation: {result['recommendation']}")
    print(f"MOND audit rows: {len(result['mond_df'])}")
    print(f"Boundary audit rows: {len(result['boundary_df'])}")
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

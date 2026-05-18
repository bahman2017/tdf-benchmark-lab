#!/usr/bin/env python3
"""
SPARC Step 7 — final synthesis report.

Aggregates corrected-MOND calibration and robustness Steps 1–6.
Does not run new fits. NOT full observational validation.
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
from tdf_obs.validation.sparc_final_synthesis import (
    BANNER_CALIBRATION,
    BANNER_SYNTHESIS,
    DEFAULT_RUN_PATHS,
    run_final_synthesis,
)

DEFAULT_RUN_ID = "sparc_final_synthesis"


def main() -> int:
    parser = argparse.ArgumentParser(description="SPARC final synthesis (Step 7)")
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
    for key in DEFAULT_RUN_PATHS:
        parser.add_argument(
            f"--{key.replace('_', '-')}-run",
            type=str,
            default=None,
            help=f"Override path name under outputs/runs/ for {key}",
        )
    args = parser.parse_args()

    print(BANNER_SYNTHESIS)
    print(BANNER_CALIBRATION)

    run_paths = dict(DEFAULT_RUN_PATHS)
    for key in DEFAULT_RUN_PATHS:
        val = getattr(args, key + "_run", None)
        if val:
            run_paths[key] = val

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

    result = run_final_synthesis(
        run_dir,
        runs_root=args.output_dir / "runs",
        run_paths=run_paths,
    )

    if layout is not None:
        manifest_path = write_run_manifest(
            layout.run_dir,
            {
                "run_id": layout.run_id,
                "script_name": Path(__file__).name,
                "command": " ".join(shlex.quote(a) for a in sys.argv),
                "output_files": collect_output_files(layout),
                "warning_banner": BANNER_SYNTHESIS,
                "assigned_claim_level": result.assigned_claim_level,
                "assigned_recommendation": result.assigned_recommendation,
                "input_runs": run_paths,
            },
        )
        print(f"Run directory: {layout.run_dir}")
        print(f"Manifest: {manifest_path}")

    print(f"\nAssigned claim level: {result.assigned_claim_level}")
    print(f"Recommendation: {result.assigned_recommendation}")
    print(f"Report: {run_dir / 'reports' / 'final_sparc_synthesis_report.md'}")
    print(f"Paper section: {run_dir / 'reports' / 'paper_ready_sparc_section.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

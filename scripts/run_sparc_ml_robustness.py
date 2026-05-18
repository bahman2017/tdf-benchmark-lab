#!/usr/bin/env python3
"""
SPARC Step 3 — mass-to-light robustness analysis.

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
from tdf_obs.validation.sparc_ml_robustness import (
    BANNER_CALIBRATION,
    BANNER_ML_ROBUSTNESS,
    run_ml_robustness_analysis,
)

DEFAULT_RUN_ID = "sparc_step_3_mass_to_light_robustness"


def main() -> int:
    parser = argparse.ArgumentParser(description="SPARC M/L robustness (Step 3)")
    parser.add_argument(
        "--input",
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
    parser.add_argument("--quality-min-points", type=int, default=5)
    args = parser.parse_args()

    print(BANNER_ML_ROBUSTNESS)
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

    result = run_ml_robustness_analysis(
        args.input,
        run_dir,
        max_galaxies=args.max_galaxies,
        quality_min_points=args.quality_min_points,
    )

    if layout is not None:
        manifest_path = write_run_manifest(
            layout.run_dir,
            {
                "run_id": layout.run_id,
                "script_name": Path(__file__).name,
                "input_path": str(args.input.resolve()),
                "command": " ".join(shlex.quote(a) for a in sys.argv),
                "output_files": collect_output_files(layout),
                "warning_banner": BANNER_ML_ROBUSTNESS,
                "regimes": result.regimes,
            },
        )
        print(f"Run directory: {layout.run_dir}")
        print(f"Manifest: {manifest_path}")

    print("\nSummary by M/L regime:")
    print(result.summary.to_string(index=False))
    print(f"\nReport: {run_dir / 'reports' / 'ml_robustness_report.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

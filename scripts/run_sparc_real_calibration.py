#!/usr/bin/env python3
"""
Phase 8A — Real SPARC rotation calibration (baryon / NFW / MOND / TDF K-essence).

Use --corrected-mond true for analytic MOND baseline (v0.20.2).
Versioned outputs default to outputs/runs/<run_id>/.

NOT full observational validation. Does not claim dark-matter replacement.
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
from tdf_obs.validation.sparc_real_calibration import (
    BANNER_CALIBRATION,
    BANNER_SPARC_CALIBRATION,
    BANNER_SPARC_CORRECTED_MOND,
    run_sparc_real_calibration,
)

DEFAULT_RUN_ID_CORRECTED = "v0.20.2_corrected_mond_sparc_calibration"
DEFAULT_RUN_ID_INITIAL = "v0.20.0_sparc_initial_calibration"


def _print_corrected_summary(stats, agg: dict) -> None:
    """Console summary for corrected-MOND rerun."""
    print("\n=== Corrected MOND SPARC calibration summary ===")
    print(f"Galaxies fitted: {stats.galaxies_fitted}/{stats.galaxies_attempted}")
    mond_key = "corrected_mond"
    for model in ("baryon_only", mond_key, "nfw", "tdf_kessence"):
        print(f"  BIC wins ({model}): {agg.get(f'bic_win_{model}', '—')}")
        if f"median_bic_{model}" in agg:
            print(f"  Median BIC ({model}): {agg[f'median_bic_{model}']:.2f}")
        if f"median_reduced_chi2_{model}" in agg:
            print(
                f"  Median reduced χ² ({model}): "
                f"{agg[f'median_reduced_chi2_{model}']:.3f}",
            )
    print(f"TDF vs NFW (ΔBIC<0 count): {agg.get('tdf_beats_nfw_count', '—')}")
    print(f"TDF vs corrected MOND (ΔBIC<0 count): {agg.get('tdf_beats_mond_count', '—')}")
    print(f"MOND active galaxies: {agg.get('mond_active_galaxy_count', stats.mond_active_galaxy_count)}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Real SPARC rotation calibration")
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "data" / "processed" / "sparc_rotation.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    parser.add_argument("--max-galaxies", type=int, default=None)
    parser.add_argument("--quality-min-points", type=int, default=5)
    parser.add_argument(
        "--fit-ml",
        type=lambda x: str(x).lower() in ("1", "true", "yes"),
        default=True,
    )
    parser.add_argument("--mode", type=str, default="real_sparc")
    parser.add_argument("--max-example-plots", type=int, default=6)
    parser.add_argument(
        "--corrected-mond",
        type=lambda x: str(x).lower() in ("1", "true", "yes"),
        default=False,
    )
    parser.add_argument("--run-id", type=str, default=None)
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
    args = parser.parse_args()

    corrected = args.corrected_mond
    run_id = args.run_id or (
        DEFAULT_RUN_ID_CORRECTED if corrected else DEFAULT_RUN_ID_INITIAL
    )

    assert_production_output_safe(
        args.input,
        args.output_dir,
        run_id,
        project_outputs_root=ROOT / "outputs",
        versioned_output=args.versioned_output,
    )

    max_galaxies = args.max_galaxies
    if corrected and max_galaxies is None:
        max_galaxies = None
    elif not corrected and max_galaxies is None:
        max_galaxies = 20

    banner = BANNER_SPARC_CORRECTED_MOND if corrected else BANNER_SPARC_CALIBRATION
    print(banner)
    print(BANNER_CALIBRATION)

    if args.versioned_output:
        layout = resolve_versioned_run_dir(
            args.output_dir,
            run_id,
            overwrite_run=args.overwrite_run,
        )
        run_dir = layout.run_dir
        file_suffix = ""
        effective_run_id = layout.run_id
    else:
        layout = None
        run_dir = args.output_dir
        file_suffix = None
        effective_run_id = run_id

    summary_df, comparison_df, stats = run_sparc_real_calibration(
        args.input,
        run_dir,
        max_galaxies=max_galaxies,
        quality_min_points=args.quality_min_points,
        fit_ml=args.fit_ml,
        mode=args.mode,
        max_example_plots=args.max_example_plots,
        corrected_mond=corrected,
        file_suffix=file_suffix,
        run_id=effective_run_id,
    )

    suffix = stats.output_suffix
    command = " ".join(shlex.quote(a) for a in sys.argv)

    if layout is not None:
        manifest_path = write_run_manifest(
            layout.run_dir,
            {
                "run_id": effective_run_id,
                "script_name": Path(__file__).name,
                "input_path": str(args.input.resolve()),
                "dataset_mode": args.mode,
                "real_observational_data": True,
                "command": command,
                "output_files": collect_output_files(layout),
                "warning_banner": banner,
                "corrected_mond": corrected,
                "galaxies_fitted": stats.galaxies_fitted,
                "galaxies_attempted": stats.galaxies_attempted,
            },
        )
        if args.write_legacy_copy:
            copy_run_to_legacy_outputs(layout, args.output_dir)
        print(f"Run directory: {layout.run_dir}")
        print(f"Manifest: {manifest_path}")
    else:
        manifest_path = None

    print(f"Input: {args.input}")
    print(
        f"Fitted {stats.galaxies_fitted}/{stats.galaxies_attempted} galaxies "
        f"(skipped {stats.galaxies_skipped}, failed {stats.galaxies_failed})",
    )
    if corrected and stats.aggregate:
        _print_corrected_summary(stats, stats.aggregate)
    elif stats.aggregate:
        print("Aggregate:", stats.aggregate)

    tables_dir = run_dir / "tables"
    reports_dir = run_dir / "reports"
    print(f"Summary: {tables_dir / f'sparc_real_calibration_summary{suffix}.csv'}")
    print(f"Comparison: {tables_dir / f'sparc_model_comparison_by_galaxy{suffix}.csv'}")
    print(f"Report: {stats.report_path or reports_dir / f'sparc_real_calibration_report{suffix}.md'}")
    print(f"Rows in summary: {len(summary_df)} | galaxies in comparison: {len(comparison_df)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

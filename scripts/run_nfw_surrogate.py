#!/usr/bin/env python3
"""
Phase 4A — ΛCDM/NFW rotation benchmark recovery.

NOT observational validation. See outputs/reports/nfw_surrogate_report.md.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tdf_obs.validation.nfw_surrogate import (
    BANNER_LCDM_NFW,
    DEFAULT_MIMIC_TOLERANCE_PERCENT,
    list_benchmark_cases,
    run_nfw_surrogate_pipeline,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run ΛCDM/NFW rotation benchmark recovery (Phase 4A).")
    p.add_argument(
        "--case",
        action="append",
        dest="cases",
        metavar="CASE_NAME",
        help="Run only this benchmark case (repeatable). Default: all registered cases.",
    )
    p.add_argument(
        "--max-cases",
        type=int,
        default=None,
        help="Limit number of cases (in registry order).",
    )
    p.add_argument(
        "--tolerance",
        type=float,
        default=DEFAULT_MIMIC_TOLERANCE_PERCENT,
        help=f"Mimic success if mean relative curve error < this %% (default {DEFAULT_MIMIC_TOLERANCE_PERCENT}).",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    print(BANNER_LCDM_NFW)
    print("Running ΛCDM/GR+DM NFW teacher vs TDF student benchmarks...")

    case_names = args.cases
    if case_names:
        available = set(list_benchmark_cases())
        unknown = [c for c in case_names if c not in available]
        if unknown:
            print(f"Unknown case(s): {unknown}", file=sys.stderr)
            print(f"Available: {sorted(available)}", file=sys.stderr)
            return 1

    records = run_nfw_surrogate_pipeline(
        outputs_root=ROOT / "outputs",
        case_names=case_names,
        max_cases=args.max_cases,
        mimic_tolerance_percent=args.tolerance,
    )
    n_mimic = sum(1 for r in records if r.mimic_success)
    print(f"Processed {len(records)} benchmark case(s); mimic success: {n_mimic}/{len(records)}")
    print(f"Tolerance: {args.tolerance}% mean relative curve error")
    print(f"Summary: {ROOT / 'outputs' / 'tables' / 'nfw_surrogate_fit_summary.csv'}")
    print(f"Report: {ROOT / 'outputs' / 'reports' / 'nfw_surrogate_report.md'}")
    print(f"Figures: {ROOT / 'outputs' / 'figures' / 'nfw_surrogate_<case>.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

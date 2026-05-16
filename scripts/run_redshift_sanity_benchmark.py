#!/usr/bin/env python3
"""
Phase 4D — Redshift/Doppler sanity benchmark.

NOT observational validation. Configured z_tau cases only.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tdf_obs.validation.redshift_sanity_benchmark import (
    BANNER_REDSHIFT_SANITY,
    list_benchmark_cases,
    run_redshift_sanity_benchmark_pipeline,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run redshift sanity benchmark (Phase 4D).")
    p.add_argument(
        "--case",
        action="append",
        dest="cases",
        metavar="CASE_NAME",
        help="Run only this case (repeatable). Default: all registered cases.",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    print(BANNER_REDSHIFT_SANITY)

    case_names = args.cases
    if case_names:
        available = set(list_benchmark_cases())
        unknown = [c for c in case_names if c not in available]
        if unknown:
            print(f"Unknown case(s): {unknown}", file=sys.stderr)
            print(f"Available: {sorted(available)}", file=sys.stderr)
            return 1

    _df, summary = run_redshift_sanity_benchmark_pipeline(
        outputs_root=ROOT / "outputs",
        case_names=case_names,
    )
    print(
        f"Cases: {summary['n_cases']} | pass: {summary['n_pass']} | "
        f"borderline: {summary['n_borderline']} | fail: {summary['n_fail']}",
    )
    print(f"Expected failures: {summary['n_failed_as_expected']}/{summary['n_expected_failures']}")
    print(f"Summary: {ROOT / 'outputs' / 'tables' / 'redshift_sanity_benchmark_summary.csv'}")
    print(f"Report: {ROOT / 'outputs' / 'reports' / 'redshift_sanity_benchmark_report.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
Phase 5A — Core–cusp stress test.

NOT observational validation. Synthetic cuspy vs cored teachers only.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tdf_obs.validation.core_cusp_stress import (
    BANNER_CORE_CUSP,
    list_benchmark_cases,
    run_core_cusp_stress_pipeline,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run core–cusp stress test (Phase 5A).")
    p.add_argument(
        "--case",
        action="append",
        dest="cases",
        metavar="CASE_NAME",
        help="Run only this case (repeatable).",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    print(BANNER_CORE_CUSP)

    if args.cases:
        available = set(list_benchmark_cases())
        unknown = [c for c in args.cases if c not in available]
        if unknown:
            print(f"Unknown case(s): {unknown}", file=sys.stderr)
            print(f"Available: {sorted(available)}", file=sys.stderr)
            return 1

    _df, results = run_core_cusp_stress_pipeline(
        outputs_root=ROOT / "outputs",
        case_names=args.cases,
    )
    n_core_adv = sum(1 for r in results if r.core_advantage_flag)
    print(f"Cases: {len(results)} | TDF core advantage: {n_core_adv}/{len(results)}")
    print(f"Summary: {ROOT / 'outputs' / 'tables' / 'core_cusp_stress_summary.csv'}")
    print(f"Report: {ROOT / 'outputs' / 'reports' / 'core_cusp_stress_report.md'}")
    print(f"Figures: {ROOT / 'outputs' / 'figures' / 'core_cusp_<case>.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

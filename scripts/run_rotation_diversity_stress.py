#!/usr/bin/env python3
"""
Phase 5B — Rotation-curve diversity stress test.

NOT observational validation. Synthetic teacher curves only.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tdf_obs.validation.rotation_diversity_stress import (
    BANNER_ROTATION_DIVERSITY,
    list_benchmark_cases,
    run_rotation_diversity_stress,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run rotation-curve diversity stress test (Phase 5B).")
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
    print(BANNER_ROTATION_DIVERSITY)

    if args.cases:
        available = set(list_benchmark_cases())
        unknown = [c for c in args.cases if c not in available]
        if unknown:
            print(f"Unknown case(s): {unknown}", file=sys.stderr)
            print(f"Available: {sorted(available)}", file=sys.stderr)
            return 1

    _df, results = run_rotation_diversity_stress(
        outputs_root=ROOT / "outputs",
        case_names=args.cases,
    )
    n_tdf = sum(1 for r in results if r.tdf_best_flag or r.tdf_core_best_flag)
    n_nfw = sum(1 for r in results if r.nfw_best_flag)
    print(f"Cases: {len(results)} | any TDF best: {n_tdf} | NFW best: {n_nfw}")
    print(f"Summary: {ROOT / 'outputs' / 'tables' / 'rotation_diversity_stress_summary.csv'}")
    print(f"Report: {ROOT / 'outputs' / 'reports' / 'rotation_diversity_stress_report.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

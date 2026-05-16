#!/usr/bin/env python3
"""
Phase 5C — Same-τ multi-observable consistency benchmark.

NOT observational validation. Synthetic teachers with shared (B, r0) only.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tdf_obs.validation.same_tau_consistency import (
    BANNER_SAME_TAU,
    list_benchmark_cases,
    run_same_tau_consistency_benchmark,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run same-τ multi-observable consistency benchmark (Phase 5C).",
    )
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
    print(BANNER_SAME_TAU)

    if args.cases:
        available = set(list_benchmark_cases())
        unknown = [c for c in args.cases if c not in available]
        if unknown:
            print(f"Unknown case(s): {unknown}", file=sys.stderr)
            print(f"Available: {sorted(available)}", file=sys.stderr)
            return 1

    _df, results = run_same_tau_consistency_benchmark(
        outputs_root=ROOT / "outputs",
        case_names=args.cases,
    )
    n_pass = sum(1 for r in results if r.same_tau_pass)
    print(f"Cases: {len(results)} | same-τ pass: {n_pass}/{len(results)}")
    print(f"Summary: {ROOT / 'outputs' / 'tables' / 'same_tau_consistency_summary.csv'}")
    print(f"Report: {ROOT / 'outputs' / 'reports' / 'same_tau_consistency_report.md'}")
    print(f"Figures: {ROOT / 'outputs' / 'figures' / 'same_tau_<case>_*.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

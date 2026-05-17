#!/usr/bin/env python3
"""
Phase 5E — CMB acoustic-scale compatibility benchmark.

NOT observational validation. ΛCDM teacher vs TDF background ε_τ(z) only.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tdf_obs.validation.cmb_acoustic_benchmark import (
    BANNER_CMB,
    DEFAULT_PASS_THRESHOLD_PERCENT,
    list_benchmark_cases,
    run_cmb_acoustic_benchmark,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run CMB acoustic-scale benchmark (Phase 5E).")
    p.add_argument(
        "--case",
        action="append",
        dest="cases",
        metavar="CASE_NAME",
        help="Run only this case (repeatable).",
    )
    p.add_argument(
        "--tolerance",
        type=float,
        default=DEFAULT_PASS_THRESHOLD_PERCENT,
        help=f"Pass if all relative errors < this %% (default {DEFAULT_PASS_THRESHOLD_PERCENT}).",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    print(BANNER_CMB)

    if args.cases:
        available = set(list_benchmark_cases())
        unknown = [c for c in args.cases if c not in available]
        if unknown:
            print(f"Unknown case(s): {unknown}", file=sys.stderr)
            print(f"Available: {sorted(available)}", file=sys.stderr)
            return 1

    _df, results = run_cmb_acoustic_benchmark(
        cases=args.cases,
        outputs_root=ROOT / "outputs",
        pass_threshold_percent=args.tolerance,
    )
    n_pass = sum(1 for r in results if r.cmb_compatibility_pass)
    print(f"Cases: {len(results)} | compatibility pass: {n_pass}/{len(results)}")
    print(f"Tolerance: {args.tolerance}% on H, r_s, D_M, ell_A")
    print(f"Summary: {ROOT / 'outputs' / 'tables' / 'cmb_acoustic_benchmark_summary.csv'}")
    print(f"Report: {ROOT / 'outputs' / 'reports' / 'cmb_acoustic_benchmark_report.md'}")
    print(f"Figure: {ROOT / 'outputs' / 'figures' / 'cmb_acoustic_epsilon_tau_cases.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

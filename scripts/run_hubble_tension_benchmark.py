#!/usr/bin/env python3
"""
Phase 5F — CMB-safe Hubble tension benchmark.

NOT observational validation. Does NOT solve the Hubble tension.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tdf_obs.validation.hubble_tension_benchmark import (
    BANNER_HUBBLE,
    DEFAULT_CMB_THRESHOLD_PERCENT,
    HUBBLE_SHIFT_MAX_PERCENT,
    HUBBLE_SHIFT_MIN_PERCENT,
    list_benchmark_cases,
    run_hubble_tension_benchmark,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run CMB-safe Hubble tension benchmark (Phase 5F).")
    p.add_argument("--case", action="append", dest="cases", metavar="CASE_NAME")
    p.add_argument(
        "--cmb-tolerance",
        type=float,
        default=DEFAULT_CMB_THRESHOLD_PERCENT,
        help="CMB-safe threshold (%%) on H(z_*), r_s, D_M, ell_A.",
    )
    p.add_argument(
        "--h0-min",
        type=float,
        default=HUBBLE_SHIFT_MIN_PERCENT,
        help="Minimum |H0 shift| %% for Hubble-shift-active.",
    )
    p.add_argument(
        "--h0-max",
        type=float,
        default=HUBBLE_SHIFT_MAX_PERCENT,
        help="Maximum |H0 shift| %% for Hubble-shift-active.",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    print(BANNER_HUBBLE)

    if args.cases:
        available = set(list_benchmark_cases())
        unknown = [c for c in args.cases if c not in available]
        if unknown:
            print(f"Unknown case(s): {unknown}", file=sys.stderr)
            print(f"Available: {sorted(available)}", file=sys.stderr)
            return 1

    _df, results = run_hubble_tension_benchmark(
        cases=args.cases,
        outputs_root=ROOT / "outputs",
        cmb_threshold_percent=args.cmb_tolerance,
        hubble_min_percent=args.h0_min,
        hubble_max_percent=args.h0_max,
    )
    n_cmb = sum(1 for r in results if r.cmb_safe_pass)
    n_hub = sum(1 for r in results if r.hubble_shift_active)
    n_ok = sum(1 for r in results if r.overall_success)
    print(f"Cases: {len(results)} | CMB-safe: {n_cmb} | Hubble-active: {n_hub} | overall: {n_ok}")
    print(f"Summary: {ROOT / 'outputs' / 'tables' / 'hubble_tension_benchmark_summary.csv'}")
    print(f"Report: {ROOT / 'outputs' / 'reports' / 'hubble_tension_benchmark_report.md'}")
    print(f"Figures: {ROOT / 'outputs' / 'figures' / 'hubble_tension_*.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

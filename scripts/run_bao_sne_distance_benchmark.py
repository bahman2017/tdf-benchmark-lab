#!/usr/bin/env python3
"""
Phase 5G — BAO/SNe late-time distance consistency benchmark.

NOT observational validation. No real BAO, Pantheon, or SH0ES data.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tdf_obs.validation.bao_sne_distance_benchmark import (
    BANNER_BAO_SNE,
    DEFAULT_DL_THRESHOLD_PERCENT,
    DEFAULT_DM_DV_THRESHOLD_PERCENT,
    DEFAULT_H_BAO_THRESHOLD_PERCENT,
    HUBBLE_SHIFT_MAX_PERCENT,
    HUBBLE_SHIFT_MIN_PERCENT,
    list_benchmark_cases,
    run_bao_sne_distance_benchmark,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run BAO/SNe distance benchmark (Phase 5G).")
    p.add_argument("--case", action="append", dest="cases", metavar="CASE_NAME")
    p.add_argument("--dl-tolerance", type=float, default=DEFAULT_DL_THRESHOLD_PERCENT)
    p.add_argument("--dm-dv-tolerance", type=float, default=DEFAULT_DM_DV_THRESHOLD_PERCENT)
    p.add_argument("--h-tolerance", type=float, default=DEFAULT_H_BAO_THRESHOLD_PERCENT)
    p.add_argument("--h0-min", type=float, default=HUBBLE_SHIFT_MIN_PERCENT)
    p.add_argument("--h0-max", type=float, default=HUBBLE_SHIFT_MAX_PERCENT)
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    print(BANNER_BAO_SNE)

    if args.cases:
        available = set(list_benchmark_cases())
        unknown = [c for c in args.cases if c not in available]
        if unknown:
            print(f"Unknown case(s): {unknown}", file=sys.stderr)
            print(f"Available: {sorted(available)}", file=sys.stderr)
            return 1

    _df, results = run_bao_sne_distance_benchmark(
        cases=args.cases,
        outputs_root=ROOT / "outputs",
        dl_threshold_percent=args.dl_tolerance,
        dm_dv_threshold_percent=args.dm_dv_tolerance,
        h_bao_threshold_percent=args.h_tolerance,
        hubble_min_percent=args.h0_min,
        hubble_max_percent=args.h0_max,
    )
    n_dist = sum(1 for r in results if r.distance_safe_pass)
    n_hub = sum(1 for r in results if r.hubble_shift_active)
    n_ok = sum(1 for r in results if r.overall_success)
    print(f"Cases: {len(results)} | distance-safe: {n_dist} | Hubble-active: {n_hub} | overall: {n_ok}")
    print(f"Summary: {ROOT / 'outputs' / 'tables' / 'bao_sne_distance_benchmark_summary.csv'}")
    print(f"Report: {ROOT / 'outputs' / 'reports' / 'bao_sne_distance_benchmark_report.md'}")
    print(f"Figures: {ROOT / 'outputs' / 'figures' / 'bao_sne_distance_*.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

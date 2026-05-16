#!/usr/bin/env python3
"""
Phase 4C — Black-hole exterior GR-limit benchmark.

NOT observational validation. Phenomenological ansatz checks only.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tdf_obs.validation.black_hole_gr_benchmark import (
    BANNER_BH_GR_LIMIT,
    DEFAULT_Q_VALUES,
    run_black_hole_gr_benchmark_pipeline,
    summarize_black_hole_benchmark,
    run_black_hole_gr_limit_benchmark,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run black-hole GR-limit benchmark (Phase 4C).")
    p.add_argument(
        "--mass-solar",
        type=float,
        default=10.0,
        help="Benchmark black-hole mass in solar masses (default 10).",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    print(BANNER_BH_GR_LIMIT)
    results = run_black_hole_gr_limit_benchmark(DEFAULT_Q_VALUES, mass_solar=args.mass_solar)
    summary = summarize_black_hole_benchmark(results)
    run_black_hole_gr_benchmark_pipeline(
        outputs_root=ROOT / "outputs",
        mass_solar=args.mass_solar,
    )
    print(f"Cases: {summary['n_cases']} | GR-like: {summary['n_gr_like']} | "
          f"strongly_modified: {summary['n_strongly_modified']} | no_horizon: {summary['n_no_horizon']}")
    print(f"Summary: {ROOT / 'outputs' / 'tables' / 'black_hole_gr_benchmark_summary.csv'}")
    print(f"Report: {ROOT / 'outputs' / 'reports' / 'black_hole_gr_benchmark_report.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

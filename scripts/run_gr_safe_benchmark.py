#!/usr/bin/env python3
"""
Phase 4B — Expanded GR-safe local benchmark.

NOT observational validation. Configurable scaffold cases only.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tdf_obs.validation.gr_safe_benchmark import (
    BANNER_GR_SAFE,
    list_benchmark_cases,
    run_gr_safe_benchmark_pipeline,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run GR-safe ΛCDM/GR local benchmarks (Phase 4B).")
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
    print(BANNER_GR_SAFE)

    case_names = args.cases
    if case_names:
        available = set(list_benchmark_cases())
        unknown = [c for c in case_names if c not in available]
        if unknown:
            print(f"Unknown case(s): {unknown}", file=sys.stderr)
            print(f"Available: {sorted(available)}", file=sys.stderr)
            return 1

    df = run_gr_safe_benchmark_pipeline(outputs_root=ROOT / "outputs", case_names=case_names)
    n_pass = int(df["pass"].sum())
    print(f"Cases: {len(df)} | pass: {n_pass} | fail: {len(df) - n_pass}")
    print(f"Summary: {ROOT / 'outputs' / 'tables' / 'gr_safe_benchmark_summary.csv'}")
    print(f"Report: {ROOT / 'outputs' / 'reports' / 'gr_safe_benchmark_report.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

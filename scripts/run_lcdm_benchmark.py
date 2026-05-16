#!/usr/bin/env python3
"""
Phase 3C — ΛCDM/GR benchmark recovery scaffolds.

NOT observational validation. Configurable diagnostic cases only.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tdf_obs.validation.lcdm_benchmark import BANNER_LCDM_BENCHMARK, run_lcdm_benchmark_pipeline


def main() -> int:
    print(BANNER_LCDM_BENCHMARK)
    df = run_lcdm_benchmark_pipeline(outputs_root=ROOT / "outputs", project_root=ROOT)
    n_pass = int((df["status"] == "pass").sum())
    n_fail = int((df["status"] == "fail").sum())
    print(f"Cases: {len(df)} | pass: {n_pass} | fail: {n_fail}")
    print(f"Summary: {ROOT / 'outputs' / 'tables' / 'lcdm_benchmark_summary.csv'}")
    print(f"Report: {ROOT / 'outputs' / 'reports' / 'lcdm_benchmark_report.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

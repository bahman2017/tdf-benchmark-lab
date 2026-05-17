#!/usr/bin/env python3
"""
Phase 6B — Dirac / spinor limit benchmark.

NOT full fermion unification. Flat-space and algebraic consistency only.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tdf_obs.validation.dirac_spinor_limit import (
    BANNER_DIRAC,
    run_dirac_spinor_limit_benchmark,
)


def main() -> int:
    print(BANNER_DIRAC)
    _df, results = run_dirac_spinor_limit_benchmark(outputs_root=ROOT / "outputs")
    n_pass = sum(1 for r in results if r.pass_)
    print(f"Cases: {len(results)} | pass: {n_pass}/{len(results)}")
    for r in results:
        print(f"  {r.case_name}: max_residual={r.max_residual:.2e} pass={r.pass_}")
    print(f"Summary: {ROOT / 'outputs' / 'tables' / 'dirac_spinor_limit_summary.csv'}")
    print(f"Report: {ROOT / 'outputs' / 'reports' / 'dirac_spinor_limit_report.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

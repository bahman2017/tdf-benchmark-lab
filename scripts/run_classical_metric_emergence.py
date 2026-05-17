#!/usr/bin/env python3
"""
Phase 6E — Classical metric emergence from τ averaging.

NOT full objective-collapse solution. Coarse-graining benchmark only.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tdf_obs.validation.classical_metric_emergence import (
    BANNER_CLASSICAL,
    run_classical_metric_emergence_benchmark,
)


def main() -> int:
    print(BANNER_CLASSICAL)
    _df, results = run_classical_metric_emergence_benchmark(outputs_root=ROOT / "outputs")
    n_ok = sum(1 for r in results if r.overall_pass == (r.expected_status == "pass"))
    print(f"Cases: {len(results)} | outcomes matched: {n_ok}/{len(results)}")
    for r in results:
        print(
            f"  {r.case_name}: classicality={r.classicality_score:.2f} "
            f"expect={r.expected_status} pass={r.overall_pass}",
        )
    print(f"Summary: {ROOT / 'outputs' / 'tables' / 'classical_metric_emergence_summary.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

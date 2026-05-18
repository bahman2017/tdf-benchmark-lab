#!/usr/bin/env python3
"""
Phase 7A — K-essence source viability from static baryonic matter.

Spherical proxy only — NOT observational validation.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tdf_obs.validation.kessence_source_viability import (
    BANNER_CALIBRATION,
    BANNER_KESSENCE_SOURCE,
    run_kessence_source_viability_benchmark,
)


def main() -> int:
    print(BANNER_KESSENCE_SOURCE)
    print(BANNER_CALIBRATION)
    _df, results = run_kessence_source_viability_benchmark(outputs_root=ROOT / "outputs")
    n_ok = sum(1 for r in results if r.expected_failure_matched)
    print(f"Cases: {len(results)} | Expected outcomes matched: {n_ok}/{len(results)}")
    for r in results:
        print(
            f"  {r.case_name}: slope={r.outer_gradient_slope:.2f} "
            f"flat={r.rotation_flatness_score:.2f} expect={r.expected_status} "
            f"matched={r.expected_failure_matched} fail_detected={r.failure_detected}",
        )
    print(f"Summary: {ROOT / 'outputs' / 'tables' / 'kessence_source_viability_summary.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

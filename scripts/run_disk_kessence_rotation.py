#!/usr/bin/env python3
"""
Phase 7B — Axisymmetric disk K-essence rotation benchmark.

Synthetic disk proxies only — NOT real SPARC validation.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tdf_obs.validation.disk_kessence_rotation import (
    BANNER_CALIBRATION,
    BANNER_DISK_KESSENCE,
    run_disk_kessence_rotation_benchmark,
)


def main() -> int:
    print(BANNER_DISK_KESSENCE)
    print(BANNER_CALIBRATION)
    _df, results = run_disk_kessence_rotation_benchmark(outputs_root=ROOT / "outputs")
    n_ok = sum(1 for r in results if r.overall_pass)
    print(f"Cases: {len(results)} | Expected outcomes matched: {n_ok}/{len(results)}")
    for r in results:
        print(
            f"  {r.case_name}: Δflat={r.flatness_improvement:+.2f} "
            f"total_flat={r.total_outer_flatness_score:.2f} "
            f"expect={r.expected_status} matched={r.overall_pass} "
            f"fail_detected={r.failure_detected}",
        )
    print(f"Summary: {ROOT / 'outputs' / 'tables' / 'disk_kessence_rotation_summary.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

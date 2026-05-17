#!/usr/bin/env python3
"""
Phase 6C — Entanglement from configuration-space τ geometry.

NOT full Bell-theorem resolution. Two-qubit toy benchmarks only.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tdf_obs.validation.entanglement_tau_geometry import (
    BANNER_ENTANGLEMENT,
    run_entanglement_tau_geometry_benchmark,
)


def main() -> int:
    print(BANNER_ENTANGLEMENT)
    _df, results = run_entanglement_tau_geometry_benchmark(outputs_root=ROOT / "outputs")
    n_pass = sum(1 for r in results if r.overall_pass)
    print(f"Cases: {len(results)} | pass: {n_pass}/{len(results)}")
    for r in results:
        print(
            f"  {r.case_name}: S={r.entropy_A:.3f} C={r.concurrence:.3f} "
            f"CHSH={r.chsh_S:.3f} pass={r.overall_pass}",
        )
    print(f"Summary: {ROOT / 'outputs' / 'tables' / 'entanglement_tau_geometry_summary.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

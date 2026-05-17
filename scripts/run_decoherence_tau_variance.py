#!/usr/bin/env python3
"""
Phase 6D — Decoherence from τ-variance benchmark.

NOT full measurement-problem solution. Two-branch toy model only.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tdf_obs.validation.decoherence_tau_variance import (
    BANNER_DECOHERENCE,
    run_decoherence_tau_variance_benchmark,
)


def main() -> int:
    print(BANNER_DECOHERENCE)
    _df, results = run_decoherence_tau_variance_benchmark(outputs_root=ROOT / "outputs")
    n_pass = sum(1 for r in results if r.overall_pass)
    print(f"Cases: {len(results)} | pass: {n_pass}/{len(results)}")
    for r in results:
        print(
            f"  {r.case_name}: C_final={r.final_coherence:.3f} "
            f"Γ_fit={r.fitted_gamma:.3f} pass={r.overall_pass}",
        )
    print(f"Summary: {ROOT / 'outputs' / 'tables' / 'decoherence_tau_variance_summary.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

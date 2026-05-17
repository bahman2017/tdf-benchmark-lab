#!/usr/bin/env python3
"""
Phase 6G — Unified microscopic quantum consistency matrix.

Integrates Phase 6A–6F reports only. NOT full quantum-gravity proof.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tdf_obs.validation.unified_microscopic_quantum_limit import (
    BANNER_UNIFIED,
    READINESS_THRESHOLD,
    run_unified_microscopic_quantum_limit_benchmark,
)


def main() -> int:
    print(BANNER_UNIFIED)
    result = run_unified_microscopic_quantum_limit_benchmark(outputs_root=ROOT / "outputs")
    n_pass = int((result.matrix["status"] == "pass").sum())
    print(f"Phases integrated: {len(result.matrix)} | status pass: {n_pass}/{len(result.matrix)}")
    print(f"Readiness score: {result.readiness_score:.2f} (threshold {READINESS_THRESHOLD})")
    for _, row in result.matrix.iterrows():
        print(f"  {row['phase']}: {row['core_result']} [{row['status']}]")
    out = ROOT / "outputs" / "tables" / "unified_microscopic_quantum_limit_matrix.csv"
    print(f"Summary: {out}")
    return 0 if result.readiness_score >= READINESS_THRESHOLD else 1


if __name__ == "__main__":
    sys.exit(main())

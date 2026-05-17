#!/usr/bin/env python3
"""
Phase 6A / 6A.1 — Schrödinger-from-TDF action benchmark.

QHJ convention cleanup (ψ = √ρ exp(-iτ), E = ℏ ∂_t τ).
NOT full quantum validation. Symbolic/numerical consistency only.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tdf_obs.validation.schrodinger_from_tdf import (
    BANNER_SCHRODINGER,
    DEFAULT_PASS_THRESHOLD,
    PHASE_CONVENTION,
    QHJ_NORMALIZED_PASS_THRESHOLD,
    run_schrodinger_from_tdf_benchmark,
)


def main() -> int:
    print(BANNER_SCHRODINGER)
    print(PHASE_CONVENTION)
    _df, results = run_schrodinger_from_tdf_benchmark(
        outputs_root=ROOT / "outputs",
        pass_threshold=DEFAULT_PASS_THRESHOLD,
        qhj_threshold=QHJ_NORMALIZED_PASS_THRESHOLD,
    )
    n_pass = sum(1 for r in results if r.pass_)
    print(f"Cases: {len(results)} | pass: {n_pass}/{len(results)}")
    for r in results:
        print(
            f"  {r.case_name}: QHJ norm={r.max_normalized_qhj_residual:.2e} "
            f"(raw={r.max_qhj_residual:.2e})",
        )
    print(f"Summary: {ROOT / 'outputs' / 'tables' / 'schrodinger_from_tdf_summary.csv'}")
    print(f"Report: {ROOT / 'outputs' / 'reports' / 'schrodinger_from_tdf_report.md'}")
    print(f"Figures: {ROOT / 'outputs' / 'figures' / 'schrodinger_*_residual.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

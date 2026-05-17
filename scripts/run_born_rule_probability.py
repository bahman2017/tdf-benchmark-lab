#!/usr/bin/env python3
"""
Phase 6F — Born-rule probability emergence proxy.

NOT full Born-rule derivation. Finite-branch consistency benchmark only.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tdf_obs.validation.born_rule_probability import (
    BANNER_BORN,
    run_born_rule_probability_benchmark,
)


def main() -> int:
    print(BANNER_BORN)
    _df, results = run_born_rule_probability_benchmark(outputs_root=ROOT / "outputs")
    n_pass = sum(1 for r in results if r.overall_pass)
    print(f"Cases: {len(results)} | passed: {n_pass}/{len(results)}")
    for r in results:
        print(f"  {r.case_name}: pass={r.overall_pass}")
    print(f"Summary: {ROOT / 'outputs' / 'tables' / 'born_rule_probability_summary.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

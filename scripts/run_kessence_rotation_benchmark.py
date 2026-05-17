#!/usr/bin/env python3
"""
TDF K-essence rotation calibration benchmark.

Phenomenological candidate-model comparison only — not observational validation.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tdf_obs.validation.kessence_rotation_benchmark import (
    BANNER_CALIBRATION,
    BANNER_KESSENCE,
    run_kessence_rotation_benchmark,
)


def main() -> int:
    print(BANNER_KESSENCE)
    print(BANNER_CALIBRATION)
    _df, result, _data = run_kessence_rotation_benchmark(
        outputs_root=ROOT / "outputs",
        project_root=ROOT,
    )
    print(f"Galaxy: {result.galaxy_id} | mode: {result.data_mode}")
    print(f"Best model (BIC): {result.best_model_by_bic}")
    print(f"a0 = {result.a0:.4g} | MSE(K-essence) = {result.mse_kessence:.2f}")
    print(f"Summary: {ROOT / 'outputs' / 'tables' / 'kessence_rotation_benchmark_summary.csv'}")
    return 0 if result.kessence_beats_baryon_by_bic else 0


if __name__ == "__main__":
    sys.exit(main())

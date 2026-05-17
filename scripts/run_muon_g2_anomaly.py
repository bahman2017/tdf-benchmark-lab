#!/usr/bin/env python3
"""
Phase 6H — Muon g-2 anomaly phenomenological QED benchmark.

Order-of-magnitude τ-geometry coupling estimate only — not full QFT derivation.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tdf_obs.validation.muon_g2_anomaly import (
    BANNER_CALIBRATION,
    BANNER_MUON_G2,
    INTERPRETATION_DISCLAIMER,
    muon_compton_wavelength_m,
    run_muon_g2_anomaly_benchmark,
)


def main() -> int:
    print(BANNER_MUON_G2)
    print(BANNER_CALIBRATION)
    print(INTERPRETATION_DISCLAIMER)
    _df, results = run_muon_g2_anomaly_benchmark(outputs_root=ROOT / "outputs")
    n_pass = sum(1 for r in results if r.overall_pass)
    print(f"Cases: {len(results)} | passed: {n_pass}/{len(results)}")
    print(f"Muon Compton wavelength (l_tau): {muon_compton_wavelength_m():.4e} m")
    for r in results:
        print(
            f"  {r.case_name}: eps_tau={r.epsilon_tau:.3e} "
            f"Delta_a_mu={r.delta_a_mu_theory:.3e} pass={r.overall_pass}",
        )
    print(f"Summary: {ROOT / 'outputs' / 'tables' / 'muon_g2_anomaly_summary.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

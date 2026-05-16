#!/usr/bin/env python3
"""
Rotation pipeline CLI.

Modes (see rotation_metadata.yaml):
  synthetic_validation       — no rotation.csv
  demo_fixture_calibration   — CSV without confirmed real metadata
  real_data_calibration        — CSV + metadata confirming real observations
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tdf_obs.io.loaders import default_processed_rotation_path, resolve_dataset_info
from tdf_obs.pipelines.run_rotation_pipeline import run_rotation_pipeline


def main() -> None:
    csv_path = default_processed_rotation_path(ROOT)
    info = resolve_dataset_info(csv_path, project_root=ROOT)

    print(f"Dataset mode: {info.dataset_mode}")
    print(f"Dataset source: {info.dataset_source}")
    print(f"Real observational data: {info.is_real_observational_data}")

    if info.warning_banner:
        print(f"WARNING: {info.warning_banner}")
    if info.dataset_mode == "synthetic_validation":
        print("No data/processed/rotation.csv — using in-memory synthetic data only.")

    result = run_rotation_pipeline(processed_csv=csv_path, outputs_root=ROOT / "outputs")

    print(f"Galaxies processed: {result.n_galaxies}")
    print(f"Summary table: {result.summary_csv_path}")
    print(f"Report: {result.report_path}")
    print(f"Figures: {ROOT / 'outputs' / 'figures'}")


if __name__ == "__main__":
    main()

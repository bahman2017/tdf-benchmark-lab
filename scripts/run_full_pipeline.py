#!/usr/bin/env python3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tdf_obs.pipelines.run_full_pipeline import run_full_pipeline

if __name__ == "__main__":
    summary = run_full_pipeline()
    print("Full pipeline summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")

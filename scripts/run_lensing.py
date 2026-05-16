#!/usr/bin/env python3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tdf_obs.pipelines.run_lensing_pipeline import run_lensing_pipeline

if __name__ == "__main__":
    print(run_lensing_pipeline())

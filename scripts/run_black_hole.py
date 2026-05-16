#!/usr/bin/env python3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tdf_obs.pipelines.run_black_hole_pipeline import run_black_hole_pipeline

if __name__ == "__main__":
    df = run_black_hole_pipeline()
    print(df.to_string(index=False) if not df.empty else "No black-hole config.")

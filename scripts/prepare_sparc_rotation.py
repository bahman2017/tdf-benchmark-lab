#!/usr/bin/env python3
"""
Prepare processed rotation.csv from user-provided SPARC raw data.

Does not download data. Does not create fake processed files if raw data is absent.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tdf_obs.io.sparc_parser import prepare_sparc_rotation, resolve_sparc_raw_input


def main() -> int:
    located = resolve_sparc_raw_input(ROOT)
    if located is None:
        print(
            "No SPARC raw data found. Place files in data/raw/ before running this script.",
        )
        print()
        print("Expected one of:")
        print("  data/raw/sparc_rotation_curves.csv")
        print("  data/raw/sparc/<GalaxyName>.csv")
        return 1

    kind, path = located
    print(f"Found SPARC raw input ({kind}): {path}")

    df = prepare_sparc_rotation(ROOT)
    n_gal = df["galaxy_id"].nunique()
    out_csv = ROOT / "data" / "processed" / "rotation.csv"
    out_meta = ROOT / "data" / "processed" / "rotation_metadata.yaml"

    print(f"Wrote {len(df)} rows for {n_gal} galaxies → {out_csv}")
    print(f"Wrote metadata → {out_meta}")
    print("Next: python scripts/run_rotation.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())

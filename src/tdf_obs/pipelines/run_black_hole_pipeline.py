"""Black-hole phenomenology demo (no observational claim)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from tdf_obs.constants import SOLAR_MASS
from tdf_obs.models.black_hole import (
    hawking_temperature,
    non_return_radius_tdf,
    schwarzschild_radius,
    tdf_temperature,
)


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def run_black_hole_pipeline() -> pd.DataFrame:
    cfg_path = project_root() / "configs" / "black_hole.yaml"
    rows = []
    if cfg_path.exists():
        with cfg_path.open() as f:
            cfg = yaml.safe_load(f) or {}
        for obj in cfg.get("objects", []):
            M = float(obj.get("mass_solar", 10.0)) * SOLAR_MASS
            rs = schwarzschild_radius(M)
            if "rc_m" in obj:
                rc = float(obj["rc_m"])
            elif "rc_fraction_of_rs" in obj:
                rc = float(obj["rc_fraction_of_rs"]) * rs
            else:
                rc = 0.0
            rows.append(
                {
                    "object_id": obj.get("id", "bh"),
                    "M_kg": M,
                    "rc_m": rc,
                    "r_s_m": rs,
                    "r_nr_TDF_m": non_return_radius_tdf(M, rc),
                    "T_H_K": hawking_temperature(M),
                    "T_TDF_K": tdf_temperature(M, rc),
                    "data_mode": "synthetic_validation",
                },
            )
    df = pd.DataFrame(rows)
    out = project_root() / "outputs" / "tables" / "black_hole_summary.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    if not df.empty:
        df.to_csv(out, index=False)
    return df

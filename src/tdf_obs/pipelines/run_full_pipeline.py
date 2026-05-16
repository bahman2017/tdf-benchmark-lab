"""Run all implemented pipeline stages."""

from __future__ import annotations

from tdf_obs.pipelines.run_black_hole_pipeline import run_black_hole_pipeline
from tdf_obs.pipelines.run_rotation_pipeline import run_rotation_pipeline
from tdf_obs.pipelines.run_solar_system_constraints import run_solar_system_constraints


def run_full_pipeline() -> dict:
    rotation = run_rotation_pipeline()
    solar = run_solar_system_constraints()
    bh = run_black_hole_pipeline()
    return {
        "rotation_mode": rotation.mode,
        "rotation_fits": len(rotation.results),
        "solar_constraints": solar,
        "black_hole_rows": len(bh),
        "lensing": "not_implemented",
        "redshift": "not_implemented",
    }

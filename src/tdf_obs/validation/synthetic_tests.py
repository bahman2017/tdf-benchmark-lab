"""Synthetic data generators for pipeline self-tests."""

from __future__ import annotations

from typing import Any

import numpy as np

from tdf_obs.io.schemas import RotationCurveData
from tdf_obs.models.rotation import v_tdf_simple


def generate_synthetic_rotation_curve(
    galaxy_id: str = "synthetic_001",
    *,
    B: float = 800.0,
    r0: float = 4.0,
    n_points: int = 25,
    r_min_kpc: float = 0.5,
    r_max_kpc: float = 30.0,
    noise_fraction: float = 0.03,
    seed: int | None = 42,
) -> tuple[RotationCurveData, dict[str, Any]]:
    """
  Generate v_obs from known (B, r0) plus Gaussian noise.

  Returns RotationCurveData with metadata data_mode=synthetic_validation
  and a truth dict with injected parameters.
  """
    rng = np.random.default_rng(seed)
    r = np.linspace(r_min_kpc, r_max_kpc, n_points)

    # Plausible declining baryon curve [km/s]
    v_baryon = 180.0 * np.sqrt(r / (r + 2.0)) + 20.0

    v_true = v_tdf_simple(r, v_baryon, B, r0)
    sigma = np.maximum(noise_fraction * v_true, 1.0)
    v_obs = v_true + rng.normal(0.0, sigma)

    truth = {
        "B_true": B,
        "r0_true": r0,
        "noise_fraction": noise_fraction,
        "seed": seed,
    }

    data = RotationCurveData(
        galaxy_id=galaxy_id,
        r_kpc=r,
        v_obs=v_obs,
        v_err=sigma,
        v_baryon=v_baryon,
        metadata={
            "dataset_mode": "synthetic_validation",
            "dataset_source": "in_memory_generator",
            "is_real_observational_data": False,
            "truth": truth,
        },
    )
    return data, truth

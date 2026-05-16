"""Data schemas for observational channels."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt


@dataclass
class RotationCurveData:
    """Galaxy rotation curve inputs."""

    galaxy_id: str
    r_kpc: npt.NDArray[np.floating]
    v_obs: npt.NDArray[np.floating]
    v_err: npt.NDArray[np.floating]
    v_baryon: npt.NDArray[np.floating]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        n = len(self.r_kpc)
        for name in ("v_obs", "v_err", "v_baryon"):
            arr = getattr(self, name)
            if len(arr) != n:
                raise ValueError(f"{name} length {len(arr)} != r_kpc length {n}")


@dataclass
class LensingData:
    """Strong/weak lensing observables (placeholder channel)."""

    lens_id: str
    r_kpc: npt.NDArray[np.floating]
    alpha_obs: npt.NDArray[np.floating]
    alpha_err: npt.NDArray[np.floating]
    baryon_potential_or_mass: npt.NDArray[np.floating]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RedshiftData:
    """Redshift / Doppler residual channel."""

    object_id: str
    z_obs: npt.NDArray[np.floating]
    z_kin: npt.NDArray[np.floating]
    z_baryon: npt.NDArray[np.floating]
    z_err: npt.NDArray[np.floating]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SolarSystemConstraint:
    """Solar-system GR-safe bound on tau potential fraction."""

    name: str
    radius_m: float
    phi_b: float
    max_epsilon_tau: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BlackHoleData:
    """Black-hole phenomenology inputs."""

    object_id: str
    mass_kg: float
    shadow_radius: float | None = None
    ringdown_freq: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

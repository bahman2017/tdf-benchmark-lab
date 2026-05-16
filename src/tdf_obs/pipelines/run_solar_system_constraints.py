"""Solar-system constraint runner (basic demo)."""

from __future__ import annotations

from pathlib import Path

import yaml

from tdf_obs.models.solar_system import epsilon_tau, passes_gr_safe_limit


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def run_solar_system_constraints() -> list[dict]:
    cfg_path = project_root() / "configs" / "solar_system.yaml"
    constraints = []
    if cfg_path.exists():
        with cfg_path.open() as f:
            cfg = yaml.safe_load(f) or {}
        for item in cfg.get("constraints", []):
            eps = epsilon_tau(item["phi_tau"], item["phi_b"])
            passed = passes_gr_safe_limit(eps, item["max_epsilon_tau"])
            constraints.append(
                {
                    "name": item["name"],
                    "epsilon_tau": eps,
                    "max_allowed": item["max_epsilon_tau"],
                    "status": "passed" if passed else "failed",
                },
            )
    return constraints

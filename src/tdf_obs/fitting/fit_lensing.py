"""Lensing fit (not yet implemented)."""

from __future__ import annotations

from tdf_obs.io.schemas import LensingData


def fit_single_lens(_data: LensingData) -> dict[str, str]:
    return {
        "status": "not_yet_tested",
        "message": "Lensing fit not implemented. See models/lensing.py and docs/TEST_PLAN.md Phase 4.",
    }

"""Redshift fit (not yet implemented)."""

from __future__ import annotations

from tdf_obs.io.schemas import RedshiftData


def fit_redshift_channel(_data: RedshiftData) -> dict[str, str]:
    return {
        "status": "not_yet_tested",
        "message": "Redshift calibration not implemented. See docs/TEST_PLAN.md Phase 5.",
    }

"""Constraint status labels for reports."""

from __future__ import annotations

from enum import Enum


class ConstraintStatus(str, Enum):
    SYNTHETIC_PASSED = "synthetic_validation_passed"
    SYNTHETIC_FAILED = "synthetic_validation_failed"
    REAL_PASSED = "real_data_calibration_passed"
    REAL_FAILED = "real_data_calibration_failed"
    NOT_TESTED = "not_yet_tested"
    NOT_IMPLEMENTED = "not_implemented"

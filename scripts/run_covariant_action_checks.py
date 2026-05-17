#!/usr/bin/env python3
"""
Phase 5D — Covariant action consistency checks on NFW surrogate TDF fits.

NOT observational validation. Requires nfw_surrogate_fit_summary.csv.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tdf_obs.validation.covariant_action_checks import (
    BANNER_COVARIANT,
    NfwSummaryMissingError,
    run_covariant_action_checks_pipeline,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run covariant action consistency checks on NFW surrogate TDF fits (Phase 5D).",
    )
    p.add_argument(
        "--no-run-nfw",
        action="store_true",
        help="Do not auto-run NFW surrogate if summary CSV is missing.",
    )
    p.add_argument(
        "--case",
        action="append",
        dest="cases",
        metavar="CASE_NAME",
        help="Check only this NFW surrogate case (repeatable).",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    print(BANNER_COVARIANT)

    try:
        _df, results = run_covariant_action_checks_pipeline(
            outputs_root=ROOT / "outputs",
            case_names=args.cases,
            run_nfw_if_missing=not args.no_run_nfw,
        )
    except NfwSummaryMissingError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except (KeyError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    n_pass = sum(1 for r in results if r.overall_action_consistency_pass)
    print(f"Cases checked: {len(results)} | overall pass: {n_pass}/{len(results)}")
    print(f"Summary: {ROOT / 'outputs' / 'tables' / 'covariant_action_checks_summary.csv'}")
    print(f"Report: {ROOT / 'outputs' / 'reports' / 'covariant_action_checks_report.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

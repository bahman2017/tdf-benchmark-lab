"""Phase 4D — Redshift/Doppler sanity benchmark (not observational validation)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

BENCHMARK_MODE = "redshift_sanity_lcdm_benchmark"
BANNER_REDSHIFT_SANITY = "REDSHIFT SANITY ΛCDM/GR BENCHMARK — NOT REAL OBSERVATIONAL DATA"

BORDERLINE_FRACTION = 0.8

REQUIRED_SUMMARY_COLUMNS: tuple[str, ...] = (
    "case_name",
    "regime",
    "z_tau",
    "max_allowed_abs_z_tau",
    "ratio_to_limit",
    "status",
    "expected_status",
    "matches_expected",
    "warning",
)


@dataclass(frozen=True)
class RedshiftSanityCase:
    case_name: str
    regime: str
    z_tau: float
    max_allowed_abs_z_tau: float
    expected_status: str

    def required_fields(self) -> dict[str, Any]:
        return {
            "case_name": self.case_name,
            "regime": self.regime,
            "z_tau": self.z_tau,
            "max_allowed_abs_z_tau": self.max_allowed_abs_z_tau,
            "expected_status": self.expected_status,
        }


BENCHMARK_CASE_REGISTRY: dict[str, RedshiftSanityCase] = {
    "negligible_tau_shift": RedshiftSanityCase(
        case_name="negligible_tau_shift",
        regime="negligible tau-induced shift",
        z_tau=1e-12,
        max_allowed_abs_z_tau=1e-8,
        expected_status="pass",
    ),
    "small_allowed_shift": RedshiftSanityCase(
        case_name="small_allowed_shift",
        regime="small allowed tau shift",
        z_tau=1e-9,
        max_allowed_abs_z_tau=1e-8,
        expected_status="pass",
    ),
    "borderline_shift": RedshiftSanityCase(
        case_name="borderline_shift",
        regime="borderline tau shift",
        z_tau=8e-9,
        max_allowed_abs_z_tau=1e-8,
        expected_status="borderline",
    ),
    "too_large_shift": RedshiftSanityCase(
        case_name="too_large_shift",
        regime="overlarge tau shift (expected fail)",
        z_tau=5e-7,
        max_allowed_abs_z_tau=1e-8,
        expected_status="fail",
    ),
    "galaxy_rotation_doppler_proxy": RedshiftSanityCase(
        case_name="galaxy_rotation_doppler_proxy",
        regime="galaxy rotation Doppler proxy",
        z_tau=1e-7,
        max_allowed_abs_z_tau=1e-6,
        expected_status="pass",
    ),
    "cluster_gravitational_redshift_proxy": RedshiftSanityCase(
        case_name="cluster_gravitational_redshift_proxy",
        regime="cluster gravitational redshift proxy",
        z_tau=5e-6,
        max_allowed_abs_z_tau=1e-5,
        expected_status="pass",
    ),
    "precision_clock_proxy": RedshiftSanityCase(
        case_name="precision_clock_proxy",
        regime="precision clock / weak-field proxy",
        z_tau=1e-13,
        max_allowed_abs_z_tau=1e-12,
        expected_status="pass",
    ),
}


@dataclass
class RedshiftSanityResult:
    case_name: str
    regime: str
    z_tau: float
    max_allowed_abs_z_tau: float
    ratio_to_limit: float
    status: str
    expected_status: str
    matches_expected: bool
    warning: str


def list_benchmark_cases() -> list[str]:
    return list(BENCHMARK_CASE_REGISTRY.keys())


def get_benchmark_case(name: str) -> RedshiftSanityCase:
    if name not in BENCHMARK_CASE_REGISTRY:
        raise KeyError(f"Unknown redshift sanity case {name!r}; available: {list_benchmark_cases()}")
    return BENCHMARK_CASE_REGISTRY[name]


def classify_redshift_case(z_tau: float, max_allowed_abs_z_tau: float) -> str:
    """
    Classify configured z_tau against benchmark tolerance.

    - pass: |z_tau| < 0.8 * max_allowed
    - borderline: 0.8 * max_allowed <= |z_tau| <= max_allowed
    - fail: |z_tau| > max_allowed
    """
    if max_allowed_abs_z_tau <= 0:
        raise ValueError("max_allowed_abs_z_tau must be positive")
    abs_z = abs(float(z_tau))
    limit = float(max_allowed_abs_z_tau)
    border = BORDERLINE_FRACTION * limit
    if abs_z < border:
        return "pass"
    if abs_z <= limit:
        return "borderline"
    return "fail"


def run_single_redshift_case(case: RedshiftSanityCase) -> RedshiftSanityResult:
    status = classify_redshift_case(case.z_tau, case.max_allowed_abs_z_tau)
    ratio = abs(case.z_tau) / case.max_allowed_abs_z_tau
    warning = ""
    if status == "borderline":
        warning = "borderline: within tolerance but near configured limit"
    if case.expected_status == "fail":
        warning = warning or "expected fail: tau redshift exceeds benchmark cap"

    return RedshiftSanityResult(
        case_name=case.case_name,
        regime=case.regime,
        z_tau=case.z_tau,
        max_allowed_abs_z_tau=case.max_allowed_abs_z_tau,
        ratio_to_limit=ratio,
        status=status,
        expected_status=case.expected_status,
        matches_expected=status == case.expected_status,
        warning=warning,
    )


def run_redshift_sanity_benchmark(
    cases: Sequence[RedshiftSanityCase] | None = None,
) -> list[RedshiftSanityResult]:
    """Run all configured redshift sanity cases (z_tau supplied directly, not fitted)."""
    case_list = list(cases) if cases is not None else list(BENCHMARK_CASE_REGISTRY.values())
    return [run_single_redshift_case(c) for c in case_list]


def summarize_redshift_benchmark(
    results: Sequence[RedshiftSanityResult],
) -> dict[str, Any]:
    """Aggregate pass/borderline/fail counts and expected-failure tracking."""
    status_counts: dict[str, int] = {"pass": 0, "borderline": 0, "fail": 0}
    for row in results:
        status_counts[row.status] = status_counts.get(row.status, 0) + 1

    expected_failures = [r.case_name for r in results if r.expected_status == "fail"]
    failed_expected = [r.case_name for r in results if r.expected_status == "fail" and r.status == "fail"]
    unexpected_fail = [r.case_name for r in results if r.expected_status != "fail" and r.status == "fail"]

    return {
        "n_cases": len(results),
        "n_pass": status_counts.get("pass", 0),
        "n_borderline": status_counts.get("borderline", 0),
        "n_fail": status_counts.get("fail", 0),
        "n_matches_expected": sum(1 for r in results if r.matches_expected),
        "expected_failure_cases": expected_failures,
        "n_expected_failures": len(expected_failures),
        "n_failed_as_expected": len(failed_expected),
        "unexpected_fail_cases": unexpected_fail,
    }


def _result_to_row(res: RedshiftSanityResult) -> dict[str, Any]:
    return {
        "case_name": res.case_name,
        "regime": res.regime,
        "z_tau": res.z_tau,
        "max_allowed_abs_z_tau": res.max_allowed_abs_z_tau,
        "ratio_to_limit": res.ratio_to_limit,
        "status": res.status,
        "expected_status": res.expected_status,
        "matches_expected": res.matches_expected,
        "warning": res.warning,
        "benchmark_mode": BENCHMARK_MODE,
        "is_real_observational_data": False,
    }


def _build_report(results: list[RedshiftSanityResult], summary: dict[str, Any]) -> str:
    lines = [
        "# Redshift/Doppler sanity benchmark report (Phase 4D)",
        "",
        f"## ⚠️ {BANNER_REDSHIFT_SANITY}",
        "",
        "## Purpose",
        "",
        "This checks whether tau-induced redshift residuals can remain **bounded** "
        "before real spectral or Doppler data are used.",
        "",
        "> **Not observational validation.** Configured z_tau values only — no spectra fitted.",
        "",
        "## Formula (TDF v0.8.1)",
        "",
        "```text",
        "z_tau = K_tau * Delta_tau_bar_l / c^2",
        "```",
        "",
        "In this benchmark, **z_tau is configured directly** for classification against "
        "`max_allowed_abs_z_tau` (not derived from a full TDF metric fit).",
        "",
        "## Classification rules",
        "",
        f"- **pass:** |z_tau| < {BORDERLINE_FRACTION} × max_allowed",
        f"- **borderline:** {BORDERLINE_FRACTION} × max_allowed ≤ |z_tau| ≤ max_allowed",
        "- **fail:** |z_tau| > max_allowed",
        "",
        "## Summary",
        "",
        f"- **Total cases:** {summary['n_cases']}",
        f"- **Pass:** {summary['n_pass']}",
        f"- **Borderline:** {summary['n_borderline']}",
        f"- **Fail:** {summary['n_fail']}",
        f"- **Matches expected status:** {summary['n_matches_expected']} / {summary['n_cases']}",
        f"- **Expected failures:** {', '.join(summary['expected_failure_cases']) or '_(none)_'}",
        "",
        "## Results table",
        "",
        "| case_name | regime | z_tau | max_allowed | ratio_to_limit | status | expected |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for res in results:
        lines.append(
            f"| {res.case_name} | {res.regime} | {res.z_tau:.3e} | {res.max_allowed_abs_z_tau:.3e} | "
            f"{res.ratio_to_limit:.4f} | {res.status} | {res.expected_status} |",
        )

    if summary["unexpected_fail_cases"]:
        lines.append("")
        lines.append(f"**Unexpected failures:** {', '.join(summary['unexpected_fail_cases'])}")

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- **Passing** means only that the configured tau residual is below the chosen benchmark tolerance.",
            "- It does **not** validate TDF against real spectra or cosmological data.",
            "- A **failure** is useful: the framework can reject overlarge tau redshift residuals.",
            "- **Borderline** cases flag configurations near the cap (see warnings in CSV).",
            "",
            "## Failure modes",
            "",
            "- z_tau is **configured**, not derived from a full TDF metric.",
            "- Real spectra, peculiar velocities, gravitational redshift, and Doppler decomposition are **not** fitted.",
            "- Rotation curves inferred from Doppler shifts require care to avoid **double-counting** tau and kinematic terms.",
            "",
            "## Disclaimer",
            "",
            "- **NOT REAL OBSERVATIONAL DATA**",
            "- Does **not** replace spectral fitting or multi-component redshift analysis.",
            "",
        ],
    )
    return "\n".join(lines)


def run_redshift_sanity_benchmark_pipeline(
    outputs_root: Path | None = None,
    *,
    case_names: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run Phase 4D benchmark and write CSV + markdown report."""
    root = Path(__file__).resolve().parents[3]
    outputs = outputs_root or (root / "outputs")
    tables_dir = outputs / "tables"
    reports_dir = outputs / "reports"
    for d in (tables_dir, reports_dir):
        d.mkdir(parents=True, exist_ok=True)

    if case_names is not None:
        cases = [get_benchmark_case(n) for n in case_names]
    else:
        cases = list(BENCHMARK_CASE_REGISTRY.values())

    results = run_redshift_sanity_benchmark(cases)
    summary = summarize_redshift_benchmark(results)
    df = pd.DataFrame([_result_to_row(r) for r in results])
    df.to_csv(tables_dir / "redshift_sanity_benchmark_summary.csv", index=False)
    (reports_dir / "redshift_sanity_benchmark_report.md").write_text(
        _build_report(results, summary),
        encoding="utf-8",
    )
    return df, summary

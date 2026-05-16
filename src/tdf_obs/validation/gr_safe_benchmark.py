"""Phase 4B — Expanded GR-safe local benchmark (not observational validation)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from tdf_obs.models.solar_system import epsilon_tau, passes_gr_safe_limit

BENCHMARK_MODE = "gr_safe_lcdm_benchmark"
BANNER_GR_SAFE = "GR-SAFE ΛCDM/GR BENCHMARK — NOT REAL OBSERVATIONAL DATA"

REQUIRED_SUMMARY_COLUMNS: tuple[str, ...] = (
    "case_name",
    "regime",
    "phi_b_scale",
    "assumed_phi_tau_scale",
    "assumed_epsilon_tau_input",
    "epsilon_tau",
    "max_allowed_epsilon",
    "status",
    "pass",
)

# Nominal |Phi_b| scales are order-of-magnitude placeholders (m^2/s^2), not fitted from ephemeris.
# Phi_tau assumptions are configurable scaffold inputs, not observational fits.


@dataclass(frozen=True)
class GrSafeBenchmarkCase:
    """One GR-safe local benchmark configuration."""

    case_name: str
    regime: str
    phi_b_scale: float
    max_allowed_epsilon: float
    assumed_phi_tau_scale: float | None = None
    assumed_epsilon_tau: float | None = None

    def resolve_phi_tau(self) -> float:
        if self.assumed_epsilon_tau is not None:
            return self.assumed_epsilon_tau * self.phi_b_scale
        if self.assumed_phi_tau_scale is not None:
            return self.assumed_phi_tau_scale
        raise ValueError(f"Case {self.case_name!r} needs assumed_phi_tau_scale or assumed_epsilon_tau")

    def required_fields(self) -> dict[str, Any]:
        return {
            "case_name": self.case_name,
            "regime": self.regime,
            "phi_b_scale": self.phi_b_scale,
            "max_allowed_epsilon": self.max_allowed_epsilon,
            "assumed_phi_tau_scale": self.assumed_phi_tau_scale,
            "assumed_epsilon_tau": self.assumed_epsilon_tau,
        }


BENCHMARK_CASE_REGISTRY: dict[str, GrSafeBenchmarkCase] = {
    "mercury_perihelion": GrSafeBenchmarkCase(
        case_name="mercury_perihelion",
        regime="Mercury perihelion regime",
        phi_b_scale=-8.5e7,
        assumed_phi_tau_scale=4.0e-11,
        max_allowed_epsilon=1e-8,
    ),
    "earth_orbit": GrSafeBenchmarkCase(
        case_name="earth_orbit",
        regime="Earth orbit regime",
        phi_b_scale=-6.2637e7,
        assumed_phi_tau_scale=1.0e-10,
        max_allowed_epsilon=1e-8,
    ),
    "gps_weak_field_clock": GrSafeBenchmarkCase(
        case_name="gps_weak_field_clock",
        regime="GPS weak-field clock regime",
        phi_b_scale=-5.8e7,
        assumed_phi_tau_scale=8.0e-11,
        max_allowed_epsilon=1e-8,
    ),
    "light_bending_sun": GrSafeBenchmarkCase(
        case_name="light_bending_sun",
        regime="light bending near Sun",
        phi_b_scale=-1.0e8,
        assumed_phi_tau_scale=1.0e-12,
        max_allowed_epsilon=1e-9,
    ),
    "shapiro_delay": GrSafeBenchmarkCase(
        case_name="shapiro_delay",
        regime="Shapiro delay regime",
        phi_b_scale=-9.5e7,
        assumed_phi_tau_scale=2.0e-12,
        max_allowed_epsilon=1e-9,
    ),
    "lunar_laser_ranging": GrSafeBenchmarkCase(
        case_name="lunar_laser_ranging",
        regime="lunar laser ranging scale",
        phi_b_scale=-1.2e7,
        assumed_phi_tau_scale=5.0e-13,
        max_allowed_epsilon=1e-8,
    ),
    "binary_pulsar_weak_field": GrSafeBenchmarkCase(
        case_name="binary_pulsar_weak_field",
        regime="binary pulsar weak-field timing proxy",
        phi_b_scale=-5.0e6,
        assumed_epsilon_tau=1.0e-12,
        max_allowed_epsilon=1e-7,
    ),
}


@dataclass
class GrSafeBenchmarkResult:
    case_name: str
    regime: str
    phi_b_scale: float
    assumed_phi_tau_scale: float
    assumed_epsilon_tau_input: float | None
    epsilon_tau: float
    max_allowed_epsilon: float
    status: str
    pass_: bool

    @property
    def pass_fail(self) -> str:
        return self.status


def list_benchmark_cases() -> list[str]:
    return list(BENCHMARK_CASE_REGISTRY.keys())


def get_benchmark_case(name: str) -> GrSafeBenchmarkCase:
    if name not in BENCHMARK_CASE_REGISTRY:
        raise KeyError(f"Unknown GR-safe case {name!r}; available: {list_benchmark_cases()}")
    return BENCHMARK_CASE_REGISTRY[name]


def run_single_gr_safe_case(case: GrSafeBenchmarkCase) -> GrSafeBenchmarkResult:
    phi_tau = case.resolve_phi_tau()
    eps = epsilon_tau(phi_tau, case.phi_b_scale)
    passed = passes_gr_safe_limit(eps, case.max_allowed_epsilon)
    return GrSafeBenchmarkResult(
        case_name=case.case_name,
        regime=case.regime,
        phi_b_scale=case.phi_b_scale,
        assumed_phi_tau_scale=phi_tau,
        assumed_epsilon_tau_input=case.assumed_epsilon_tau,
        epsilon_tau=eps,
        max_allowed_epsilon=case.max_allowed_epsilon,
        status="pass" if passed else "fail",
        pass_=passed,
    )


def _result_to_row(res: GrSafeBenchmarkResult) -> dict[str, Any]:
    return {
        "case_name": res.case_name,
        "regime": res.regime,
        "phi_b_scale": res.phi_b_scale,
        "assumed_phi_tau_scale": res.assumed_phi_tau_scale,
        "assumed_epsilon_tau_input": res.assumed_epsilon_tau_input,
        "epsilon_tau": res.epsilon_tau,
        "max_allowed_epsilon": res.max_allowed_epsilon,
        "status": res.status,
        "pass": res.pass_,
        "benchmark_mode": BENCHMARK_MODE,
        "is_real_observational_data": False,
    }


def _build_report(results: list[GrSafeBenchmarkResult]) -> str:
    n_pass = sum(1 for r in results if r.pass_)
    n_fail = len(results) - n_pass

    lines = [
        "# GR-safe local benchmark report (Phase 4B)",
        "",
        f"## ⚠️ {BANNER_GR_SAFE}",
        "",
        "> **This is not observational validation.** Configurable assumed Φ_τ / ε_τ inputs only. "
        "> **Real observational constraints are not yet fitted** (no ephemeris, Cassini, or LLR fits).",
        "",
        "## Scientific interpretation",
        "",
        "**Passing** means |ε_τ| = |Φ_τ/Φ_b| stays below the configured cap for that GR-success regime — "
        "a **compatibility scaffold** showing TDF corrections can remain suppressed where GR is already strong.",
        "",
        "**Passing does not validate TDF** against solar-system observations or ΛCDM.",
        "",
        "## Summary",
        "",
        f"- **Cases:** {len(results)}",
        f"- **Pass:** {n_pass}",
        f"- **Fail:** {n_fail}",
        "",
        "## Pass/fail table",
        "",
        "| Case | Regime | ε_τ | max allowed | Status |",
        "| --- | --- | --- | --- | --- |",
    ]
    for res in results:
        lines.append(
            f"| {res.case_name} | {res.regime} | {res.epsilon_tau:.3e} | "
            f"{res.max_allowed_epsilon:.3e} | {res.status} |",
        )

    lines.extend(
        [
            "",
            "## Per-case detail",
            "",
        ],
    )
    for res in results:
        lines.append(f"### {res.case_name}")
        lines.append("")
        lines.append(f"- **Regime:** {res.regime}")
        lines.append(f"- **Φ_b scale (nominal):** {res.phi_b_scale:.6g}")
        lines.append(f"- **Assumed Φ_τ:** {res.assumed_phi_tau_scale:.6g}")
        if res.assumed_epsilon_tau_input is not None:
            lines.append(f"- **Assumed ε_τ input:** {res.assumed_epsilon_tau_input:.3e}")
        lines.append(f"- **Computed ε_τ:** {res.epsilon_tau:.6e}")
        lines.append(f"- **max_allowed_ε:** {res.max_allowed_epsilon:.3e}")
        lines.append(f"- **Result:** {res.status}")
        lines.append("")

    lines.extend(
        [
            "## Failure modes",
            "",
            "- Assumed Φ_τ is a **placeholder**, not derived from a dynamical TDF metric.",
            "- Φ_b scales are order-of-magnitude only; real ephemeris coupling is future work.",
            "- A fail here would indicate the scaffold assumption violates the configured GR-safe cap.",
            "",
            "## Disclaimer",
            "",
            "- **NOT REAL OBSERVATIONAL DATA**",
            "- Does **not** replace PPN bounds, Cassini, GPS clock fits, or LLR analysis.",
            "- Phase 4C/4D (BH exterior, redshift) are separate benchmarks.",
            "",
        ],
    )
    return "\n".join(lines)


def run_gr_safe_benchmark_pipeline(
    outputs_root: Path | None = None,
    *,
    case_names: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Run Phase 4B GR-safe benchmarks and write CSV + markdown report."""
    root = Path(__file__).resolve().parents[3]
    outputs = outputs_root or (root / "outputs")
    tables_dir = outputs / "tables"
    reports_dir = outputs / "reports"
    for d in (tables_dir, reports_dir):
        d.mkdir(parents=True, exist_ok=True)

    names = list(case_names) if case_names is not None else list_benchmark_cases()
    results = [run_single_gr_safe_case(get_benchmark_case(n)) for n in names]
    rows = [_result_to_row(r) for r in results]
    df = pd.DataFrame(rows)
    df.to_csv(tables_dir / "gr_safe_benchmark_summary.csv", index=False)
    (reports_dir / "gr_safe_benchmark_report.md").write_text(_build_report(results), encoding="utf-8")
    return df

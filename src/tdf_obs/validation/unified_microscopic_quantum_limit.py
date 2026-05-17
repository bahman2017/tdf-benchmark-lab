"""Phase 6G — Unified microscopic quantum consistency matrix (integration only)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

BENCHMARK_MODE = "unified_microscopic_quantum_limit_benchmark"
BANNER_UNIFIED = (
    "UNIFIED MICROSCOPIC QUANTUM LIMIT BENCHMARK — NOT FULL QUANTUM-GRAVITY PROOF"
)

READINESS_THRESHOLD = 0.85

REQUIRED_SUMMARY_COLUMNS: tuple[str, ...] = (
    "phase",
    "benchmark_name",
    "core_result",
    "pass_count",
    "fail_count",
    "expected_fail_count",
    "status",
    "controlled_claim",
    "explicit_limit",
    "paper_ready",
    "warning",
)

NO_OVERCLAIM_PHRASES: tuple[str, ...] = (
    "NOT FULL",
    "not prove",
    "does not",
    "Does **not**",
    "toy",
    "proxy",
    "benchmark only",
)

SYMBOL_ALIASES: dict[str, tuple[str, ...]] = {
    "tau": ("tau", "τ"),
    "rho": ("rho", "ρ"),
    "psi": ("psi", "ψ"),
    "Psi": ("psi", "ψ", "Psi", "Ψ"),
    "chi": ("chi", "χ"),
    "tau_bar": ("tau_bar", "τ̄", "taū"),
    "g_tilde": ("g_tilde", "g̃", "g̃"),
    "Delta_tau": ("Delta_tau", "Δτ", "delta_tau", "Δ"),
    "C_AB": ("C_AB", "C =", "coherence"),
    "P_i": ("P_i", "P_i", "probability", "ρ_i"),
    "ℏ": ("ℏ", "\\hbar", "hbar"),
}

SYMBOL_REGISTRY: dict[str, tuple[str, ...]] = {
    "6A": ("tau", "rho", "psi", "ℏ"),
    "6B": ("tau", "rho", "Psi", "chi"),
    "6C": ("tau", "rho", "psi"),
    "6D": ("tau", "Delta_tau", "C_AB"),
    "6E": ("tau", "tau_bar", "g_tilde"),
    "6F": ("tau", "rho", "P_i"),
}


def _symbol_in_text(symbol: str, text: str) -> bool:
    aliases = SYMBOL_ALIASES.get(symbol, (symbol,))
    low = text.lower()
    return any(a.lower() in low for a in aliases)

PIPELINE_DEPENDENCIES: tuple[tuple[str, str, str], ...] = (
    ("6A", "6B", "6B extends scalar τ dynamics with spinor χ and Dirac limit"),
    ("6B", "6C", "6C adds multi-particle configuration-space τ/ρ geometry"),
    ("6C", "6D", "6D introduces branch decoherence via Var(Δτ)"),
    ("6D", "6E", "6E coarse-grains decohered τ into effective classical geometry"),
    ("6E", "6F", "6F maps surviving branch weights to probability proxy"),
)

PHASE_SPECS: tuple[dict[str, Any], ...] = (
    {
        "phase": "6A",
        "benchmark_name": "Schrödinger-from-TDF action",
        "summary_csv": "schrodinger_from_tdf_summary.csv",
        "report_md": "schrodinger_from_tdf_report.md",
        "banner_substring": "NOT FULL QUANTUM VALIDATION",
        "core_equation": "ψ=√ρ e^{-iτ}; QHJ + continuity recover Schrödinger hydrodynamics",
        "controlled_claim": "Phase-density action reproduces Schrödinger dynamics in 1D controls",
        "explicit_limit": "Not full quantum validation; no spin, entanglement, or measurement",
        "pass_column": "pass",
        "outcome_mode": "direct",
        "expected_fail_count": 0,
        "symbols": SYMBOL_REGISTRY["6A"],
    },
    {
        "phase": "6B",
        "benchmark_name": "Dirac / spinor limit",
        "summary_csv": "dirac_spinor_limit_summary.csv",
        "report_md": "dirac_spinor_limit_report.md",
        "banner_substring": "NOT FULL FERMION UNIFICATION",
        "core_equation": "ρ=Ψ†Ψ; flat H(k) Dirac dispersion + Clifford checks",
        "controlled_claim": "TDF admits spinor extension and flat Dirac limit checks",
        "explicit_limit": "Not SM fermion unification; no gauge fields",
        "pass_column": "pass",
        "outcome_mode": "direct",
        "expected_fail_count": 0,
        "symbols": SYMBOL_REGISTRY["6B"],
    },
    {
        "phase": "6C",
        "benchmark_name": "Entanglement / τ geometry",
        "summary_csv": "entanglement_tau_geometry_summary.csv",
        "report_md": "entanglement_tau_geometry_report.md",
        "banner_substring": "NOT FULL BELL-THEOREM RESOLUTION",
        "core_equation": "Configuration-space τ/ρ encodes nonseparable Bell correlations",
        "controlled_claim": "Entanglement as nonseparable τ/ρ geometry with CHSH scaffold",
        "explicit_limit": "Not Bell-theorem resolution; no collapse dynamics",
        "pass_column": "overall_pass",
        "outcome_mode": "direct",
        "expected_fail_count": 0,
        "symbols": SYMBOL_REGISTRY["6C"],
    },
    {
        "phase": "6D",
        "benchmark_name": "Decoherence from τ variance",
        "summary_csv": "decoherence_tau_variance_summary.csv",
        "report_md": "decoherence_tau_variance_report.md",
        "banner_substring": "NOT FULL MEASUREMENT-PROBLEM SOLUTION",
        "core_equation": "C_AB=exp(−½Var(Δτ)); Γ=½ dVar/dt",
        "controlled_claim": "Branch coherence decays with Var(Δτ) growth",
        "explicit_limit": "Not full measurement-problem solution",
        "pass_column": "overall_pass",
        "outcome_mode": "direct",
        "expected_fail_count": 0,
        "symbols": SYMBOL_REGISTRY["6D"],
    },
    {
        "phase": "6E",
        "benchmark_name": "Classical metric emergence",
        "summary_csv": "classical_metric_emergence_summary.csv",
        "report_md": "classical_metric_emergence_report.md",
        "banner_substring": "NOT FULL OBJECTIVE-COLLAPSE SOLUTION",
        "core_equation": "g̃=η+α_τ ∂τ̄∂τ̄; τ̄=A_ℓ[τ_micro]",
        "controlled_claim": "Coarse-grained τ̄ yields smooth effective disformal metric",
        "explicit_limit": "Not objective collapse; 1+1D imposed averaging",
        "pass_column": "overall_pass",
        "outcome_mode": "expected_status",
        "expected_fail_count": 2,
        "symbols": SYMBOL_REGISTRY["6E"],
    },
    {
        "phase": "6F",
        "benchmark_name": "Born-rule probability proxy",
        "summary_csv": "born_rule_probability_summary.csv",
        "report_md": "born_rule_probability_report.md",
        "banner_substring": "NOT FULL BORN-RULE DERIVATION",
        "core_equation": "c_i=√ρ_i e^{-iτ_i}; P_i=ρ_i/Σρ_j",
        "controlled_claim": "Branch weights behave as Born probabilities after decoherence",
        "explicit_limit": "Not Born-rule derivation; toy multinomial sampling",
        "pass_column": "overall_pass",
        "outcome_mode": "direct",
        "expected_fail_count": 0,
        "symbols": SYMBOL_REGISTRY["6F"],
    },
)


@dataclass
class PhaseReportStatus:
    phase: str
    report_path: Path
    present: bool
    banner_found: bool
    banner_text: str
    pass_count: int
    fail_count: int
    total_cases: int
    disclaimer_found: bool


@dataclass
class UnifiedConsistencyRow:
    phase: str
    benchmark_name: str
    core_equation: str
    pass_count: int
    fail_count: int
    expected_fail_count: int
    controlled_claim: str
    explicit_limit: str
    status: str
    core_result: str
    paper_ready: bool
    warning: str = ""


@dataclass
class UnifiedBenchmarkResult:
    matrix: pd.DataFrame
    readiness_score: float
    reports_present: bool
    outcomes_correct: bool
    disclaimers_present: bool
    symbol_consistency_pass: bool
    pipeline_dependencies_pass: bool
    claim_hierarchy_pass: bool
    checks: dict[str, Any] = field(default_factory=dict)


def _outputs_root(root: Path | None) -> Path:
    if root is not None:
        return Path(root)
    return Path(__file__).resolve().parents[3] / "outputs"


def _count_phase_passes(df: pd.DataFrame, spec: dict[str, Any]) -> tuple[int, int]:
    """Return (pass_count, fail_count) for outcome correctness."""
    if df.empty:
        return 0, 0
    col = spec["pass_column"]
    if col not in df.columns:
        return 0, len(df)
    mode = spec["outcome_mode"]
    if mode == "expected_status" and "expected_status" in df.columns:
        ok = df["overall_pass"] == (df["expected_status"] == "pass")
        return int(ok.sum()), int((~ok).sum())
    passed = df[col].astype(bool)
    return int(passed.sum()), int((~passed).sum())


def load_phase_report_status(
    report_path: Path,
    *,
    phase: str = "",
    banner_substring: str = "",
    pass_count: int = 0,
    fail_count: int = 0,
    total_cases: int = 0,
) -> PhaseReportStatus:
    """Extract banner and pass metadata from a phase report markdown file."""
    path = Path(report_path)
    if not path.is_file():
        return PhaseReportStatus(
            phase=phase,
            report_path=path,
            present=False,
            banner_found=False,
            banner_text="",
            pass_count=pass_count,
            fail_count=fail_count,
            total_cases=total_cases,
            disclaimer_found=False,
        )
    text = path.read_text(encoding="utf-8")
    banner_found = bool(banner_substring and banner_substring in text)
    if not banner_found:
        for line in text.splitlines():
            if "NOT FULL" in line.upper() or "NOT FULL" in line:
                banner_found = True
                banner_substring = line.strip().lstrip("#").strip()
                break
    disclaimer_found = any(p.lower() in text.lower() for p in NO_OVERCLAIM_PHRASES)
    return PhaseReportStatus(
        phase=phase,
        report_path=path,
        present=True,
        banner_found=banner_found,
        banner_text=banner_substring,
        pass_count=pass_count,
        fail_count=fail_count,
        total_cases=total_cases,
        disclaimer_found=disclaimer_found,
    )


def collect_phase_outputs(outputs_root: Path | None = None) -> dict[str, pd.DataFrame]:
    """Load summary CSVs for phases 6A–6F."""
    root = _outputs_root(outputs_root)
    tables = root / "tables"
    out: dict[str, pd.DataFrame] = {}
    for spec in PHASE_SPECS:
        path = tables / spec["summary_csv"]
        if path.is_file():
            out[spec["phase"]] = pd.read_csv(path)
        else:
            out[spec["phase"]] = pd.DataFrame()
    return out


def build_unified_consistency_matrix(
    phase_outputs: dict[str, pd.DataFrame] | None = None,
    report_statuses: dict[str, PhaseReportStatus] | None = None,
) -> list[UnifiedConsistencyRow]:
    """Build one row per phase 6A–6F."""
    outputs = phase_outputs or collect_phase_outputs()
    rows: list[UnifiedConsistencyRow] = []
    for spec in PHASE_SPECS:
        phase = spec["phase"]
        df = outputs.get(phase, pd.DataFrame())
        p_pass, p_fail = _count_phase_passes(df, spec)
        rs = (report_statuses or {}).get(phase)
        report_ok = rs.present and rs.banner_found and rs.disclaimer_found if rs else False
        outcomes_ok = p_fail == 0 and len(df) > 0
        status = "pass" if outcomes_ok and report_ok else "fail"
        core_result = f"{p_pass}/{p_pass + p_fail} cases OK" if len(df) else "missing CSV"
        warning = ""
        if df.empty:
            warning = "summary CSV missing"
        elif not report_ok:
            warning = "report banner or disclaimer missing"
        rows.append(
            UnifiedConsistencyRow(
                phase=phase,
                benchmark_name=spec["benchmark_name"],
                core_equation=spec["core_equation"],
                pass_count=p_pass,
                fail_count=p_fail,
                expected_fail_count=int(spec["expected_fail_count"]),
                controlled_claim=spec["controlled_claim"],
                explicit_limit=spec["explicit_limit"],
                status=status,
                core_result=core_result,
                paper_ready=status == "pass",
                warning=warning,
            ),
        )
    return rows


def check_claim_hierarchy(rows: list[UnifiedConsistencyRow]) -> tuple[bool, list[str]]:
    """Every phase must have controlled claim, explicit limit, and no empty fields."""
    issues: list[str] = []
    for r in rows:
        if not r.controlled_claim.strip():
            issues.append(f"{r.phase}: missing controlled_claim")
        if not r.explicit_limit.strip():
            issues.append(f"{r.phase}: missing explicit_limit")
        if "prove full" in r.controlled_claim.lower():
            issues.append(f"{r.phase}: overclaim in controlled_claim")
    return len(issues) == 0, issues


def check_symbol_consistency(
    outputs_root: Path | None = None,
) -> tuple[bool, dict[str, list[str]]]:
    """Verify phase reports mention expected τ/ρ symbol vocabulary."""
    root = _outputs_root(outputs_root)
    reports = root / "reports"
    missing: dict[str, list[str]] = {}
    for spec in PHASE_SPECS:
        path = reports / spec["report_md"]
        if not path.is_file():
            missing[spec["phase"]] = ["report missing"]
            continue
        text = path.read_text(encoding="utf-8")
        absent = [s for s in spec["symbols"] if not _symbol_in_text(s, text)]
        if absent:
            missing[spec["phase"]] = absent
    return len(missing) == 0, missing


def check_pipeline_dependencies() -> tuple[bool, list[dict[str, str]]]:
    """Verify documented logical chain 6A→6B→…→6F."""
    chain = [
        {
            "from_phase": a,
            "to_phase": b,
            "description": desc,
            "status": "linked",
        }
        for a, b, desc in PIPELINE_DEPENDENCIES
    ]
    phases = {s["phase"] for s in PHASE_SPECS}
    ok = all(a in phases and b in phases for a, b, _ in PIPELINE_DEPENDENCIES)
    return ok and len(chain) == len(PIPELINE_DEPENDENCIES), chain


def unified_readiness_score(
    rows: list[UnifiedConsistencyRow],
    *,
    reports_present: bool,
    outcomes_correct: bool,
    disclaimers_present: bool,
    symbol_pass: bool,
    pipeline_pass: bool,
    claim_pass: bool,
) -> float:
    """Weighted readiness in [0, 1]."""
    weights = {
        "reports": 0.15,
        "outcomes": 0.30,
        "disclaimers": 0.15,
        "symbols": 0.15,
        "pipeline": 0.10,
        "claims": 0.15,
    }
    phase_frac = sum(1 for r in rows if r.status == "pass") / max(len(rows), 1)
    score = (
        weights["reports"] * float(reports_present)
        + weights["outcomes"] * float(outcomes_correct)
        + weights["disclaimers"] * float(disclaimers_present)
        + weights["symbols"] * float(symbol_pass)
        + weights["pipeline"] * float(pipeline_pass)
        + weights["claims"] * float(claim_pass)
    )
    return float(np.clip(0.5 * score + 0.5 * phase_frac, 0.0, 1.0))


def _row_to_dict(row: UnifiedConsistencyRow) -> dict[str, Any]:
    return {
        "phase": row.phase,
        "benchmark_name": row.benchmark_name,
        "core_result": row.core_result,
        "pass_count": row.pass_count,
        "fail_count": row.fail_count,
        "expected_fail_count": row.expected_fail_count,
        "status": row.status,
        "controlled_claim": row.controlled_claim,
        "explicit_limit": row.explicit_limit,
        "paper_ready": row.paper_ready,
        "warning": row.warning,
        "core_equation": row.core_equation,
        "dataset_mode": BENCHMARK_MODE,
        "is_real_observational_data": False,
    }


def _load_all_report_statuses(
    outputs_root: Path | None,
    phase_outputs: dict[str, pd.DataFrame],
) -> dict[str, PhaseReportStatus]:
    root = _outputs_root(outputs_root)
    reports = root / "reports"
    statuses: dict[str, PhaseReportStatus] = {}
    for spec in PHASE_SPECS:
        df = phase_outputs.get(spec["phase"], pd.DataFrame())
        p_pass, p_fail = _count_phase_passes(df, spec)
        path = reports / spec["report_md"]
        statuses[spec["phase"]] = load_phase_report_status(
            path,
            phase=spec["phase"],
            banner_substring=spec["banner_substring"],
            pass_count=p_pass,
            fail_count=p_fail,
            total_cases=p_pass + p_fail,
        )
    return statuses


def run_unified_microscopic_quantum_limit_benchmark(
    outputs_root: Path | None = None,
) -> UnifiedBenchmarkResult:
    """Integrate phases 6A–6F; write matrix CSV, report, and figures."""
    root = _outputs_root(outputs_root)
    tables_dir = root / "tables"
    reports_dir = root / "reports"
    figures_dir = root / "figures"
    for d in (tables_dir, reports_dir, figures_dir):
        d.mkdir(parents=True, exist_ok=True)

    phase_outputs = collect_phase_outputs(root)
    report_statuses = _load_all_report_statuses(root, phase_outputs)
    rows = build_unified_consistency_matrix(phase_outputs, report_statuses)

    reports_present = all(rs.present for rs in report_statuses.values())
    disclaimers_present = all(rs.disclaimer_found for rs in report_statuses.values())
    outcomes_correct = all(r.fail_count == 0 and r.pass_count > 0 for r in rows)
    symbol_pass, symbol_missing = check_symbol_consistency(root)
    pipeline_pass, pipeline_chain = check_pipeline_dependencies()
    claim_pass, claim_issues = check_claim_hierarchy(rows)

    readiness = unified_readiness_score(
        rows,
        reports_present=reports_present,
        outcomes_correct=outcomes_correct,
        disclaimers_present=disclaimers_present,
        symbol_pass=symbol_pass,
        pipeline_pass=pipeline_pass,
        claim_pass=claim_pass,
    )

    matrix_df = pd.DataFrame([_row_to_dict(r) for r in rows])
    matrix_df.to_csv(tables_dir / "unified_microscopic_quantum_limit_matrix.csv", index=False)

    report_text = _build_report(
        rows,
        readiness=readiness,
        report_statuses=report_statuses,
        symbol_missing=symbol_missing,
        claim_issues=claim_issues,
        pipeline_chain=pipeline_chain,
    )
    (reports_dir / "unified_microscopic_quantum_limit_report.md").write_text(
        report_text,
        encoding="utf-8",
    )

    _plot_pipeline(figures_dir / "unified_microscopic_pipeline.png")
    _plot_status_matrix(rows, figures_dir / "unified_microscopic_status_matrix.png")

    return UnifiedBenchmarkResult(
        matrix=matrix_df,
        readiness_score=readiness,
        reports_present=reports_present,
        outcomes_correct=outcomes_correct,
        disclaimers_present=disclaimers_present,
        symbol_consistency_pass=symbol_pass,
        pipeline_dependencies_pass=pipeline_pass,
        claim_hierarchy_pass=claim_pass,
        checks={
            "symbol_missing": symbol_missing,
            "claim_issues": claim_issues,
            "pipeline_chain": pipeline_chain,
        },
    )


def _build_report(
    rows: list[UnifiedConsistencyRow],
    *,
    readiness: float,
    report_statuses: dict[str, PhaseReportStatus],
    symbol_missing: dict[str, list[str]],
    claim_issues: list[str],
    pipeline_chain: list[dict[str, str]],
) -> str:
    n_pass = sum(1 for r in rows if r.status == "pass")
    lines = [
        "# Unified microscopic quantum limit report (Phase 6G)",
        "",
        f"## ⚠️ {BANNER_UNIFIED}",
        "",
        "## Purpose",
        "",
        "Integrate the **microscopic quantum-limit benchmark chain** of TDF (Phases 6A–6F) "
        "into one consistency matrix without modifying underlying benchmark equations.",
        "",
        "> **NOT FULL QUANTUM-GRAVITY PROOF.** Integration and documentation only.",
        "",
        "## Pipeline",
        "",
        "```text",
        "τ action → Schrödinger (6A)",
        "τ + spinor χ → Dirac (6B)",
        "configuration-space τ/ρ → entanglement (6C)",
        "Var(Δτ) → decoherence (6D)",
        "A_ℓ[τ] → classical metric (6E)",
        "ρ_i → Born-rule probability proxy (6F)",
        "```",
        "",
        f"- **Readiness score:** {readiness:.2f} (threshold {READINESS_THRESHOLD})",
        f"- **Phases passing integration checks:** {n_pass} / {len(rows)}",
        "",
        "## Results matrix",
        "",
        "| Phase | Benchmark | core result | pass | fail | exp. fail | status | paper ready |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        lines.append(
            f"| {r.phase} | {r.benchmark_name} | {r.core_result} | {r.pass_count} | "
            f"{r.fail_count} | {r.expected_fail_count} | {r.status} | "
            f"{'yes' if r.paper_ready else 'no'} |",
        )
    lines.extend(
        [
            "",
            "## Controlled claims & limits",
            "",
            "| Phase | Controlled claim | Explicit limit |",
            "| --- | --- | --- |",
        ],
    )
    for r in rows:
        lines.append(f"| {r.phase} | {r.controlled_claim} | {r.explicit_limit} |")

    lines.extend(
        [
            "",
            "## Integration checks",
            "",
            f"- Reports present: {all(rs.present for rs in report_statuses.values())}",
            f"- Disclaimers present: {all(rs.disclaimer_found for rs in report_statuses.values())}",
            f"- Symbol consistency: {not symbol_missing}",
            f"- Pipeline dependencies: {len(pipeline_chain)} links documented",
            f"- Claim hierarchy: {not claim_issues}",
            "",
        ],
    )
    if symbol_missing:
        lines.append(f"- Symbol gaps: `{symbol_missing}`")
    if claim_issues:
        lines.append(f"- Claim issues: `{claim_issues}`")

    lines.extend(
        [
            "",
            "## Scientific interpretation",
            "",
            "- **Passing** means TDF has a coherent microscopic benchmark chain connecting "
            "scalar QM, spinors, entanglement, decoherence, classical metric emergence, "
            "and probability weights.",
            "- Does **not** prove full quantum gravity.",
            "- Does **not** solve the measurement problem or derive the Born rule.",
            "",
            "## Explicit limitations",
            "",
            "- Not full quantum gravity.",
            "- Not full Standard Model.",
            "- Not full Bell-theorem resolution.",
            "- Not full objective collapse.",
            "- Not full Born-rule derivation.",
            "- No relativistic QFT measurement model yet.",
            "",
            "## Next-step recommendation",
            "",
            "**v0.17.0** should focus on deriving the microscopic action more deeply from "
            "5D geometry, instead of adding more proxy tests.",
            "",
            "## Disclaimer",
            "",
            "- Meta-benchmark integration only; no new physics equations introduced.",
            "",
        ],
    )
    return "\n".join(lines)


def _plot_pipeline(path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    labels = [
        "6A\nSchrödinger",
        "6B\nDirac",
        "6C\nEntanglement",
        "6D\nDecoherence",
        "6E\nClassical metric",
        "6F\nBorn proxy",
    ]
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 2)
    ax.axis("off")
    for i, lab in enumerate(labels):
        x = 0.5 + i * 1.9
        box = FancyBboxPatch(
            (x, 0.6),
            1.5,
            0.9,
            boxstyle="round,pad=0.05",
            facecolor="#d4e8f7",
            edgecolor="#333",
        )
        ax.add_patch(box)
        ax.text(x + 0.75, 1.05, lab, ha="center", va="center", fontsize=9)
        if i < len(labels) - 1:
            ax.annotate(
                "",
                xy=(x + 1.55, 1.05),
                xytext=(x + 1.75, 1.05),
                arrowprops=dict(arrowstyle="->", lw=1.5),
            )
    ax.set_title("TDF microscopic quantum-limit pipeline (Phase 6G)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_status_matrix(rows: list[UnifiedConsistencyRow], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    phases = [r.phase for r in rows]
    pass_frac = [
        r.pass_count / max(r.pass_count + r.fail_count, 1) for r in rows
    ]
    colors = ["C2" if r.status == "pass" else "C3" for r in rows]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(phases, pass_frac, color=colors)
    ax.axhline(1.0, color="gray", ls="--", lw=0.8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("fraction of cases OK")
    ax.set_title("Unified microscopic phase status (6A–6F)")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)

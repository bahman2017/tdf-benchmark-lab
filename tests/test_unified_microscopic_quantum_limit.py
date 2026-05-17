"""Phase 6G — Unified microscopic quantum limit integration tests."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from tdf_obs.validation.unified_microscopic_quantum_limit import (
    BANNER_UNIFIED,
    PHASE_SPECS,
    READINESS_THRESHOLD,
    REQUIRED_SUMMARY_COLUMNS,
    build_unified_consistency_matrix,
    check_claim_hierarchy,
    check_pipeline_dependencies,
    check_symbol_consistency,
    collect_phase_outputs,
    load_phase_report_status,
    run_unified_microscopic_quantum_limit_benchmark,
    unified_readiness_score,
)


def test_collect_phase_outputs_loads_six_summaries() -> None:
    outputs = collect_phase_outputs()
    assert len(outputs) == 6
    for spec in PHASE_SPECS:
        assert spec["phase"] in outputs
        if (Path(__file__).resolve().parents[1] / "outputs" / "tables" / spec["summary_csv"]).is_file():
            assert not outputs[spec["phase"]].empty


def test_every_phase_has_controlled_claim_and_limit() -> None:
    rows = build_unified_consistency_matrix()
    assert len(rows) == 6
    ok, issues = check_claim_hierarchy(rows)
    assert ok, issues
    for r in rows:
        assert r.controlled_claim
        assert r.explicit_limit


def test_symbol_consistency_passes_on_repo_outputs() -> None:
    ok, missing = check_symbol_consistency()
    assert ok, missing


def test_pipeline_dependency_check_passes() -> None:
    ok, chain = check_pipeline_dependencies()
    assert ok
    assert len(chain) == 5


def test_readiness_score_finite_and_above_threshold() -> None:
    result = run_unified_microscopic_quantum_limit_benchmark(outputs_root=None)
    assert result.readiness_score == pytest.approx(result.readiness_score)
    assert 0.0 <= result.readiness_score <= 1.0
    assert result.readiness_score >= READINESS_THRESHOLD


def test_load_phase_report_status_extracts_banner(tmp_path: Path) -> None:
    report = tmp_path / "r.md"
    report.write_text(
        "## ⚠️ SCHRÖDINGER — NOT FULL QUANTUM VALIDATION\n\nDoes not prove full QG.\n",
        encoding="utf-8",
    )
    st = load_phase_report_status(
        report,
        phase="6A",
        banner_substring="NOT FULL QUANTUM VALIDATION",
        pass_count=3,
        fail_count=0,
        total_cases=3,
    )
    assert st.present
    assert st.banner_found
    assert st.disclaimer_found


def test_pipeline_report_csv_and_figures(tmp_path: Path) -> None:
    root = tmp_path / "out"
    tables = root / "tables"
    reports = root / "reports"
    tables.mkdir(parents=True)
    reports.mkdir(parents=True)
    for spec in PHASE_SPECS:
        df = pd.DataFrame({spec["pass_column"]: [True, True]})
        if spec["outcome_mode"] == "expected_status":
            df["expected_status"] = ["pass", "pass"]
            df["overall_pass"] = [True, True]
        df.to_csv(tables / spec["summary_csv"], index=False)
        (reports / spec["report_md"]).write_text(
            f"## ⚠️ {spec['banner_substring']}\n\n"
            f"τ ρ ψ Ψ χ g̃ Δτ C_AB P_i probability proxy benchmark only.\n"
            f"Does not prove full quantum gravity.\n",
            encoding="utf-8",
        )
    result = run_unified_microscopic_quantum_limit_benchmark(outputs_root=root)
    for col in REQUIRED_SUMMARY_COLUMNS:
        assert col in result.matrix.columns
    report = (root / "reports" / "unified_microscopic_quantum_limit_report.md").read_text(
        encoding="utf-8",
    )
    assert "NOT FULL QUANTUM-GRAVITY PROOF" in report
    assert BANNER_UNIFIED in report
    assert (root / "figures" / "unified_microscopic_pipeline.png").is_file()
    assert (root / "figures" / "unified_microscopic_status_matrix.png").is_file()


def test_unified_matrix_all_phases_pass_when_outputs_present() -> None:
    result = run_unified_microscopic_quantum_limit_benchmark(outputs_root=None)
    assert len(result.matrix) == 6
    if result.outcomes_correct and result.reports_present:
        assert (result.matrix["status"] == "pass").all()

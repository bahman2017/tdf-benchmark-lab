"""Phase 4A expanded ΛCDM/NFW rotation benchmark tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tdf_obs.validation.nfw_surrogate import (
    BANNER_LCDM_NFW,
    BENCHMARK_CASE_REGISTRY,
    REQUIRED_SUMMARY_COLUMNS,
    generate_baryon_profile,
    get_benchmark_case,
    list_benchmark_cases,
    run_nfw_surrogate_pipeline,
    run_single_benchmark_case,
)


def test_registry_has_at_least_ten_cases() -> None:
    assert len(BENCHMARK_CASE_REGISTRY) >= 10


def test_each_case_has_required_fields() -> None:
    required_keys = {
        "case_name",
        "r_min_kpc",
        "r_max_kpc",
        "n_points",
        "baryon_profile_type",
        "baryon_params",
        "Vh2",
        "rs",
        "noise_std",
        "random_seed",
    }
    for name, case in BENCHMARK_CASE_REGISTRY.items():
        fields = case.required_fields()
        assert required_keys <= set(fields.keys())
        assert fields["case_name"] == name
        assert fields["n_points"] >= 10
        assert fields["r_max_kpc"] > fields["r_min_kpc"]


@pytest.mark.parametrize(
    "kind,params",
    [
        ("saturating_disk", {"v_max": 60.0, "r_disk": 2.0}),
        ("exponential_like", {"v_max": 50.0, "r_disk": 3.0}),
        (
            "compact_bulge_disk",
            {"v_bulge_max": 40.0, "r_bulge": 0.4, "v_disk_max": 55.0, "r_disk": 2.0},
        ),
    ],
)
def test_baryon_profiles_finite_nonnegative(kind: str, params: dict[str, float]) -> None:
    r = np.linspace(0.5, 25.0, 30)
    v = generate_baryon_profile(r, kind=kind, **params)
    assert np.all(np.isfinite(v))
    assert np.all(v >= 0)


def test_run_single_case_produces_outputs(tmp_path: Path) -> None:
    case = get_benchmark_case("spiral_mid_mass")
    rec = run_single_benchmark_case(case, mimic_tolerance_percent=15.0, noise_std=0.0)
    assert rec.case_name == "spiral_mid_mass"
    assert np.isfinite(rec.teacher_student_mse)
    assert rec.mimic_success or rec.relative_curve_error_percent >= 0


def test_mimic_success_uses_configurable_tolerance() -> None:
    case = get_benchmark_case("dwarf_low_mass")
    rec_strict = run_single_benchmark_case(case, mimic_tolerance_percent=0.01, noise_std=0.0)
    rec_loose = run_single_benchmark_case(case, mimic_tolerance_percent=50.0, noise_std=0.0)
    if rec_strict.relative_curve_error_percent < 50.0:
        assert rec_loose.mimic_success
    if rec_strict.relative_curve_error_percent >= 0.01:
        assert not rec_strict.mimic_success or rec_strict.relative_curve_error_percent < 0.01


def test_pipeline_one_case_report_and_csv(tmp_path: Path) -> None:
    run_nfw_surrogate_pipeline(
        outputs_root=tmp_path / "out",
        case_names=["milky_way_like"],
        mimic_tolerance_percent=5.0,
        noise_std=0.0,
    )
    report = (tmp_path / "out" / "reports" / "nfw_surrogate_report.md").read_text(encoding="utf-8")
    assert "NOT REAL OBSERVATIONAL DATA" in report
    assert BANNER_LCDM_NFW in report

    csv_path = tmp_path / "out" / "tables" / "nfw_surrogate_fit_summary.csv"
    assert csv_path.is_file()
    import pandas as pd

    df = pd.read_csv(csv_path)
    for col in REQUIRED_SUMMARY_COLUMNS:
        assert col in df.columns, f"missing column {col}"
    assert len(df) == 1
    assert df["case_name"].iloc[0] == "milky_way_like"

    figs = list((tmp_path / "out" / "figures").glob("nfw_surrogate_milky_way_like.png"))
    assert len(figs) == 1


def test_all_registry_names_listable() -> None:
    names = list_benchmark_cases()
    assert len(names) == len(BENCHMARK_CASE_REGISTRY)
    assert "concentrated_halo" in names

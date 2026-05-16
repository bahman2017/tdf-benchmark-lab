"""Phase 3B NFW surrogate validation tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tdf_obs.fitting.fit_rotation import fit_single_galaxy_rotation
from tdf_obs.models.rotation import v_tdf_simple
from tdf_obs.validation.nfw_surrogate import (
    BANNER_NFW_SURROGATE,
    DATASET_MODE,
    generate_baryon_profile,
    generate_nfw_surrogate_rotation_curve,
    dataframe_to_rotation_data,
    run_nfw_surrogate_pipeline,
    teacher_student_mse,
)


def test_surrogate_generator_columns() -> None:
    r = np.linspace(0.5, 20.0, 15)
    v_b = generate_baryon_profile(r, v_max=60.0, r_disk=2.0)
    df = generate_nfw_surrogate_rotation_curve(
        "test_gal",
        r,
        v_b,
        Vh2=3000.0,
        rs=5.0,
        noise_std=0.5,
        random_seed=1,
    )
    required = {
        "galaxy_id",
        "r_kpc",
        "v_obs",
        "v_err",
        "v_baryon",
        "v_teacher",
        "nfw_Vh2_true",
        "nfw_rs_true",
        "dataset_mode",
    }
    assert required <= set(df.columns)
    assert (df["dataset_mode"] == DATASET_MODE).all()
    assert df["dataset_mode"].iloc[0] != "real_data_calibration"


def test_no_fake_real_data_label() -> None:
    r = np.linspace(1.0, 10.0, 8)
    v_b = generate_baryon_profile(r)
    df = generate_nfw_surrogate_rotation_curve("g1", r, v_b, 1000.0, 3.0)
    rot = dataframe_to_rotation_data(df)
    assert rot.metadata["dataset_mode"] == DATASET_MODE
    assert rot.metadata["is_real_observational_data"] is False


def test_generated_velocities_finite_positive() -> None:
    r = np.linspace(0.5, 25.0, 30)
    v_b = generate_baryon_profile(r)
    df = generate_nfw_surrogate_rotation_curve("g2", r, v_b, 5000.0, 8.0, noise_std=1.0, random_seed=2)
    for col in ("v_obs", "v_err", "v_baryon", "v_teacher"):
        assert np.all(np.isfinite(df[col]))
        assert np.all(df[col] >= 0)
    assert np.all(df["v_err"] > 0)


def test_tdf_fitting_runs_on_surrogate() -> None:
    r = np.linspace(0.5, 20.0, 20)
    v_b = generate_baryon_profile(r)
    df = generate_nfw_surrogate_rotation_curve("g3", r, v_b, 4000.0, 6.0, random_seed=3)
    fit = fit_single_galaxy_rotation(dataframe_to_rotation_data(df))
    assert fit.success_tdf
    assert np.isfinite(fit.tdf_B)
    v_teacher = df["v_teacher"].to_numpy()
    v_tdf = v_tdf_simple(r, v_b, fit.tdf_B, fit.tdf_r0)
    assert teacher_student_mse(v_teacher, v_tdf) < teacher_student_mse(v_teacher, v_b)


def test_report_contains_banner(tmp_path: Path) -> None:
    run_nfw_surrogate_pipeline(outputs_root=tmp_path / "outputs", noise_std=0.5)
    report = (tmp_path / "outputs" / "reports" / "nfw_surrogate_report.md").read_text(encoding="utf-8")
    assert BANNER_NFW_SURROGATE in report
    assert "does not validate tdf against real observations" in report.lower()
    assert "Phase 3B" in report


def test_outputs_created(tmp_path: Path) -> None:
    run_nfw_surrogate_pipeline(outputs_root=tmp_path / "out")
    assert (tmp_path / "out" / "tables" / "nfw_surrogate_fit_summary.csv").is_file()
    assert (tmp_path / "out" / "reports" / "nfw_surrogate_report.md").is_file()
    figs = list((tmp_path / "out" / "figures").glob("nfw_surrogate_*.png"))
    assert len(figs) >= 3

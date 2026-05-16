"""Phase 3 rotation baseline comparison tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tdf_obs.fitting.fit_rotation import fit_single_galaxy_rotation
from tdf_obs.fitting.metrics import aic, bic
from tdf_obs.io.dataset_metadata import BANNER_DEMO_FIXTURE
from tdf_obs.pipelines.run_rotation_pipeline import run_rotation_pipeline
from tdf_obs.plotting.plot_rotation import plot_rotation_fit
from tdf_obs.validation.synthetic_tests import generate_synthetic_rotation_curve


def test_fit_returns_tdf_and_nfw_metrics() -> None:
    data, _ = generate_synthetic_rotation_curve(seed=99, noise_fraction=0.02)
    fit = fit_single_galaxy_rotation(data)

    assert np.isfinite(fit.tdf_B)
    assert np.isfinite(fit.tdf_r0)
    assert np.isfinite(fit.nfw_Vh2)
    assert np.isfinite(fit.nfw_rs)
    assert fit.mse_tdf <= fit.mse_baryon
    assert fit.best_model_by_bic in ("baryon_only", "tdf_simple", "nfw_simple")
    assert fit.success_tdf
    assert fit.success_nfw


def test_aic_bic_parameter_counts() -> None:
    chi2 = 10.0
    n = 20
    assert aic(chi2, 0) == pytest.approx(chi2)
    assert aic(chi2, 2) == pytest.approx(chi2 + 4.0)
    assert bic(chi2, n, 0) == pytest.approx(chi2)
    assert bic(chi2, n, 2) == pytest.approx(chi2 + 2 * np.log(n))


def test_report_includes_baseline_comparison(tmp_path: Path) -> None:
    result = run_rotation_pipeline(
        processed_csv=tmp_path / "missing.csv",
        outputs_root=tmp_path / "outputs",
    )
    report = result.report_path.read_text(encoding="utf-8")
    assert "## Baseline comparison (Phase 3)" in report
    assert "### Baseline comparison" in report
    assert "Best model by BIC" in report
    assert "Lower MSE" in report


def test_plot_includes_nfw_curve(tmp_path: Path) -> None:
    data, _ = generate_synthetic_rotation_curve()
    fit = fit_single_galaxy_rotation(data)
    out = plot_rotation_fit(data, fit, tmp_path / "fig.png")
    assert out.is_file()
    assert out.stat().st_size > 1000


def test_demo_fixture_report_keeps_banner(tmp_path: Path) -> None:
    """Demo fixture labeling must remain visible after Phase 3."""
    csv = Path(__file__).parent / "fixtures" / "sample_rotation.csv"
    proc = tmp_path / "data" / "processed"
    proc.mkdir(parents=True)
    (proc / "rotation.csv").write_text(csv.read_text(encoding="utf-8"), encoding="utf-8")
    (proc / "rotation_metadata.yaml").write_text(
        "dataset_type: demo_fixture\n"
        "source: test_fixture\n"
        'description: "demo"\n'
        "is_real_observational_data: false\n",
        encoding="utf-8",
    )

    result = run_rotation_pipeline(
        processed_csv=proc / "rotation.csv",
        outputs_root=tmp_path / "outputs",
    )
    report = result.report_path.read_text(encoding="utf-8")
    assert BANNER_DEMO_FIXTURE in report
    summary = __import__("pandas").read_csv(result.summary_csv_path)
    assert "nfw_Vh2" in summary.columns
    assert "best_model_by_bic" in summary.columns

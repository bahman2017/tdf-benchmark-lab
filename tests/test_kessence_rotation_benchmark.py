"""K-essence rotation calibration benchmark tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tdf_obs.fitting.fit_rotation import fit_kessence_galaxy_rotation
from tdf_obs.validation.kessence_rotation_benchmark import (
    BANNER_CALIBRATION,
    compare_rotation_models,
    generate_synthetic_kessence_rotation_curve,
    run_kessence_rotation_benchmark,
)


def test_fit_kessence_recovers_a0_on_synthetic() -> None:
    data, truth = generate_synthetic_kessence_rotation_curve(
        a0=1500.0,
        noise_fraction=0.02,
        seed=7,
    )
    fit = fit_kessence_galaxy_rotation(data)
    assert fit.success
    assert fit.a0 == pytest.approx(truth["a0_true"], rel=0.3)


def test_compare_models_kessence_beats_baryon_on_synthetic() -> None:
    data, _ = generate_synthetic_kessence_rotation_curve(seed=1)
    res = compare_rotation_models(data)
    assert res.mse_kessence < res.mse_baryon
    assert res.kessence_beats_baryon_by_bic


def test_benchmark_outputs_and_banner(tmp_path: Path) -> None:
    out = tmp_path / "outputs"
    data, _ = generate_synthetic_kessence_rotation_curve()
    res = compare_rotation_models(data)
    from tdf_obs.validation import kessence_rotation_benchmark as krb

    krb._plot_comparison(data, res, out / "figures" / "test.png")
    report = krb._build_report(res)
    assert BANNER_CALIBRATION in report
    assert "phenomenological" in report.lower()
    assert "do not constitute observational validation" in report


def test_run_benchmark_pipeline(tmp_path: Path) -> None:
    df, res, _ = run_kessence_rotation_benchmark(outputs_root=tmp_path / "out")
    assert len(df) == 1
    assert (tmp_path / "out" / "tables" / "kessence_rotation_benchmark_summary.csv").is_file()
    report = (tmp_path / "out" / "reports" / "kessence_rotation_benchmark_report.md").read_text(
        encoding="utf-8",
    )
    assert BANNER_CALIBRATION in report
    assert np.isfinite(res.a0)

"""K-essence rotation calibration benchmark — phenomenological model comparison."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tdf_obs.fitting.fit_rotation import (
    N_PARAMS_BARYON,
    N_PARAMS_KESSENCE,
    N_PARAMS_TDF,
    KessenceFitResult,
    RotationFitResult,
    _model_metrics,
    fit_kessence_galaxy_rotation,
    fit_single_galaxy_rotation,
)
from tdf_obs.io.loaders import default_processed_rotation_path, load_rotation_csv
from tdf_obs.io.schemas import RotationCurveData
from tdf_obs.models.rotation import (
    baryon_only_model,
    v_tdf_kessence,
    v_tdf_simple,
)

BENCHMARK_MODE = "kessence_rotation_calibration_benchmark"
BANNER_CALIBRATION = (
    "These results are calibration diagnostics for a phenomenological TDF model. "
    "They do not constitute observational validation of TDF unless all required "
    "multi-channel constraints are passed."
)
BANNER_KESSENCE = (
    "TDF K-ESSENCE ROTATION CALIBRATION — PHENOMENOLOGICAL CANDIDATE MODEL ONLY"
)


@dataclass
class KessenceRotationBenchmarkResult:
    galaxy_id: str
    data_mode: str
    a0: float
    tdf_B: float
    tdf_r0: float
    a0_true: float | None
    mse_baryon: float
    mse_tdf_simple: float
    mse_kessence: float
    chi2_baryon: float
    chi2_tdf_simple: float
    chi2_kessence: float
    bic_baryon: float
    bic_tdf_simple: float
    bic_kessence: float
    best_model_by_bic: str
    kessence_beats_baryon_by_bic: bool
    kessence_beats_tdf_by_bic: bool
    overall_pass: bool
    warnings: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


def generate_synthetic_kessence_rotation_curve(
    galaxy_id: str = "kessence_synthetic_001",
    *,
    a0: float = 1200.0,
    n_points: int = 30,
    r_min_kpc: float = 0.5,
    r_max_kpc: float = 35.0,
    noise_fraction: float = 0.03,
    seed: int = 42,
) -> tuple[RotationCurveData, dict[str, Any]]:
    """Synthetic curve from K-essence truth (calibration recovery test)."""
    rng = np.random.default_rng(seed)
    r = np.linspace(r_min_kpc, r_max_kpc, n_points)
    v_baryon = 200.0 * np.sqrt(r / (r + 1.5)) + 15.0
    v_true = v_tdf_kessence(r, v_baryon, a0)
    sigma = np.maximum(noise_fraction * v_true, 1.0)
    v_obs = v_true + rng.normal(0.0, sigma)
    truth = {"a0_true": a0, "noise_fraction": noise_fraction, "seed": seed}
    data = RotationCurveData(
        galaxy_id=galaxy_id,
        r_kpc=r,
        v_obs=v_obs,
        v_err=sigma,
        v_baryon=v_baryon,
        metadata={
            "dataset_mode": "synthetic_validation",
            "dataset_source": "kessence_in_memory_generator",
            "is_real_observational_data": False,
            "truth": truth,
        },
    )
    return data, truth


def _resolve_benchmark_data(
    project_root: Path,
) -> tuple[RotationCurveData, dict[str, Any]]:
    csv_path = default_processed_rotation_path(project_root)
    if csv_path.is_file():
        curves = load_rotation_csv(csv_path)
        if curves:
            data = curves[0]
            truth = data.metadata.get("truth", {})
            return data, dict(truth) if isinstance(truth, dict) else {}
    return generate_synthetic_kessence_rotation_curve()


def compare_rotation_models(data: RotationCurveData) -> KessenceRotationBenchmarkResult:
    """Compare baryon-only, TDF simple (B, r0), and TDF K-essence (a0)."""
    r = np.asarray(data.r_kpc, dtype=float)
    v_obs = np.asarray(data.v_obs, dtype=float)
    v_err = np.maximum(np.asarray(data.v_err, dtype=float), 1e-3)
    v_baryon = np.asarray(data.v_baryon, dtype=float)

    v_bary = baryon_only_model(v_baryon)
    mse_b, chi2_b, _, _, bic_b = _model_metrics(v_obs, v_bary, v_err, N_PARAMS_BARYON)

    tdf_fit: RotationFitResult = fit_single_galaxy_rotation(data)
    kess_fit: KessenceFitResult = fit_kessence_galaxy_rotation(data)

    if tdf_fit.success_tdf:
        v_tdf = v_tdf_simple(r, v_baryon, tdf_fit.tdf_B, tdf_fit.tdf_r0)
        B, r0 = tdf_fit.tdf_B, tdf_fit.tdf_r0
    else:
        B, r0 = float("nan"), float("nan")
        v_tdf = v_bary

    if kess_fit.success:
        v_kess = v_tdf_kessence(r, v_baryon, kess_fit.a0)
        a0 = kess_fit.a0
    else:
        a0 = float("nan")
        v_kess = v_bary

    mse_t, chi2_t, _, _, bic_t = _model_metrics(v_obs, v_tdf, v_err, N_PARAMS_TDF)
    mse_k, chi2_k, _, _, bic_k = _model_metrics(v_obs, v_kess, v_err, N_PARAMS_KESSENCE)

    bic_scores = {
        "baryon_only": bic_b,
        "tdf_simple": bic_t,
        "tdf_kessence": bic_k,
    }
    best = min(bic_scores, key=bic_scores.get)  # type: ignore[arg-type]

    truth = data.metadata.get("truth", {})
    a0_true = None
    if isinstance(truth, dict) and "a0_true" in truth:
        a0_true = float(truth["a0_true"])

    warnings_list: list[str] = []
    warnings_list.extend(tdf_fit.warnings)
    warnings_list.extend(kess_fit.warnings)

    recovery_ok = True
    if a0_true is not None and kess_fit.success:
        rel_err = abs(a0 - a0_true) / max(a0_true, 1e-6)
        recovery_ok = rel_err < 0.35
        if not recovery_ok:
            warnings_list.append(
                f"K-essence a0 recovery |fit−true|/true = {rel_err:.2f} (>35% tolerance).",
            )

    overall = kess_fit.success and tdf_fit.success_tdf and recovery_ok

    data_mode = str(
        data.metadata.get("dataset_mode", data.metadata.get("data_mode", "unknown")),
    )

    return KessenceRotationBenchmarkResult(
        galaxy_id=data.galaxy_id,
        data_mode=data_mode,
        a0=a0,
        tdf_B=B,
        tdf_r0=r0,
        a0_true=a0_true,
        mse_baryon=mse_b,
        mse_tdf_simple=mse_t,
        mse_kessence=mse_k,
        chi2_baryon=chi2_b,
        chi2_tdf_simple=chi2_t,
        chi2_kessence=chi2_k,
        bic_baryon=bic_b,
        bic_tdf_simple=bic_t,
        bic_kessence=bic_k,
        best_model_by_bic=best,
        kessence_beats_baryon_by_bic=bic_k < bic_b,
        kessence_beats_tdf_by_bic=bic_k < bic_t,
        overall_pass=overall,
        warnings="; ".join(warnings_list),
        metadata={"bic_scores": bic_scores},
    )


def _result_to_row(res: KessenceRotationBenchmarkResult) -> dict[str, Any]:
    return {
        "galaxy_id": res.galaxy_id,
        "data_mode": res.data_mode,
        "a0": res.a0,
        "tdf_B": res.tdf_B,
        "tdf_r0": res.tdf_r0,
        "a0_true": res.a0_true if res.a0_true is not None else "",
        "mse_baryon": res.mse_baryon,
        "mse_tdf_simple": res.mse_tdf_simple,
        "mse_kessence": res.mse_kessence,
        "chi2_baryon": res.chi2_baryon,
        "chi2_tdf_simple": res.chi2_tdf_simple,
        "chi2_kessence": res.chi2_kessence,
        "bic_baryon": res.bic_baryon,
        "bic_tdf_simple": res.bic_tdf_simple,
        "bic_kessence": res.bic_kessence,
        "best_model_by_bic": res.best_model_by_bic,
        "kessence_beats_baryon_by_bic": res.kessence_beats_baryon_by_bic,
        "kessence_beats_tdf_by_bic": res.kessence_beats_tdf_by_bic,
        "overall_pass": res.overall_pass,
        "warnings": res.warnings,
        "dataset_mode": BENCHMARK_MODE,
        "is_real_observational_data": False,
    }


def _plot_comparison(
    data: RotationCurveData,
    res: KessenceRotationBenchmarkResult,
    path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    r = np.asarray(data.r_kpc, dtype=float)
    v_obs = np.asarray(data.v_obs, dtype=float)
    v_err = np.asarray(data.v_err, dtype=float)
    v_baryon = np.asarray(data.v_baryon, dtype=float)
    r_fine = np.linspace(r.min(), r.max(), 200)
    vb_fine = np.interp(r_fine, r, v_baryon)

    v_bary = baryon_only_model(v_baryon)
    v_tdf = (
        v_tdf_simple(r_fine, vb_fine, res.tdf_B, res.tdf_r0)
        if np.isfinite(res.tdf_B)
        else np.interp(r_fine, r, v_bary)
    )
    v_kess = (
        v_tdf_kessence(r_fine, vb_fine, res.a0)
        if np.isfinite(res.a0)
        else np.interp(r_fine, r, v_bary)
    )

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.errorbar(r, v_obs, yerr=v_err, fmt="o", capsize=3, label="observed", color="C0")
    ax.plot(r, v_bary, "--", label="baryon-only", color="C1", lw=1.5)
    ax.plot(r_fine, v_tdf, "-", label="TDF simple (B, r0)", color="C2", lw=1.5)
    ax.plot(r_fine, v_kess, "-", label=f"TDF K-essence (a0={res.a0:.1f})", color="C4", lw=1.5)
    ax.set_xlabel("r [kpc]")
    ax.set_ylabel("v [km/s]")
    ax.set_title(
        f"{res.galaxy_id} — K-essence calibration diagnostic\n"
        f"best BIC: {res.best_model_by_bic}",
    )
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _build_report(res: KessenceRotationBenchmarkResult) -> str:
    a0_true_line = (
        f"| a0_true (injected) | {res.a0_true:.2f} |"
        if res.a0_true is not None
        else "| a0_true (injected) | — (not synthetic K-essence truth) |"
    )
    return "\n".join(
        [
            "# TDF K-essence rotation calibration report",
            "",
            f"## ⚠️ {BANNER_KESSENCE}",
            "",
            f"## ⚠️ {BANNER_CALIBRATION}",
            "",
            "## Purpose",
            "",
            "Phenomenological calibration diagnostic comparing **baryon-only**, "
            "**TDF simple** (weak-field ansatz), and **TDF non-linear K-essence** "
            "candidate rotation models on a single galaxy curve.",
            "",
            "The K-essence limit tested here is:",
            "",
            "```text",
            "v^2(r) = v_b^2(r) + v_b(r) * sqrt(a0 * r)",
            "```",
            "",
            "with `a0` in units of (km/s)²/kpc. This is a **candidate model** fit only; "
            "it does not prove dark matter is absent or that TDF is observationally validated.",
            "",
            "## Galaxy",
            "",
            f"- **ID:** {res.galaxy_id}",
            f"- **Data mode:** {res.data_mode}",
            "",
            "## Fitted parameters",
            "",
            "| Parameter | Value |",
            "| --- | --- |",
            f"| a0 (K-essence) | {res.a0:.4g} |",
            f"| B (TDF simple) | {res.tdf_B:.4g} |",
            f"| r0 (TDF simple) | {res.tdf_r0:.4g} |",
            a0_true_line,
            "",
            "## Model comparison",
            "",
            "| Model | MSE | χ² | BIC |",
            "| --- | --- | --- | --- |",
            f"| baryon-only | {res.mse_baryon:.2f} | {res.chi2_baryon:.2f} | {res.bic_baryon:.2f} |",
            f"| TDF simple | {res.mse_tdf_simple:.2f} | {res.chi2_tdf_simple:.2f} | {res.bic_tdf_simple:.2f} |",
            f"| TDF K-essence | {res.mse_kessence:.2f} | {res.chi2_kessence:.2f} | {res.bic_kessence:.2f} |",
            "",
            f"- **Best model by BIC:** {res.best_model_by_bic}",
            f"- **K-essence beats baryon-only (BIC):** {res.kessence_beats_baryon_by_bic}",
            f"- **K-essence beats TDF simple (BIC):** {res.kessence_beats_tdf_by_bic}",
            "",
            "## Interpretation",
            "",
            "- Lower MSE/χ²/BIC indicates a better **phenomenological fit** under this diagnostic.",
            "- A flat outer rotation curve from baryons alone is not expected; the K-essence "
            "correction is designed as a MOND-like enhancement candidate.",
            "- Passing this benchmark only checks numerical consistency and fit competitiveness "
            "in the chosen test set — not astrophysical completeness.",
            "",
            "## Warnings",
            "",
            res.warnings or "(none)",
            "",
            "## Disclaimer",
            "",
            f"- {BANNER_CALIBRATION}",
            "",
        ],
    )


def run_kessence_rotation_benchmark(
    outputs_root: Path | None = None,
    project_root: Path | None = None,
) -> tuple[pd.DataFrame, KessenceRotationBenchmarkResult, RotationCurveData]:
    """Run K-essence rotation benchmark; write CSV, report, figure."""
    proj = Path(project_root or Path(__file__).resolve().parents[3])
    out = Path(outputs_root or proj / "outputs")
    tables = out / "tables"
    reports = out / "reports"
    figures = out / "figures"
    for d in (tables, reports, figures):
        d.mkdir(parents=True, exist_ok=True)

    data, _truth = _resolve_benchmark_data(proj)
    result = compare_rotation_models(data)

    df = pd.DataFrame([_result_to_row(result)])
    csv_path = tables / "kessence_rotation_benchmark_summary.csv"
    df.to_csv(csv_path, index=False)

    report_path = reports / "kessence_rotation_benchmark_report.md"
    report_path.write_text(_build_report(result), encoding="utf-8")

    fig_path = figures / f"{result.galaxy_id}_kessence_rotation_benchmark.png"
    _plot_comparison(data, result, fig_path)

    return df, result, data

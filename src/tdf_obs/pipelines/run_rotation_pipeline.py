"""Rotation calibration pipeline — synthetic, demo fixture, or real observational CSV."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd
import yaml

from tdf_obs.fitting.fit_rotation import RotationFitResult, fit_single_galaxy_rotation
from tdf_obs.io.dataset_metadata import (
    BANNER_DEMO_FIXTURE,
    BANNER_SYNTHETIC,
    RotationDatasetInfo,
    default_processed_rotation_path,
    resolve_dataset_info,
)
from tdf_obs.io.loaders import load_rotation_csv
from tdf_obs.plotting.plot_rotation import plot_rotation_fit
from tdf_obs.validation.synthetic_tests import generate_synthetic_rotation_curve

ROTATION_MODEL_EQUATION = (
    "v_model^2(r) = v_baryon^2(r) + B * r / (r + r0)   "
    "[tau_bar_l(r) = A log(1 + r/r0), B = K_tau * A; r in kpc, v in km/s, B in km^2/s^2]"
)

NFW_MODEL_EQUATION = (
    "v_DM^2(r) = v_baryon^2(r) + Vh2 * [ln(1+r/rs) - (r/rs)/(1+r/rs)] / (r/rs)   "
    "[phenomenological NFW-like baseline; Vh2 in km^2/s^2, rs in kpc]"
)

REPORT_LIMITATIONS = [
    "This is a calibration diagnostic, not a validation of TDF.",
    "B and K_tau * A are not separated; only the combined parameter B is fitted.",
    "v_baryon(r) is an input, not fitted here.",
    "NFW simple is a phenomenological comparison halo, not a unique physical fit.",
    "Lower MSE alone does not establish a better model when parameter counts differ; use BIC.",
    "No lensing, redshift, or solar-system consistency checks in this pipeline.",
    "Poor fits do not imply TDF failure until multi-channel tests are run.",
]

BASELINE_COMPARISON_NOTE = (
    "Models compared: baryon-only (0 parameters), TDF simple (2), NFW simple (2). "
    "Selection by BIC uses chi² as likelihood proxy. This does not prove TDF is correct."
)


@dataclass
class RotationPipelineResult:
    """Summary returned by the rotation pipeline."""

    mode: str
    dataset_info: RotationDatasetInfo
    data_source: str
    processed_csv: Path | None
    n_galaxies: int
    results: list[RotationFitResult] = field(default_factory=list)
    report_path: Path | None = None
    summary_csv_path: Path | None = None


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def load_config(config_name: str = "rotation_sparc.yaml") -> dict:
    path = project_root() / "configs" / config_name
    if path.exists():
        with path.open() as f:
            return yaml.safe_load(f) or {}
    return {}


def _figure_path(figures_dir: Path, galaxy_id: str) -> Path:
    safe_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in galaxy_id)
    return figures_dir / f"{safe_id}_rotation.png"


def _write_summary_csv(
    results: list[RotationFitResult],
    path: Path,
    dataset_info: RotationDatasetInfo,
) -> None:
    rows = [asdict(r) for r in results]
    for row in rows:
        row["warnings"] = "; ".join(row.get("warnings") or [])
        row["dataset_mode"] = dataset_info.dataset_mode
        row["dataset_source"] = dataset_info.dataset_source
        row["is_real_observational_data"] = dataset_info.is_real_observational_data
    pd.DataFrame(rows).to_csv(path, index=False)


def _format_baseline_comparison(fit: RotationFitResult) -> list[str]:
    return [
        "### Baseline comparison",
        "",
        f"- **Best model by BIC:** `{fit.best_model_by_bic}`",
        f"- **TDF beats baryon-only (BIC):** {fit.tdf_beats_baryon_by_bic}",
        f"- **TDF beats NFW simple (BIC):** {fit.tdf_beats_nfw_by_bic}",
        f"- MSE baryon = {fit.mse_baryon:.6g}, TDF = {fit.mse_tdf:.6g}, NFW = {fit.mse_nfw:.6g}",
        f"- BIC baryon = {fit.bic_baryon:.4f}, TDF = {fit.bic_tdf:.4f}, NFW = {fit.bic_nfw:.4f}",
        f"- TDF vs baryon MSE improvement = {fit.tdf_vs_baryon_improvement_percent:.2f}%",
        f"- TDF vs NFW MSE improvement = {fit.tdf_vs_nfw_improvement_percent:.2f}%",
        "",
        "_Lower MSE with more parameters can mislead; prefer BIC for model ranking._",
        "",
    ]


def _format_fit_metrics(fit: RotationFitResult) -> list[str]:
    lines = [
        "#### TDF simple",
        f"- tdf_B = {fit.tdf_B:.6g} km²/s²",
        f"- tdf_r0 = {fit.tdf_r0:.6g} kpc",
        f"- success_tdf = {fit.success_tdf}",
        "",
        "#### NFW simple",
        f"- nfw_Vh2 = {fit.nfw_Vh2:.6g} km²/s²",
        f"- nfw_rs = {fit.nfw_rs:.6g} kpc",
        f"- success_nfw = {fit.success_nfw}",
        "",
        "#### Metrics",
        f"- χ² baryon = {fit.chi2_baryon:.6g} (reduced {fit.chi2_red_baryon:.4f}, n_params=0)",
        f"- χ² TDF = {fit.chi2_tdf:.6g} (reduced {fit.chi2_red_tdf:.4f}, n_params=2)",
        f"- χ² NFW = {fit.chi2_nfw:.6g} (reduced {fit.chi2_red_nfw:.4f}, n_params=2)",
        f"- AIC: baryon={fit.aic_baryon:.4f}, TDF={fit.aic_tdf:.4f}, NFW={fit.aic_nfw:.4f}",
    ]
    lines.extend(_format_baseline_comparison(fit))
    return lines


def _build_report_markdown(
    *,
    dataset_info: RotationDatasetInfo,
    n_galaxies: int,
    results: list[RotationFitResult],
    synthetic_truth: dict | None = None,
) -> str:
    lines = [
        "# Rotation pipeline report",
        "",
        "> **Disclaimer:** Calibration diagnostics only. **This is not a claim that TDF is validated.**",
        "",
    ]

    if dataset_info.warning_banner:
        lines.extend(
            [
                f"## ⚠️ {dataset_info.warning_banner}",
                "",
            ],
        )

    lines.extend(
        [
            "## Dataset labeling",
            "",
            "| Field | Value |",
            "| --- | --- |",
            f"| **Dataset mode** | `{dataset_info.dataset_mode}` |",
            f"| **Dataset type** | `{dataset_info.dataset_type}` |",
            f"| **Dataset source** | {dataset_info.dataset_source} |",
            f"| **Real observational data** | {dataset_info.is_real_observational_data} |",
            f"| **Description** | {dataset_info.description} |",
            f"| **Galaxies** | {n_galaxies} |",
        ],
    )
    if dataset_info.csv_path:
        lines.append(f"| **Processed CSV** | `{dataset_info.csv_path}` |")
    if dataset_info.metadata_path:
        lines.append(f"| **Metadata sidecar** | `{dataset_info.metadata_path}` |")
    elif dataset_info.csv_path and dataset_info.dataset_mode == "demo_fixture_calibration":
        lines.append("| **Metadata sidecar** | _(missing — defaulted to demo fixture)_ |")
    lines.extend(
        [
            "",
            "## Baseline comparison (Phase 3)",
            "",
            BASELINE_COMPARISON_NOTE,
            "",
            "## Models",
            "",
            "**TDF simple:**",
            "",
            "```text",
            ROTATION_MODEL_EQUATION,
            "```",
            "",
            "**NFW simple (comparison halo):**",
            "",
            "```text",
            NFW_MODEL_EQUATION,
            "```",
            "",
        ],
    )

    if dataset_info.dataset_mode == "synthetic_validation" and synthetic_truth:
        lines.extend(
            [
                "### Injected truth (synthetic)",
                "",
                f"- B_true = {synthetic_truth['B_true']:.6g} km²/s²",
                f"- r0_true = {synthetic_truth['r0_true']:.6g} kpc",
                "",
            ],
        )

    for fit in results:
        lines.append(f"## Galaxy: `{fit.galaxy_id}`")
        lines.append("")
        lines.extend(_format_fit_metrics(fit))
        lines.append("")
        if fit.warnings:
            lines.append("**Warnings:**")
            for w in fit.warnings:
                lines.append(f"- {w}")
            lines.append("")

    lines.extend(["## Limitations", ""])
    for item in REPORT_LIMITATIONS:
        lines.append(f"- {item}")
    lines.append("")

    return "\n".join(lines)


def _run_synthetic_pipeline(
    outputs: Path,
    cfg: dict,
    dataset_info: RotationDatasetInfo,
) -> tuple[list[RotationFitResult], dict]:
    figures_dir = outputs / "figures"
    syn = cfg.get("synthetic", {})
    data, truth = generate_synthetic_rotation_curve(
        galaxy_id=syn.get("galaxy_id", "synthetic_001"),
        B=float(syn.get("B_true", 800.0)),
        r0=float(syn.get("r0_true", 4.0)),
        n_points=int(syn.get("n_points", 25)),
        noise_fraction=float(syn.get("noise_fraction", 0.03)),
        seed=syn.get("seed", 42),
    )
    fit = fit_single_galaxy_rotation(data)
    plot_rotation_fit(data, fit, _figure_path(figures_dir, data.galaxy_id))
    return [fit], truth


def _run_csv_pipeline(
    csv_path: Path,
    outputs: Path,
    dataset_info: RotationDatasetInfo,
) -> list[RotationFitResult]:
    figures_dir = outputs / "figures"
    results: list[RotationFitResult] = []
    for data in load_rotation_csv(csv_path, dataset_info=dataset_info):
        fit = fit_single_galaxy_rotation(data)
        results.append(fit)
        plot_rotation_fit(data, fit, _figure_path(figures_dir, data.galaxy_id))
    return results


def run_rotation_pipeline(
    *,
    processed_csv: Path | None = None,
    outputs_root: Path | None = None,
) -> RotationPipelineResult:
    """
    Run rotation pipeline in exactly one mode:

    - ``synthetic_validation`` — no rotation.csv
    - ``demo_fixture_calibration`` — CSV without confirmed real metadata
    - ``real_data_calibration`` — CSV + rotation_metadata.yaml confirming real data
    """
    root = project_root()
    csv_path = Path(processed_csv) if processed_csv else default_processed_rotation_path(root)
    dataset_info = resolve_dataset_info(csv_path, project_root=root)
    outputs = outputs_root or (root / "outputs")
    tables_dir = outputs / "tables"
    reports_dir = outputs / "reports"
    for d in (outputs / "figures", tables_dir, reports_dir):
        d.mkdir(parents=True, exist_ok=True)

    cfg = load_config()
    synthetic_truth: dict | None = None

    if dataset_info.dataset_mode == "synthetic_validation":
        results, synthetic_truth = _run_synthetic_pipeline(outputs, cfg, dataset_info)
        data_source = f"{BANNER_SYNTHETIC} (in-memory generator)"
    else:
        assert dataset_info.csv_path is not None
        results = _run_csv_pipeline(dataset_info.csv_path, outputs, dataset_info)
        data_source = (
            f"{dataset_info.dataset_source} — `{dataset_info.csv_path}`"
        )

    summary_path = tables_dir / "rotation_fit_summary.csv"
    _write_summary_csv(results, summary_path, dataset_info)

    report_path = reports_dir / "rotation_report.md"
    report_path.write_text(
        _build_report_markdown(
            dataset_info=dataset_info,
            n_galaxies=len(results),
            results=results,
            synthetic_truth=synthetic_truth,
        ),
        encoding="utf-8",
    )

    return RotationPipelineResult(
        mode=dataset_info.dataset_mode,
        dataset_info=dataset_info,
        data_source=data_source,
        processed_csv=dataset_info.csv_path,
        n_galaxies=len(results),
        results=results,
        report_path=report_path,
        summary_csv_path=summary_path,
    )

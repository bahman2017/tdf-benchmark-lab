"""Rotation-curve diagnostic plots with baseline comparison."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tdf_obs.fitting.fit_rotation import RotationFitResult
from tdf_obs.io.schemas import RotationCurveData
from tdf_obs.models.dark_matter import v_nfw_simple
from tdf_obs.models.rotation import baryon_only_model, v_tdf_simple


def plot_rotation_fit(
    data: RotationCurveData,
    fit: RotationFitResult,
    output_path: Path,
) -> Path:
    """
    Plot v_obs with errors, baryon-only, TDF fit, and NFW simple fit.
    """
    r = np.asarray(data.r_kpc, dtype=float)
    v_obs = np.asarray(data.v_obs, dtype=float)
    v_err = np.asarray(data.v_err, dtype=float)
    v_baryon = np.asarray(data.v_baryon, dtype=float)

    r_fine = np.linspace(r.min(), r.max(), 200)
    v_baryon_fine = np.interp(r_fine, r, v_baryon)
    v_bary = baryon_only_model(v_baryon)

    if fit.success_tdf and np.isfinite(fit.tdf_B):
        v_tdf_curve = v_tdf_simple(r_fine, v_baryon_fine, fit.tdf_B, fit.tdf_r0)
    else:
        v_tdf_curve = np.interp(r_fine, r, v_bary)

    if fit.success_nfw and np.isfinite(fit.nfw_Vh2):
        v_nfw_curve = v_nfw_simple(r_fine, v_baryon_fine, fit.nfw_Vh2, fit.nfw_rs)
    else:
        v_nfw_curve = np.interp(r_fine, r, v_bary)

    mode = data.metadata.get("dataset_mode", data.metadata.get("data_mode", "unknown"))
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.errorbar(r, v_obs, yerr=v_err, fmt="o", capsize=3, label="observed", color="C0")
    ax.plot(r, v_bary, "--", label="baryon-only", color="C1", linewidth=1.5)
    ax.plot(r_fine, v_tdf_curve, "-", label="TDF fit", color="C2", linewidth=1.5)
    ax.plot(r_fine, v_nfw_curve, "-.", label="NFW simple fit", color="C3", linewidth=1.5)
    ax.set_xlabel("r [kpc]")
    ax.set_ylabel("v [km/s]")
    ax.set_title(f"{data.galaxy_id} — {mode}\n(best BIC: {fit.best_model_by_bic})")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path

# TDF Benchmark Lab

A reproducible **benchmark and calibration suite** for testing the [Time Delay Field (TDF)](./INSTRUCTIONS.md) v0.8.1 framework against ΛCDM/GR recovery regimes, controlled stress tests, and (future) observational constraints.

**Repository:** [github.com/bahman2017/tdf-benchmark-lab](https://github.com/bahman2017/tdf-benchmark-lab)

| | |
|---|---|
| **Status** | Research benchmark scaffold |
| **Validation** | Not observationally validated |
| **Tests** | `pytest` suite (+ GitHub Actions on push) |

Before contributing, read [INSTRUCTIONS.md](./INSTRUCTIONS.md).

---

## Scientific disclaimer

This repository provides **calibration diagnostics** and **controlled benchmark tests** for a phenomenological TDF model. It does **not** constitute observational validation of TDF, does **not** prove TDF, and does **not** disprove ΛCDM or dark matter.

Passing a benchmark means **compatibility or recovery in that controlled setting only**. Every generated report includes an explicit warning banner (e.g. `NOT REAL OBSERVATIONAL DATA`).

---

## What this repository does

- Implements TDF v0.8.1 **phenomenological** equations in testable form (rotation, redshift formula, solar-system ε_τ, BH exterior ansatz).
- Runs **synthetic and teacher-based** benchmarks where ΛCDM/GR+DM is already successful (NFW-like rotation, GR-safe caps, BH limits, redshift bounds).
- Runs **stress tests** where ΛCDM is strained (core–cusp, rotation-curve diversity).
- Compares **baryon-only**, **TDF simple**, **TDF core proxy** (stress diagnostic), and **NFW simple** models with BIC-aware reporting.
- Provides a **pytest** suite and scripts that regenerate tables, reports, and figures under `outputs/`.

---

## What this repository does not claim

- That TDF is correct in nature or validated on real galaxies.
- That dark matter is disproven or unnecessary.
- That SPARC or other catalogs are used for calibration in the current release (real data is **postponed** to Phase 6).
- That black-hole formulas are full nonlinear GR solutions.

---

## Core equations tested

**Rotation (TDF simple, unchanged in this repo):**

```text
v_TDF²(r) = v_baryon²(r) + B · r / (r + r0)
τ̄_l(r) = A · log(1 + r/r0),   B = K_τ · A
```

**TDF core proxy (stress diagnostic only):**

```text
v_TDF_core²(r) = v_baryon²(r) + C · r² / (r² + rc_tau²)
```

**Redshift (formula):** `z_τ = K_τ · Δτ̄_l / c²`

**Solar system:** `ε_τ = Φ_τ / Φ_b` (configured scaffold)

**Black hole (ansatz):** `T_TDF = T_H · √(1 − (rc/rs)²)`, `r_nr = √(rs² − rc²)`

See [docs/SCIENTIFIC_ASSUMPTIONS.md](./docs/SCIENTIFIC_ASSUMPTIONS.md).

---

## Implemented benchmark phases

| Phase | Description |
|-------|-------------|
| 1 | Synthetic / demo rotation validation |
| 2 | SPARC ingestion **scaffold** only (no auto-download) |
| 3 | Baryon / TDF / NFW baseline comparison |
| 3B | NFW surrogate recovery |
| 3C | ΛCDM/GR combined benchmark scaffold |
| 4A | Expanded NFW/ΛCDM rotation benchmark (10 cases) |
| 4B | GR-safe local benchmark |
| 4C | Black-hole exterior GR-limit benchmark |
| 4D | Redshift / Doppler sanity benchmark |
| 5A | Core–cusp stress test |
| 5B | Rotation-curve diversity stress test |
| 5C | Same-τ multi-observable consistency (rotation → lensing/redshift proxies) |
| 5D | Covariant action consistency checks (NFW surrogate TDF fits) |
| 5E | CMB acoustic-scale compatibility (ΛCDM teacher background) |
| 5F | CMB-safe Hubble tension (late-time ε_τ background) |
| 5G | BAO/SNe late-time distance consistency |

Details: [docs/TEST_PLAN.md](./docs/TEST_PLAN.md), [docs/BENCHMARK_MANIFEST.md](./docs/BENCHMARK_MANIFEST.md).

---

## Installation

```bash
git clone https://github.com/bahman2017/tdf-benchmark-lab.git
cd tdf-benchmark-lab
python3.11 -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

Requires **Python ≥ 3.10** (CI uses 3.11).

---

## Quick start

```bash
pytest -q
python scripts/run_nfw_surrogate.py
python scripts/run_core_cusp_stress.py
```

Outputs are written under `outputs/` (regenerated locally; not all files are committed — see [outputs/README.md](./outputs/README.md)).

---

## Running tests

```bash
pytest
pytest -v tests/test_nfw_surrogate_expanded.py
```

---

## Running benchmark scripts

| Script | Phase |
|--------|-------|
| `python scripts/run_rotation.py` | 1 / 3 — **synthetic or demo** unless real metadata |
| `python scripts/run_kessence_rotation_benchmark.py` | 3K — baryon vs TDF-simple vs TDF K-essence (`a0`) |
| `python scripts/run_nfw_surrogate.py` | 3B / 4A |
| `python scripts/run_lcdm_benchmark.py` | 3C |
| `python scripts/run_gr_safe_benchmark.py` | 4B |
| `python scripts/run_black_hole_gr_benchmark.py` | 4C |
| `python scripts/run_redshift_sanity_benchmark.py` | 4D |
| `python scripts/run_core_cusp_stress.py` | 5A |
| `python scripts/run_rotation_diversity_stress.py` | 5B |
| `python scripts/run_same_tau_consistency.py` | 5C |
| `python scripts/run_covariant_action_checks.py` | 5D |
| `python scripts/run_cmb_acoustic_benchmark.py` | 5E |
| `python scripts/run_hubble_tension_benchmark.py` | 5F |
| `python scripts/run_bao_sne_distance_benchmark.py` | 5G |
| `python scripts/run_schrodinger_from_tdf.py` | 6A |
| `python scripts/run_dirac_spinor_limit.py` | 6B |
| `python scripts/run_entanglement_tau_geometry.py` | 6C |
| `python scripts/run_decoherence_tau_variance.py` | 6D |
| `python scripts/run_classical_metric_emergence.py` | 6E |
| `python scripts/run_born_rule_probability.py` | 6F |
| `python scripts/run_unified_microscopic_quantum_limit.py` | 6G |
| `python scripts/run_muon_g2_anomaly.py` | 6H — μon (g−2) / ε_τ phenomenology |
| `PYTHONPATH=src python3 scripts/run_kessence_source_viability.py` | 7A — K-essence source viability (spherical proxy; **not** observational validation) |
| `PYTHONPATH=src python3 scripts/run_disk_kessence_rotation.py` | 7B — Disk K-essence rotation (axisymmetric proxy; **not** real SPARC validation) |
| `PYTHONPATH=src python3 scripts/parse_sparc_real_data.py --input data/raw/16284118/Rotmod_LTG --output data/processed/sparc_rotation.csv` | 8A.0 — Parse real SPARC rotmod (**not** model validation) |
| `PYTHONPATH=src python3 scripts/run_sparc_real_calibration.py --input data/processed/sparc_rotation.csv --output-dir outputs` | 8A — Real SPARC calibration (**not** full observational validation) |
| `PYTHONPATH=src python3 scripts/run_sparc_parameter_audit.py --output-dir outputs` | 8A.1 — SPARC parameter / MOND baseline audit (**not** full observational validation; does not overwrite 8A tables) |
| `PYTHONPATH=src python3 scripts/run_sparc_real_calibration.py --corrected-mond true --run-id v0.20.2_corrected_mond_sparc_calibration --overwrite-run true` | 8A.2 — Corrected analytic MOND rerun → `outputs/runs/v0.20.2_corrected_mond_sparc_calibration/` (**not** full validation) |
| `PYTHONPATH=src python3 scripts/run_sparc_boundary_filtered_analysis.py` | SPARC Step 1 — boundary-filtered BIC analysis (**not** full validation; no new fitting) |
| `PYTHONPATH=src python3 scripts/run_sparc_galaxy_class_analysis.py` | SPARC Step 2 — galaxy-class BIC analysis (**not** full validation) |
| `PYTHONPATH=src python3 scripts/run_sparc_ml_robustness.py` | SPARC Step 3 — M/L robustness reruns (**not** full validation) |
| `PYTHONPATH=src python3 scripts/run_sparc_cored_halo_baseline.py` | SPARC Step 4 — Burkert / pseudo-isothermal baseline (**not** full validation) |
| `PYTHONPATH=src python3 scripts/run_sparc_tdf_parameter_stability.py` | SPARC Step 5 — TDF β/M stability diagnostics (**not** full validation) |
| `PYTHONPATH=src python3 scripts/run_sparc_residual_diagnostics.py` | SPARC Step 6 — residual diagnostics by radius/class (**not** full validation) |
| `PYTHONPATH=src python3 scripts/run_sparc_tau_inverse_design.py` | SPARC Step 6A — τ inverse design from DM phenomenology proxy (**not** full validation) |
| `PYTHONPATH=src python3 scripts/run_sparc_inverse_tau_response.py` | SPARC Step 6B — inverse-designed τ response benchmark (**not** full validation) |
| `PYTHONPATH=src python3 scripts/run_sparc_baryon_constrained_tau_law.py` | SPARC Step 6C — baryon-constrained τ coupling laws (**not** full validation) |
| `PYTHONPATH=src python3 scripts/run_sparc_tau_field_solver.py` | SPARC Step 6D — τ field-equation solver (**not** full validation) |
| `PYTHONPATH=src python3 scripts/run_sparc_tau_field_robustness.py` | SPARC Step 6E — τ field robustness audit (**not** full validation) |
| `PYTHONPATH=src python3 scripts/run_sparc_5d_projection_kernel.py` | SPARC Step 6F — 5D projection kernel (**theoretical proxy**, not full validation) |
| `PYTHONPATH=src python3 scripts/run_sparc_final_synthesis.py` | SPARC Step 7 — final synthesis & paper-ready section (**not** full validation) |

Full command list and appendix guidance: [docs/PAPER_APPENDIX_GUIDE.md](./docs/PAPER_APPENDIX_GUIDE.md).

---

## Output structure

### Versioned benchmark outputs (default since v0.20.2.1)

Production SPARC and benchmark runs write to isolated folders:

```text
outputs/runs/<run_id>/
  tables/
  reports/
  figures/
  metadata/run_manifest.json
```

Examples:

- `outputs/runs/v0.20.2_corrected_mond_sparc_calibration/` — corrected analytic MOND calibration
- `outputs/runs/sparc_step_1_boundary_filtered/` — Step 1 boundary-filtered BIC comparison (analysis only)
- `outputs/runs/sparc_step_2_galaxy_class_analysis/` — Step 2 galaxy-class BIC comparison (analysis only)
- `outputs/runs/sparc_step_3_mass_to_light_robustness/` — Step 3 M/L robustness reruns (analysis only)
- `outputs/runs/sparc_step_4_cored_halo_baseline/` — Step 4 Burkert / pseudo-isothermal baseline (analysis only)
- `outputs/runs/sparc_step_5_tdf_parameter_stability/` — Step 5 TDF parameter stability (analysis only)
- `outputs/runs/sparc_step_6_residual_diagnostics/` — Step 6 residual diagnostics (analysis only)
- `outputs/runs/sparc_step_6a_tau_inverse_design/` — Step 6A τ inverse-design diagnostics (analysis only)
- `outputs/runs/sparc_step_6b_inverse_tau_response/` — Step 6B inverse τ response benchmark (analysis only)
- `outputs/runs/sparc_step_6c_baryon_constrained_tau_law/` — Step 6C baryon-constrained τ laws (analysis only)
- `outputs/runs/sparc_step_6d_tau_field_equation_solver/` — Step 6D τ field-equation solver (analysis only)
- `outputs/runs/sparc_step_6e_tau_field_robustness/` — Step 6E τ field robustness audit (analysis only)
- `outputs/runs/sparc_step_6f_5d_projection_kernel/` — Step 6F 5D projection kernel (analysis only)
- `outputs/runs/sparc_final_synthesis/` — Step 7 final SPARC synthesis (analysis only)
- `outputs/runs/v0.20.1_sparc_parameter_audit/` — parameter / MOND baseline audit
- `outputs/runs/v0.20.0_sparc_initial_calibration/` — initial Phase 8A calibration

Legacy flat paths (`outputs/tables/`, etc.) are **not** written unless `--write-legacy-copy true`.

### Legacy layout (optional copy only)

```text
outputs/
  tables/     # optional legacy copy
  reports/
  figures/
```

Optional paper snapshots: [docs/results_snapshot/](./docs/results_snapshot/).

---

## Data policy

| Path | Policy |
|------|--------|
| `data/raw/` | User-supplied only; **not** in Git |
| `data/processed/` | Demo fixture may exist locally; **not** shipped as real SPARC |
| `data/synthetic/` | Synthetic generators / notes |

No automatic download. No fake “real observational” labels. See [docs/DATA_REQUIREMENTS.md](./docs/DATA_REQUIREMENTS.md).

---

## Reproducibility

1. Clone repo and install (above).
2. Run `pytest`.
3. Run benchmark scripts (see [docs/BENCHMARK_MANIFEST.md](./docs/BENCHMARK_MANIFEST.md)).
4. Compare regenerated CSV/MD under `outputs/` with appendix tables.

Record `git rev-parse HEAD` and Python version when citing results.

---

## How to cite

See [CITATION.cff](./CITATION.cff). If you use this benchmark suite, cite the associated TDF paper/preprint when available.

```bibtex
@software{tdf_benchmark_lab,
  author = {Masarrat, Bahman},
  title = {TDF Benchmark Lab: Controlled ΛCDM/GR Recovery and Stress Tests},
  year = {2026},
  url = {https://github.com/bahman2017/tdf-benchmark-lab}
}
```

---

## Roadmap

Implemented phases 1–5B; real observational calibration (Phase 6) and lensing/redshift pipelines deferred.

See [docs/ROADMAP.md](./docs/ROADMAP.md) and [docs/LCDM_COMPATIBILITY_STRATEGY.md](./docs/LCDM_COMPATIBILITY_STRATEGY.md).

---

## License

[MIT License](./LICENSE) — Copyright (c) 2026 Bahman Masarrat.

---

## Repository layout

```text
tdf-benchmark-lab/
  README.md  INSTRUCTIONS.md  LICENSE  CITATION.cff  CHANGELOG.md
  pyproject.toml  requirements.txt  .gitignore
  configs/  data/  docs/  scripts/  src/tdf_obs/  tests/  outputs/
  .github/workflows/tests.yml
```

# Changelog

All notable changes to this project are documented here.

Format: versions track **benchmark-lab releases** (code + docs), not a claim that TDF theory is validated.

---

## SPARC Step 6F — 5D projection kernel (2026-05-17)

**Summary:** Nonlocal projection kernel S_τ(R)=λ₀∫K(R,R')Σ_b(R')R'dR' with kernel families A–F; compares to Step 6D/6E. Theoretical proxy only.

**New:** `sparc_5d_projection_kernel.py`, `run_sparc_5d_projection_kernel.py`, `tests/test_sparc_5d_projection_kernel.py`.

**Output:** `outputs/runs/sparc_step_6f_5d_projection_kernel/`

---

## SPARC Step 6E — τ field robustness audit (2026-05-17)

**Summary:** Audits Step 6D for λ_b stability, global-λ test, M/L regimes, boundary-filtered BIC, and zone residuals. Robustness diagnostic only.

**New:** `sparc_tau_field_robustness.py`, `run_sparc_tau_field_robustness.py`, `tests/test_sparc_tau_field_robustness.py`.

**Output:** `outputs/runs/sparc_step_6e_tau_field_robustness/`

---

## SPARC Step 6D — τ field-equation solver (2026-05-17)

**Summary:** Radial/cylindrical proxy field equation with cumulative source form; tests μ variants A–E via per-radius σ′ root solve. Numerical field diagnostic only.

**New:** `sparc_tau_field_solver.py`, `run_sparc_tau_field_solver.py`, `tests/test_sparc_tau_field_solver.py`.

**Output:** `outputs/runs/sparc_step_6d_tau_field_equation_solver/`

---

## SPARC Step 6C — Baryon-constrained τ coupling law (2026-05-17)

**Summary:** Global β_eff laws (A–D) with R_core and cohort-fitted parameters vs baselines and Step 6B; fewer per-galaxy DOF than Variant C. Formula-revision diagnostics only.

**New:** `sparc_baryon_constrained_tau_law.py`, `run_sparc_baryon_constrained_tau_law.py`, `tests/test_sparc_baryon_constrained_tau_law.py`.

**Output:** `outputs/runs/sparc_step_6c_baryon_constrained_tau_law/`

---

## SPARC Step 6B — Inverse-designed τ response benchmark (2026-05-17)

**Summary:** Benchmarks candidate inverse τ response (global/class/baryon-feature β with R_core) vs old TDF k-essence, NFW, MOND, and pseudo-isothermal on SPARC. Formula-revision benchmark only.

**New:** `sparc_inverse_tau_response.py`, `run_sparc_inverse_tau_response.py`, `tests/test_sparc_inverse_tau_response.py`.

**Output:** `outputs/runs/sparc_step_6b_inverse_tau_response/`

---

## SPARC Step 6A — τ inverse design (2026-05-17)

**Summary:** Reverse-engineers required `a_τ(r)` and β_eff proxies from SPARC baryonic residuals using corrected-MOND calibration Υ. Cored-τ and class-dependent coupling diagnostics only; no new fits.

**New:** `sparc_tau_inverse_design.py`, `run_sparc_tau_inverse_design.py`, `tests/test_sparc_tau_inverse_design.py`.

**Output:** `outputs/runs/sparc_step_6a_tau_inverse_design/`

---

## SPARC Step 7 — Final SPARC synthesis (2026-05-17)

**Summary:** Aggregates corrected-MOND calibration and Steps 1–6 into summary tables, decision matrix, claim level (0–3), recommendation, and paper-ready rotation section. No new fits.

**New:** `sparc_final_synthesis.py`, `run_sparc_final_synthesis.py`, `tests/test_sparc_final_synthesis.py`.

**Output:** `outputs/runs/sparc_final_synthesis/`

---

## SPARC Step 6 — Residual diagnostics (2026-05-17)

**Summary:** Point- and galaxy-level weighted residuals for baryon, corrected MOND, NFW, and TDF; inner/middle/outer zones and TDF vs NFW failure modes. Analysis only.

**New:** `sparc_residual_diagnostics.py`, `run_sparc_residual_diagnostics.py`, `tests/test_sparc_residual_diagnostics.py`.

**Output:** `outputs/runs/sparc_step_6_residual_diagnostics/`

---

## SPARC Step 5 — TDF parameter stability (2026-05-17)

**Summary:** Analyzes per-galaxy TDF β/M and M/L from corrected-MOND calibration; correlations, boundary hits, outliers, and approximate global-β BIC diagnostic. Analysis only.

**New:** `sparc_tdf_parameter_stability.py`, `run_sparc_tdf_parameter_stability.py`, `tests/test_sparc_tdf_parameter_stability.py`.

**Output:** `outputs/runs/sparc_step_5_tdf_parameter_stability/`

---

## SPARC Step 4 — Cored halo baseline (2026-05-17)

**Summary:** Adds Burkert and pseudo-isothermal halo baselines alongside NFW and corrected MOND on SPARC; compares TDF vs cored and cuspy halos. Analysis only.

**New:** `sparc_cored_halo_baseline.py`, `run_sparc_cored_halo_baseline.py`, `tests/test_sparc_cored_halo_baseline.py`; Burkert and pseudo-isothermal helpers in `dark_matter.py`.

**Output:** `outputs/runs/sparc_step_4_cored_halo_baseline/`

---

## SPARC Step 3 — Mass-to-light robustness (2026-05-16)

**Summary:** Reruns corrected-MOND SPARC fits under fixed, narrow, standard, and shared Υ priors; compares TDF competitiveness vs M/L flexibility. Analysis only.

**New:** `sparc_ml_robustness.py`, `run_sparc_ml_robustness.py`, `tests/test_sparc_ml_robustness.py`.

**Output:** `outputs/runs/sparc_step_3_mass_to_light_robustness/`

---

## SPARC Step 2 — Galaxy-class model comparison (2026-05-16)

**Summary:** BIC / reduced χ² comparison by dwarf / intermediate / massive (v_max proxy) on corrected-MOND calibration. Analysis only.

**New:** `sparc_galaxy_class_analysis.py`, `run_sparc_galaxy_class_analysis.py`, `tests/test_sparc_galaxy_class_analysis.py`.

**Output:** `outputs/runs/sparc_step_2_galaxy_class_analysis/`

---

## SPARC Step 1 — Boundary-filtered model comparison (2026-05-16)

**Summary:** Analysis-only comparison of corrected-MOND SPARC BIC outcomes across `all_galaxies`, `exclude_any_nfw_or_tdf_boundary_hit`, and `exclude_severe_boundary_pressure` filters. **Not** observational validation.

**New:** `src/tdf_obs/validation/sparc_boundary_filtered_analysis.py`, `scripts/run_sparc_boundary_filtered_analysis.py`, `tests/test_sparc_boundary_filtered_analysis.py`.

**Output run:** `outputs/runs/sparc_step_1_boundary_filtered/`

---

## v0.20.2.1 — Versioned output directory hygiene (2026-05-16)

**Summary:** Benchmark outputs now default to `outputs/runs/<run_id>/{tables,reports,figures,metadata}/` with `run_manifest.json`, production guards for pytest/tmp inputs, and optional `--write-legacy-copy`. No physics changes.

**New:** `src/tdf_obs/utils/run_outputs.py`; CLI flags `--run-id`, `--versioned-output`, `--overwrite-run`, `--write-legacy-copy` on SPARC scripts.

**Production run paths:** `v0.20.0_sparc_initial_calibration`, `v0.20.1_sparc_parameter_audit`, `v0.20.2_corrected_mond_sparc_calibration`.

---

## v0.20.2 — Corrected MOND SPARC rerun / Phase 8A.2 (2026-05-16)

**Summary:** Rerun real SPARC calibration with **analytic** simple-μ MOND (`corrected_mond` model); new outputs use `_corrected_mond` suffix without overwriting Phase 8A tables. **Not** observational validation.

**Code:** `sparc_real_calibration.py` — `mond_g_baryon_analytic`, `v_mond_analytic`, `check_mond_activity`; `--corrected-mond` CLI flag.

**Outputs:** `sparc_real_calibration_summary_corrected_mond.csv`, `sparc_model_comparison_by_galaxy_corrected_mond.csv`, `sparc_real_calibration_report_corrected_mond.md`, figures `*_corrected_mond.png`.

---

## v0.20.1 — SPARC parameter audit / Phase 8A.1 (2026-05-16)

**Summary:** Audits Phase 8A calibration outputs for **MOND baseline activity** (analytic simple-μ, unit-consistent a₀), **parameter boundary hits**, **BIC/AIC parameter-count fairness**, and **TDF vs NFW** comparison before/after excluding boundary-limited galaxies. **Does not** modify prior benchmark tables or claim observational validation.

**New code:** `src/tdf_obs/validation/sparc_parameter_audit.py`, `scripts/run_sparc_parameter_audit.py`, `tests/test_sparc_parameter_audit.py`.

**Outputs:** `sparc_parameter_audit_summary.csv`, `sparc_parameter_boundary_flags.csv`, `sparc_mond_baseline_audit.csv`, `sparc_parameter_count_audit.csv`, `sparc_parameter_audit_report.md`, figures `sparc_mond_vs_baryon_boost.png`, `sparc_parameter_boundary_counts.png`, `sparc_delta_bic_tdf_nfw_audited.png`, `sparc_audit_by_galaxy_class.png`.

**Banner:** `SPARC PARAMETER AUDIT — NOT FULL OBSERVATIONAL VALIDATION`

---

## v0.21.0 — Real SPARC calibration / Phase 8A (2026-05-16)

**Summary:** Per-galaxy fits on **real** SPARC rotation curves comparing baryon-only (Υ), NFW, MOND (μ=x/(1+x)), and TDF K-essence disk proxy; BIC model selection. **Not** full observational validation or dark-matter replacement.

**New code:** `src/tdf_obs/validation/sparc_real_calibration.py`, `scripts/run_sparc_real_calibration.py`, `tests/test_sparc_real_calibration.py`.

**Outputs:** `sparc_real_calibration_summary.csv`, `sparc_model_comparison_by_galaxy.csv`, `sparc_real_calibration_report.md`, figures under `outputs/figures/sparc_*`.

---

## v0.20.0 — SPARC raw data parser / Phase 8A.0 (2026-05-16)

**Summary:** Robust parser for **real** SPARC `Rotmod_LTG/*_rotmod.dat` files → `data/processed/sparc_rotation.csv` with schema validation and parser report. **Not** model fitting or observational validation.

**New code:** `src/tdf_obs/data/sparc_parser.py`, `scripts/parse_sparc_real_data.py`, `tests/test_sparc_parser.py`.

**Outputs:** `data/processed/sparc_rotation.csv` (local), `outputs/reports/sparc_parser_report.md`.

---

## v0.19.0 — Disk K-essence rotation benchmark / Phase 7B (2026-05-16)

**Summary:** Axisymmetric **thin-disk** extension of Phase 7A: cylindrical K-essence equation with conformal trace source Σ_b(R), Freeman-style baryon rotation proxy, and eight synthetic disk cases (pass + expected-fail).

**Not claimed:** real SPARC validation, dark-matter replacement, lensing consistency.

**New code:** `src/tdf_obs/validation/disk_kessence_rotation.py`, `scripts/run_disk_kessence_rotation.py`, `tests/test_disk_kessence_rotation.py`.

**Outputs (gitignored):** `disk_kessence_rotation_summary.csv`, `disk_kessence_rotation_report.md`, five `disk_kessence_*.png` figures.

---

## v0.18.1 — K-essence source viability expected-fail fix / Phase 7A.1 (2026-05-16)

**Summary:** Correct benchmark scoring for expected-fail cases (`failure_detected`, `failure_reason`, `expected_failure_matched`); stability column includes gradient bound; report shows **Expected outcomes matched: 7/7**.

---

## v0.18.0 — K-essence source viability benchmark / Phase 7A (2026-05-16)

**Summary:** Spherical **source-structure viability** check: pure disformal coupling with static dust gives **zero** spatial τ source (`div J = 0`), while a **conformal trace** correction `S_b ∝ (β/M) ρ_b` can drive K-essence radial profiles with deep-MOND-like `σ' ∝ 1/r` and flat rotation **proxies** in controlled cases.

**Theoretical motivation (calibration only):** Addresses the fatal flaw that static baryons cannot source a spatial τ halo through disformal coupling alone. This does **not** derive the conformal factor from 5D geometry, **not** observational validation, and **not** a replacement for SPARC fitting.

**New code:**

| Path | Purpose |
|------|---------|
| `src/tdf_obs/validation/kessence_source_viability.py` | Baryon profiles, disformal vs conformal sources, μ models, seven benchmark cases |
| `scripts/run_kessence_source_viability.py` | CLI |
| `tests/test_kessence_source_viability.py` | Source zero/nonzero, slopes, fail detection, pipeline |

**Documentation:** `docs/ROADMAP.md`, `docs/TEST_PLAN.md` (Phase 7A), `docs/BENCHMARK_MANIFEST.md`, `README.md`.

**Outputs (gitignored):** `kessence_source_viability_summary.csv`, `kessence_source_viability_report.md`, `kessence_*.png` (five figures).

---

## v0.17.0 — Muon g-2 anomaly benchmark / Phase 6H (2026-05-16)

**Summary:** Phenomenological **precision QED coupling** check: sub-Compton τ fluctuations with ε_τ = (σ_τ/ℓ_τ)² yield Δa_μ = (α/2π)ε_τ, compared to the Fermilab/BNL consensus Δa_μ^exp ≈ 2.51×10⁻¹⁰.

**Theoretical motivation (calibration only):** TDF posits α_eff = α(1 + ε_τ) at order-of-magnitude level, giving a one-loop proxy shift in the muon anomalous magnetic moment. This does **not** derive the anomaly from a full TDF–QED Lagrangian.

**New code:**

| Path | Purpose |
|------|---------|
| `src/tdf_obs/validation/muon_g2_anomaly.py` | `calculate_delta_a_mu`, `estimate_epsilon_tau`, Compton-scale σ_τ estimate |
| `scripts/run_muon_g2_anomaly.py` | CLI |
| `tests/test_muon_g2_anomaly.py` | Reference ε_τ match; positive-input guards; pipeline |

**Documentation:** `docs/ROADMAP.md`, `docs/TEST_PLAN.md` (Phase 6H), `docs/BENCHMARK_MANIFEST.md`, `README.md`.

**Outputs (gitignored):** `muon_g2_anomaly_summary.csv`, `muon_g2_anomaly_report.md`, `muon_g2_epsilon_sweep.png`.

---

## v0.9.0 — K-essence rotation benchmark (2026-05-16)

**Summary:** Galaxy-scale **phenomenological** extension testing the TDF non-linear Horndeski K-essence **effective** rotation limit against baryon-only and TDF-simple baselines.

**Theoretical motivation (calibration only):** In the deep non-linear K-essence limit, the circular-speed ansatz

\[
v^2(r) = v_b^2(r) + v_b(r)\sqrt{a_0 r}
\]

provides a one-parameter (\(a_0\)) candidate for MOND-like outer enhancement relative to baryons alone. This is implemented for **controlled fitting and BIC comparison**, not as a full covariant Horndeski derivation or observational proof.

**New / updated code:**

| Path | Purpose |
|------|---------|
| `src/tdf_obs/models/rotation.py` | `v2_tdf_kessence`, `v_tdf_kessence` |
| `src/tdf_obs/fitting/fit_rotation.py` | `fit_kessence_galaxy_rotation`, `KessenceFitResult` |
| `src/tdf_obs/validation/kessence_rotation_benchmark.py` | Three-model comparison, synthetic K-essence generator, report/plot |
| `scripts/run_kessence_rotation_benchmark.py` | CLI entry point |
| `tests/test_rotation_model.py` | K-essence finite values; `a0=0` → baryon-only |
| `tests/test_kessence_rotation_benchmark.py` | Pipeline, banner, recovery tests |

**Documentation:** `docs/ROADMAP.md`, `docs/TEST_PLAN.md` (Phase 3K), `docs/BENCHMARK_MANIFEST.md`, `README.md`, `INSTRUCTIONS.md` (§25 continuous documentation rule).

**Outputs (generated, gitignored):** `outputs/tables/kessence_rotation_benchmark_summary.csv`, `outputs/reports/kessence_rotation_benchmark_report.md`, `outputs/figures/*_kessence_rotation_benchmark.png`.

---

## v0.1.0 — Benchmark scaffold release

- Synthetic / demo rotation validation (Phase 1)
- SPARC ingestion scaffold only; real observational calibration intentionally postponed (Phase 2)
- Baryon / TDF / NFW baseline comparison (Phase 3)
- NFW surrogate recovery (Phase 3B)
- ΛCDM / GR benchmark recovery scaffold (Phase 3C)
- Expanded NFW / ΛCDM rotation benchmark (Phase 4A)
- GR-safe local benchmark (Phase 4B)
- Black-hole exterior GR-limit benchmark (Phase 4C)
- Redshift / Doppler sanity benchmark (Phase 4D)
- Core–cusp stress test (Phase 5A)
- Rotation-curve diversity stress test (Phase 5B)
- Documentation for GitHub, paper appendix, and reproducibility (`PAPER_APPENDIX_GUIDE.md`, `BENCHMARK_MANIFEST.md`)
- Pytest suite and GitHub Actions CI workflow

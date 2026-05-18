# Benchmark manifest

Reproducibility index for all implemented benchmark pipelines.  
**Status:** controlled benchmarks only — **not** observational validation.

| Benchmark | Script | Output table | Output report | Figures | Status |
|-----------|--------|--------------|---------------|---------|--------|
| Rotation demo / synthetic | `scripts/run_rotation.py` | `outputs/tables/rotation_fit_summary.csv` | `outputs/reports/rotation_report.md` | `outputs/figures/<galaxy_id>_rotation.png` | ✅ Implemented |
| K-essence rotation (3K) | `scripts/run_kessence_rotation_benchmark.py` | `outputs/tables/kessence_rotation_benchmark_summary.csv` | `outputs/reports/kessence_rotation_benchmark_report.md` | `outputs/figures/<galaxy_id>_kessence_rotation_benchmark.png` | ✅ Implemented |
| NFW surrogate (4A) | `scripts/run_nfw_surrogate.py` | `outputs/tables/nfw_surrogate_fit_summary.csv` | `outputs/reports/nfw_surrogate_report.md` | `outputs/figures/nfw_surrogate_<case>.png` | ✅ Implemented |
| ΛCDM combined scaffold (3C) | `scripts/run_lcdm_benchmark.py` | `outputs/tables/lcdm_benchmark_summary.csv` | `outputs/reports/lcdm_benchmark_report.md` | — | ✅ Implemented |
| GR-safe local (4B) | `scripts/run_gr_safe_benchmark.py` | `outputs/tables/gr_safe_benchmark_summary.csv` | `outputs/reports/gr_safe_benchmark_report.md` | — | ✅ Implemented |
| Black-hole GR-limit (4C) | `scripts/run_black_hole_gr_benchmark.py` | `outputs/tables/black_hole_gr_benchmark_summary.csv` | `outputs/reports/black_hole_gr_benchmark_report.md` | — | ✅ Implemented |
| Redshift sanity (4D) | `scripts/run_redshift_sanity_benchmark.py` | `outputs/tables/redshift_sanity_benchmark_summary.csv` | `outputs/reports/redshift_sanity_benchmark_report.md` | — | ✅ Implemented |
| Core–cusp stress (5A) | `scripts/run_core_cusp_stress.py` | `outputs/tables/core_cusp_stress_summary.csv` | `outputs/reports/core_cusp_stress_report.md` | `outputs/figures/core_cusp_<case>.png` | ✅ Implemented |
| Rotation diversity (5B) | `scripts/run_rotation_diversity_stress.py` | `outputs/tables/rotation_diversity_stress_summary.csv` | `outputs/reports/rotation_diversity_stress_report.md` | `outputs/figures/rotation_diversity_<case>.png` | ✅ Implemented |
| Same-τ multi-observable (5C) | `scripts/run_same_tau_consistency.py` | `outputs/tables/same_tau_consistency_summary.csv` | `outputs/reports/same_tau_consistency_report.md` | `outputs/figures/same_tau_<case>_*.png` | ✅ Implemented |
| Covariant action checks (5D) | `scripts/run_covariant_action_checks.py` | `outputs/tables/covariant_action_checks_summary.csv` | `outputs/reports/covariant_action_checks_report.md` | — | ✅ Implemented |
| CMB acoustic scale (5E) | `scripts/run_cmb_acoustic_benchmark.py` | `outputs/tables/cmb_acoustic_benchmark_summary.csv` | `outputs/reports/cmb_acoustic_benchmark_report.md` | `outputs/figures/cmb_acoustic_epsilon_tau_cases.png` | ✅ Implemented |
| CMB-safe Hubble (5F) | `scripts/run_hubble_tension_benchmark.py` | `outputs/tables/hubble_tension_benchmark_summary.csv` | `outputs/reports/hubble_tension_benchmark_report.md` | `outputs/figures/hubble_tension_*.png` | ✅ Implemented |
| BAO/SNe distance (5G) | `scripts/run_bao_sne_distance_benchmark.py` | `outputs/tables/bao_sne_distance_benchmark_summary.csv` | `outputs/reports/bao_sne_distance_benchmark_report.md` | `outputs/figures/bao_sne_distance_*.png` | ✅ Implemented |
| Schrödinger-from-TDF (6A) | `scripts/run_schrodinger_from_tdf.py` | `outputs/tables/schrodinger_from_tdf_summary.csv` | `outputs/reports/schrodinger_from_tdf_report.md` | `outputs/figures/schrodinger_*_residual.png` | ✅ Implemented |
| Dirac / spinor limit (6B) | `scripts/run_dirac_spinor_limit.py` | `outputs/tables/dirac_spinor_limit_summary.csv` | `outputs/reports/dirac_spinor_limit_report.md` | `outputs/figures/dirac_dispersion_relation.png`, `compact_tau_mass_ladder.png` | ✅ Implemented |
| Entanglement / τ geometry (6C) | `scripts/run_entanglement_tau_geometry.py` | `outputs/tables/entanglement_tau_geometry_summary.csv` | `outputs/reports/entanglement_tau_geometry_report.md` | `outputs/figures/entanglement_*.png`, `tau_nonseparability_scores.png` | ✅ Implemented |
| Decoherence / τ variance (6D) | `scripts/run_decoherence_tau_variance.py` | `outputs/tables/decoherence_tau_variance_summary.csv` | `outputs/reports/decoherence_tau_variance_report.md` | `outputs/figures/decoherence_*.png` | ✅ Implemented |
| Classical metric emergence (6E) | `scripts/run_classical_metric_emergence.py` | `outputs/tables/classical_metric_emergence_summary.csv` | `outputs/reports/classical_metric_emergence_report.md` | `outputs/figures/classical_metric_*.png` | ✅ Implemented |
| Born-rule probability (6F) | `scripts/run_born_rule_probability.py` | `outputs/tables/born_rule_probability_summary.csv` | `outputs/reports/born_rule_probability_report.md` | `outputs/figures/born_rule_*.png` | ✅ Implemented |
| Unified microscopic quantum limit (6G) | `scripts/run_unified_microscopic_quantum_limit.py` | `outputs/tables/unified_microscopic_quantum_limit_matrix.csv` | `outputs/reports/unified_microscopic_quantum_limit_report.md` | `outputs/figures/unified_microscopic_*.png` | ✅ Implemented |
| Muon g-2 anomaly (6H) | `scripts/run_muon_g2_anomaly.py` | `outputs/tables/muon_g2_anomaly_summary.csv` | `outputs/reports/muon_g2_anomaly_report.md` | `outputs/figures/muon_g2_epsilon_sweep.png` | ✅ Implemented |
| K-essence source viability (7A) | `scripts/run_kessence_source_viability.py` | `outputs/tables/kessence_source_viability_summary.csv` | `outputs/reports/kessence_source_viability_report.md` | `outputs/figures/kessence_*.png` (5 figures) | ✅ Implemented |
| Disk K-essence rotation (7B) | `scripts/run_disk_kessence_rotation.py` | `outputs/tables/disk_kessence_rotation_summary.csv` | `outputs/reports/disk_kessence_rotation_report.md` | `outputs/figures/disk_kessence_*.png` (5 figures) | ✅ Implemented |
| SPARC raw parser (8A.0) | `scripts/parse_sparc_real_data.py` | `data/processed/sparc_rotation.csv` | `outputs/reports/sparc_parser_report.md` | — | ✅ Implemented |
| Real SPARC calibration (8A) | `scripts/run_sparc_real_calibration.py --run-id v0.20.0_sparc_initial_calibration` | `outputs/runs/v0.20.0_sparc_initial_calibration/tables/*.csv` | `outputs/runs/v0.20.0_sparc_initial_calibration/reports/sparc_real_calibration_report.md` | `outputs/runs/.../figures/*.png` | ✅ Implemented |
| SPARC Step 7 final synthesis | `scripts/run_sparc_final_synthesis.py` | `outputs/runs/sparc_final_synthesis/tables/*.csv` | `outputs/runs/sparc_final_synthesis/reports/final_sparc_synthesis_report.md` | `outputs/runs/sparc_final_synthesis/figures/*.png` | ✅ Implemented |
| SPARC Step 6F 5D projection kernel | `scripts/run_sparc_5d_projection_kernel.py` | `outputs/runs/sparc_step_6f_5d_projection_kernel/tables/*.csv` | `outputs/runs/sparc_step_6f_5d_projection_kernel/reports/projection_kernel_report.md` | `outputs/runs/sparc_step_6f_5d_projection_kernel/figures/*.png` | ✅ Implemented |
| SPARC Step 6E τ field robustness | `scripts/run_sparc_tau_field_robustness.py` | `outputs/runs/sparc_step_6e_tau_field_robustness/tables/*.csv` | `outputs/runs/sparc_step_6e_tau_field_robustness/reports/tau_field_robustness_report.md` | `outputs/runs/sparc_step_6e_tau_field_robustness/figures/*.png` | ✅ Implemented |
| SPARC Step 6D τ field solver | `scripts/run_sparc_tau_field_solver.py` | `outputs/runs/sparc_step_6d_tau_field_equation_solver/tables/*.csv` | `outputs/runs/sparc_step_6d_tau_field_equation_solver/reports/tau_field_solver_report.md` | `outputs/runs/sparc_step_6d_tau_field_equation_solver/figures/*.png` | ✅ Implemented |
| SPARC Step 6C baryon-constrained τ law | `scripts/run_sparc_baryon_constrained_tau_law.py` | `outputs/runs/sparc_step_6c_baryon_constrained_tau_law/tables/*.csv` | `outputs/runs/sparc_step_6c_baryon_constrained_tau_law/reports/tau_law_report.md` | `outputs/runs/sparc_step_6c_baryon_constrained_tau_law/figures/*.png` | ✅ Implemented |
| SPARC Step 6B inverse τ benchmark | `scripts/run_sparc_inverse_tau_response.py` | `outputs/runs/sparc_step_6b_inverse_tau_response/tables/*.csv` | `outputs/runs/sparc_step_6b_inverse_tau_response/reports/inverse_tau_response_report.md` | `outputs/runs/sparc_step_6b_inverse_tau_response/figures/*.png` | ✅ Implemented |
| SPARC Step 6A τ inverse design | `scripts/run_sparc_tau_inverse_design.py` | `outputs/runs/sparc_step_6a_tau_inverse_design/tables/*.csv` | `outputs/runs/sparc_step_6a_tau_inverse_design/reports/tau_inverse_design_report.md` | `outputs/runs/sparc_step_6a_tau_inverse_design/figures/*.png` | ✅ Implemented |
| SPARC Step 6 residual diagnostics | `scripts/run_sparc_residual_diagnostics.py` | `outputs/runs/sparc_step_6_residual_diagnostics/tables/*.csv` | `outputs/runs/sparc_step_6_residual_diagnostics/reports/residual_diagnostics_report.md` | `outputs/runs/sparc_step_6_residual_diagnostics/figures/*.png` | ✅ Implemented |
| SPARC Step 5 TDF parameter stability | `scripts/run_sparc_tdf_parameter_stability.py` | `outputs/runs/sparc_step_5_tdf_parameter_stability/tables/*.csv` | `outputs/runs/sparc_step_5_tdf_parameter_stability/reports/tdf_parameter_stability_report.md` | `outputs/runs/sparc_step_5_tdf_parameter_stability/figures/*.png` | ✅ Implemented |
| SPARC Step 4 cored halo baseline | `scripts/run_sparc_cored_halo_baseline.py` | `outputs/runs/sparc_step_4_cored_halo_baseline/tables/*.csv` | `outputs/runs/sparc_step_4_cored_halo_baseline/reports/cored_halo_baseline_report.md` | `outputs/runs/sparc_step_4_cored_halo_baseline/figures/*.png` | ✅ Implemented |
| SPARC Step 3 M/L robustness | `scripts/run_sparc_ml_robustness.py` | `outputs/runs/sparc_step_3_mass_to_light_robustness/tables/*.csv` | `outputs/runs/sparc_step_3_mass_to_light_robustness/reports/ml_robustness_report.md` | `outputs/runs/sparc_step_3_mass_to_light_robustness/figures/*.png` | ✅ Implemented |
| SPARC Step 2 galaxy-class analysis | `scripts/run_sparc_galaxy_class_analysis.py` | `outputs/runs/sparc_step_2_galaxy_class_analysis/tables/*.csv` | `outputs/runs/sparc_step_2_galaxy_class_analysis/reports/galaxy_class_analysis_report.md` | `outputs/runs/sparc_step_2_galaxy_class_analysis/figures/*.png` | ✅ Implemented |
| SPARC Step 1 boundary-filtered analysis | `scripts/run_sparc_boundary_filtered_analysis.py` | `outputs/runs/sparc_step_1_boundary_filtered/tables/*.csv` | `outputs/runs/sparc_step_1_boundary_filtered/reports/boundary_filtered_analysis_report.md` | `outputs/runs/sparc_step_1_boundary_filtered/figures/*.png` | ✅ Implemented |
| Corrected MOND SPARC rerun (8A.2) | `scripts/run_sparc_real_calibration.py --corrected-mond true --run-id v0.20.2_corrected_mond_sparc_calibration` | `outputs/runs/v0.20.2_corrected_mond_sparc_calibration/tables/*.csv` | `outputs/runs/v0.20.2_corrected_mond_sparc_calibration/reports/sparc_real_calibration_report.md` | `outputs/runs/.../figures/*.png` | ✅ Implemented |
| SPARC parameter audit (8A.1) | `scripts/run_sparc_parameter_audit.py --run-id v0.20.1_sparc_parameter_audit` | `outputs/runs/v0.20.1_sparc_parameter_audit/tables/*.csv` | `outputs/runs/v0.20.1_sparc_parameter_audit/reports/sparc_parameter_audit_report.md` | `outputs/runs/.../figures/*.png` | ✅ Implemented |

## Regenerate all primary benchmarks

```bash
pytest
python scripts/run_rotation.py
python scripts/run_kessence_rotation_benchmark.py
python scripts/run_nfw_surrogate.py
python scripts/run_lcdm_benchmark.py
python scripts/run_gr_safe_benchmark.py
python scripts/run_black_hole_gr_benchmark.py
python scripts/run_redshift_sanity_benchmark.py
python scripts/run_core_cusp_stress.py
python scripts/run_rotation_diversity_stress.py
python scripts/run_same_tau_consistency.py
python scripts/run_covariant_action_checks.py
python scripts/run_cmb_acoustic_benchmark.py
python scripts/run_hubble_tension_benchmark.py
python scripts/run_bao_sne_distance_benchmark.py
python scripts/run_schrodinger_from_tdf.py
python scripts/run_dirac_spinor_limit.py
python scripts/run_entanglement_tau_geometry.py
python scripts/run_decoherence_tau_variance.py
python scripts/run_classical_metric_emergence.py
python scripts/run_born_rule_probability.py
python scripts/run_unified_microscopic_quantum_limit.py
python scripts/run_muon_g2_anomaly.py
PYTHONPATH=src python3 scripts/run_kessence_source_viability.py
PYTHONPATH=src python3 scripts/run_disk_kessence_rotation.py
PYTHONPATH=src python3 scripts/parse_sparc_real_data.py \
  --input data/raw/16284118/Rotmod_LTG \
  --output data/processed/sparc_rotation.csv
PYTHONPATH=src python3 scripts/run_sparc_real_calibration.py \
  --input data/processed/sparc_rotation.csv --output-dir outputs
```

See [PAPER_APPENDIX_GUIDE.md](./PAPER_APPENDIX_GUIDE.md) for appendix wording and per-phase notes.

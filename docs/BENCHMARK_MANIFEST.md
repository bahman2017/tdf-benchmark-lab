# Benchmark manifest

Reproducibility index for all implemented benchmark pipelines.  
**Status:** controlled benchmarks only — **not** observational validation.

| Benchmark | Script | Output table | Output report | Figures | Status |
|-----------|--------|--------------|---------------|---------|--------|
| Rotation demo / synthetic | `scripts/run_rotation.py` | `outputs/tables/rotation_fit_summary.csv` | `outputs/reports/rotation_report.md` | `outputs/figures/<galaxy_id>_rotation.png` | ✅ Implemented |
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

## Regenerate all primary benchmarks

```bash
pytest
python scripts/run_rotation.py
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
```

See [PAPER_APPENDIX_GUIDE.md](./PAPER_APPENDIX_GUIDE.md) for appendix wording and per-phase notes.

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
```

See [PAPER_APPENDIX_GUIDE.md](./PAPER_APPENDIX_GUIDE.md) for appendix wording and per-phase notes.

# Benchmark outputs

This directory holds **generated** tables, reports, and figures from benchmark scripts.

## Policy

- Files here are produced by `scripts/run_*.py` and are **not** observational validation.
- By default, generated files under `figures/`, `tables/`, and `reports/` are **not** committed to Git (see root `.gitignore`).
- Regenerate locally after cloning:

```bash
pytest
python scripts/run_nfw_surrogate.py
python scripts/run_lcdm_benchmark.py
python scripts/run_gr_safe_benchmark.py
python scripts/run_black_hole_gr_benchmark.py
python scripts/run_redshift_sanity_benchmark.py
python scripts/run_core_cusp_stress.py
python scripts/run_rotation_diversity_stress.py
python scripts/run_same_tau_consistency.py
python scripts/run_covariant_action_checks.py
python scripts/run_rotation.py   # synthetic/demo unless real metadata confirmed
```

## Paper / appendix snapshots

Selected frozen outputs for publication may be stored under `docs/results_snapshot/` (see that README). Those are still **controlled benchmarks**, not real-sky validation.

## Layout

| Subfolder | Contents |
|-----------|----------|
| `tables/` | CSV summaries per benchmark |
| `reports/` | Markdown reports with warning banners |
| `figures/` | PNG plots per case or galaxy |

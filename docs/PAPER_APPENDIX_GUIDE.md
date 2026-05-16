# Paper appendix and reproducibility guide

Use this document when citing **TDF Benchmark Lab** in a paper, preprint, or reviewer response.

Repository: [https://github.com/bahman2017/tdf-benchmark-lab](https://github.com/bahman2017/tdf-benchmark-lab)

---

## Scientific caution (recommended wording)

> The benchmark suite provides **calibration diagnostics** and **controlled synthetic tests** for a phenomenological Time Delay Field (TDF) model. Results reported here are **compatibility and stress-test outcomes**, not observational validation of TDF. The suite does **not** prove TDF, does **not** disprove ΛCDM, and does **not** establish that dark matter is unnecessary.

Per-benchmark banners in generated reports must be preserved (e.g. `NOT REAL OBSERVATIONAL DATA`).

---

## Environment setup

```bash
git clone https://github.com/bahman2017/tdf-benchmark-lab.git
cd tdf-benchmark-lab
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

---

## Full test suite

```bash
pytest
```

---

## Benchmark commands (by phase)

| Phase | Command | Notes |
|-------|---------|-------|
| 1 / 3 rotation | `python scripts/run_rotation.py` | **Synthetic or demo fixture** unless `rotation_metadata.yaml` confirms real SPARC |
| 3B / 4A NFW | `python scripts/run_nfw_surrogate.py` | ΛCDM/NFW teacher; not real galaxies |
| 3C ΛCDM scaffold | `python scripts/run_lcdm_benchmark.py` | Combined solar/BH/redshift scaffolds |
| 4B GR-safe | `python scripts/run_gr_safe_benchmark.py` | Configured ε_τ caps only |
| 4C BH exterior | `python scripts/run_black_hole_gr_benchmark.py` | Phenomenological BH ansatz |
| 4D Redshift | `python scripts/run_redshift_sanity_benchmark.py` | Configured z_τ only |
| 5A Core–cusp | `python scripts/run_core_cusp_stress.py` | Cuspy vs cored teachers |
| 5B Diversity | `python scripts/run_rotation_diversity_stress.py` | Ten shape families |
| 5C Same-τ | `python scripts/run_same_tau_consistency.py` | Rotation fit → lensing/redshift proxies |

Optional flags (examples):

```bash
python scripts/run_nfw_surrogate.py --tolerance 5.0
python scripts/run_nfw_surrogate.py --case milky_way_like
```

---

## Expected output files

| Benchmark | Table | Report |
|-----------|-------|--------|
| Rotation | `outputs/tables/rotation_fit_summary.csv` | `outputs/reports/rotation_report.md` |
| NFW surrogate | `outputs/tables/nfw_surrogate_fit_summary.csv` | `outputs/reports/nfw_surrogate_report.md` |
| ΛCDM scaffold | `outputs/tables/lcdm_benchmark_summary.csv` | `outputs/reports/lcdm_benchmark_report.md` |
| GR-safe | `outputs/tables/gr_safe_benchmark_summary.csv` | `outputs/reports/gr_safe_benchmark_report.md` |
| Black-hole | `outputs/tables/black_hole_gr_benchmark_summary.csv` | `outputs/reports/black_hole_gr_benchmark_report.md` |
| Redshift | `outputs/tables/redshift_sanity_benchmark_summary.csv` | `outputs/reports/redshift_sanity_benchmark_report.md` |
| Core–cusp | `outputs/tables/core_cusp_stress_summary.csv` | `outputs/reports/core_cusp_stress_report.md` |
| Diversity | `outputs/tables/rotation_diversity_stress_summary.csv` | `outputs/reports/rotation_diversity_stress_report.md` |
| Same-τ (5C) | `outputs/tables/same_tau_consistency_summary.csv` | `outputs/reports/same_tau_consistency_report.md` |

Figures: `outputs/figures/` (per-case PNGs where applicable).

---

## Suggested appendix table structure

**Table A1 — Benchmark suite overview**  
Columns: Phase | Purpose | Data type | Pass criterion | Validation claim?

**Table A2 — NFW surrogate recovery (Phase 4A)**  
From `nfw_surrogate_fit_summary.csv`: case, mimic_success, relative_curve_error_percent, best_model_by_bic.

**Table A3 — Stress tests (Phases 5A–5B)**  
From `core_cusp_stress_summary.csv` and `rotation_diversity_stress_summary.csv`: teacher type, best_model_by_bic, relative errors.

**Figure A1–A2**  
Example: `nfw_surrogate_milky_way_like.png`, `core_cusp_core_small_rc.png` (copy to `docs/results_snapshot/` if freezing for publication).

---

## What may be cited in the paper

| Allowed | Not allowed |
|---------|-------------|
| “TDF mimics NFW-like teacher within X% in synthetic benchmark” | “TDF validated on galaxies” |
| “Compatibility with GR-limit BH ansatz at small q” | “TDF proves dark matter is unnecessary” |
| “Core proxy improves fit vs simple TDF in cored teacher cases” | “SPARC confirms TDF” (without real metadata) |

---

## Frozen snapshots for publication

Copy selected CSV/PNG into `docs/results_snapshot/` and record:

- Git commit hash: `git rev-parse HEAD`
- Python version: `python --version`
- Date generated

See [results_snapshot/README.md](./results_snapshot/README.md).

---

## Related documents

- [BENCHMARK_MANIFEST.md](./BENCHMARK_MANIFEST.md)
- [SCIENTIFIC_ASSUMPTIONS.md](./SCIENTIFIC_ASSUMPTIONS.md)
- [TEST_PLAN.md](./TEST_PLAN.md)
- [INSTRUCTIONS.md](../INSTRUCTIONS.md) (developer rules)

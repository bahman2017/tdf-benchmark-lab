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
| `python scripts/run_nfw_surrogate.py` | 3B / 4A |
| `python scripts/run_lcdm_benchmark.py` | 3C |
| `python scripts/run_gr_safe_benchmark.py` | 4B |
| `python scripts/run_black_hole_gr_benchmark.py` | 4C |
| `python scripts/run_redshift_sanity_benchmark.py` | 4D |
| `python scripts/run_core_cusp_stress.py` | 5A |
| `python scripts/run_rotation_diversity_stress.py` | 5B |
| `python scripts/run_same_tau_consistency.py` | 5C |
| `python scripts/run_covariant_action_checks.py` | 5D |

Full command list and appendix guidance: [docs/PAPER_APPENDIX_GUIDE.md](./docs/PAPER_APPENDIX_GUIDE.md).

---

## Output structure

```text
outputs/
  README.md
  tables/     # CSV summaries (generated)
  reports/    # Markdown reports with warning banners
  figures/    # PNG plots (generated)
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

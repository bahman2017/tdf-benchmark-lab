# TDF Observational Calibration

Before modifying this project, read [INSTRUCTIONS.md](./INSTRUCTIONS.md).

Numerical **testing and calibration framework** for the Time Delay Field (TDF) v0.8.1 phenomenological equations.

> **Warning:** Outputs are calibration diagnostics only. This project does **not** claim that TDF is validated. Labels distinguish synthetic validation, real-data calibration, passed/failed constraints, and not-yet-tested assumptions.

## Current project mode: ΛCDM Compatibility and Recovery Testing

We are **not** using real observational data for calibration in the current phase.

Instead:

- **ΛCDM / GR+Dark Matter** is used as a **teacher benchmark** in regimes where it already works well (solar system, exterior black-hole limits, NFW-like rotation surrogates, bounded redshift scaffolds).
- TDF is tested on whether it can **recover or mimic** those successful effective behaviors in controlled, labeled benchmarks.
- **Passing benchmark tests does not validate TDF** — it only checks compatibility with known successful phenomenology before we stress TDF in ΛCDM problem regimes or run real-sky calibration.

Real SPARC / observational workflows are **postponed** until Phase 4 compatibility expansion is complete. See [docs/LCDM_COMPATIBILITY_STRATEGY.md](./docs/LCDM_COMPATIBILITY_STRATEGY.md) and [docs/ROADMAP.md](./docs/ROADMAP.md).

## Purpose

- Implement the first weak-field rotation ansatz:  
  `v_model²(r) = v_baryon²(r) + B·r/(r + r0)` with `τ̄_l(r) = A·log(1 + r/r0)` and `B = K_τ·A`.
- Provide modular hooks for lensing, redshift, solar-system, and black-hole channels.
- Run **Phase 1** synthetic rotation recovery tests and **ΛCDM compatibility benchmarks** (Phases 3B–3C, expanding in Phase 4) before real observational calibration (Phase 6).

## Installation

```bash
cd tdf_observational_calibration
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

## Rotation pipeline (Phase 1; Phase 2 scaffold; Phase 6 real data later)

```bash
pytest
python scripts/run_rotation.py            # synthetic or demo fixture by default
# Real SPARC (Phase 6 only, after ΛCDM compatibility work):
# python scripts/prepare_sparc_rotation.py   # user-supplied raw data in data/raw/
# python scripts/run_rotation.py
```

| Condition | Mode | Report banner |
|-----------|------|----------------|
| No `rotation.csv` | `synthetic_validation` | SYNTHETIC ONLY — NO REAL DATA USED |
| CSV without real metadata | `demo_fixture_calibration` | DEMO FIXTURE DATA — NOT REAL SPARC |
| CSV + `rotation_metadata.yaml` (real confirmed) | `real_data_calibration` | _(none)_ |

Outputs:

- `outputs/tables/rotation_fit_summary.csv`
- `outputs/reports/rotation_report.md` (data source, model, B, r0, metrics, warnings, limitations)
- `outputs/figures/<galaxy_id>_rotation.png` (one per galaxy)

### Phase 2 — SPARC ingestion scaffold (not active calibration path)

Infrastructure exists for when **Phase 6 — Real observational calibration** opens. Until then, do not treat processed demo files as real SPARC. See `docs/DATA_REQUIREMENTS.md` for formats.

### Phase 3B–3C — ΛCDM compatibility benchmarks (current focus)

Tests whether TDF can mimic ΛCDM/GR+DM-like effective behavior — **not** real observations.

```bash
python scripts/run_nfw_surrogate.py
python scripts/run_lcdm_benchmark.py
python scripts/run_gr_safe_benchmark.py
python scripts/run_black_hole_gr_benchmark.py
python scripts/run_redshift_sanity_benchmark.py
python scripts/run_core_cusp_stress.py
python scripts/run_rotation_diversity_stress.py
```

Banners: **ΛCDM/NFW BENCHMARK — NOT REAL OBSERVATIONAL DATA** · **ΛCDM BENCHMARK RECOVERY — NOT REAL OBSERVATIONAL DATA**

## Full pipeline (partial)

```bash
python scripts/run_full_pipeline.py
```

Runs rotation synthetic test, solar-system demo constraints, and black-hole summary table. Lensing/redshift remain placeholders.

## Tests

```bash
pytest -v
```

## Project layout

See repository tree: `src/tdf_obs/` (models, fitting, validation, plotting, pipelines), `configs/`, `docs/`, `scripts/`, `tests/`.

## Scientific status

| Channel        | Status (v0.1)        |
|----------------|----------------------|
| Rotation synth | Implemented (Phase 1)|
| SPARC ingest   | Scaffold only (Phase 2); real calibration Phase 6 |
| Baseline compare | Implemented (Phase 3: baryon / TDF / NFW) |
| NFW surrogate    | Implemented (Phase 3B: teacher/student) |
| ΛCDM benchmarks  | Implemented scaffold (Phase 3C) |
| ΛCDM NFW rotation (4A) | Implemented (10 benchmark cases) |
| GR-safe local (4B) | Implemented (7 benchmark cases) |
| BH exterior GR-limit (4C) | Implemented (q sweep) |
| Redshift sanity (4D) | Implemented (7 benchmark cases) |
| Core–cusp stress (5A) | Implemented (8 synthetic cases) |
| Rotation diversity (5B) | Implemented (10 synthetic cases) |
| Lensing        | Not implemented      |
| Redshift       | Partial formula only |
| Solar system   | Basic epsilon checks |
| Black hole     | Formula demos        |

See `docs/LCDM_COMPATIBILITY_STRATEGY.md`, `docs/TEST_PLAN.md`, `docs/ROADMAP.md`, and `docs/SCIENTIFIC_ASSUMPTIONS.md`.

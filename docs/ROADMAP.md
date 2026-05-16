# Roadmap — ΛCDM Compatibility and Recovery Testing

**Current near-term focus:** use ΛCDM / GR+Dark Matter as a **teacher benchmark** in regimes where it already works well. Test whether TDF can recover or mimic those successful effective behaviors — **without** claiming observational validation.

> **Direction change (temporary):** Real SPARC / observational calibration is **not** the immediate next step. Do not download or ingest real sky data for calibration until Phase 4 compatibility work is substantially complete.

---

## Implemented (Phases 1–3C)

| Version | Milestone | Status |
|---------|-----------|--------|
| v0.1 | Phase 1 — Synthetic rotation recovery + scaffold | ✅ |
| v0.2 | Phase 2 — SPARC ingestion **scaffold** (no auto-download; no fake real labels) | ✅ infrastructure only |
| v0.3 | Phase 3 — Baryon / TDF / NFW baseline comparison (BIC) | ✅ |
| v0.3B | Phase 3B — NFW surrogate teacher/student (3 benchmark profiles) | ✅ |
| v0.3C | Phase 3C — ΛCDM/GR benchmark scaffolds (solar system, BH exterior, redshift sanity) | ✅ |

All implemented benchmark outputs must remain labeled as **not real observational validation**.

---

## Phase 4 — ΛCDM Compatibility Expansion *(next)*

**Purpose:** Before using real observational data, test TDF against benchmark outputs from ΛCDM/GR+DM in regimes where ΛCDM is already successful.

**Banner (all Phase 4 outputs):**  
`ΛCDM/NFW BENCHMARK — NOT REAL OBSERVATIONAL DATA`

### Phase 4A — Rotation NFW recovery expansion ✅ (implemented)

- **10** benchmark galaxy-like profiles in `BENCHMARK_CASE_REGISTRY`.
- Include representative archetypes: dwarf, LSB, Milky-Way-like, high-surface-brightness, extended disk.
- Teacher: GR+DM/NFW-like effective rotation; student: TDF simple ansatz.
- Pass/fail: TDF mimics teacher within configured relative curve error (same spirit as Phase 3B).

### Phase 4B — GR-safe local tests ✅ (implemented)

- Seven regimes: Mercury perihelion, Earth orbit, GPS clock, light bending, Shapiro delay, LLR, binary pulsar proxy.
- Outputs: `gr_safe_benchmark_summary.csv`, `gr_safe_benchmark_report.md`.
- Metric: ε_τ = Φ_τ/Φ_b vs configurable `max_allowed_epsilon`.
- Goal: confirm TDF corrections can be **suppressed** where GR is already strongly tested.

### Phase 4C — Black-hole exterior GR recovery ✅ (implemented)

- q = rc/rs sweep; `black_hole_gr_benchmark` outputs and status classification.
- r_nr/r_s and T/T_H vs √(1 − (rc/rs)²).
- Strong-field formulas remain **phenomenological** — not derived metric claims.

### Phase 4D — Redshift sanity expansion ✅ (implemented)

- Seven configured z_τ cases; `redshift_sanity_benchmark` outputs and pass/borderline/fail classification.
- Confirm tau-induced residuals stay bounded in GR-success regimes.
- No real line-of-sight or cosmological data in this phase.

---

## Phase 5 — ΛCDM Stress / Problem Regimes

Phase 4 complete. Stress tests examine ΛCDM tension regimes (not observational validation).

### Phase 5A — Core–cusp stress test ✅ (implemented)

- Cuspy NFW-like vs cored halo teachers; TDF simple vs TDF core proxy diagnostic.
- Outputs: `core_cusp_stress_summary.csv`, `core_cusp_stress_report.md`.

### Phase 5B — Rotation-curve diversity ✅ (implemented)

- Ten synthetic teacher shape families; TDF simple vs TDF core vs NFW by BIC.

Further topics:

- Core–cusp (extended)
- Missing satellites
- Hubble tension (phenomenological placeholders)
- Black-hole singularity / information paradox (ansatz-level only)

Outputs: stress-test reports with explicit **benchmark / tension** labels, not validation claims.

---

## Phase 6 — Real observational calibration *(postponed)*

**Intentionally deferred** until ΛCDM compatibility tests (Phase 4) are completed.

- Real SPARC rotation ingestion and per-galaxy calibration.
- Requires confirmed `rotation_metadata.yaml` (`real_observational`).
- No auto-download; no fake real-data labels.

See `docs/TEST_PLAN.md` for detailed steps when this phase opens.

---

## Later channels (after Phase 6)

| Version | Milestone |
|---------|-----------|
| v0.7 | Lensing consistency (shared τ parameters) |
| v0.8 | Full redshift residual pipeline vs kinematics |
| v0.9 | Solar-system coupling to ephemeris-scale constraints |
| v1.0 | Multi-channel consistency report — still **not** "TDF validated" unless external review agrees |

---

## What this roadmap does **not** mean

- Passing NFW surrogate or ΛCDM benchmark tests does **not** validate TDF.
- Teacher benchmarks are **effective phenomenology** from ΛCDM/GR+DM, not proof that TDF is correct on the sky.
- Real-data workflows are out of scope until Phase 6.

See also: `docs/LCDM_COMPATIBILITY_STRATEGY.md`, `docs/TEST_PLAN.md`.

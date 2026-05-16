# ΛCDM Compatibility and Recovery Testing — Strategy

**Project mode:** ΛCDM Compatibility and Recovery Testing  
**Banner for benchmark outputs:** `ΛCDM/NFW BENCHMARK — NOT REAL OBSERVATIONAL DATA` (rotation/NFW family) or `ΛCDM BENCHMARK RECOVERY — NOT REAL OBSERVATIONAL DATA` (multi-channel scaffold)

---

## Motivation

TDF is a phenomenological framework that must be constrained **honestly** and **incrementally**. Jumping straight to real galaxy catalogs risks overfitting narratives to noisy data before we know whether TDF can even reproduce behaviors that ΛCDM+GR already explains well.

This phase therefore asks a narrower, falsifiable question first:

> In controlled benchmarks, can TDF **recover or mimic** the **successful effective behaviors** of ΛCDM/GR+DM — without claiming we have validated TDF against the real Universe?

Only after that baseline compatibility work should we stress TDF in ΛCDM **problem** regimes, and only much later run **real** observational calibration (Phase 6).

---

## Why use ΛCDM as teacher first

1. **Known successes:** Solar-system GR, exterior black-hole limits, NFW-like halo rotation, and small cosmological redshift shifts are regimes where ΛCDM+GR is already empirically strong. TDF must not obviously fail there.
2. **Clear teacher curves:** Synthetic NFW teachers and configurable ε_τ / z_τ cases give reproducible pass/fail criteria without downloading catalogs.
3. **Separation of concerns:** Teacher–student mismatch isolates **shape / limit** failure from **data noise / systematics** failure.
4. **Honest sequencing:** Real SPARC calibration is postponed until we document where TDF matches ΛCDM-like phenomenology and where it does not.

---

## What counts as pass / fail

### Rotation (Phase 3B → 4A)

| Outcome | Criterion (configurable) |
|---------|---------------------------|
| **Pass (mimic)** | TDF student curve vs NFW teacher: mean relative curve error below tolerance (default &lt; 5%) |
| **Fail** | Error above tolerance, or fit unstable / non-finite |

**Phase 4A (implemented)** expands the NFW rotation benchmark family to **10+ registry cases** covering dwarf, LSB, Milky-Way-like, HSB, massive spiral, extended disk, and concentrated-halo archetypes, with three baryon profile types (`saturating_disk`, `exponential_like`, `compact_bulge_disk`). All outputs use the banner `ΛCDM/NFW BENCHMARK — NOT REAL OBSERVATIONAL DATA`.

Teacher data are **surrogate** (baryon profile + NFW halo term + optional noise), not SPARC.

### GR-safe local (Phase 3C → 4B)

| Outcome | Criterion |
|---------|-----------|
| **Pass** | \|ε_τ\| = \|Φ_τ/Φ_b\| ≤ `max_allowed_epsilon` for each configured case |
| **Fail** | Assumed or predicted ε_τ exceeds cap |

Assumed Φ_τ values in scaffold runs are **placeholders**, not fitted from ephemeris.

### Black-hole exterior (Phase 3C → 4C)

| Outcome | Criterion |
|---------|-----------|
| **GR-like** | q ≪ 1: r_nr/r_s and T/T_H ≈ 1 (deviation from Hawking limit &lt; 0.1%) |
| **Mildly modified** | 0.1% ≤ deviation &lt; 5% |
| **Strongly modified** | deviation ≥ 5% |
| **No horizon** | q ≥ 1 (exterior ansatz undefined) |

**Phase 4C (implemented):** sweep q ∈ {0, 1e−8, …, 0.99, 1.0} via `run_black_hole_gr_benchmark.py`; outputs `black_hole_gr_benchmark_summary.csv` with status per q. Banner: `BLACK-HOLE GR-LIMIT BENCHMARK — NOT REAL OBSERVATIONAL DATA`.

Formulas are **phenomenological** (TDF v0.8.1 ansatz level). Ratios track √(1 − q²) by construction; classification measures departure from the **q → 0** Hawking limit.

### Redshift sanity (Phase 3C → 4D)

| Outcome | Criterion |
|---------|-----------|
| **Pass** | \|z_τ\| < 0.8 × `max_allowed_abs_z_tau` |
| **Borderline** | 0.8 × max ≤ \|z_τ\| ≤ max |
| **Fail** | \|z_τ\| > max |

**Phase 4D (implemented):** seven configured cases (negligible through cluster/clock proxies) via `run_redshift_sanity_benchmark.py`. Banner: `REDSHIFT SANITY ΛCDM/GR BENCHMARK — NOT REAL OBSERVATIONAL DATA`. z_tau is **configured**, not fitted from spectra.

---

## What does **not** count as validation

- Passing NFW surrogate or ΛCDM benchmark tables.
- Lower MSE than baryon-only without BIC / parameter-count context.
- Matching a **teacher** we built ourselves from ΛCDM-like formulas.
- Any output labeled synthetic, demo fixture, NFW surrogate, or ΛCDM benchmark recovery.

**Validation** (in the strong sense used in this project) would require multi-channel consistency on **real** data with independent systematics — explicitly **Phase 6+** and external scrutiny. We do not claim that here.

---

## Planned benchmark families

| Family | Teacher / reference | TDF object | Phase |
|--------|---------------------|------------|-------|
| NFW rotation surrogate | v²_teacher = v²_baryon + NFW halo term | v²_TDF = v²_baryon + B·r/(r+r0) | 3B → **4A** |
| Solar-system GR-safe | GR-dominated Φ_b; tiny Φ_τ | ε_τ cap | 3C → **4B** |
| BH exterior GR limit | Schwarzschild/Hawking scaling | r_nr, T_TDF ansatz | 3C → **4C** |
| Redshift sanity | z_τ bound in weak field | z_τ = K_τ Δτ̄_l/c² | 3C → **4D** |
| Lensing (future) | ΛCDM deflection integral | shared Φ_τ | post–Phase 6 |
| ΛCDM stress (later) | Known tension scenarios | TDF vs effective tension metrics | **Phase 5** |

All families use **configurable synthetic or surrogate** inputs unless Phase 6 explicitly opens real-data mode.

---

## Later transition to ΛCDM problem regimes (Phase 5)

**Phase 4 (4A–4D) compatibility recovery is complete.** Stress testing now examines known ΛCDM problem regimes.

**Phase 5A (implemented):** Core–cusp stress test — cuspy NFW-like vs cored Burkert-like teachers; TDF simple vs TDF core proxy `C·r²/(r²+rc_tau²)`. Banner: `CORE-CUSP STRESS TEST — NOT REAL OBSERVATIONAL DATA`. Command: `python scripts/run_core_cusp_stress.py`.

**Phase 5B (implemented):** Rotation-curve diversity stress — ten synthetic teacher shapes (fast/slow rise, baryon-dominated, outer decline/rise, flat extended, core/cusp+flat outer, dwarf). Banner: `ROTATION-CURVE DIVERSITY STRESS TEST — NOT REAL OBSERVATIONAL DATA`.

**Phase 5C (implemented):** Same-τ multi-observable consistency — one fitted `Φ_τ(r)=B log(1+r/r0)` from rotation must predict a fixed lensing proxy `α_τ ∝ B/(R+r0)` and redshift proxy `z_τ=ΔΦ_τ/c²` without refitting τ. Banner: `SAME-TAU MULTI-OBSERVABLE BENCHMARK — NOT REAL OBSERVATIONAL DATA`. Command: `python scripts/run_same_tau_consistency.py`. This tests whether a **single τ profile** can connect rotation, lensing proxy, and redshift proxy in a controlled synthetic setting only — **not** real lensing or redshift data.

Further Phase 5 topics:

- Core–cusp (extended / real-data later) (spread of teacher classes TDF can/cannot mimic)
- Missing satellites (order-of-magnitude scaffold only at first)
- Hubble tension (phenomenological H₀–scale tests, not full cosmology fit)
- BH singularity / information (ansatz reminders only — no claim of resolution)

These are **stress tests**, not wins. Failures are scientifically informative.

---

## Later transition to real observational calibration (Phase 6)

**Gate:** Phase 4 substantially complete; strategy doc and reports reviewed.

Then:

1. User supplies real SPARC (or equivalent) under `data/raw/` — no auto-download.
2. `prepare_sparc_rotation.py` writes processed tables + honest metadata.
3. `run_rotation.py` runs only in `real_data_calibration` when metadata confirms real source.
4. Reports compare TDF to baryon and halo baselines with full self-criticism section.

Until then: **do not** present demo fixtures or surrogates as real SPARC.

---

## Reporting requirements

Every benchmark report must include:

1. Prominent banner (ΛCDM/NFW or ΛCDM benchmark recovery — not real observational data).
2. Statement that this is **not** observational validation.
3. Teacher definition and pass/fail thresholds used.
4. Self-criticism and failure modes.
5. Next step aligned with `docs/ROADMAP.md`.

---

## Related documents

- [ROADMAP.md](./ROADMAP.md) — versioned milestones
- [TEST_PLAN.md](./TEST_PLAN.md) — phase commands and status
- [INSTRUCTIONS.md](../INSTRUCTIONS.md) — equations, labeling, anti-fake-data rules
- [SCIENTIFIC_ASSUMPTIONS.md](./SCIENTIFIC_ASSUMPTIONS.md) — phenomenological scope

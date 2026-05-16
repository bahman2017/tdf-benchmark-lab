# Scientific assumptions — TDF Benchmark Lab

## What this project is

A **phenomenological testing harness** for TDF-inspired observables. It produces labeled diagnostics (synthetic vs demo vs real, pass vs fail). It does **not** establish that TDF is correct, validated, or superior to ΛCDM on the sky.

## What passing a benchmark means

- **Compatibility / recovery** in a **controlled** setting with a defined teacher or tolerance.
- **Not** observational validation, **not** proof of TDF, **not** disproof of dark matter.

## TDF version

**TDF v0.8.1 — Consistency Cleanup** (see [INSTRUCTIONS.md](../INSTRUCTIONS.md)).

## Core equations (reference)

1. `Φ_τ = K_τ τ̄_l`
2. `a_τ^i = -K_τ ∂^i τ̄_l`
3. Weak-field rotation: `v_model²(r) = v_baryon²(r) + B·r/(r+r0)` with `τ̄_l = A·log(1+r/r0)`, `B = K_τ A`
4. Lensing: `α_lens ≈ (2/c²) ∫ ∇⊥(Φ_b + Φ_τ) dl` — **not implemented** in benchmarks
5. Redshift: `z_τ = K_τ Δτ̄_l / c²` — benchmark uses **configured** z_τ in Phase 4D
6. Solar system: `ε_τ = Φ_τ/Φ_b` — configured scaffold (Phase 4B)
7. Black hole: phenomenological `T_TDF`, `r_nr,TDF` — **ansatz-level** (Phase 4C)

## Explicit assumptions

| ID | Assumption |
|----|------------|
| A1 | TDF v0.8.1 is **phenomenological** in strong-field / black-hole regimes; formulas are not derived from a full metric. |
| A2 | Rotation fits are **controlled benchmarks** unless `rotation_metadata.yaml` explicitly confirms real observational data. |
| A3 | Fitted parameter **B** combines `K_τ` and **A**; they are not separated in rotation benchmarks. |
| A4 | Units: `r` [kpc], `v` [km/s], **B** [km²/s²], `r0` [kpc] — document SI conversions when mixing channels. |
| A5 | Baryon curves `v_baryon(r)` are **inputs** in rotation benchmarks, not fitted from photometry here. |
| A6 | TDF **core proxy** (`C`, `rc_tau`) is a **stress-test diagnostic**; it does not replace the simple TDF model in the theory. |
| A7 | `rc > rs` for BH ansatz is unphysical; code returns NaN with warning. |
| A8 | **No validation claim** until independent real-data channels pass jointly with shared parameters and solar-system safety — **Phase 6+**. |

## Black-hole and strong-field

Black-hole temperature and non-return radius are **ansatz-level** comparisons to Hawking/Schwarzschild limits. They are **not** full nonlinear black-hole solutions, shadow predictions, or ringdown fits to data.

## Rotation and SPARC

- Phase 2 provides an ingestion **scaffold** only.
- Demo `data/processed/rotation.csv` must **not** be described as SPARC validation.
- Real SPARC calibration is **intentionally postponed** (Phase 6).

## What would count as progress (not proof)

| Milestone | Meaning |
|-----------|---------|
| Synthetic recovery of `(B, r0)` | Pipeline integrity |
| NFW teacher mimic within tolerance | ΛCDM-like phenomenology recovery in synthetic setting |
| GR-safe ε_τ below configured cap | Compatibility scaffold only |
| Stress-test pass | Shape class representable by TDF variant — **not** cosmological conclusion |

Failures must be reported as **failed constraints** or informative stress outcomes, not hidden.

## Multi-channel validation (future)

Real validation would require, at minimum:

- Shared τ parameters across rotation, lensing, and redshift on **real** data
- Solar-system bounds from dynamics or clocks, not configured placeholders
- External review and pre-registered pass/fail criteria

This repository **does not** meet that bar in v0.1.0.

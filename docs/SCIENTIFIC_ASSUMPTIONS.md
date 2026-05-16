# Scientific Assumptions — TDF v0.8.1 Calibration Framework

## What this project is

A **phenomenological testing harness** for TDF-inspired observables. It produces labeled diagnostics (synthetic vs real, pass vs fail). It does **not** establish that TDF is correct or validated.

## Core equations (reference)

1. `Φ_τ = K_τ τ̄_l`
2. `a_τ^i = -K_τ ∂^i τ̄_l`
3. `v_model²(r) = v_baryon²(r) + r dΦ_τ/dr`
4. First ansatz: `τ̄_l(r) = A log(1 + r/r0)` → `v_model² = v_baryon² + B r/(r+r0)`, `B = K_τ A`
5. Lensing: `α_lens ≈ (2/c²) ∫ ∇⊥(Φ_b + Φ_τ) dl` — **not implemented**
6. Redshift: `z_τ ≈ K_τ Δτ̄_l / c²` — implemented as formula only
7. Solar system: `ε_τ = Φ_τ/Φ_b` must be ≪ 1 on solar scales
8. BH core ansatz: `Φ_τ,core = -GM/√(r² + r_c²)`; modified `T_TDF`, `r_nr,TDF`

## Explicit assumptions

| ID | Assumption |
|----|------------|
| A1 | TDF v0.8.1 is **phenomenological** in strong-field / BH regimes until a full metric is derived. |
| A2 | The rotation model is the **first weak-field test** only. |
| A3 | Fitting uses combined parameter **B**; **K_τ** and **A** are not separated in Phase 1. |
| A4 | Units: `r` [kpc], `v` [km/s], **B** [km²/s²], `r0` [kpc] — document conversions if mixing SI. |
| A5 | Baryon curves `v_baryon(r)` are **inputs**, not fitted here. |
| A6 | Lensing, redshift, and multi-channel consistency are **not_yet_tested** unless a report says otherwise. |
| A7 | BH temperature and non-return radius formulas are **ansatz-level**; `rc > r_s` is unphysical and returns NaN with warning. |
| A8 | No claim of validation until **independent channels** pass with shared parameters and solar-system safety. |

## What would count as progress (not proof)

- Synthetic recovery of `(B, r0)` within tolerance → pipeline integrity
- Real SPARC fits that improve χ² vs baryon-only without unphysical `(B, r0)`
- Lensing and redshift using **same** tau parameters within errors
- Solar-system `ε_τ` below experimental bounds

Failures must be reported as **failed constraints**, not hidden by defaults.

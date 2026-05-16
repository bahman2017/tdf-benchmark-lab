# ΛCDM benchmark recovery report (Phase 3C)

## ⚠️ ΛCDM BENCHMARK RECOVERY — NOT REAL OBSERVATIONAL DATA

> **This is not observational validation.** Configurable scaffold cases only. > No real SPARC, solar-system ephemeris, lensing, or cosmological data were used.

## Purpose

Extend benchmark recovery beyond rotation (Phase 3B NFW surrogate) with:

1. Solar-system GR-safe checks on ε_τ = Φ_τ/Φ_b
2. Black-hole ansatz GR-limit recovery when r_c/r_s → 0
3. Redshift sanity on z_τ = K_τ Δτ̄_l / c²

## NFW rotation surrogate status (Phase 3B)

NFW surrogate summary available: 3 galaxies in `nfw_surrogate_fit_summary.csv`.
TDF mimics teacher (Phase 3B): 3/3.
See `nfw_surrogate_report.md` for details.
_Surrogate success does not imply observational validation._

## Summary

- **Total cases:** 15
- **Pass:** 14
- **Fail:** 1 _(expected for intentional `too_large` redshift case)_

## Solar System Gr Safe

| Case | Status | Key values |
| --- | --- | --- |
| earth_orbit | pass | ε_τ=-1.60e-18, max=1.00e-08 |
| mercury_orbit | pass | ε_τ=-5.88e-19, max=1.00e-08 |
| light_bending_sun | pass | ε_τ=-1.00e-20, max=1.00e-09 |
| gps_weak_field_clock | pass | ε_τ=-1.67e-19, max=1.00e-08 |

## Black Hole Gr Limit

| Case | Status | Key values |
| --- | --- | --- |
| rc_over_rs_0 | pass | r_nr/rs=1.0000, T/T_H=1.0000 |
| rc_over_rs_1e-06 | pass | r_nr/rs=1.0000, T/T_H=1.0000 |
| rc_over_rs_0.0001 | pass | r_nr/rs=1.0000, T/T_H=1.0000 |
| rc_over_rs_0.01 | pass | r_nr/rs=0.9999, T/T_H=0.9999 |
| rc_over_rs_0.1 | pass | r_nr/rs=0.9950, T/T_H=0.9950 |
| rc_over_rs_0.5 | pass | r_nr/rs=0.8660, T/T_H=0.8660 |
| rc_over_rs_0.9 | pass | r_nr/rs=0.4359, T/T_H=0.4359 |

## Redshift Sanity

| Case | Status | Key values |
| --- | --- | --- |
| negligible | pass | z_τ=1.11e-17, max=1.00e-15 |
| small_allowed | pass | z_τ=1.11e-09, max=1.00e-06 |
| borderline | pass | z_τ=9.90e-07, max=1.00e-06 |
| too_large | fail | z_τ=1.11e-05, max=1.00e-08 |

## Failure modes

- Assumed Φ_τ in solar-system cases are placeholders until a dynamical TDF metric is coupled.
- Black-hole formulas are phenomenological ansatz, not derived from a full metric.
- Redshift scaffold uses synthetic Δτ̄_l; no line-of-sight or velocity degeneracy model yet.
- Passing here does not constrain lensing, CMB, or structure formation.

## Next steps

1. Ingest real SPARC rotation data (`prepare_sparc_rotation.py`).
2. Implement lensing consistency with shared τ parameters (Phase 4).
3. Replace assumed solar-system ε_τ with predictions from a weak-field TDF metric.
4. Tie redshift tests to independent kinematic baselines without double-counting.

## Disclaimer

- Does **not** validate TDF against ΛCDM or observations.
- Phase 3B NFW surrogate success is a **shape** benchmark only.

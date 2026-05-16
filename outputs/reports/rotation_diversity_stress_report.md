# Rotation-curve diversity stress test (Phase 5B)

## ⚠️ ROTATION-CURVE DIVERSITY STRESS TEST — NOT REAL OBSERVATIONAL DATA

## Purpose

This stress test checks whether TDF variants can represent **diverse synthetic** rotation-curve shapes before real data are used.

> **Not observational validation.**

## Equations

**TDF simple:** `v_TDF^2 = v_baryon^2 + B * r / (r + r0)`

**TDF core (diagnostic):** `v_TDF_core^2 = v_baryon^2 + C * r^2 / (r^2 + rc_tau^2)`

**NFW simple:** `v_NFW^2 = v_baryon^2 + Vh2 * [ln(1+x) - x/(1+x)] / x`

**Teacher families** include: fast_rising_flat, slow_rising_lsb, compact/diffuse baryon-dominated, declining/rising outer, flat extended, inner core/cusp + outer flat, low-velocity dwarf.

## Summary

- **Cases:** 10
- **TDF simple best by BIC:** 2
- **TDF core best by BIC:** 3
- **Any TDF variant best:** 5
- **NFW best by BIC:** 5
- **Median rel. error TDF simple:** 2.05%
- **Median rel. error TDF core:** 4.01%
- **Median rel. error NFW:** 1.57%
- **Hardest case (min TDF rel. err):** inner_cusp_outer_flat

## Per-case table

| Case | Shape | Best BIC | rel% simple | rel% core | rel% NFW |
| --- | --- | --- | --- | --- | --- |
| fast_rising_flat | fast_rising_flat | tdf_core | 3.25 | 0.50 | 2.08 |
| slow_rising_lsb | slow_rising_lsb | nfw_simple | 0.50 | 4.95 | 0.98 |
| compact_baryon_dominated | compact_baryon_dominated | tdf_core | 0.26 | 0.20 | 0.23 |
| diffuse_baryon_dominated | diffuse_baryon_dominated | tdf_simple | 0.54 | 5.91 | 1.32 |
| declining_outer_curve | declining_outer_curve | nfw_simple | 4.16 | 3.87 | 1.83 |
| rising_outer_curve | rising_outer_curve | tdf_simple | 0.53 | 2.63 | 0.63 |
| flat_extended_curve | flat_extended_curve | nfw_simple | 0.86 | 4.36 | 0.50 |
| inner_core_outer_flat | inner_core_outer_flat | nfw_simple | 5.72 | 4.16 | 2.35 |
| inner_cusp_outer_flat | inner_cusp_outer_flat | nfw_simple | 12.54 | 12.48 | 9.14 |
| low_velocity_dwarf_diverse | low_velocity_dwarf_diverse | tdf_core | 6.59 | 0.45 | 5.17 |

## Scientific interpretation

- Passing means TDF variants can represent **synthetic diversity patterns** in this registry.
- It does **not** validate TDF against real galaxies or SPARC.
- Cases where NFW wins are **informative**, not test failures.

## Failure modes

- Teacher curves are **synthetic** shape families only.
- TDF core has a core-scale parameter and may overfit some cases.
- NFW is a simple comparison baseline, not a full halo model survey.
- Real diversity requires SPARC or similar data in a later phase.

## Disclaimer

- **NOT REAL OBSERVATIONAL DATA**

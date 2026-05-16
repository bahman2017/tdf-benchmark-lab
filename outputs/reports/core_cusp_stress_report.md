# Core–cusp stress test report (Phase 5A)

## ⚠️ CORE-CUSP STRESS TEST — NOT REAL OBSERVATIONAL DATA

## Purpose

This test checks whether TDF-style **core smoothing** can represent cored inner rotation behavior in controlled stress cases, compared to cuspy NFW-like teachers.

> **Not observational validation.** Synthetic teachers only.

## Equations

**Cuspy NFW-like teacher:**

```text
v_NFW^2(r) = Vh2 * [ln(1+x) - x/(1+x)] / x,  x = r/rs
v_teacher^2 = v_baryon^2 + v_NFW^2
```

**Cored teacher proxy:**

```text
v_core^2(r) = Vc2 * r^2 / (r^2 + rc^2)
v_teacher^2 = v_baryon^2 + v_core^2
```

**TDF simple (unchanged):**

```text
v_TDF^2 = v_baryon^2 + B * r / (r + r0)
```

**TDF core proxy (stress diagnostic only):**

```text
v_TDF_core^2 = v_baryon^2 + C * r^2 / (r^2 + rc_tau^2)
```

## Summary

- **Total cases:** 8
- **Cuspy teachers:** 4
- **Cored teachers:** 4
- **TDF core beats TDF simple (BIC + error):** 6 / 8
- **TDF core beats NFW in cored cases:** 4 / 4
- **Worst TDF core rel. error:** nfw_strong_cusp (4.26%)

## Results table

| Case | Teacher | Best BIC | rel err TDF simple % | rel err TDF core % | rel err NFW % | core adv. |
| --- | --- | --- | --- | --- | --- | --- |
| nfw_mild_cusp | cuspy | nfw_simple | 1.11 | 1.28 | 0.28 | no |
| nfw_strong_cusp | cuspy | nfw_simple | 4.71 | 4.26 | 0.30 | yes |
| nfw_concentrated_inner | cuspy | nfw_simple | 2.73 | 2.29 | 0.05 | yes |
| nfw_extended_outer | cuspy | tdf_simple | 0.66 | 1.67 | 0.62 | no |
| core_small_rc | cored | cored_proxy | 3.05 | 0.27 | 2.20 | yes |
| core_large_rc | cored | tdf_core | 3.15 | 0.06 | 2.78 | yes |
| core_lsb_like | cored | cored_proxy | 7.72 | 1.04 | 6.76 | yes |
| core_diffuse_dwarf | cored | tdf_core | 7.81 | 1.56 | 6.89 | yes |

## Scientific interpretation

- Passing cored stress tests does **not** validate TDF against real galaxies.
- It only suggests that tau-core smoothing can represent **core-like phenomenology** in these controlled benchmarks.
- BIC penalizes extra parameters; compare TDF core (2 params) fairly against TDF simple and NFW.

## Failure modes

- TDF simple may fail near inner-core behavior (cuspy vs cored mismatch).
- TDF core proxy adds a physically motivated core scale — more flexibility by design.
- More parameters can improve fit; always compare **BIC**, not MSE alone.
- Real galaxies and SPARC data are **not** tested here.

## Disclaimer

- **NOT REAL OBSERVATIONAL DATA**
- Does not resolve the core–cusp problem observationally.

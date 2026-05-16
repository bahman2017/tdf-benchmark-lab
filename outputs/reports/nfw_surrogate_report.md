# ΛCDM/NFW rotation benchmark report (Phase 4A)

## ⚠️ ΛCDM/NFW BENCHMARK — NOT REAL OBSERVATIONAL DATA

> **This is not observational validation.** Controlled ΛCDM/GR+DM teacher benchmarks only. > This does not validate TDF against real observations. > Builds on the Phase 3B NFW surrogate scaffold.

## Scientific interpretation

**Passing** means TDF can approximate NFW-like rotation phenomenology in these controlled benchmarks when the mean relative curve error vs the teacher is below the configured tolerance.

**Passing does not mean** real observational validation, nor that TDF is correct in nature.

## Models

**Teacher (ΛCDM/NFW + baryon):**

```text
v_teacher^2(r) = v_baryon^2(r) + v_NFW^2(r)  [NFW halo: Vh2 * (ln(1+x) - x/(1+x)) / x,  x = r/rs]
```

**Student (TDF simple):**

```text
v_TDF^2(r) = v_baryon^2(r) + B * r / (r + r0)
```

**Mimic tolerance:** mean relative curve error < 5.0%

## Summary

- **Number of cases:** 10
- **Mimicked by TDF:** 10
- **Pass fraction:** 100.0%
- **Median relative curve error:** 0.93%
- **Worst-case relative curve error:** 1.89%

- **Failed mimic threshold:** _(none)_

## Per-case table

| Case | Baryon profile | Vh2 true | rs true | rel. error % | mimic | best BIC |
| --- | --- | --- | --- | --- | --- | --- |
| dwarf_low_mass | saturating_disk | 800 | 1.2 | 0.97 | yes | nfw_simple |
| dwarf_extended_lsb | exponential_like | 600 | 2.8 | 1.71 | yes | nfw_simple |
| compact_dwarf | compact_bulge_disk | 1500 | 0.9 | 1.52 | yes | tdf_simple |
| lsb_diffuse | exponential_like | 2500 | 6 | 1.89 | yes | tdf_simple |
| spiral_mid_mass | saturating_disk | 5500 | 7 | 0.60 | yes | nfw_simple |
| milky_way_like | saturating_disk | 9000 | 10 | 0.50 | yes | tdf_simple |
| high_surface_brightness | compact_bulge_disk | 1.2e+04 | 5 | 0.48 | yes | nfw_simple |
| massive_spiral | saturating_disk | 1.8e+04 | 15 | 0.89 | yes | tdf_simple |
| extended_disk | exponential_like | 8000 | 12 | 0.54 | yes | tdf_simple |
| concentrated_halo | saturating_disk | 1.4e+04 | 3.5 | 1.56 | yes | nfw_simple |

## Failure modes

The simple TDF ansatz `B·r/(r+r0)` may struggle when:

- **Inner steep halos** — small r_s with rapid rise in v_NFW at small r.
- **Very compact baryons** — bulge+disk quadrature dominates inner radii.
- **Extreme concentration** — concentrated_halo-like teachers with small rs.
- **Outer slowly rising curves** — extended exponential disks + extended radii.
- **Noisy low-mass profiles** — dwarf cases with higher noise_std.

A failed mimic does not imply TDF fails on real data; these are synthetic teachers only.

## Per-case detail

### dwarf_low_mass

- Baryon profile: `saturating_disk`
- True NFW: Vh2 = 800, rs = 1.2 kpc
- Fitted TDF: B = 160.875 km²/s², r0 = 0.427668 kpc
- Teacher–student MSE: 0.135575
- Relative curve error: 0.97%
- Max fractional error: 0.0204
- Mimic success: **True**

### dwarf_extended_lsb

- Baryon profile: `exponential_like`
- True NFW: Vh2 = 600, rs = 2.8 kpc
- Fitted TDF: B = 119.995 km²/s², r0 = 0.29438 kpc
- Teacher–student MSE: 0.179055
- Relative curve error: 1.71%
- Max fractional error: 0.2540
- Mimic success: **True**

### compact_dwarf

- Baryon profile: `compact_bulge_disk`
- True NFW: Vh2 = 1500, rs = 0.9 kpc
- Fitted TDF: B = 249.394 km²/s², r0 = 0.067364 kpc
- Teacher–student MSE: 0.466218
- Relative curve error: 1.52%
- Max fractional error: 0.0489
- Mimic success: **True**

### lsb_diffuse

- Baryon profile: `exponential_like`
- True NFW: Vh2 = 2500, rs = 6 kpc
- Fitted TDF: B = 748.412 km²/s², r0 = 3.00796 kpc
- Teacher–student MSE: 0.710115
- Relative curve error: 1.89%
- Max fractional error: 0.0572
- Mimic success: **True**

### spiral_mid_mass

- Baryon profile: `saturating_disk`
- True NFW: Vh2 = 5500, rs = 7 kpc
- Fitted TDF: B = 1263.57 km²/s², r0 = 1.83775 kpc
- Teacher–student MSE: 0.227033
- Relative curve error: 0.60%
- Max fractional error: 0.0344
- Mimic success: **True**

### milky_way_like

- Baryon profile: `saturating_disk`
- True NFW: Vh2 = 9000, rs = 10 kpc
- Fitted TDF: B = 2225.51 km²/s², r0 = 3.47515 kpc
- Teacher–student MSE: 0.19444
- Relative curve error: 0.50%
- Max fractional error: 0.0241
- Mimic success: **True**

### high_surface_brightness

- Baryon profile: `compact_bulge_disk`
- True NFW: Vh2 = 12000, rs = 5 kpc
- Fitted TDF: B = 2960.04 km²/s², r0 = 1.94715 kpc
- Teacher–student MSE: 0.380591
- Relative curve error: 0.48%
- Max fractional error: 0.0091
- Mimic success: **True**

### massive_spiral

- Baryon profile: `saturating_disk`
- True NFW: Vh2 = 18000, rs = 15 kpc
- Fitted TDF: B = 4553.48 km²/s², r0 = 4.40064 kpc
- Teacher–student MSE: 1.31597
- Relative curve error: 0.89%
- Max fractional error: 0.0431
- Mimic success: **True**

### extended_disk

- Baryon profile: `exponential_like`
- True NFW: Vh2 = 8000, rs = 12 kpc
- Fitted TDF: B = 2122.39 km²/s², r0 = 4.90085 kpc
- Teacher–student MSE: 0.195602
- Relative curve error: 0.54%
- Max fractional error: 0.0519
- Mimic success: **True**

### concentrated_halo

- Baryon profile: `saturating_disk`
- True NFW: Vh2 = 14000, rs = 3.5 kpc
- Fitted TDF: B = 3096.89 km²/s², r0 = 1.02238 kpc
- Teacher–student MSE: 2.22926
- Relative curve error: 1.56%
- Max fractional error: 0.0419
- Mimic success: **True**

## Disclaimer

- **NOT REAL OBSERVATIONAL DATA** — ΛCDM/NFW benchmark teachers only.
- Does **not** validate TDF against the real Universe.
- Proceed to ΛCDM stress regimes (Phase 5) and real calibration (Phase 6) only after reviewing patterns here.

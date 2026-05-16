# Redshift/Doppler sanity benchmark report (Phase 4D)

## ⚠️ REDSHIFT SANITY ΛCDM/GR BENCHMARK — NOT REAL OBSERVATIONAL DATA

## Purpose

This checks whether tau-induced redshift residuals can remain **bounded** before real spectral or Doppler data are used.

> **Not observational validation.** Configured z_tau values only — no spectra fitted.

## Formula (TDF v0.8.1)

```text
z_tau = K_tau * Delta_tau_bar_l / c^2
```

In this benchmark, **z_tau is configured directly** for classification against `max_allowed_abs_z_tau` (not derived from a full TDF metric fit).

## Classification rules

- **pass:** |z_tau| < 0.8 × max_allowed
- **borderline:** 0.8 × max_allowed ≤ |z_tau| ≤ max_allowed
- **fail:** |z_tau| > max_allowed

## Summary

- **Total cases:** 7
- **Pass:** 5
- **Borderline:** 1
- **Fail:** 1
- **Matches expected status:** 7 / 7
- **Expected failures:** too_large_shift

## Results table

| case_name | regime | z_tau | max_allowed | ratio_to_limit | status | expected |
| --- | --- | --- | --- | --- | --- | --- |
| negligible_tau_shift | negligible tau-induced shift | 1.000e-12 | 1.000e-08 | 0.0001 | pass | pass |
| small_allowed_shift | small allowed tau shift | 1.000e-09 | 1.000e-08 | 0.1000 | pass | pass |
| borderline_shift | borderline tau shift | 8.000e-09 | 1.000e-08 | 0.8000 | borderline | borderline |
| too_large_shift | overlarge tau shift (expected fail) | 5.000e-07 | 1.000e-08 | 50.0000 | fail | fail |
| galaxy_rotation_doppler_proxy | galaxy rotation Doppler proxy | 1.000e-07 | 1.000e-06 | 0.1000 | pass | pass |
| cluster_gravitational_redshift_proxy | cluster gravitational redshift proxy | 5.000e-06 | 1.000e-05 | 0.5000 | pass | pass |
| precision_clock_proxy | precision clock / weak-field proxy | 1.000e-13 | 1.000e-12 | 0.1000 | pass | pass |

## Interpretation

- **Passing** means only that the configured tau residual is below the chosen benchmark tolerance.
- It does **not** validate TDF against real spectra or cosmological data.
- A **failure** is useful: the framework can reject overlarge tau redshift residuals.
- **Borderline** cases flag configurations near the cap (see warnings in CSV).

## Failure modes

- z_tau is **configured**, not derived from a full TDF metric.
- Real spectra, peculiar velocities, gravitational redshift, and Doppler decomposition are **not** fitted.
- Rotation curves inferred from Doppler shifts require care to avoid **double-counting** tau and kinematic terms.

## Disclaimer

- **NOT REAL OBSERVATIONAL DATA**
- Does **not** replace spectral fitting or multi-component redshift analysis.

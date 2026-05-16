# GR-safe local benchmark report (Phase 4B)

## ⚠️ GR-SAFE ΛCDM/GR BENCHMARK — NOT REAL OBSERVATIONAL DATA

> **This is not observational validation.** Configurable assumed Φ_τ / ε_τ inputs only. > **Real observational constraints are not yet fitted** (no ephemeris, Cassini, or LLR fits).

## Scientific interpretation

**Passing** means |ε_τ| = |Φ_τ/Φ_b| stays below the configured cap for that GR-success regime — a **compatibility scaffold** showing TDF corrections can remain suppressed where GR is already strong.

**Passing does not validate TDF** against solar-system observations or ΛCDM.

## Summary

- **Cases:** 7
- **Pass:** 7
- **Fail:** 0

## Pass/fail table

| Case | Regime | ε_τ | max allowed | Status |
| --- | --- | --- | --- | --- |
| mercury_perihelion | Mercury perihelion regime | -4.706e-19 | 1.000e-08 | pass |
| earth_orbit | Earth orbit regime | -1.597e-18 | 1.000e-08 | pass |
| gps_weak_field_clock | GPS weak-field clock regime | -1.379e-18 | 1.000e-08 | pass |
| light_bending_sun | light bending near Sun | -1.000e-20 | 1.000e-09 | pass |
| shapiro_delay | Shapiro delay regime | -2.105e-20 | 1.000e-09 | pass |
| lunar_laser_ranging | lunar laser ranging scale | -4.167e-20 | 1.000e-08 | pass |
| binary_pulsar_weak_field | binary pulsar weak-field timing proxy | 1.000e-12 | 1.000e-07 | pass |

## Per-case detail

### mercury_perihelion

- **Regime:** Mercury perihelion regime
- **Φ_b scale (nominal):** -8.5e+07
- **Assumed Φ_τ:** 4e-11
- **Computed ε_τ:** -4.705882e-19
- **max_allowed_ε:** 1.000e-08
- **Result:** pass

### earth_orbit

- **Regime:** Earth orbit regime
- **Φ_b scale (nominal):** -6.2637e+07
- **Assumed Φ_τ:** 1e-10
- **Computed ε_τ:** -1.596500e-18
- **max_allowed_ε:** 1.000e-08
- **Result:** pass

### gps_weak_field_clock

- **Regime:** GPS weak-field clock regime
- **Φ_b scale (nominal):** -5.8e+07
- **Assumed Φ_τ:** 8e-11
- **Computed ε_τ:** -1.379310e-18
- **max_allowed_ε:** 1.000e-08
- **Result:** pass

### light_bending_sun

- **Regime:** light bending near Sun
- **Φ_b scale (nominal):** -1e+08
- **Assumed Φ_τ:** 1e-12
- **Computed ε_τ:** -1.000000e-20
- **max_allowed_ε:** 1.000e-09
- **Result:** pass

### shapiro_delay

- **Regime:** Shapiro delay regime
- **Φ_b scale (nominal):** -9.5e+07
- **Assumed Φ_τ:** 2e-12
- **Computed ε_τ:** -2.105263e-20
- **max_allowed_ε:** 1.000e-09
- **Result:** pass

### lunar_laser_ranging

- **Regime:** lunar laser ranging scale
- **Φ_b scale (nominal):** -1.2e+07
- **Assumed Φ_τ:** 5e-13
- **Computed ε_τ:** -4.166667e-20
- **max_allowed_ε:** 1.000e-08
- **Result:** pass

### binary_pulsar_weak_field

- **Regime:** binary pulsar weak-field timing proxy
- **Φ_b scale (nominal):** -5e+06
- **Assumed Φ_τ:** -5e-06
- **Assumed ε_τ input:** 1.000e-12
- **Computed ε_τ:** 1.000000e-12
- **max_allowed_ε:** 1.000e-07
- **Result:** pass

## Failure modes

- Assumed Φ_τ is a **placeholder**, not derived from a dynamical TDF metric.
- Φ_b scales are order-of-magnitude only; real ephemeris coupling is future work.
- A fail here would indicate the scaffold assumption violates the configured GR-safe cap.

## Disclaimer

- **NOT REAL OBSERVATIONAL DATA**
- Does **not** replace PPN bounds, Cassini, GPS clock fits, or LLR analysis.
- Phase 4C/4D (BH exterior, redshift) are separate benchmarks.

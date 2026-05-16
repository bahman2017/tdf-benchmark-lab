# Rotation pipeline report

> **Disclaimer:** Calibration diagnostics only. **This is not a claim that TDF is validated.**

## ⚠️ DEMO FIXTURE DATA — NOT REAL SPARC

## Dataset labeling

| Field | Value |
| --- | --- |
| **Dataset mode** | `demo_fixture_calibration` |
| **Dataset type** | `demo_fixture` |
| **Dataset source** | test_fixture |
| **Real observational data** | False |
| **Description** | Small demo dataset used to test pipeline behavior. |
| **Galaxies** | 2 |
| **Processed CSV** | `/Users/bahmanmasarratbakhsh/Library/Mobile Documents/com~apple~CloudDocs/TDF/grProjection/tdf_observational_calibration/data/processed/rotation.csv` |
| **Metadata sidecar** | `/Users/bahmanmasarratbakhsh/Library/Mobile Documents/com~apple~CloudDocs/TDF/grProjection/tdf_observational_calibration/data/processed/rotation_metadata.yaml` |

## Baseline comparison (Phase 3)

Models compared: baryon-only (0 parameters), TDF simple (2), NFW simple (2). Selection by BIC uses chi² as likelihood proxy. This does not prove TDF is correct.

## Models

**TDF simple:**

```text
v_model^2(r) = v_baryon^2(r) + B * r / (r + r0)   [tau_bar_l(r) = A log(1 + r/r0), B = K_tau * A; r in kpc, v in km/s, B in km^2/s^2]
```

**NFW simple (comparison halo):**

```text
v_DM^2(r) = v_baryon^2(r) + Vh2 * [ln(1+r/rs) - (r/rs)/(1+r/rs)] / (r/rs)   [phenomenological NFW-like baseline; Vh2 in km^2/s^2, rs in kpc]
```

## Galaxy: `DDO154`

#### TDF simple
- tdf_B = 1852.77 km²/s²
- tdf_r0 = 4.79976 kpc
- success_tdf = True

#### NFW simple
- nfw_Vh2 = 5859.16 km²/s²
- nfw_rs = 7.77991 kpc
- success_nfw = True

#### Metrics
- χ² baryon = 40.8956 (reduced 8.1791, n_params=0)
- χ² TDF = 0.282036 (reduced 0.0940, n_params=2)
- χ² NFW = 0.212156 (reduced 0.0707, n_params=2)
- AIC: baryon=40.8956, TDF=4.2820, NFW=4.2122
### Baseline comparison

- **Best model by BIC:** `nfw_simple`
- **TDF beats baryon-only (BIC):** True
- **TDF beats NFW simple (BIC):** False
- MSE baryon = 60.8, TDF = 0.380608, NFW = 0.271117
- BIC baryon = 40.8956, TDF = 3.5009, NFW = 3.4310
- TDF vs baryon MSE improvement = 99.37%
- TDF vs NFW MSE improvement = -40.38%

_Lower MSE with more parameters can mislead; prefer BIC for model ranking._


## Galaxy: `NGC2403`

#### TDF simple
- tdf_B = 5753.89 km²/s²
- tdf_r0 = 12.0019 kpc
- success_tdf = True

#### NFW simple
- nfw_Vh2 = 17860.7 km²/s²
- nfw_rs = 18.9971 kpc
- success_nfw = True

#### Metrics
- χ² baryon = 117.264 (reduced 23.4528, n_params=0)
- χ² TDF = 0.885172 (reduced 0.2951, n_params=2)
- χ² NFW = 0.733772 (reduced 0.2446, n_params=2)
- AIC: baryon=117.2642, TDF=4.8852, NFW=4.7338
### Baseline comparison

- **Best model by BIC:** `nfw_simple`
- **TDF beats baryon-only (BIC):** True
- **TDF beats NFW simple (BIC):** False
- MSE baryon = 259.6, TDF = 1.51813, NFW = 1.20081
- BIC baryon = 117.2642, TDF = 4.1040, NFW = 3.9526
- TDF vs baryon MSE improvement = 99.42%
- TDF vs NFW MSE improvement = -26.42%

_Lower MSE with more parameters can mislead; prefer BIC for model ranking._


## Limitations

- This is a calibration diagnostic, not a validation of TDF.
- B and K_tau * A are not separated; only the combined parameter B is fitted.
- v_baryon(r) is an input, not fitted here.
- NFW simple is a phenomenological comparison halo, not a unique physical fit.
- Lower MSE alone does not establish a better model when parameter counts differ; use BIC.
- No lensing, redshift, or solar-system consistency checks in this pipeline.
- Poor fits do not imply TDF failure until multi-channel tests are run.

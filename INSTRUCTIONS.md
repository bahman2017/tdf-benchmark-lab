# TDF Observational Calibration — Cursor Instructions

## 1. Project Identity

This project is a numerical validation and calibration framework for the **Time Delay Field (TDF)** theory.

The goal is **not** to prove TDF immediately.

The goal is to build a clean, testable, falsifiable computational framework that checks whether **TDF v0.8.1 — Consistency Cleanup** can survive observational constraints.

The project must preserve scientific honesty, reproducibility, and falsifiability.

---

## 2. Current Theoretical Version

```text
TDF v0.8.1 — Consistency Cleanup
```

The v0.8.1 version corrected the phase convention and unified the gravitational potential convention.

Use the corrected wavefunction convention:

```math
\psi=\sqrt{\rho}e^{-i\tau}
```

with:

```math
E=\hbar\partial_t\tau
```

Do **not** revert to:

```math
\psi=\sqrt{\rho}e^{i\tau}
```

unless explicitly testing sign conventions.

The older positive-phase form appears in earlier TDF notes, including the additional phase geometry relation document, so be careful not to copy that convention into the v0.8.1 calibration project without correction.

---

## 3. Core TDF Equations

### 3.1 Tau potential

Always use:

```math
\Phi_\tau = K_\tau \bar{\tau}_\ell
```

Do **not** write:

```math
\Phi_\tau \sim \tau
```

except in historical notes.

If using a simplified fit model, combine:

```math
B=K_\tau A
```

and document that \(B\) is an effective fitted parameter.

---

### 3.2 Tau acceleration

```math
a_\tau^i = -K_\tau \partial^i \bar{\tau}_\ell
```

This is the calibrated version of the earlier conceptual relation:

```math
a \sim -\nabla\tau
```

---

### 3.3 Rotation curve model

The first-pass galaxy-scale weak-field ansatz is:

```math
\bar{\tau}_\ell(r)=A\log(1+r/r_0)
```

Therefore:

```math
v_{\rm TDF}^2(r)
=
v_b^2(r)
+
B\frac{r}{r+r_0}
```

where:

```math
B=K_\tau A
```

Use this as the initial weak-field galaxy-scale model.

Do **not** introduce unnecessary complex profiles until this simple model is tested.

---

### 3.4 Lensing consistency

The same \(\tau\) profile used for rotation must be used for lensing.

Core condition:

```math
\Psi_\tau\simeq\Phi_\tau=K_\tau\bar{\tau}_\ell
```

Lensing must see the same effective metric potential that matter sees:

```math
\boldsymbol{\alpha}_{\rm lens}
\simeq
\frac{2}{c^2}
\int\nabla_\perp(\Phi_b+K_\tau\bar{\tau}_\ell)\,dl
```

Do not fit lensing with an unrelated independent \(\tau\) field unless explicitly labeled as a failed consistency fallback.

---

### 3.5 Doppler/redshift consistency

Use:

```math
z_{\rm obs}
\simeq
z_{\rm kin}
+
z_b
+
z_\tau
```

with:

```math
z_\tau\simeq\frac{K_\tau\Delta\bar{\tau}_\ell}{c^2}
```

Important:

Rotation curves are often inferred from Doppler shifts. Redshift tests must be treated carefully to avoid double-counting velocity effects.

---

### 3.6 Solar-system GR-safe limit

TDF must not break known GR successes.

Use:

```math
\epsilon_\tau=\frac{\Phi_\tau}{\Phi_b}
```

In solar-system regimes:

```math
\epsilon_\tau \ll 1
```

Prefer strict constraints.

If a model requires large \(\tau\) corrections in the solar system, mark it as failed.

---

### 3.7 Black-hole phenomenological ansatz

Black-hole formulas in v0.8.1 are **phenomenological strong-field ansatz**, not full nonlinear TDF black-hole solutions.

Always label them as:

```text
phenomenological strong-field ansatz
```

Core formulas:

```math
\Phi_\tau^{\rm core}(r)
=
-\frac{GM}{\sqrt{r^2+r_c^2}}
```

```math
r_s=\frac{2GM}{c^2}
```

```math
r_{\rm nr}^{\rm TDF}
=
\sqrt{r_s^2-r_c^2}
```

```math
T_H=
\frac{\hbar c^3}{8\pi GM k_B}
```

```math
T_{\rm TDF}
\simeq
T_H
\sqrt{1-\frac{r_c^2}{r_s^2}}
```

```math
M_{\rm rem}
\simeq
\frac{c^2r_c}{2G}
```

Rules:

- If \(r_c=0\), recover the GR/Hawking limit.
- If \(r_c\ll r_s\), TDF must be close to GR.
- If \(r_c\to r_s\), \(T_{\rm TDF}\to0\).
- If \(r_c>r_s\), return `NaN`, zero, or structured warning; do not produce fake horizon values.

---

## 4. Scientific Rule: No Overclaiming

Never write code, documentation, reports, plot titles, or output summaries that claim:

- TDF is proven
- TDF is validated
- TDF replaces ΛCDM
- TDF solves dark matter
- TDF solves black-hole information paradox
- TDF is experimentally confirmed
- GR is disproven
- dark matter is disproven

Allowed language:

- calibration test
- synthetic validation
- phenomenological fit
- constraint check
- diagnostic result
- candidate model
- not yet validated
- requires multi-channel consistency
- preliminary numerical result
- falsifiability framework

Forbidden language:

- proof
- confirmed
- solved
- verified
- final theory
- dark matter disproven
- GR disproven

Every report must include this warning:

```text
These results are calibration diagnostics for a phenomenological TDF model. They do not constitute observational validation of TDF unless all required multi-channel constraints are passed.
```

---

## 5. Development Philosophy

Build step by step.

Do not implement all future science at once.

Priority order:

1. Synthetic rotation curve test
2. Real SPARC rotation curve ingestion
3. Baseline comparison
4. Lensing consistency
5. Doppler/redshift consistency
6. Solar-system GR-safe limits
7. Black-hole phenomenological checks
8. Multi-channel report

Every phase must be independently runnable and testable.

Do not move to a later phase unless the previous phase has:

- working code
- unit tests
- clear outputs
- report file
- documented assumptions

---

## 6. Project Structure

Create and maintain this structure:

```text
tdf_observational_calibration/
  INSTRUCTIONS.md
  README.md
  pyproject.toml
  requirements.txt
  configs/
    rotation_sparc.yaml
    lensing.yaml
    redshift.yaml
    solar_system.yaml
    black_hole.yaml
    full_pipeline.yaml
  data/
    raw/
      README.md
    processed/
      README.md
    synthetic/
      README.md
  src/
    tdf_obs/
      __init__.py
      constants.py
      io/
        __init__.py
        schemas.py
        loaders.py
      models/
        __init__.py
        tau_profiles.py
        rotation.py
        lensing.py
        redshift.py
        solar_system.py
        black_hole.py
      fitting/
        __init__.py
        metrics.py
        optimizers.py
        fit_rotation.py
        fit_lensing.py
        fit_redshift.py
      validation/
        __init__.py
        constraints.py
        synthetic_tests.py
        consistency_checks.py
      plotting/
        __init__.py
        plot_rotation.py
        plot_lensing.py
        plot_redshift.py
        plot_black_hole.py
      pipelines/
        __init__.py
        run_rotation_pipeline.py
        run_lensing_pipeline.py
        run_redshift_pipeline.py
        run_solar_system_constraints.py
        run_black_hole_pipeline.py
        run_full_pipeline.py
  scripts/
    run_rotation.py
    run_lensing.py
    run_redshift.py
    run_solar_system.py
    run_black_hole.py
    run_full_pipeline.py
  tests/
    test_tau_profiles.py
    test_rotation_model.py
    test_metrics.py
    test_solar_constraints.py
    test_black_hole_formulas.py
    test_synthetic_pipeline.py
  outputs/
    figures/
    tables/
    reports/
  docs/
    TEST_PLAN.md
    SCIENTIFIC_ASSUMPTIONS.md
    DATA_REQUIREMENTS.md
    ROADMAP.md
```

---

## 7. Data Rules

### 7.1 Never fake real data

If real data is missing, run synthetic tests only.

Clearly label outputs:

```text
SYNTHETIC ONLY — NO REAL DATA USED
```

Do not silently generate fake “real” results.

---

### 7.2 Data folders

Use:

```text
data/raw/
data/processed/
data/synthetic/
```

Rules:

- `raw/` contains untouched downloaded or provided files.
- `processed/` contains standardized CSVs.
- `synthetic/` contains generated test data.

Each folder should have a README explaining expected files.

---

### 7.3 Rotation data schema

Use this schema for processed rotation data:

```text
galaxy_id,r_kpc,v_obs,v_err,v_baryon
```

Optional columns may include:

```text
v_gas,v_disk,v_bulge,source,quality_flag
```

Never assume missing uncertainty is zero.

If `v_err` is missing, use unweighted metrics and clearly report that.

---

### 7.4 Lensing data schema

Use:

```text
lens_id,r_kpc,alpha_obs,alpha_err,baryon_potential_or_mass
```

If required baryonic mass/potential information is missing, do not compute fake lensing. Return a structured “insufficient data” report.

---

### 7.5 Redshift data schema

Use:

```text
object_id,z_obs,z_kin,z_baryon,z_err
```

Then compute residual:

```text
z_residual = z_obs - z_kin - z_baryon
```

Compare against predicted:

```text
z_tau
```

---

## 8. Coding Standards

Use Python.

Use:

- numpy
- pandas
- scipy
- matplotlib
- pydantic or dataclasses
- pytest
- pathlib
- typing

Do not use notebooks for core logic.

Notebooks are allowed only for exploration, not production pipeline.

Core logic must live under:

```text
src/tdf_obs/
```

Scripts must live under:

```text
scripts/
```

Tests must live under:

```text
tests/
```

Outputs must go to:

```text
outputs/figures/
outputs/tables/
outputs/reports/
```

---

## 9. Unit Handling

Be explicit about units.

For the first rotation model:

- radius \(r\): kpc
- velocity \(v\): km/s
- \(B\): km²/s²
- \(r_0\): kpc

Document this clearly.

For black-hole formulas, use SI units:

- mass: kg
- radius: meter
- temperature: kelvin

Do not mix SI and astrophysical units silently.

Each model function must state expected units in its docstring.

---

## 10. Implementation Requirements

### 10.1 constants.py

Define physical constants in SI units:

- `c`
- `G`
- `hbar`
- `k_B`
- `solar_mass`

---

### 10.2 io/schemas.py

Define dataclasses or pydantic models for:

- `RotationCurveData`
- `LensingData`
- `RedshiftData`
- `SolarSystemConstraint`
- `BlackHoleData`

Expected fields:

```text
RotationCurveData:
  galaxy_id, r_kpc, v_obs, v_err, v_baryon

LensingData:
  lens_id, r_kpc, alpha_obs, alpha_err, baryon_potential_or_mass

RedshiftData:
  object_id, z_obs, z_kin, z_baryon, z_err

SolarSystemConstraint:
  name, radius_m, phi_b, max_epsilon_tau

BlackHoleData:
  object_id, mass_kg, optional_shadow_radius, optional_ringdown_freq
```

---

### 10.3 models/tau_profiles.py

Implement:

- `tau_log_profile(r, A, r0)`
- `d_tau_log_dr(r, A, r0)`
- `tau_core_potential(r, M, rc)`
- `d_tau_core_potential_dr(r, M, rc)`

Use numpy arrays.

---

### 10.4 models/rotation.py

Implement:

- `v2_tdf_simple(r, v_baryon, B, r0)`
- `v_tdf_simple(r, v_baryon, B, r0)`
- `baryon_only_model(v_baryon)`

Formula:

```math
v^2 = v_b^2 + B\frac{r}{r+r_0}
```

Use radius in kpc, velocity in km/s, and \(B\) in km²/s² for the first version.

---

### 10.5 fitting/metrics.py

Implement:

- `mse`
- `weighted_mse`
- `chi_square`
- `reduced_chi_square`
- `aic`
- `bic`
- `percent_improvement`

---

### 10.6 fitting/fit_rotation.py

Implement:

- `fit_single_galaxy_rotation(data)`

Fit \(B\) and \(r_0\) using `scipy.optimize.curve_fit` or `scipy.optimize.least_squares`.

Compare:

- baryon-only baseline
- TDF simple model

Return:

- `best_B`
- `best_r0`
- `mse_baryon`
- `mse_tdf`
- `chi2_baryon`
- `chi2_tdf`
- `improvement_percent`
- `success`
- `warning_messages`

---

### 10.7 validation/synthetic_tests.py

Create synthetic data generator:

- `generate_synthetic_rotation_curve(...)`

Generate `v_obs` from known \(B\) and \(r_0\) plus noise.

This lets us test whether the fitter can recover the injected parameters.

---

### 10.8 plotting/plot_rotation.py

Create plot:

- observed data with error bars
- baryon-only curve
- TDF fitted curve

Save to:

```text
outputs/figures/
```

---

### 10.9 scripts/run_rotation.py

Run synthetic test first if no real data exists.

If:

```text
data/processed/rotation.csv
```

exists, then run real-data fitting.

Save:

```text
outputs/tables/rotation_fit_summary.csv
outputs/reports/rotation_report.md
outputs/figures/<galaxy_id>_rotation.png
```

---

### 10.10 models/lensing.py

Prepare placeholder functions but do not fake results.

Allowed:

- `alpha_lens_tdf_placeholder(...)`
- structured warning if data is missing
- `NotImplementedError` if the required lensing model is not implemented

Do not invent lensing results.

---

### 10.11 models/redshift.py

Implement:

```python
z_tau(delta_tau, K_tau)
```

Formula:

```math
z_\tau=\frac{K_\tau\Delta\tau}{c^2}
```

Also implement placeholders for observational residual testing.

---

### 10.12 models/solar_system.py

Implement:

- `epsilon_tau(phi_tau, phi_b)`
- `passes_gr_safe_limit(epsilon, max_allowed)`

This checks whether tau corrections are small enough in solar-system regimes.

---

### 10.13 models/black_hole.py

Implement:

- `schwarzschild_radius(M)`
- `non_return_radius_tdf(M, rc)`
- `hawking_temperature(M)`
- `tdf_temperature(M, rc)`
- `remnant_mass(rc)`

If \(r_c>r_s\), `non_return_radius_tdf` should return `np.nan` or `0` with a warning.

---

## 11. Testing Requirements

Every implemented model must have tests.

### 11.1 Tau profile tests

- `tau_log_profile` returns finite values.
- `d_tau_log_dr` matches numerical derivative.
- derivative is positive for positive \(A,r_0,r\).

### 11.2 Rotation tests

- `v2_tdf_simple` returns finite positive values for valid inputs.
- `v_tdf_simple` equals sqrt of v².
- baryon-only model matches input baryonic velocity.
- synthetic fitter recovers injected \(B,r_0\) within tolerance.

### 11.3 Metrics tests

- MSE works.
- Weighted MSE works.
- Chi-square works.
- Reduced chi-square handles degrees of freedom correctly.
- AIC/BIC handle sample size and parameter count.

### 11.4 Solar tests

- `epsilon_tau` computes correct ratio.
- `passes_gr_safe_limit` returns true/false correctly.
- zero baryonic potential handled safely.

### 11.5 Black-hole tests

- `schwarzschild_radius` positive for positive mass.
- `non_return_radius_tdf(M,0)=r_s`.
- `non_return_radius_tdf` approaches zero when \(r_c\to r_s\).
- invalid \(r_c>r_s\) handled with warning/NaN.
- `tdf_temperature(M,0)=hawking_temperature(M)`.
- `tdf_temperature` approaches zero when \(r_c\to r_s\).
- `remnant_mass(rc)` positive for positive \(r_c\).

### 11.6 Pipeline tests

- synthetic rotation pipeline runs end-to-end.
- outputs are created.
- synthetic outputs are labeled synthetic.
- missing real data does not crash pipeline.

---

## 12. Reporting Rules

Every pipeline must produce:

1. CSV summary table
2. Markdown report
3. Diagnostic plot if applicable

Reports must include:

- model equations used
- data source
- whether data is synthetic or real
- fitted parameters
- metrics
- warnings
- limitations
- next steps
- self-criticism and failure modes

Reports should never overclaim.

Example report language:

```text
The TDF simple rotation model improves the synthetic fit relative to the baryon-only baseline under the injected test conditions. This confirms code recovery behavior only; it does not validate TDF observationally.
```

---

## 13. Baseline Comparison Rules

Always compare TDF against baselines.

For Phase 1:

- baryon-only baseline

For later phases:

- baryon-only
- simple NFW halo
- MOND-like acceleration law if implemented

Do not claim success unless TDF is compared to at least one baseline.

---

## 14. Optimization Rules

Use stable fitting.

For rotation, fit:

```math
v^2_{\rm model}=v_b^2+B\frac{r}{r+r_0}
```

Parameters:

```text
B > 0
r0 > 0
```

Use reasonable bounds.

If fit fails:

- do not crash
- return `success=false`
- store warning
- report failed fit

Do not hide failed galaxies.

---

## 15. Plotting Rules

Use matplotlib only.

Do not use seaborn.

Each plot must include:

- title
- axis labels with units
- legend
- data source label
- synthetic/real label

Rotation plot must show:

- observed data with error bars
- baryon-only curve
- TDF fitted curve

Save plots to:

```text
outputs/figures/
```

---

## 16. Documentation Requirements

Maintain these files:

```text
docs/TEST_PLAN.md
docs/SCIENTIFIC_ASSUMPTIONS.md
docs/DATA_REQUIREMENTS.md
docs/ROADMAP.md
```

### 16.1 TEST_PLAN.md

Must include phases:

1. Synthetic rotation validation
2. SPARC ingestion
3. TDF vs baselines
4. Lensing consistency
5. Doppler/redshift residuals
6. Solar-system GR-safe checks
7. Black-hole phenomenological checks
8. Multi-channel falsifiability report

### 16.2 SCIENTIFIC_ASSUMPTIONS.md

Must state:

- TDF v0.8.1 is phenomenological in strong field.
- Rotation test is weak-field and galaxy-scale.
- \(B=K_\tau A\) is an effective parameter.
- No validation claim until multi-channel consistency passes.
- Black-hole formulas are ansatz-level.

### 16.3 DATA_REQUIREMENTS.md

Must state exact required data formats.

### 16.4 ROADMAP.md

Must describe what is implemented now and what is future work.

---

## 17. Safety Against Self-Deception

When results look good, automatically ask:

1. Did we compare against baryon-only?
2. Did we compare against a dark-matter halo baseline?
3. Did we use real data or synthetic data?
4. Did we fit too many parameters?
5. Did the same \(\tau\) explain lensing?
6. Did redshift constraints remain acceptable?
7. Did solar-system constraints pass?
8. Are we using a model that only works because of flexible fitting?

Reports should include a section:

```text
Self-criticism and failure modes
```

---

## 18. Versioning Policy

Use scientific versioning:

```text
v0.8.1 — theory consistency cleanup
v0.9.0 — observational calibration framework
v0.9.1 — SPARC ingestion
v0.9.2 — baseline comparison
v0.9.3 — lensing consistency
v0.9.4 — redshift residual constraints
v0.9.5 — solar-system GR-safe constraints
v0.9.6 — black-hole phenomenological diagnostics
v1.0.0 — first multi-channel falsifiability report
```

Do not call anything v1.0.0 until there is a complete report across multiple observational channels.

---

## 19. Cursor Workflow Rules

When implementing:

1. Read this file first.
2. Implement only the requested phase.
3. Add or update tests.
4. Run tests.
5. Summarize changed files.
6. Summarize how to run.
7. State what remains incomplete.

Do not jump ahead.

Do not silently rewrite scientific assumptions.

Do not rename core variables without updating documentation.

Do not remove warnings from reports.

---

## 20. First Implementation Target

Current target:

```text
Phase 1 — Synthetic rotation curve validation
```

Implement:

- project structure
- core constants
- tau log profile
- simple TDF rotation model
- synthetic data generator
- fitting of \(B,r_0\)
- baryon-only comparison
- metrics
- rotation plot
- CSV summary
- Markdown report
- unit tests

Stop after Phase 1.

Do not fully implement real SPARC ingestion yet.

Create placeholders and docs for future phases only.

---

## 21. Expected Commands

The project should support:

```bash
pip install -r requirements.txt
pytest
python scripts/run_rotation.py
python scripts/run_full_pipeline.py
```

If real data is missing, `run_rotation.py` must run synthetic validation and clearly label outputs as synthetic.

---

## 22. README Requirement

Add this line near the top of `README.md`:

```markdown
Before modifying this project, read [INSTRUCTIONS.md](./INSTRUCTIONS.md).
```

---

## 23. Cursor Prompt Reminder

Before coding, Cursor should be given this reminder:

```text
Before coding, read INSTRUCTIONS.md and follow it strictly. Do not overclaim results. Implement only Phase 1 unless I explicitly request the next phase.
```

---

## 24. Final Scientific Reminder

This project is successful if it can honestly answer:

```text
Where does TDF pass, where does it fail, and what data would falsify it?
```

Not:

```text
How can we make TDF look correct?
```

The goal is scientific honesty, reproducibility, and falsifiability.

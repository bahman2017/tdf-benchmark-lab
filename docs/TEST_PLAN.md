# TDF Observational Calibration — Test Plan

**Framework goal:** build constraints and calibration diagnostics. **Not** a claim of theoretical validation.

**Current project mode:** **ΛCDM Compatibility and Recovery Testing** — see `docs/LCDM_COMPATIBILITY_STRATEGY.md`.

> ⚠️ **Real observational calibration is intentionally postponed until ΛCDM compatibility tests (Phase 4) are completed.** Do not treat demo fixtures, surrogates, or benchmark teachers as real sky data.

Status labels used in reports:

- `synthetic_validation` — injected parameters / noise tests
- `nfw_surrogate_validation` — GR+DM/NFW teacher, not real sky data
- `lcdm_benchmark_recovery` — ΛCDM/GR scaffold benchmarks, not real sky data
- `real_data_calibration` — fits to observational CSVs (**Phase 6 only**, when metadata confirms real source)
- `passed` / `failed` — constraint outcome
- `not_yet_tested` — channel not implemented
- `not_implemented` — placeholder API only

---

## Phase 1 — Synthetic rotation curve validation ✅ (implemented)

**Objective:** Verify the fitter recovers known `(B, r0)` from noisy synthetic data.

**Steps:**

1. `generate_synthetic_rotation_curve()` injects `v_obs` from true `B`, `r0`.
2. `fit_single_galaxy_rotation()` fits TDF simple model vs baryon-only.
3. Assert `|B_fit - B_true| / B_true < 15%`, `|r0_fit - r0_true| / r0_true < 25%` (noise-dependent).
4. Plot observed / baryon / TDF curves → `outputs/figures/`.
5. Write `rotation_report.md` with **synthetic_validation** label.

**Commands:** `pytest tests/test_synthetic_pipeline.py`, `python scripts/run_rotation.py`

---

## Phase 2 — SPARC ingestion scaffold ✅ (infrastructure only)

**Objective:** Parse user-supplied SPARC-like files into processed tables **when the user provides raw data**.

**Steps:**

1. User places raw files in `data/raw/` (no auto-download).
2. `prepare_sparc_rotation.py` → `data/processed/rotation.csv` + metadata.
3. Loader and batch fit hooks exist for `run_rotation.py`.

**Status:** implemented as **scaffold**. **Not** the active calibration path until Phase 6.

**Note:** Demo fixture mode (`demo_fixture_calibration`) must never be labeled as real SPARC.

---

## Phase 3 — Baseline comparison ✅ (implemented)

Compare per-galaxy:

- Baryon-only (0 parameters)
- TDF simple `(B, r0)` — 2 parameters
- NFW simple `(Vh2, r_s)` — 2 parameters

Selection by **BIC** (chi² proxy); report warns that lower MSE alone is insufficient.

**Commands:** `python scripts/run_rotation.py`

**Status:** implemented.

---

## Phase 3B — NFW surrogate recovery ✅ (implemented)

**Question:** Can TDF reproduce GR+DM/NFW-like phenomenology in controlled benchmarks?

**Method:**

1. Fix a smooth baryon profile `v_baryon(r)`.
2. Teacher: `v_teacher² = v_baryon² + Vh2·f(r/rs)` (NFW halo term).
3. `v_obs = v_teacher + noise` (surrogate, not SPARC).
4. Fit student TDF: `v_TDF² = v_baryon² + B·r/(r+r0)`.
5. Compare TDF curve to teacher (MSE, relative error, mimic tolerance).

**Banner:** `NFW SURROGATE TEST — NOT REAL OBSERVATIONAL DATA`

**Commands:** `python scripts/run_nfw_surrogate.py`

**Outputs:** `nfw_surrogate_fit_summary.csv`, `nfw_surrogate_report.md`, `nfw_surrogate_*.png`

**Status:** implemented (legacy 3-preset scaffold; superseded by Phase 4A registry).

---

## Phase 3C — ΛCDM/GR benchmark recovery ✅ (implemented)

**Question:** Do TDF scaffolds recover GR/ΛCDM limits outside rotation (solar system, BH exterior, redshift)?

**Method (configurable synthetic cases only):**

1. Solar-system GR-safe: ε_τ = Φ_τ/Φ_b vs `max_allowed_epsilon` (Earth, Mercury, light-bending, GPS-like).
2. Black-hole GR-limit: r_nr/r_s and T/T_H vs √(1 − (r_c/r_s)²) at listed r_c/r_s values.
3. Redshift sanity: z_τ = K_τ Δτ̄_l / c² vs tolerance (negligible, small, borderline, too_large).

**Banner:** `ΛCDM BENCHMARK RECOVERY — NOT REAL OBSERVATIONAL DATA`

**Commands:** `python scripts/run_lcdm_benchmark.py`

**Outputs:** `lcdm_benchmark_summary.csv`, `lcdm_benchmark_report.md`

**Status:** implemented scaffold. Expansion → **Phase 4B–4D**.

---

## Phase 4 — ΛCDM Compatibility Expansion

**Purpose:** Before real observational data, test TDF against ΛCDM/GR+DM benchmark outputs in regimes where ΛCDM already succeeds.

**Banner:** `ΛCDM/NFW BENCHMARK — NOT REAL OBSERVATIONAL DATA`

### Phase 4A — Rotation NFW recovery expansion ✅ (implemented)

**Objective:** ≥10 galaxy-like ΛCDM/NFW teacher profiles; TDF student mimic vs teacher.

**Registry cases:** `dwarf_low_mass`, `dwarf_extended_lsb`, `compact_dwarf`, `lsb_diffuse`, `spiral_mid_mass`, `milky_way_like`, `high_surface_brightness`, `massive_spiral`, `extended_disk`, `concentrated_halo`.

**Baryon profiles:** `saturating_disk`, `exponential_like`, `compact_bulge_disk`.

**Mimic default:** mean relative curve error &lt; 5% (CLI `--tolerance`).

**Commands:**

```bash
python scripts/run_nfw_surrogate.py
python scripts/run_nfw_surrogate.py --case milky_way_like --tolerance 5.0
python scripts/run_nfw_surrogate.py --max-cases 3
```

**Outputs:** `nfw_surrogate_fit_summary.csv`, `nfw_surrogate_report.md`, `nfw_surrogate_<case>.png`

**Status:** implemented. **Not** observational validation.

| Subphase | Objective | Status |
|----------|-----------|--------|
| **4B** | Expanded solar-system GR-safe ε_τ tests | ✅ implemented |
| **4C** | Black-hole exterior GR recovery (q = rc/rs sweep) | ✅ implemented |
| **4D** | Redshift/Doppler sanity benchmark expansion | ✅ implemented |

### Phase 4B — GR-safe local benchmark expansion ✅ (implemented)

**Objective:** Confirm |ε_τ| = |Φ_τ/Φ_b| stays below caps in GR-success regimes (Mercury, Earth, GPS, light bending, Shapiro, LLR, binary pulsar proxy).

**Banner:** `GR-SAFE ΛCDM/GR BENCHMARK — NOT REAL OBSERVATIONAL DATA`

**Commands:** `python scripts/run_gr_safe_benchmark.py`

**Outputs:** `gr_safe_benchmark_summary.csv`, `gr_safe_benchmark_report.md`

**Status:** implemented. Assumed Φ_τ only — **not** observational validation.

### Phase 4C — Black-hole exterior GR recovery ✅ (implemented)

**Objective:** TDF ansatz recovers GR/Hawking when q = r_c/r_s → 0; classify departures vs q.

**Banner:** `BLACK-HOLE GR-LIMIT BENCHMARK — NOT REAL OBSERVATIONAL DATA`

**Commands:** `python scripts/run_black_hole_gr_benchmark.py`

**Outputs:** `black_hole_gr_benchmark_summary.csv`, `black_hole_gr_benchmark_report.md`

**Status:** implemented. Phenomenological formulas only — **not** observational validation.

### Phase 4D — Redshift/Doppler sanity expansion ✅ (implemented)

**Objective:** Configured z_tau values stay within benchmark tolerances in GR-success regimes.

**Banner:** `REDSHIFT SANITY ΛCDM/GR BENCHMARK — NOT REAL OBSERVATIONAL DATA`

**Formula:** z_tau = K_tau Δτ̄_l / c² (z_tau configured directly in benchmark)

**Commands:** `python scripts/run_redshift_sanity_benchmark.py`

**Outputs:** `redshift_sanity_benchmark_summary.csv`, `redshift_sanity_benchmark_report.md`

**Status:** implemented. **Not** spectral or real-data validation.

---

## Phase 5 — ΛCDM Stress / Problem Regimes

**Objective:** Diagnostic tests where ΛCDM has known tensions — **not** validation of TDF.

Phase 4 (compatibility recovery) is complete; stress tests now examine problem regimes.

### Phase 5A — Core–Cusp Stress Test ✅ (implemented)

**Objective:** Compare cuspy NFW-like teachers vs cored halo teachers; test TDF simple vs TDF core proxy.

**Banner:** `CORE-CUSP STRESS TEST — NOT REAL OBSERVATIONAL DATA`

**Commands:** `python scripts/run_core_cusp_stress.py`

**Outputs:** `core_cusp_stress_summary.csv`, `core_cusp_stress_report.md`, `core_cusp_<case>.png`

**Status:** implemented (8 synthetic cases). **Not** real SPARC.

| Subphase | Topic | Status |
|----------|-------|--------|
| **5B** | Rotation-curve diversity stress test | ✅ implemented |
| **5C** | Same-τ multi-observable consistency | ✅ implemented |
| **5D** | Covariant action consistency checks | ✅ implemented |
| **5E** | CMB acoustic-scale compatibility | ✅ implemented |
| **5F** | CMB-safe Hubble tension benchmark | ✅ implemented |
| **5G** | BAO/SNe distance consistency | ✅ implemented |
| **6+** | Missing satellites, real data, etc. | not_yet_tested |

### Phase 5B — Rotation-curve diversity stress test ✅ (implemented)

**Objective:** Ten synthetic teacher shape families; fit baryon, TDF simple, TDF core, NFW; compare BIC.

**Banner:** `ROTATION-CURVE DIVERSITY STRESS TEST — NOT REAL OBSERVATIONAL DATA`

**Commands:** `python scripts/run_rotation_diversity_stress.py`

**Outputs:** `rotation_diversity_stress_summary.csv`, `rotation_diversity_stress_report.md`, `rotation_diversity_<case>.png`

**Status:** implemented. **Not** SPARC / real-sky validation.

### Phase 5C — Same-τ multi-observable consistency benchmark ✅ (implemented)

**Objective:** Fit **(B, r0)** from synthetic rotation only; freeze; predict lensing-proxy
`α_τ ∝ B/(R+r0)` and redshift-proxy `z_τ = ΔΦ_τ/c²` from the same `Φ_τ = B log(1+r/r0)`.

**Banner:** `SAME-TAU MULTI-OBSERVABLE BENCHMARK — NOT REAL OBSERVATIONAL DATA`

**Commands:** `python scripts/run_same_tau_consistency.py`

**Outputs:** `same_tau_consistency_summary.csv`, `same_tau_consistency_report.md`,
`same_tau_<case>_rotation.png`, `same_tau_<case>_lensing.png`, `same_tau_<case>_redshift.png`

**Pass rule:** rotation, lensing, and redshift relative errors all &lt; 5% (controlled synthetic).

**Status:** implemented (6 cases). **Not** observational validation; no separate τ fits per channel.

### Phase 5D — Covariant action consistency checks ✅ (implemented)

**Objective:** Apply EFT/action-level **proxy** sanity checks to fitted **(B, r0)** from the NFW surrogate
benchmark: weak-field identity `r dΦ/dr = B r/(r+r0)`, τ-gradient, disformal safety, k-essence stability,
effective-density finiteness.

**Banner:** `COVARIANT ACTION CONSISTENCY CHECK — NOT REAL OBSERVATIONAL DATA`

**Prerequisite:** `outputs/tables/nfw_surrogate_fit_summary.csv` (auto-runs NFW surrogate if missing unless `--no-run-nfw`).

**Commands:** `python scripts/run_covariant_action_checks.py`

**Outputs:** `covariant_action_checks_summary.csv`, `covariant_action_checks_report.md`

**Status:** implemented. **Not** a proof of the covariant action; **not** observational validation.

### Phase 5E — CMB acoustic-scale compatibility benchmark ✅ (implemented)

**Objective:** Compare ΛCDM teacher background to TDF student `H_TDF² = H_ΛCDM² [1+ε_τ(z)]` on
sound-horizon proxy, comoving distance, and acoustic scale `ℓ_A = π D_M / r_s`.

**Banner:** `CMB ACOUSTIC SCALE BENCHMARK — NOT REAL OBSERVATIONAL DATA`

**Commands:** `python scripts/run_cmb_acoustic_benchmark.py`

**Outputs:** `cmb_acoustic_benchmark_summary.csv`, `cmb_acoustic_benchmark_report.md`,
`cmb_acoustic_epsilon_tau_cases.png`

**Pass rule:** relative errors on H(z_*), r_s, D_M, ℓ_A each &lt; 1% (configurable).

**Status:** implemented (8 cases, including intentional fails). **Not** Planck/ACT/SPT validation.

### Phase 5F — CMB-safe Hubble tension benchmark ✅ (implemented)

**Objective:** Late-time `ε_τ(z)` shifts low-z H(z)/H₀ while preserving CMB acoustic proxies (r_s, D_M, ℓ_A, H(z_*)).

**Banner:** `CMB-SAFE HUBBLE TENSION BENCHMARK — NOT REAL OBSERVATIONAL DATA`

**Commands:** `python scripts/run_hubble_tension_benchmark.py`

**Outputs:** `hubble_tension_benchmark_summary.csv`, `hubble_tension_benchmark_report.md`,
`hubble_tension_epsilon_tau_cases.png`, `hubble_tension_H_ratio_cases.png`

**Overall success:** CMB-safe (all errors &lt; 1%) **and** |H₀ shift| in [2%, 10%].

**Status:** implemented. **Does not** solve the Hubble tension.

### Phase 5G — BAO/SNe late-time distance consistency ✅ (implemented)

**Objective:** Late-time `ε_τ(z)` must preserve SNe-like `D_L` and BAO-like `D_M`, `D_V`, `H(z)` proxies vs ΛCDM teacher.

**Banner:** `BAO/SNe DISTANCE CONSISTENCY BENCHMARK — NOT REAL OBSERVATIONAL DATA`

**Commands:** `python scripts/run_bao_sne_distance_benchmark.py`

**Outputs:** `bao_sne_distance_benchmark_summary.csv`, `bao_sne_distance_benchmark_report.md`, `bao_sne_distance_*.png`

**Overall success:** distance-safe (ΔD_L, ΔD_M, ΔD_V, ΔH within thresholds) **and** |H₀ shift| in [2%, 10%].

**Status:** implemented. **Not** BAO/SNe likelihood validation.

---

## Phase 6A — Schrödinger-from-TDF action benchmark ✅ (implemented)

**Objective:** Numerically verify that the TDF **phase-density action** (ρ, τ with ψ = √ρ exp(-iτ)) reproduces continuity, quantum Hamilton–Jacobi, and Schrödinger residuals in controlled 1D cases.

**Banner:** `SCHRÖDINGER-FROM-TDF ACTION BENCHMARK — NOT FULL QUANTUM VALIDATION`

**Commands:** `python scripts/run_schrodinger_from_tdf.py`

**Module:** `src/tdf_obs/validation/schrodinger_from_tdf.py`

**Derivation note:** [docs/quantum_limit/SCHRODINGER_DERIVATION.md](./quantum_limit/SCHRODINGER_DERIVATION.md)

**Cases:**

1. Free plane wave — ρ = 1, τ = ωt − kx, ω = ℏk²/(2m)
2. Gaussian snapshot — smooth ρ, τ = kx (static slice)
3. Harmonic oscillator ground state — stationary eigenvalue residual

**Outputs:** `schrodinger_from_tdf_summary.csv`, `schrodinger_from_tdf_report.md`, `schrodinger_plane_wave_residual.png`, `schrodinger_gaussian_residual.png`, `schrodinger_harmonic_ground_state.png`

**Tests:** `pytest tests/test_schrodinger_from_tdf.py`

**Status:** implemented. **Not** full quantum gravity; no Dirac/spin/entanglement/decoherence.

---

## Phase 6B — Dirac / spinor limit benchmark ✅ (implemented)

**Objective:** Check whether TDF admits a **covariant spinor** extension (Ψ = √ρ e^{−iτ} χ) compatible with the flat-space Dirac equation, Clifford algebra, disformal metric, and τ-momentum mass ladder.

**Banner:** `DIRAC / SPINOR LIMIT BENCHMARK — NOT FULL FERMION UNIFICATION`

**Commands:** `python scripts/run_dirac_spinor_limit.py`

**Module:** `src/tdf_obs/validation/dirac_spinor_limit.py`

**Cases:** Clifford algebra; flat dispersion; positive-energy plane wave; TDF spinor ansatz density; disformal metric + tetrad; compact τ mass ladder.

**Outputs:** `dirac_spinor_limit_summary.csv`, `dirac_spinor_limit_report.md`, `dirac_dispersion_relation.png`, `compact_tau_mass_ladder.png`

**Tests:** `pytest tests/test_dirac_spinor_limit.py`

**Status:** implemented. **Not** SM unification; no gauge fields, entanglement, or decoherence.

---

## Phase 6C — Entanglement from configuration-space τ benchmark ✅ (implemented)

**Objective:** Test whether **nonseparable configuration-space τ geometry** can represent entangled two-qubit states while preserving **no-signaling** and standard CHSH/Bell statistics.

**Banner:** `ENTANGLEMENT / NONLOCAL CORRELATION BENCHMARK — NOT FULL BELL-THEOREM RESOLUTION`

**Commands:** `python scripts/run_entanglement_tau_geometry.py`

**Module:** `src/tdf_obs/validation/entanglement_tau_geometry.py`

**Cases:** product control; Bell Φ⁺/Ψ⁻; partially entangled; random product; nonseparable τ demo.

**Outputs:** `entanglement_tau_geometry_summary.csv`, `entanglement_tau_geometry_report.md`, `entanglement_chsh_values.png`, `entanglement_entropy_concurrence.png`, `tau_nonseparability_scores.png`

**Tests:** `pytest tests/test_entanglement_tau_geometry.py`

**Status:** implemented. **Not** Bell-theorem resolution; no decoherence or collapse.

---

## Phase 6D — Decoherence from τ-variance benchmark ✅ (implemented)

**Objective:** Test whether growth in **Var(Δτ_AB)** between quantum branches suppresses coherence C_AB = exp(−½ Var) while preserving controlled-rate decay Γ = ½ dVar/dt.

**Banner:** `DECOHERENCE FROM TAU VARIANCE BENCHMARK — NOT FULL MEASUREMENT-PROBLEM SOLUTION`

**Commands:** `python scripts/run_decoherence_tau_variance.py`

**Module:** `src/tdf_obs/validation/decoherence_tau_variance.py`

**Cases:** coherent control; linear decoherence; Gaussian τ noise; correlated-noise protection; environment σ sweep; mass-proxy Γ ∝ M².

**Outputs:** `decoherence_tau_variance_summary.csv`, `decoherence_tau_variance_report.md`, `decoherence_*.png`

**Tests:** `pytest tests/test_decoherence_tau_variance.py`

**Status:** implemented. **Not** collapse or Born-rule derivation.

---

## Phase 6E — Classical metric emergence from τ averaging ✅ (implemented)

**Objective:** Test whether microscopic τ fluctuations can be coarse-grained into a stable effective classical τ field and smooth **disformal metric** g̃ = η + α_τ ∂τ̄ ∂τ̄ (1+1D toy).

**Banner:** `CLASSICAL METRIC EMERGENCE BENCHMARK — NOT FULL OBJECTIVE-COLLAPSE SOLUTION`

**Commands:** `python scripts/run_classical_metric_emergence.py`

**Module:** `src/tdf_obs/validation/classical_metric_emergence.py`

**Cases:** smooth control; noisy microscopic τ; correlated-noise protected; two-branch decohered metric merge; insufficient averaging (intentional fail); excessive gradient (intentional fail).

**Outputs:** `classical_metric_emergence_summary.csv`, `classical_metric_emergence_report.md`, `classical_metric_*.png`

**Tests:** `pytest tests/test_classical_metric_emergence.py`

**Status:** implemented. **Not** objective collapse, Born-rule derivation, or full measurement theory.

---

## Phase 6F — Born-rule probability emergence proxy ✅ (implemented)

**Objective:** Test whether branch weights **ρ_i** behave as stable probabilities **P_i = ρ_i / Σρ_j** after decoherence, using **c_i = √(ρ_i) e^(−iτ_i)** and multinomial frequency convergence.

**Banner:** `BORN-RULE PROBABILITY EMERGENCE BENCHMARK — NOT FULL BORN-RULE DERIVATION`

**Commands:** `python scripts/run_born_rule_probability.py`

**Module:** `src/tdf_obs/validation/born_rule_probability.py`

**Cases:** balanced / unequal two-branch; three-branch distribution; decoherence preserves diagonals; wrong-rule χ² comparison; coarse-graining additivity; zero-weight branch; phase invariance.

**Outputs:** `born_rule_probability_summary.csv`, `born_rule_probability_report.md`, `born_rule_*.png`

**Tests:** `pytest tests/test_born_rule_probability.py`

**Status:** implemented. **Not** full Born-rule derivation or measurement-problem solution.

---

## Phase 6 — Real observational calibration *(postponed)*

> ⚠️ **Real observational calibration is intentionally postponed until ΛCDM compatibility tests (Phase 4) are completed.**

**Objective:** Calibrate `(B, r0)` per galaxy from **confirmed** observational tables (e.g. SPARC).

**Steps:**

1. User supplies real SPARC (or equivalent) in `data/raw/` — no auto-download.
2. `prepare_sparc_rotation.py` → processed CSV + `rotation_metadata.yaml` with `real_observational`.
3. Batch-fit all galaxies; `data_mode=real_data_calibration`.
4. Flag galaxies where TDF does not beat baryon-only; full self-criticism in reports.

**Commands:** `python scripts/prepare_sparc_rotation.py`, `python scripts/run_rotation.py`

**Status:** infrastructure exists; **active calibration deferred**.

---

## Phase 7 — Lensing consistency *(future)*

**Objective:** Same tau parameters predict deflection angles  
`α_lens ≈ (2/c²) ∫ ∇⊥(Φ_b + Φ_τ) dl`.

**Status:** not_implemented — see `models/lensing.py`.

---

## Phase 8 — Doppler / redshift residuals vs kinematics *(future)*

**Objective:** Test `z_τ = K_τ Δτ̄_l / c²` against kinematic residuals on real or controlled data.

**Status:** formula + Phase 3C sanity scaffold only — see `models/redshift.py`.

---

## Phase 9 — Solar-system ephemeris coupling *(future)*

**Objective:** Replace assumed ε_τ with predictions from weak-field TDF metric; compare to ephemeris-scale bounds.

**Steps:** `python scripts/run_solar_system.py` with `configs/solar_system.yaml`.

**Status:** basic checks in Phase 3C; real ephemeris coupling not_yet_tested.

---

## Phase 10 — Black-hole observational comparison *(future)*

Checks (ansatz-level):

- `r_nr,TDF = √(r_s² - r_c²)`
- `T_TDF = T_H √(1 - r_c²/r_s²)`
- Remnant mass scale from `r_c`
- Shadow / ringdown constraints (future)

**Commands:** `pytest tests/test_black_hole_formulas.py`, `python scripts/run_black_hole.py`

**Status:** formulas + synthetic table; observational comparison not_yet_tested.

---

## Multi-channel sign-off (future)

TDF v0.8.1 is **not** validated until independent channels pass jointly with consistent `(K_τ, A)` or `(B, r0)` and solar-system safety on appropriate data. Document failures explicitly in `outputs/reports/`.

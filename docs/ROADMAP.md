# Roadmap — TDF Benchmark Lab

**Repository:** [tdf-benchmark-lab](https://github.com/bahman2017/tdf-benchmark-lab)

This roadmap tracks **benchmark and calibration** work only. Passing a phase does **not** mean TDF is observationally validated.

### Galaxy-scale rotation models (current)

| Model | Parameters | Role |
|-------|------------|------|
| **Baryon-only** | 0 | Baseline (no τ correction) |
| **TDF simple** | `B`, `r0` | Weak-field ansatz \(v^2 = v_b^2 + B r/(r+r_0)\) |
| **TDF K-essence** | `a0` | Non-linear Horndeski **effective** limit \(v^2 = v_b^2 + v_b\sqrt{a_0 r}\) |

Phase 3 batch fitting also compares **NFW simple** \((V_{h}^2, r_s)\) as a ΛCDM/GR+DM phenomenological baseline. K-essence is a **candidate** TDF extension, not a replacement for the simple ansatz or for full covariant Horndeski theory.

---

## Implemented

| Phase | Milestone | Status |
|-------|-----------|--------|
| **1** | Synthetic / demo rotation validation | ✅ |
| **2** | SPARC ingestion **scaffold** (no auto-download; no fake real labels) | ✅ |
| **3** | Baryon / TDF-simple / NFW baseline comparison (BIC) | ✅ |
| **3K** | K-essence non-linear effective rotation limit (`a0`; vs baryon & TDF-simple) | ✅ |
| **3B** | NFW surrogate teacher/student recovery | ✅ |
| **3C** | ΛCDM/GR combined benchmark scaffold (solar, BH, redshift) | ✅ |
| **4A** | Expanded NFW/ΛCDM rotation benchmark (10 cases) | ✅ |
| **4B** | GR-safe local benchmark (7 regimes) | ✅ |
| **4C** | Black-hole exterior GR-limit benchmark (q sweep) | ✅ |
| **4D** | Redshift / Doppler sanity benchmark | ✅ |
| **5A** | Core–cusp stress test (cuspy vs cored teachers) | ✅ |
| **5B** | Rotation-curve diversity stress test (10 shapes) | ✅ |
| **5C** | Same-τ multi-observable consistency (rotation / lensing / redshift proxies) | ✅ |
| **5D** | Covariant action consistency checks on NFW surrogate fits | ✅ |
| **5E** | CMB acoustic-scale compatibility (ΛCDM teacher, ε_τ background) | ✅ |
| **5F** | CMB-safe Hubble tension (late-time ε_τ, background proxies) | ✅ |
| **5G** | BAO/SNe late-time distance consistency | ✅ |
| **6A** | Schrödinger-from-TDF action benchmark (phase-density action → QM hydrodynamics) | ✅ |
| **6B** | Dirac / spinor limit (Clifford, flat Dirac, tetrad, τ-momentum mass ladder) | ✅ |
| **6C** | Entanglement from configuration-space τ (CHSH, concurrence, no-signaling) | ✅ |
| **6D** | Decoherence from τ-variance (branch coherence suppression) | ✅ |
| **6E** | Classical metric emergence from τ averaging | ✅ |
| **6F** | Born-rule probability emergence proxy | ✅ |
| **6G** | Unified microscopic quantum consistency matrix | ✅ |
| **6H** | Muon g-2 / precision QED coupling phenomenology | ✅ |

Supporting docs: [BENCHMARK_MANIFEST.md](./BENCHMARK_MANIFEST.md), [PAPER_APPENDIX_GUIDE.md](./PAPER_APPENDIX_GUIDE.md), GitHub Actions `pytest` CI.

---

## v0.9.0 — K-essence non-linear effective rotation limit ✅ (completed)

**Goal:** Add a **phenomenological** galaxy-scale rotation candidate derived from the TDF non-linear Horndeski K-essence limit, and compare it to baryon-only and TDF-simple fits using standard information criteria — **without** claiming dark matter is excluded or that TDF is observationally validated.

| Item | Status |
|------|--------|
| `v2_tdf_kessence` / `v_tdf_kessence` in `models/rotation.py` | ✅ |
| `fit_kessence_galaxy_rotation` (`a0 > 0`) | ✅ |
| `run_kessence_rotation_benchmark.py` (MSE, χ², BIC; report + figure) | ✅ |

**Effective limit (calibration form):** \(v^2(r) = v_b^2(r) + v_b(r)\sqrt{a_0 r}\), with \(a_0\) in \((\mathrm{km/s})^2/\mathrm{kpc}\).

Command: `python scripts/run_kessence_rotation_benchmark.py`

---

## v0.10.0 — Microscopic quantum limit of TDF

**Goal:** Move from macroscopic benchmark compatibility toward a **microscopic quantum derivation** — without claiming full quantum gravity.

| Phase | Milestone | Status |
|-------|-----------|--------|
| **6A** | Schrödinger-from-TDF action benchmark (ρ, τ phase-density action; 1D numerical consistency) | ✅ |
| **6E+** | Measurement collapse, Born-rule derivation | Planned |

**Phase 6A** does **not** modify rotation, cosmology, or NFW equations. See [quantum_limit/SCHRODINGER_DERIVATION.md](./quantum_limit/SCHRODINGER_DERIVATION.md).

Command: `python scripts/run_schrodinger_from_tdf.py`

---

## v0.11.0 — Dirac / spinor limit of TDF

**Goal:** Extend the microscopic quantum-limit program to **fermionic** degrees of freedom via spinors and tetrads — without claiming full fermion unification.

| Phase | Milestone | Status |
|-------|-----------|--------|
| **6B** | Dirac / spinor limit benchmark (γ matrices, H(k), Ψ = √ρ e^{−iτ} χ, g̃, m = p_τ/c) | ✅ |
| **6E+** | Measurement collapse, Born rule | Planned |

Command: `python scripts/run_dirac_spinor_limit.py`

---

## v0.12.0 — Entanglement / nonlocal correlations from TDF

**Goal:** Test whether TDF can encode **entangled** correlations as nonseparable τ-phase geometry in configuration space — without claiming Bell resolution or superluminal signaling.

| Phase | Milestone | Status |
|-------|-----------|--------|
| **6C** | Entanglement / τ geometry (two-qubit, CHSH, concurrence, no-signaling) | ✅ |
| **6E+** | Measurement collapse, Born rule | Planned |

Command: `python scripts/run_entanglement_tau_geometry.py`

---

## v0.13.0 — Decoherence and measurement dynamics from τ variance

**Goal:** Model **decoherence** as suppression of branch coherence from growth in Var(Δτ) — without claiming full measurement-problem solution.

| Phase | Milestone | Status |
|-------|-----------|--------|
| **6D** | Decoherence from τ-variance (C = exp(−½Var), Γ = ½ dVar/dt) | ✅ |
| **6E** | Classical metric emergence / objective-collapse proxy (τ averaging → g̃) | ✅ |
| **6F** | Born-rule / probability emergence proxy (ρ_i → P_i, χ² rule comparison) | ✅ |
| **6G** | Unified microscopic quantum consistency matrix (6A–6F integration) | ✅ |
| **6H** | Muon g-2 anomaly / precision QED coupling proxy (ε_τ → Δa_μ) | ✅ |
| **7+** | 5D action derivation; full QG / measurement theory | Planned |

Commands: `python scripts/run_unified_microscopic_quantum_limit.py` (plus individual 6A–6F scripts)

---

## v0.14.0 — Classical metric emergence / objective collapse proxy

**Goal:** Test whether noisy microscopic τ configurations can be averaged into a stable classical effective τ field and corresponding disformal metric — **without** claiming full objective collapse or Born-rule derivation.

| Phase | Milestone | Status |
|-------|-----------|--------|
| **6E** | Classical metric emergence from τ averaging (1+1D g̃, variance suppression, branch metric merge) | ✅ |

Command: `python scripts/run_classical_metric_emergence.py`

---

## v0.15.0 — Born rule / probability emergence proxy

**Goal:** Test whether branch weights **ρ_i** behave as stable probabilities after τ-variance decoherence and coarse-graining — **without** claiming full Born-rule derivation or solving the measurement problem.

| Phase | Milestone | Status |
|-------|-----------|--------|
| **6F** | Born-rule probability emergence (P_i = ρ_i/Σρ, decoherence diagonals, χ² vs wrong rules) | ✅ |

Command: `python scripts/run_born_rule_probability.py`

---

## v0.16.0 — Unified microscopic quantum limit of TDF

**Goal:** Integrate the microscopic quantum benchmark chain (6A–6F) into one consistency report — **without** claiming full quantum gravity or modifying prior benchmark logic.

| Phase | Milestone | Status |
|-------|-----------|--------|
| **6G** | Unified microscopic quantum consistency matrix | ✅ |

Command: `python scripts/run_unified_microscopic_quantum_limit.py`

**Next (v0.18.0+):** Derive microscopic action more deeply from 5D geometry rather than adding further proxy tests.

---

## v0.17.0 — Muon g-2 anomaly / precision QED benchmark (Phase 6H) ✅ (completed)

**Goal:** Test whether sub-Compton **τ-phase geometric variance** ε_τ = (σ_τ/ℓ_τ)² can produce a muon **(g−2)** shift Δa_μ = (α/2π)ε_τ at the order of magnitude of the experimental anomaly — **without** claiming a full QFT derivation.

| Phase | Milestone | Status |
|-------|-----------|--------|
| **6H** | Muon g-2 phenomenological QED coupling (α_eff proxy, Compton-scale ℓ_τ) | ✅ |

Command: `python scripts/run_muon_g2_anomaly.py`

---

## Next

| Phase | Description | Status |
|-------|-------------|--------|
| **5B.1** | Failure-mode refinement (declining outer, inner-cusp/outer-flat teachers) | Planned |
| **5H** | Missing-satellites / small-scale suppression scaffold | Planned |
| **6** | Real observational calibration (SPARC etc.) — **postponed** | Deferred |
| **7+** | Lensing consistency; redshift vs kinematics; solar-system ephemeris; BH observational constraints | Future |

---

## Phase 6 — Real observational calibration (postponed)

**Gate:** Benchmark suite documented; reviewer-facing appendix complete; explicit metadata workflow for real SPARC.

- User supplies raw data under `data/raw/` (no auto-download).
- `rotation_metadata.yaml` must confirm `real_observational`.
- Reports must **not** use demo/synthetic banners for real runs.

See [TEST_PLAN.md](./TEST_PLAN.md), [DATA_REQUIREMENTS.md](./DATA_REQUIREMENTS.md).

---

## Later channels (after Phase 6)

| Milestone | Content |
|-----------|---------|
| Lensing | Shared τ parameters; deflection integral |
| Redshift | Full residual pipeline vs kinematics |
| Solar system | Ephemeris-coupled ε_τ |
| Multi-channel report | Joint constraints — still **not** “TDF validated” without external review |

---

## What this roadmap does not mean

- Benchmark success ≠ TDF validated.
- Teacher curves ≠ real Universe.
- Stress-test wins ≠ ΛCDM replaced.

See [LCDM_COMPATIBILITY_STRATEGY.md](./LCDM_COMPATIBILITY_STRATEGY.md).

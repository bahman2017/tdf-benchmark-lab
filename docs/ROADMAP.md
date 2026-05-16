# Roadmap — TDF Benchmark Lab

**Repository:** [tdf-benchmark-lab](https://github.com/bahman2017/tdf-benchmark-lab)

This roadmap tracks **benchmark and calibration** work only. Passing a phase does **not** mean TDF is observationally validated.

---

## Implemented

| Phase | Milestone | Status |
|-------|-----------|--------|
| **1** | Synthetic / demo rotation validation | ✅ |
| **2** | SPARC ingestion **scaffold** (no auto-download; no fake real labels) | ✅ |
| **3** | Baryon / TDF / NFW baseline comparison (BIC) | ✅ |
| **3B** | NFW surrogate teacher/student recovery | ✅ |
| **3C** | ΛCDM/GR combined benchmark scaffold (solar, BH, redshift) | ✅ |
| **4A** | Expanded NFW/ΛCDM rotation benchmark (10 cases) | ✅ |
| **4B** | GR-safe local benchmark (7 regimes) | ✅ |
| **4C** | Black-hole exterior GR-limit benchmark (q sweep) | ✅ |
| **4D** | Redshift / Doppler sanity benchmark | ✅ |
| **5A** | Core–cusp stress test (cuspy vs cored teachers) | ✅ |
| **5B** | Rotation-curve diversity stress test (10 shapes) | ✅ |
| **5C** | Same-τ multi-observable consistency (rotation / lensing / redshift proxies) | ✅ |

Supporting docs: [BENCHMARK_MANIFEST.md](./BENCHMARK_MANIFEST.md), [PAPER_APPENDIX_GUIDE.md](./PAPER_APPENDIX_GUIDE.md), GitHub Actions `pytest` CI.

---

## Next

| Phase | Description | Status |
|-------|-------------|--------|
| **5B.1** | Failure-mode refinement (declining outer, inner-cusp/outer-flat teachers) | Planned |
| **5D** | Hubble-tension style expansion benchmark (phenomenological) | Planned |
| **5E** | Missing-satellites / small-scale suppression scaffold | Planned |
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

# Changelog

All notable changes to this project are documented here.

Format: versions track **benchmark-lab releases** (code + docs), not a claim that TDF theory is validated.

---

## v0.17.0 — Muon g-2 anomaly benchmark / Phase 6H (2026-05-16)

**Summary:** Phenomenological **precision QED coupling** check: sub-Compton τ fluctuations with ε_τ = (σ_τ/ℓ_τ)² yield Δa_μ = (α/2π)ε_τ, compared to the Fermilab/BNL consensus Δa_μ^exp ≈ 2.51×10⁻¹⁰.

**Theoretical motivation (calibration only):** TDF posits α_eff = α(1 + ε_τ) at order-of-magnitude level, giving a one-loop proxy shift in the muon anomalous magnetic moment. This does **not** derive the anomaly from a full TDF–QED Lagrangian.

**New code:**

| Path | Purpose |
|------|---------|
| `src/tdf_obs/validation/muon_g2_anomaly.py` | `calculate_delta_a_mu`, `estimate_epsilon_tau`, Compton-scale σ_τ estimate |
| `scripts/run_muon_g2_anomaly.py` | CLI |
| `tests/test_muon_g2_anomaly.py` | Reference ε_τ match; positive-input guards; pipeline |

**Documentation:** `docs/ROADMAP.md`, `docs/TEST_PLAN.md` (Phase 6H), `docs/BENCHMARK_MANIFEST.md`, `README.md`.

**Outputs (gitignored):** `muon_g2_anomaly_summary.csv`, `muon_g2_anomaly_report.md`, `muon_g2_epsilon_sweep.png`.

---

## v0.9.0 — K-essence rotation benchmark (2026-05-16)

**Summary:** Galaxy-scale **phenomenological** extension testing the TDF non-linear Horndeski K-essence **effective** rotation limit against baryon-only and TDF-simple baselines.

**Theoretical motivation (calibration only):** In the deep non-linear K-essence limit, the circular-speed ansatz

\[
v^2(r) = v_b^2(r) + v_b(r)\sqrt{a_0 r}
\]

provides a one-parameter (\(a_0\)) candidate for MOND-like outer enhancement relative to baryons alone. This is implemented for **controlled fitting and BIC comparison**, not as a full covariant Horndeski derivation or observational proof.

**New / updated code:**

| Path | Purpose |
|------|---------|
| `src/tdf_obs/models/rotation.py` | `v2_tdf_kessence`, `v_tdf_kessence` |
| `src/tdf_obs/fitting/fit_rotation.py` | `fit_kessence_galaxy_rotation`, `KessenceFitResult` |
| `src/tdf_obs/validation/kessence_rotation_benchmark.py` | Three-model comparison, synthetic K-essence generator, report/plot |
| `scripts/run_kessence_rotation_benchmark.py` | CLI entry point |
| `tests/test_rotation_model.py` | K-essence finite values; `a0=0` → baryon-only |
| `tests/test_kessence_rotation_benchmark.py` | Pipeline, banner, recovery tests |

**Documentation:** `docs/ROADMAP.md`, `docs/TEST_PLAN.md` (Phase 3K), `docs/BENCHMARK_MANIFEST.md`, `README.md`, `INSTRUCTIONS.md` (§25 continuous documentation rule).

**Outputs (generated, gitignored):** `outputs/tables/kessence_rotation_benchmark_summary.csv`, `outputs/reports/kessence_rotation_benchmark_report.md`, `outputs/figures/*_kessence_rotation_benchmark.png`.

---

## v0.1.0 — Benchmark scaffold release

- Synthetic / demo rotation validation (Phase 1)
- SPARC ingestion scaffold only; real observational calibration intentionally postponed (Phase 2)
- Baryon / TDF / NFW baseline comparison (Phase 3)
- NFW surrogate recovery (Phase 3B)
- ΛCDM / GR benchmark recovery scaffold (Phase 3C)
- Expanded NFW / ΛCDM rotation benchmark (Phase 4A)
- GR-safe local benchmark (Phase 4B)
- Black-hole exterior GR-limit benchmark (Phase 4C)
- Redshift / Doppler sanity benchmark (Phase 4D)
- Core–cusp stress test (Phase 5A)
- Rotation-curve diversity stress test (Phase 5B)
- Documentation for GitHub, paper appendix, and reproducibility (`PAPER_APPENDIX_GUIDE.md`, `BENCHMARK_MANIFEST.md`)
- Pytest suite and GitHub Actions CI workflow

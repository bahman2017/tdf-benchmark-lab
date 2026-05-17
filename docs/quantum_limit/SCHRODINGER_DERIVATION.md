# Schrödinger equation from the TDF phase-density action

**Status:** Phase 6A / 6A.1 — symbolic sketch + numerical consistency checks  
**Not:** full quantum validation, quantum gravity, or axiomatic replacement of QM

---

## Wavefunction ansatz (TDF v0.8.1)

```text
ψ(x,t) = √ρ(x,t) exp(-i τ(x,t))
E      = ℏ ∂_t τ
```

ρ is a non-negative density; τ is a real phase field. This matches INSTRUCTIONS.md
(ψ = √ρ e^{-iτ}, not the older positive-phase notes).

**Plane-wave check:** ψ = exp(i(kx − ωt)) ⇒ τ = ωt − kx with ω = ℏk²/(2m).

---

## Phase-density action (1D sketch)

```text
L = -ℏ ρ ∂_t τ
    - (ℏ²/2m) ρ |∂_x τ|²
    - V ρ
    + (ℏ²/8m) |∂_x ρ|² / ρ
```

The last term is a **Fisher-information** form. Its variation with respect to ρ produces the
**quantum potential**.

---

## Variation with respect to τ → continuity equation

```text
δ/δτ ∫ L dx dt  →  ∂_t ρ + ∂_x( ρ (ℏ/m) ∂_x τ ) = 0
```

---

## Variation with respect to ρ → quantum Hamilton–Jacobi equation

With ψ = √ρ exp(-iτ) and E = ℏ ∂_t τ:

```text
ℏ ∂_t τ − (ℏ²/2m) |∂_x τ|² − V − Q = 0
```

```text
Q = -(ℏ²/2m) ∂_x²(√ρ) / √ρ
```

**Sign note (Phase 6A.1):** An equivalent Madelung form uses action phase S = −ℏτ:

```text
-ℏ ∂_t τ + (ℏ²/2m) |∂_x τ|² + V + Q = 0
```

The pairing `ℏ∂_tτ + (ℏ²/2m)|∇τ|² + V + Q = 0` is **not** consistent with ψ = √ρ exp(-iτ)
and ω = ℏk²/(2m) for the plane wave.

The Fisher term

```text
(ℏ²/8m) |∂_x ρ|² / ρ
```

generates Q upon variation in ρ (integration by parts).

---

## Stationary eigenstates

For ψ(x,t) = ψ₀(x) exp(−iEt/ℏ):

```text
τ = (E/ℏ) t + χ(x),    ℏ ∂_t τ = E
```

Harmonic-oscillator ground state: E₀ = ½ℏω, spatially uniform τ at fixed t,
and QHJ reduces to **V + Q = E₀** when |∇τ| = 0.

---

## Link to the Schrödinger equation

The pair (continuity, quantum Hamilton–Jacobi) is equivalent to

```text
iℏ ∂_t ψ = [ -ℏ²/(2m) ∂_x² + V ] ψ
```

(1D shown; 3D generalization is direct.)

---

## Numerical residuals (Phase 6A.1)

- **Raw QHJ:** max|ℏ∂_tτ − kinetic − V − Q| on interior grid points.
- **Normalized QHJ:** raw max divided by max(1, max(|ℏ∂_tτ|, |kinetic|, |V|, |Q|)).
- Pass threshold for normalized QHJ: **10⁻³** (see `schrodinger_from_tdf.py`).

---

## Limitations (explicit)

| Topic | Status |
|-------|--------|
| 1D numerical benchmarks only | implemented |
| 3D, curved backgrounds | not yet |
| Relativistic Klein–Gordon / Dirac | not yet |
| Spinors, entanglement, measurement | not yet |
| Macroscopic TDF rotation / cosmology | separate benchmarks (unchanged) |

Banner: **SCHRÖDINGER-FROM-TDF ACTION BENCHMARK — NOT FULL QUANTUM VALIDATION**

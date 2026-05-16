# Black-hole exterior GR-limit benchmark report (Phase 4C)

## ⚠️ BLACK-HOLE GR-LIMIT BENCHMARK — NOT REAL OBSERVATIONAL DATA

## Purpose

This checks whether the **phenomenological** TDF black-hole ansatz recovers GR/Hawking behavior in the exterior limit **q = r_c/r_s ≪ 1**.

> **Not observational validation.** No ringdown, shadow, or evaporation data are fitted.

## Formulas (TDF v0.8.1 ansatz level)

```text
r_s = 2 G M / c^2
r_nr_TDF = sqrt(r_s^2 - r_c^2)
T_TDF = T_H * sqrt(1 - r_c^2 / r_s^2)
M_rem = c^2 r_c / (2 G)
q = r_c / r_s
r_nr_TDF / r_s = sqrt(1 - q^2)
T_TDF / T_H = sqrt(1 - q^2)
```

**Benchmark mass:** 10.0 M☉ (configurable; not from observation).

## Summary

- **Cases:** 13
- **GR-like:** 5
- **Mildly modified:** 3
- **Strongly modified:** 4
- **No horizon (q ≥ 1):** 1
- **Largest q still GR-like:** 0.01

## Results table

| q | r_nr/r_s | T/T_H | expected | dev. from GR % | status |
| --- | --- | --- | --- | --- | --- |
| 0 | 1 | 1 | 1 | 0 | GR-like |
| 1e-08 | 1 | 1 | 1 | 1.11e-14 | GR-like |
| 1e-06 | 1 | 1 | 1 | 5e-11 | GR-like |
| 0.0001 | 1 | 1 | 1 | 5e-07 | GR-like |
| 0.01 | 0.99995 | 0.99995 | 0.99995 | 0.005 | GR-like |
| 0.05 | 0.998749 | 0.998749 | 0.998749 | 0.1251 | mildly_modified |
| 0.1 | 0.994987 | 0.994987 | 0.994987 | 0.5013 | mildly_modified |
| 0.25 | 0.968246 | 0.968246 | 0.968246 | 3.175 | mildly_modified |
| 0.5 | 0.866025 | 0.866025 | 0.866025 | 13.4 | strongly_modified |
| 0.75 | 0.661438 | 0.661438 | 0.661438 | 33.86 | strongly_modified |
| 0.9 | 0.43589 | 0.43589 | 0.43589 | 56.41 | strongly_modified |
| 0.99 | 0.141067 | 0.141067 | 0.141067 | 85.89 | strongly_modified |
| 1 | — | — | — | — | no_horizon |

## Interpretation

- **q → 0** recovers Hawking/GR exterior ratios (r_nr/r_s → 1, T/T_H → 1).
- **Large q** produces strong departures from the q=0 limit and explores remnant scale M_rem.
- This is **not** a full nonlinear black-hole solution; Kerr spin, backreaction, and real data are out of scope.

## Failure modes

- Formulas are **phenomenological ansatz-level** only.
- Real ringdown, shadow, and evaporation constraints are **not** fitted.
- **Kerr spin** is not included.
- **Backreaction** is not included.
- q ≥ 1 is flagged `no_horizon` for this exterior ansatz.

## Disclaimer

- **NOT REAL OBSERVATIONAL DATA**
- Passing GR-limit recovery in this table does **not** validate TDF.

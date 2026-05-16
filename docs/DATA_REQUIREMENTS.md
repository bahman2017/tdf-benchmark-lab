# Data Requirements

## SPARC raw input (Phase 2 ingestion)

The pipeline does **not** download SPARC. Place files manually under `data/raw/`.

### Option A — single combined CSV

Path: `data/raw/sparc_rotation_curves.csv`

| Column | Required | Unit | Description |
|--------|----------|------|-------------|
| `galaxy_id` | yes | — | Galaxy name |
| `r_kpc` | yes | kpc | Galactocentric radius |
| `v_obs` | yes | km/s | Observed circular speed |
| `v_err` | yes | km/s | 1σ uncertainty (> 0) |
| `v_gas` | yes | km/s | Gas component (sign may follow SPARC convention) |
| `v_disk` | yes | km/s | Disk component |
| `v_bulge` | no | km/s | Bulge component (defaults to 0 if absent) |

### Option B — per-galaxy CSV folder

Path: `data/raw/sparc/<GalaxyName>.csv`

Same columns as above **except** `galaxy_id` (taken from the filename stem).

Example:

```text
data/raw/sparc/DDO154.csv
data/raw/sparc/NGC2403.csv
```

### Prepare command

```bash
python scripts/prepare_sparc_rotation.py
```

Writes:

- `data/processed/rotation.csv`
- `data/processed/rotation_metadata.yaml` (only after successful parse of **your** raw files)

If no raw data is present, the script exits with a message and **does not** create fake processed files.

---

## Mass-to-light assumptions (Υ)

Baryonic speed is derived from components:

```text
v_baryon² = |v_gas|² + Υ_disk·|v_disk|² + Υ_bulge·|v_bulge|²   (each term uses |v|·|v|)
v_baryon = sqrt(max(v_baryon², 0))
```

Defaults (configurable in `prepare_sparc_rotation.py` / parser API):

| Parameter | Default | Typical SPARC usage |
|-----------|---------|---------------------|
| `upsilon_disk` (Υ_disk) | 0.5 | Disk stellar M/L |
| `upsilon_bulge` (Υ_bulge) | 0.7 | Bulge stellar M/L |

The `|v|·v` form keeps contributions sign-safe when SPARC stores negative components for direction conventions.

Processed CSV stores `upsilon_disk` and `upsilon_bulge` per export for reproducibility.

---

## Processed rotation schema

Path: `data/processed/rotation.csv` (long format, one row per radius point).

### Required columns

| Field | Unit | Description |
|-------|------|-------------|
| `galaxy_id` | — | Unique identifier |
| `r_kpc` | kpc | Galactocentric radius (> 0) |
| `v_obs` | km/s | Observed circular speed |
| `v_err` | km/s | 1σ uncertainty (> 0) |
| `v_baryon` | km/s | Baryonic model speed from components |

### Optional preserved columns

`v_gas`, `v_disk`, `v_bulge`, `upsilon_disk`, `upsilon_bulge`, `source`

Minimum **3** points per galaxy.

---

## Dataset labeling (`rotation_metadata.yaml`)

Processed output is used in the rotation fit only after honest labeling:

| Metadata | Pipeline mode |
|----------|----------------|
| Missing or non-real metadata | `demo_fixture_calibration` |
| `dataset_type: real_observational`, `is_real_observational_data: true`, explicit `source` (e.g. SPARC) | `real_data_calibration` |

`prepare_sparc_rotation.py` writes real observational metadata **only** when it successfully parses **user-provided** raw SPARC files. Copying test fixtures into `data/raw/` without real survey data is still your responsibility to label correctly.

No `rotation.csv` → `synthetic_validation` (in-memory generator).

---

## Lensing (Phase 4) — not yet collected

- Deflection angle `α` [arcsec] vs impact parameter
- Projected baryon potential or surface density
- Same `(B, r0)` or `(K_τ, A)` as rotation fit

## Redshift (Phase 5)

- `z_obs`, `z_kin`, `z_baryon`, `z_err`
- Map to `Δτ̄_l` along line of sight

## Solar system (Phase 6)

- `Φ_b`, `Φ_τ` or parametric model at test radii
- Published upper bounds on `ε_τ` or PPN parameters

## Black hole (Phase 7)

- Mass `M`, optional shadow size, ringdown frequency
- Compare to `r_nr,TDF`, `T_TDF` predictions (ansatz only)

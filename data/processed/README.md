# Processed data (Phase 2)

Place **`rotation.csv`** here for CSV-based fitting. The pipeline does **not** download SPARC automatically.

## Required files

| File | Purpose |
|------|---------|
| `rotation.csv` | Long-format table (see schema below) |
| `rotation_metadata.yaml` | **Required for honest labeling** (see example) |

If `rotation.csv` exists but `rotation_metadata.yaml` is **missing**, the run is labeled **`demo_fixture_calibration`** — not real SPARC.

## CSV schema

```text
galaxy_id,r_kpc,v_obs,v_err,v_baryon
```

## Metadata schema (`rotation_metadata.yaml`)

### Demo / test fixture

```yaml
dataset_type: demo_fixture
source: test_fixture
description: "Small demo dataset used to test pipeline behavior."
is_real_observational_data: false
```

### Real observational data (e.g. SPARC)

All of the following are required for `real_data_calibration`:

```yaml
dataset_type: real_observational
source: SPARC
description: "Processed SPARC rotation curves."
is_real_observational_data: true
```

## Modes

| Condition | Mode | Report banner |
|-----------|------|----------------|
| No `rotation.csv` | `synthetic_validation` | SYNTHETIC ONLY — NO REAL DATA USED |
| CSV, no / non-real metadata | `demo_fixture_calibration` | DEMO FIXTURE DATA — NOT REAL SPARC |
| CSV + confirmed real metadata | `real_data_calibration` | _(no demo/synthetic banner)_ |

If no `rotation.csv`, run `python scripts/run_rotation.py` for **synthetic validation only**.

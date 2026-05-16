# Raw observational data

Place **user-provided** SPARC files here. The project does **not** download catalogs.

## Supported layouts

1. **Single file:** `sparc_rotation_curves.csv`  
   Columns: `galaxy_id`, `r_kpc`, `v_obs`, `v_err`, `v_gas`, `v_disk` [, `v_bulge`]

2. **Per-galaxy folder:** `sparc/<GalaxyName>.csv`  
   Same columns without `galaxy_id` (name from filename).

## Prepare processed data

```bash
python scripts/prepare_sparc_rotation.py
```

Creates `data/processed/rotation.csv` and metadata only when raw files exist.

**Do not commit proprietary SPARC data without license clearance.**

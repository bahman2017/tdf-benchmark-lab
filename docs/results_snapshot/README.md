# Results snapshot (optional, for paper appendix)

This folder may contain **selected frozen benchmark outputs** referenced in a TDF paper or preprint appendix.

## Important

- Snapshots here are **controlled benchmark results**, not observational validation of TDF.
- They do **not** prove TDF, disprove ΛCDM, or replace dark matter.
- Full, up-to-date outputs should be **regenerated** from the repository using the commands in [PAPER_APPENDIX_GUIDE.md](../PAPER_APPENDIX_GUIDE.md).

## What to store here (if needed)

Examples suitable for an appendix:

- One representative table per benchmark phase (CSV copy)
- One or two key figures (PNG copy)
- A short `MANIFEST.txt` listing commit hash and script versions used

Do **not** store raw SPARC or other observational catalogs here unless explicitly labeled with verified metadata.

## Regeneration

From the repository root:

```bash
pip install -r requirements.txt && pip install -e .
pytest
# Then run individual benchmark scripts — see PAPER_APPENDIX_GUIDE.md
```

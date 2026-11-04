# NYC 311 technical fixes — implementation notes

## Included replacements
- `01_download.py`: writes `reports/data_manifest.json`; records UTC run start/completion, per-day row counts and origin, combined row count, distinct IDs, date range, and SHA-256. It compares every cached daily file and every new download with NYC Open Data's source-side `count(*)`. A mismatched cached file is redownloaded; a mismatched new download is discarded and the run stops.
- `06_subgroup_audit.py`: reports both ROC-AUC and average precision as `NaN` when a subgroup contains only one target class.
- `README.md`: explains the cohort-only backlog ramp-up, clarifies generated local model/report files, and documents the manifest.

## Important provenance limitation
The original download timestamp for the existing 3,655,042-row extract was not preserved in the available project materials. Do not invent one. The new manifest records the time a future run is assembled/verified; it does not retroactively establish when already-cached files were first downloaded.

## Apply on Windows PowerShell
From the root of the GitHub project, copy these three replacement files over the corresponding project files. Then run:

```powershell
python -m py_compile .\01_download.py .\06_subgroup_audit.py
python .\06_subgroup_audit.py
```

The subgroup script requires the locally generated database, model artifact, and `reports/model_summary.json`. It does not retrain the model. The downloader is updated for future dataset refreshes; do not rerun it just to apply the code fix.

After confirming the subgroup output, commit the replacements and regenerated `results/subgroup_metrics.csv` in one Git commit. The GitHub repository description should be changed manually to:

`NYC 311 service performance analysis using Python, SQL, SQLite and LightGBM, with chronological model evaluation and operational insights.`

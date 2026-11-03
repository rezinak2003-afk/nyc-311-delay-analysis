# NYC 311 Service Performance & 72-Hour Delay Analysis

An end-to-end analytics and machine-learning project using NYC 311 service requests created during 2025. The workflow downloads the source data, validates and loads it into SQLite, runs SQL analysis, trains a chronological LightGBM model, and evaluates model performance across boroughs and agencies.

## Project scope and target definition

The target is `slow_72h`:

- `1`: the request was closed more than 72 hours after creation, or it had no closure timestamp and its status was not `Closed` in the downloaded extract.
- `0`: the request was closed within 72 hours.
- `NULL`: the outcome is unknown because creation/closure chronology is invalid, or the status is `Closed` but the closure timestamp is missing. Unknown outcomes are excluded from model training and evaluation.

This is an operational 72-hour outcome, **not an official NYC agency SLA breach**. It is a retrospective analysis of the downloaded 2025 cohort, not a live prediction service. The model uses only request-creation fields: agency, complaint type, borough, creation hour, and creation weekday. It does not use closure timestamp, status, due date, or resolution duration as predictors.

The source dataset is updated over time. Re-running the downloader later may retrieve revised historical records, so exact counts and metrics can differ from this recorded run.

## Recorded dataset and data-quality results

| Measure | Recorded result |
|---|---:|
| Requests downloaded | 3,655,042 |
| Distinct request IDs | 3,655,042 |
| Known 72-hour outcomes | 3,648,102 |
| Unknown outcomes excluded | 6,940 |
| Slow 72-hour outcomes | 912,161 |
| Slow outcome rate among known outcomes | 25.00% |
| Invalid closure chronology, retained and flagged | 916 |
| `Closed` status with missing closure timestamp | 6,024 |
| Requests with a due date | 0.45% |

The due-date field is too sparsely populated to support a citywide SLA claim. The project reports due-date comparisons only for records where the relevant timestamps are available and does not treat `due_date` as a universal closure deadline.

## SQL analysis highlights

- HPD accounts for 45.82% of all known slow-72-hour outcomes; the top five agencies account for 82.06%.
- The highest borough slow-outcome rates are Staten Island (28.85%) and Manhattan (28.65%); Queens is lowest among the five boroughs (21.26%).
- July has the highest monthly slow-outcome rate in the 2025 cohort (27.80%).
- The reconstructed December 2025 month-end backlog is 148,348 requests **from the 2025-created cohort only**, not the entire NYC backlog.
- Resolution-time medians and percentiles use requests with valid closure timestamps only. They should not be interpreted as the eventual resolution-time distribution for all requests because still-open requests do not have an observed final duration.

These are descriptive associations, not causal explanations.

## Model methodology and results

The model uses a chronological split by creation date, rather than a random row split:

| Split | Rows | Period boundary |
|---|---:|---|
| Train | 2,491,006 | Before 2025-09-13 |
| Validation | 582,219 | 2025-09-13 to before 2025-11-07 |
| Held-out test | 574,877 | From 2025-11-07 |

The classification threshold is selected on validation data only. The final test period is not used to tune the threshold.

| Model | ROC-AUC | Average precision | Precision | Recall | F1 | Accuracy |
|---|---:|---:|---:|---:|---:|---:|
| Majority-class baseline | 0.5000 | 0.2398 | 0.00% | 0.00% | 0.00% | 76.02% |
| Complaint-type-rate baseline | 0.9411 | 0.8249 | 76.52% | 80.39% | 78.41% | 89.38% |
| LightGBM | **0.9491** | **0.8511** | **77.66%** | **81.27%** | **79.42%** | **89.90%** |

LightGBM improves modestly over the strong complaint-type-rate baseline. The result should be presented as an incremental improvement, not a dramatic performance breakthrough.

### Agency ablation

The same LightGBM configuration and chronological split were evaluated without the agency feature:

| Metric | With agency | Without agency |
|---|---:|---:|
| ROC-AUC | 0.9491 | 0.9476 |
| Average precision | 0.8511 | 0.8494 |
| F1 | 79.42% | 79.25% |
| Accuracy | 89.90% | 89.68% |

The model retains most of its performance without agency. Keep the full model as the primary model and report the no-agency experiment as a robustness check.

### Subgroup and temporal evaluation

Subgroup metrics use the held-out test period and the same validation-selected threshold (0.67).

- Borough F1 ranges from 73.90% (Staten Island) to 83.81% (Bronx).
- Agency-level performance varies substantially. For example, F1 is 83.33% for HPD, 58.38% for DSNY, and 47.35% for DEP.
- NYPD has an extremely low positive-outcome rate in the test period; its 99.996% accuracy is misleading because recall and F1 are zero.
- EDC has only one target class in the test subset, so ROC-AUC is undefined.
- November covers only November 7–30, while December covers the full month. Test-period F1 is 82.03% for November 7–30 and 77.58% for December; this is a descriptive temporal comparison, not a comparison of two full calendar months.

Small agency groups should not be ranked without considering sample size and class distribution. Subgroup differences are diagnostic and do not establish unfair treatment or causal effects.

## Repository files

- `01_download.py` — resumable day-by-day download of 2025-created requests.
- `02_load_sqlite.py` — chunked validation, derived fields, target creation, and SQLite loading.
- `03_analysis.sql` — 11 analytical queries covering data quality, due dates, agencies, complaint types, boroughs, time patterns, backlog, and status reconciliation.
- `04_model.py` — LightGBM, majority and complaint-type-rate baselines, chronological evaluation, agency ablation, and permutation importance.
- `05_run_sql.py` — read-only SQL execution and CSV export.
- `06_audit_sample.py` — basic audit of a supplied CSV file.
- `06_subgroup_audit.py` — subgroup metrics using the saved model and held-out test data.
- `results/` — SQL outputs, model metrics, confusion matrix, feature importance, agency ablation, and subgroup metrics.
- `reports/model_summary.json` — machine-readable model methodology and results.
- `models/slow_72h_model.joblib` — trained LightGBM pipeline.

## Setup and reproduction (Windows PowerShell)

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Run from the project root:

```powershell
python 01_download.py
python 02_load_sqlite.py
python 05_run_sql.py
python 04_model.py
python 06_subgroup_audit.py
```

The downloader writes `data/raw_311_2025.csv` and daily files under `data/days/`. The loader writes `data/nyc311.db`. The database and raw data are not included in this lightweight repository package; rerun the downloader and loader to recreate them. The downloader queries NYC Open Data, so later runs may differ if the source revises historical records.

## Data source

NYC Open Data — 311 Service Requests from 2020 to Present, dataset identifier `erm2-nwe9`:

https://data.cityofnewyork.us/Social-Services/311-Service-Requests-from-2020-to-Present/erm2-nwe9

Check the current data dictionary and source terms before publication.

## Responsible interpretation

- Do not call `slow_72h` an official SLA breach.
- Do not describe this retrospective model as a live real-time prediction system.
- Do not interpret feature importance as causation.
- Do not use accuracy alone, particularly for agencies with highly imbalanced outcomes.
- Treat subgroup metrics as diagnostics and disclose low sample sizes or single-class groups.
- Distinguish the reconstructed 2025-created cohort backlog from the full citywide open backlog.

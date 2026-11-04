# NYC 311 Service Performance & 72-Hour Delay Analysis
## Business Findings & Recommendations

**Analysis period:** Requests created January 1–December 31, 2025  
**Dataset:** NYC Open Data — 311 Service Requests from 2020 to Present (`erm2-nwe9`)

## 1. Executive summary

This project examines **3,655,042 NYC 311 service requests created during 2025**. Of these, **3,648,102 had a known 72-hour outcome**. Among requests with known outcomes, **912,161 (25.00%) were not closed within 72 hours** under the project's retrospective target definition.

The analysis finds that slow outcomes are concentrated among a small number of agencies, vary across boroughs and complaint categories, and show monthly variation. A LightGBM model achieved a **0.9491 ROC-AUC** and **0.7942 F1 score** on a chronological held-out test period. Its improvement over a strong complaint-type historical-rate baseline was modest.

The practical value of the work is to help analysts and operational teams identify where to investigate delays, monitor differences in outcomes, and assess whether risk-based prioritisation could be useful. These results are descriptive and predictive; they do not establish the causes of delays or prove that a proposed intervention will improve service.

## 2. Scope and target definition

The target, `slow_72h`, is defined as follows:

- **1 — Slow:** Closed more than 72 hours after creation, or no closure timestamp was present and the request status was not `Closed` in the downloaded extract.
- **0 — Not slow:** Closed within 72 hours.
- **NULL — Unknown:** Invalid creation/closure chronology, or status `Closed` with a missing closure timestamp.

Unknown outcomes are excluded from model training and evaluation. The 72-hour target is an **analytical definition**, not an official NYC agency service-level agreement (SLA) or SLA breach. This is a retrospective analysis of the 2025-created cohort, not a live prediction service.

| Data-quality measure | Recorded result |
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

The due-date field is too sparsely populated to support a citywide SLA assessment.

## 3. Key business findings

### Finding 1 — Slow outcomes are concentrated among a small number of agencies

- HPD accounts for **45.82%** of all known slow-72-hour outcomes.
- The five largest contributing agencies account for **82.06%** of slow outcomes.

**Why this matters:** A small number of agencies account for a large share of the observed slow-case volume. Agency-level investigation may therefore be a more actionable starting point than treating all requests as one citywide operational pool.

**Recommended follow-up:** For the highest-volume agencies, break results down by complaint type, borough, request volume and case mix. Review workflow stages and staffing/capacity data before attributing differences to agency performance.

### Finding 2 — Slow-outcome rates differ across boroughs

| Borough | Slow-outcome rate |
|---|---:|
| Staten Island | 28.85% |
| Manhattan | 28.65% |
| Brooklyn | 25.92% |
| Bronx | 23.90% |
| Queens | 21.26% |

**Why this matters:** Borough-level rates identify geographic variation that merits further investigation.

**Recommended follow-up:** Compare boroughs within similar complaint categories and agencies. These unadjusted rates may reflect differences in request mix and should not be interpreted as direct rankings of operational efficiency.

### Finding 3 — Complaint categories have materially different outcomes

The complaint-type analysis reports eligible request volume, slow-outcome rate and average resolution hours among validly closed requests. These measures can help distinguish high-volume categories from categories with elevated slow-outcome rates.

**Recommended follow-up:** Prioritise categories that combine substantial volume with high slow-outcome rates. Review their operational handling with the responsible agency. Average resolution hours are calculated only for requests with valid closure timestamps, so they are not the eventual resolution-time distribution for all requests.

### Finding 4 — Monthly patterns and backlog provide planning signals

- **July** had the highest monthly slow-outcome rate in the 2025 cohort: **27.80%**.
- The reconstructed December 2025 month-end backlog was **148,348 requests** from the 2025-created cohort.

**Why this matters:** Seasonal variation and accumulation of unresolved requests can inform questions about workload and capacity.

**Recommended follow-up:** Examine monthly demand, complaint mix and staffing/capacity information together. The backlog series is a reconstructed **2025-created cohort** measure, not the full citywide open backlog. The series accumulates as the 2025 cohort enters the dataset, so the early-year ramp-up should not be read as a standalone citywide trend.

## 4. Predictive model results

The project uses a chronological split rather than a random row split. The model uses request-creation fields: agency, complaint type, borough, creation hour and creation weekday. The classification threshold was selected on validation data only.

| Model | ROC-AUC | Average precision | Precision | Recall | F1 | Accuracy |
|---|---:|---:|---:|---:|---:|---:|
| Majority-class baseline | 0.5000 | 0.2398 | 0.00% | 0.00% | 0.00% | 76.02% |
| Complaint-type-rate baseline | 0.9411 | 0.8249 | 76.52% | 80.39% | 78.41% | 89.38% |
| LightGBM | **0.9491** | **0.8511** | **77.66%** | **81.27%** | **79.42%** | **89.90%** |

LightGBM improves on the complaint-type-rate baseline, but the improvement is incremental: F1 increases from **78.41% to 79.42%**. This should be presented as a useful but modest lift, not a dramatic breakthrough.

### Agency ablation

| Metric | With agency | Without agency |
|---|---:|---:|
| ROC-AUC | 0.9491 | 0.9476 |
| Average precision | 0.8511 | 0.8494 |
| F1 | 79.42% | 79.25% |
| Accuracy | 89.90% | 89.68% |

Removing agency causes only a small reduction in performance. The full model remains the primary model; the no-agency run is a robustness comparison. Because agency and complaint type may be correlated, this experiment does not prove that the model is independent of agency-related information.

### Feature importance

Permutation importance ranked **agency** as the strongest feature by mean ROC-AUC drop, followed by **complaint type**. Weekday, borough and creation hour had much smaller measured effects in this experiment.

Feature importance describes how model performance changes when a feature is permuted. It does not establish that a feature causes delays, nor does it identify an intervention by itself.

## 5. Subgroup evaluation and responsible interpretation

Subgroup metrics were calculated on the held-out test period using the validation-selected threshold of **0.67**.

- Borough F1 ranged from **73.90% (Staten Island)** to **83.81% (Bronx)**.
- Agency-level performance varied: F1 was **83.33% for HPD**, **58.38% for DSNY**, and **47.35% for DEP**.
- NYPD had an extremely low positive-outcome rate in the test subset. Its **99.996% accuracy** is misleading because recall and F1 were zero.
- EDC had only one target class in the test subset, so ROC-AUC and average precision were undefined.
- November's test results cover **November 7–30**, while December covers the full month. The two months are not directly comparable as full calendar months.

These differences are diagnostic signals for further review. They do not, by themselves, establish unfair treatment, discriminatory impact, or causal operational differences. Small groups and highly imbalanced target distributions require particular caution.

## 6. Recommendations

1. **Start with agency-level investigation.** Review the highest-volume contributors to slow outcomes, beginning with HPD, and drill down into complaint type and borough.
2. **Use case-mix-adjusted comparisons.** Compare agencies and boroughs within similar complaint categories before interpreting rate differences as performance gaps.
3. **Prioritise high-volume, high-delay complaint categories.** Combine request volume and slow-outcome rate to identify categories for operational review.
4. **Use monthly patterns as planning prompts.** Investigate July's elevated rate and review whether demand, case mix or capacity changed during that period.
5. **Pilot risk-based triage only after operational validation.** Test whether model scores help teams identify requests needing earlier attention; do not automate decisions based on this retrospective model alone.
6. **Monitor subgroup performance.** Track precision, recall and F1 alongside sample sizes and positive-outcome rates. Avoid relying on accuracy alone for imbalanced groups.
7. **Strengthen source and timestamp governance.** Preserve extraction metadata and confirm that predictor fields are available at the intended decision time before any deployment.

## 7. Limitations

- The 72-hour target is a retrospective project definition, not an official NYC SLA breach.
- The model is not a live or deployed prediction service.
- The analysis covers requests created in 2025 with known outcomes; unknown outcomes are excluded from model evaluation.
- The downloaded extract does not establish that all predictor fields are point-in-time snapshots as they existed at request creation.
- Agency and complaint-type features may be correlated; feature importance and ablation should not be interpreted causally.
- Borough and agency comparisons are affected by case mix, group size and class imbalance.
- Valid-closure resolution-time summaries exclude requests without valid closure timestamps.
- The reconstructed backlog covers only the 2025-created cohort, not the entire citywide open backlog.
- NYC Open Data may revise historical records. A future re-download may produce different counts and metrics.

## 8. Conclusion

The analysis shows how a large public-service dataset can be transformed into validated operational measures, SQL findings, a chronological predictive evaluation and an interactive dashboard. The most actionable next step is targeted investigation of agency and complaint-category patterns, supported by case-mix-aware monitoring.

The model may help prioritise further review, but operational use would require point-in-time feature validation, prospective testing, monitoring and human oversight. The strongest portfolio outcome is the complete analytical workflow and the careful translation of evidence into practical, appropriately qualified recommendations.

---

**Source:** NYC Open Data, *311 Service Requests from 2020 to Present*, dataset `erm2-nwe9`.  
**Project repository:** https://github.com/rezinak2003-afk/nyc-311-delay-analysis

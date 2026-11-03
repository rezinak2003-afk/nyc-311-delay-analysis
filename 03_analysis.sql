-- NYC 311 2025 analytical queries
-- Run from project root with: python 05_run_sql.py
--
-- slow_72h: closed after 72 hours OR still not closed after 72 hours.
-- Unknown outcomes (invalid chronology/creation date, or status=Closed but no
-- closed_date) are NULL and excluded from rates.
-- closed_after_due is a limited comparison to the supplied due_date field.
-- NYC data dictionary describes due_date as an expected update date; it is NOT
-- assumed to be a universal closure SLA.
-- Month-end backlog is reconstructed only for requests created in 2025.

-- 1. Dataset coverage and data-quality reconciliation
SELECT
    COUNT(*) AS total_requests,
    COUNT(DISTINCT unique_key) AS distinct_request_ids,
    SUM(is_closed) AS requests_with_closed_date,
    SUM(CASE WHEN is_closed = 0 THEN 1 ELSE 0 END) AS requests_without_closed_date,
    SUM(status_closed_missing_closed_date) AS status_closed_missing_close_timestamp,
    ROUND(100.0 * SUM(has_due_date) / COUNT(*), 2) AS pct_with_due_date,
    SUM(created_date_invalid) AS invalid_created_dates,
    SUM(closure_order_invalid) AS invalid_closure_chronology,
    COUNT(slow_72h) AS known_72h_outcomes,
    SUM(slow_72h) AS slow_72h_requests
FROM requests;

-- 2. Due-date coverage and late closure, shown only for comparable records
SELECT
    agency,
    COUNT(*) AS requests,
    SUM(has_due_date) AS requests_with_due_date,
    ROUND(100.0 * SUM(has_due_date) / COUNT(*), 2) AS pct_with_due_date,
    SUM(closed_after_due) AS closed_after_due_count,
    ROUND(100.0 * AVG(closed_after_due), 2) AS pct_closed_after_due_among_comparable
FROM requests
GROUP BY agency
ORDER BY requests DESC;

-- 3. Agency scorecard. P90 uses nearest-rank definition: rank = CEILING(0.9*n).
WITH closed_ranked AS (
    SELECT
        agency,
        resolution_hours,
        ROW_NUMBER() OVER (PARTITION BY agency ORDER BY resolution_hours) AS rn,
        COUNT(*) OVER (PARTITION BY agency) AS n
    FROM requests
    WHERE resolution_hours IS NOT NULL
),
closed_stats AS (
    SELECT
        agency,
        MAX(n) AS valid_closed_requests,
        AVG(CASE WHEN rn IN ((n + 1) / 2, (n + 2) / 2)
                 THEN resolution_hours END) AS median_resolution_hours,
        MAX(CASE
            WHEN rn = CAST(0.9 * n AS INT)
                     + CASE WHEN 0.9 * n > CAST(0.9 * n AS INT) THEN 1 ELSE 0 END
            THEN resolution_hours END) AS p90_resolution_hours
    FROM closed_ranked
    GROUP BY agency
),
delay_stats AS (
    SELECT
        agency,
        COUNT(slow_72h) AS eligible_requests,
        SUM(slow_72h) AS slow_72h_requests,
        ROUND(100.0 * AVG(slow_72h), 2) AS pct_not_closed_within_72h
    FROM requests
    GROUP BY agency
)
SELECT
    c.agency,
    c.valid_closed_requests,
    ROUND(c.median_resolution_hours, 1) AS median_resolution_hours,
    ROUND(c.p90_resolution_hours, 1) AS p90_resolution_hours,
    d.eligible_requests,
    d.slow_72h_requests,
    d.pct_not_closed_within_72h
FROM closed_stats c
LEFT JOIN delay_stats d USING (agency)
ORDER BY c.valid_closed_requests DESC;

-- 4. Pareto: each agency's contribution to all known slow-72h outcomes
WITH agency_slow AS (
    SELECT agency, SUM(slow_72h) AS slow_requests
    FROM requests
    GROUP BY agency
)
SELECT
    agency,
    slow_requests,
    ROUND(100.0 * slow_requests / SUM(slow_requests) OVER (), 2) AS pct_of_all_slow,
    ROUND(100.0 * SUM(slow_requests) OVER (
        ORDER BY slow_requests DESC ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) / SUM(slow_requests) OVER (), 2) AS cumulative_pct
FROM agency_slow
ORDER BY slow_requests DESC;

-- 5. Complaint types with highest slow-72h rates (minimum 500 known outcomes)
SELECT
    agency, complaint_type,
    COUNT(slow_72h) AS eligible_requests,
    SUM(slow_72h) AS slow_72h_requests,
    ROUND(AVG(slow_72h) * 100.0, 2) AS pct_not_closed_within_72h,
    ROUND(AVG(resolution_hours), 1) AS avg_hours_among_closed_valid_requests
FROM requests
GROUP BY agency, complaint_type
HAVING COUNT(slow_72h) >= 500
ORDER BY pct_not_closed_within_72h DESC, eligible_requests DESC
LIMIT 25;

-- 6. Borough comparison
SELECT
    borough,
    COUNT(slow_72h) AS eligible_requests,
    SUM(slow_72h) AS slow_72h_requests,
    ROUND(AVG(slow_72h) * 100.0, 2) AS pct_not_closed_within_72h
FROM requests
GROUP BY borough
ORDER BY eligible_requests DESC;

-- 7. Weekday pattern (0=Monday ... 6=Sunday; Python weekday convention)
SELECT
    created_weekday,
    CASE created_weekday
        WHEN 0 THEN 'Monday' WHEN 1 THEN 'Tuesday' WHEN 2 THEN 'Wednesday'
        WHEN 3 THEN 'Thursday' WHEN 4 THEN 'Friday' WHEN 5 THEN 'Saturday'
        WHEN 6 THEN 'Sunday'
    END AS weekday_name,
    COUNT(slow_72h) AS eligible_requests,
    ROUND(AVG(slow_72h) * 100.0, 2) AS pct_not_closed_within_72h
FROM requests
GROUP BY created_weekday
ORDER BY created_weekday;

-- 8. Hour-of-day pattern
SELECT
    created_hour AS hour_of_day,
    COUNT(slow_72h) AS eligible_requests,
    ROUND(AVG(slow_72h) * 100.0, 2) AS pct_not_closed_within_72h
FROM requests
GROUP BY created_hour
ORDER BY created_hour;

-- 9. Monthly request volume and 72-hour slow rate by creation month
SELECT
    created_month,
    COUNT(*) AS requests_created,
    COUNT(slow_72h) AS known_72h_outcomes,
    SUM(slow_72h) AS slow_72h_requests,
    ROUND(AVG(slow_72h) * 100.0, 2) AS pct_not_closed_within_72h
FROM requests
GROUP BY created_month
ORDER BY created_month;

-- 10. Reconstructed month-end backlog for the 2025-created cohort.
WITH month_ends(month_end) AS (
    VALUES
    ('2025-01-31 23:59:59'), ('2025-02-28 23:59:59'),
    ('2025-03-31 23:59:59'), ('2025-04-30 23:59:59'),
    ('2025-05-31 23:59:59'), ('2025-06-30 23:59:59'),
    ('2025-07-31 23:59:59'), ('2025-08-31 23:59:59'),
    ('2025-09-30 23:59:59'), ('2025-10-31 23:59:59'),
    ('2025-11-30 23:59:59'), ('2025-12-31 23:59:59')
)
SELECT
    strftime('%Y-%m', m.month_end) AS month,
    COUNT(r.unique_key) AS month_end_backlog
FROM month_ends m
LEFT JOIN requests r
    ON r.created_date <= m.month_end
   AND (r.closed_date IS NULL OR r.closed_date > m.month_end)
   AND r.created_date_invalid = 0
   AND r.closure_order_invalid = 0
   AND r.status_closed_missing_closed_date = 0
GROUP BY m.month_end
ORDER BY m.month_end;

-- 11. Reconcile status with closed_date; exposes status/closure timestamp conflicts.
SELECT
    agency,
    status,
    CASE WHEN closed_date IS NULL THEN 'Missing closed_date'
         ELSE 'Has closed_date' END AS closure_timestamp_state,
    COUNT(*) AS request_count
FROM requests
GROUP BY agency, status, closure_timestamp_state
ORDER BY request_count DESC;

"""Generate held-out subgroup metrics without retraining or replacing 04_model.py."""
import json
import sqlite3
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, average_precision_score, f1_score,
    precision_score, recall_score, roc_auc_score
)

DB_PATH = Path("data/nyc311.db")
MODEL_PATH = Path("models/slow_72h_model.joblib")
SUMMARY_PATH = Path("reports/model_summary.json")
OUT_PATH = Path("results/subgroup_metrics.csv")


def metrics_for_group(name, group_name, part, scores, threshold):
    y = part["y"].to_numpy()
    pred = (scores >= threshold).astype(int)
    return {
        "dimension": name,
        "group": str(group_name),
        "rows": int(len(part)),
        "actual_slow_rate": float(y.mean()),
        "predicted_slow_rate": float(pred.mean()),
        # Both ranking metrics require positive and negative examples to be
        # interpretable for subgroup comparisons. Mark single-class groups NA.
        "roc_auc": float(roc_auc_score(y, scores)) if len(np.unique(y)) == 2 else np.nan,
        "average_precision": float(average_precision_score(y, scores)) if len(np.unique(y)) == 2 else np.nan,
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "accuracy": float(accuracy_score(y, pred)),
        "threshold": float(threshold),
    }


def main():
    for p in (DB_PATH, MODEL_PATH, SUMMARY_PATH):
        if not p.exists():
            raise FileNotFoundError(f"Required file missing: {p}")

    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    threshold = float(summary["thresholds_selected_on_validation"]["lightgbm"])
    test_start = pd.Timestamp(summary["test_start"])

    with sqlite3.connect(DB_PATH) as con:
        df = pd.read_sql_query(
            """
            SELECT created_date, agency, complaint_type, borough,
                   created_hour, created_weekday, slow_72h
            FROM requests
            WHERE slow_72h IS NOT NULL
            """, con
        )

    df["created_date"] = pd.to_datetime(df["created_date"], errors="coerce")
    df["slow_72h"] = pd.to_numeric(df["slow_72h"], errors="coerce")
    df = df.dropna(subset=["created_date", "slow_72h"]).copy()
    df["y"] = df["slow_72h"].astype("int8")
    for col in ["agency", "complaint_type", "borough"]:
        df[col] = df[col].fillna("Unknown").astype(str)
    for col in ["created_hour", "created_weekday"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Recreate the exact chronological split used by 04_model.py.
    dates = np.array(sorted(df["created_date"].dt.normalize().unique()))
    valid_idx = max(max(1, int(len(dates) * 0.70)) + 1, int(len(dates) * 0.85))
    if valid_idx >= len(dates):
        valid_idx = len(dates) - 1
    recreated_test_start = pd.Timestamp(dates[valid_idx])
    if recreated_test_start.normalize() != test_start.normalize():
        raise RuntimeError(
            f"Split mismatch: summary says {test_start.date()}, "
            f"database recreates {recreated_test_start.date()}. Stop and investigate."
        )

    test = df[df["created_date"] >= test_start].copy()
    features = ["complaint_type", "borough", "agency", "created_hour", "created_weekday"]
    model = joblib.load(MODEL_PATH)
    scores = model.predict_proba(test[features])[:, 1]

    rows = []
    # Boroughs: include all groups; small/unknown categories remain visible.
    for group, part in test.groupby("borough", dropna=False):
        idx = part.index
        pos = test.index.get_indexer(idx)
        rows.append(metrics_for_group("borough", group, part, scores[pos], threshold))

    # Agencies: top 15 by test volume, plus a pooled Other group.
    agency_counts = test["agency"].value_counts()
    top_agencies = set(agency_counts.head(15).index)
    agency_labels = test["agency"].where(test["agency"].isin(top_agencies), "Other (remaining agencies)")
    for group, part in test.assign(_agency_group=agency_labels).groupby("_agency_group"):
        pos = test.index.get_indexer(part.index)
        rows.append(metrics_for_group("agency", group, part, scores[pos], threshold))

    # Test-period month comparison. November is only Nov 7-30.
    month_labels = test["created_date"].dt.to_period("M").astype(str)
    for group, part in test.assign(_month=month_labels).groupby("_month"):
        pos = test.index.get_indexer(part.index)
        rows.append(metrics_for_group("test_month", group, part, scores[pos], threshold))

    out = pd.DataFrame(rows)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False)
    print(f"Test rows evaluated: {len(test):,}")
    print(f"Test period: {test['created_date'].min()} to {test['created_date'].max()}")
    print(f"Threshold (from saved validation result): {threshold:.2f}")
    print(f"Saved: {OUT_PATH}")
    print("\nSUBGROUP METRICS")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()

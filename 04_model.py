"""NYC 311: predict the operational 72-hour delay outcome from creation-time fields.

Target:
  slow_72h = 1 when a request was not closed within 72 hours
              (closed after 72 hours, or still had no closure timestamp
               and was not marked Closed in the downloaded extract).
  slow_72h = 0 when closed within 72 hours.
  NULL outcomes are excluded by the loader.

This is a retrospective model using the downloaded 2025 cohort. It is not an
official NYC SLA model and should not be described as a live real-time system.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


DB_PATH = Path("data/nyc311.db")
RESULTS = Path("results")
MODELS = Path("models")
REPORTS = Path("reports")
RANDOM_STATE = 42

for folder in (RESULTS, MODELS, REPORTS):
    folder.mkdir(parents=True, exist_ok=True)


def choose_threshold(y_true: pd.Series, scores: np.ndarray) -> float:
    """Choose the threshold with best F1 on validation data only."""
    candidates = np.linspace(0.05, 0.95, 91)
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in candidates:
        labels = (scores >= threshold).astype(int)
        score = f1_score(y_true, labels, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_threshold = float(threshold)
    return best_threshold


def score_metrics(name: str, y_true: pd.Series, scores: np.ndarray,
                  threshold: float) -> dict:
    labels = (scores >= threshold).astype(int)
    return {
        "model": name,
        "threshold_selected_on_validation": round(float(threshold), 4),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "average_precision": float(average_precision_score(y_true, scores)),
        "precision": float(precision_score(y_true, labels, zero_division=0)),
        "recall": float(recall_score(y_true, labels, zero_division=0)),
        "f1": float(f1_score(y_true, labels, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, labels)),
    }


def main() -> None:
    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"Database not found: {DB_PATH}. Run 02_load_sqlite.py first."
        )

    with sqlite3.connect(DB_PATH) as con:
        columns = {
            row[1] for row in con.execute("PRAGMA table_info(requests)").fetchall()
        }

        required = {
            "unique_key", "created_date", "agency", "complaint_type",
            "borough", "created_hour", "created_weekday", "slow_72h",
        }
        missing = sorted(required - columns)
        if missing:
            raise RuntimeError(
                "Database schema does not match the current loader. "
                f"Missing columns: {missing}. Re-run 02_load_sqlite.py."
            )

        unknown_outcome_rows = con.execute(
            "SELECT COUNT(*) FROM requests WHERE slow_72h IS NULL"
        ).fetchone()[0]

        df = pd.read_sql_query(
            """
            SELECT
                unique_key,
                created_date,
                agency,
                complaint_type,
                borough,
                created_hour,
                created_weekday,
                slow_72h
            FROM requests
            WHERE slow_72h IS NOT NULL
            """,
            con,
        )

    if df.empty:
        raise SystemExit("No known slow_72h outcomes found in the database.")

    df["created_date"] = pd.to_datetime(df["created_date"], errors="coerce")
    df["slow_72h"] = pd.to_numeric(df["slow_72h"], errors="coerce")
    df = df.dropna(subset=["created_date", "slow_72h"]).copy()
    df["y"] = df["slow_72h"].astype("int8")

    categorical = ["complaint_type", "borough", "agency"]
    numeric = ["created_hour", "created_weekday"]
    features = categorical + numeric

    for col in categorical:
        df[col] = df[col].fillna("Unknown").astype(str)
    for col in numeric:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Split by unique creation dates: earliest 70% train, next 15% validation,
    # latest 15% final test. No random row-level split.
    dates = np.array(sorted(df["created_date"].dt.normalize().unique()))
    if len(dates) < 20:
        raise SystemExit(
            f"Only {len(dates)} distinct creation dates; too few for "
            "a reliable chronological train/validation/test split."
        )

    train_idx = max(1, int(len(dates) * 0.70))
    valid_idx = max(train_idx + 1, int(len(dates) * 0.85))
    if valid_idx >= len(dates):
        valid_idx = len(dates) - 1

    train_end = pd.Timestamp(dates[train_idx])
    valid_end = pd.Timestamp(dates[valid_idx])

    train = df[df["created_date"] < train_end].copy()
    valid = df[
        (df["created_date"] >= train_end)
        & (df["created_date"] < valid_end)
    ].copy()
    test = df[df["created_date"] >= valid_end].copy()

    for name, part in (("train", train), ("validation", valid), ("test", test)):
        if part.empty or part["y"].nunique() < 2:
            raise SystemExit(
                f"{name.title()} split is empty or contains only one target class. "
                "Check the date range and outcome distribution."
            )

    X_train, y_train = train[features], train["y"]
    X_valid, y_valid = valid[features], valid["y"]
    X_test, y_test = test[features], test["y"]

    print(
        f"Known outcomes: {len(df):,} | train: {len(train):,} | "
        f"validation: {len(valid):,} | test: {len(test):,}"
    )
    print(
        f"Date boundaries: train before {train_end.date()} | "
        f"validation before {valid_end.date()} | "
        f"test from {valid_end.date()}"
    )
    print(
        f"Slow rate: train {y_train.mean():.2%} | "
        f"validation {y_valid.mean():.2%} | test {y_test.mean():.2%}"
    )

    # Baseline 1: majority-class probability from training data.
    dummy = DummyClassifier(strategy="prior")
    dummy.fit(np.zeros((len(y_train), 1)), y_train)
    majority_valid = dummy.predict_proba(np.zeros((len(y_valid), 1)))[:, 1]
    majority_test = dummy.predict_proba(np.zeros((len(y_test), 1)))[:, 1]

    # Baseline 2: complaint-type rate learned only from training data.
    type_rates = train.groupby("complaint_type")["y"].mean()
    type_valid = valid["complaint_type"].map(type_rates).fillna(y_train.mean()).to_numpy()
    type_test = test["complaint_type"].map(type_rates).fillna(y_train.mean()).to_numpy()

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            ),
            (
                "numeric",
                Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]),
                numeric,
            ),
        ]
    )

    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise SystemExit(
            "LightGBM is not installed in this environment. "
            "Activate the project virtual environment and run: "
            "python -m pip install lightgbm"
        ) from exc

    classifier = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        max_bin=127,
        min_child_samples=100,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=-1,
    )
    model = Pipeline(
        steps=[("preprocessor", preprocessor), ("classifier", classifier)]
    )

    print("\nTraining LightGBM...")
    model.fit(X_train, y_train)

    # Select thresholds on validation only; final test is not used for tuning.
    model_valid = model.predict_proba(X_valid)[:, 1]
    model_test = model.predict_proba(X_test)[:, 1]
    # Keep the majority baseline as a true majority-class classifier:
    # its probability score is constant and its 0.5 decision predicts class 0.
    majority_threshold = 0.5
    type_threshold = choose_threshold(y_valid, type_valid)
    model_threshold = choose_threshold(y_valid, model_valid)

    metric_rows = [
        score_metrics(
            "Majority-class baseline", y_test, majority_test, majority_threshold
        ),
        score_metrics(
            "Complaint-type-rate baseline", y_test, type_test, type_threshold
        ),
        score_metrics("LightGBM", y_test, model_test, model_threshold),
    ]
    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(RESULTS / "model_metrics.csv", index=False)

    print("\nHELD-OUT TEST METRICS")
    print(metrics_df.to_string(index=False))

    # Agency ablation: retrain with the same dates, target, preprocessing,
    # hyperparameters, and validation-only threshold selection, but exclude
    # agency from the feature set. This isolates the contribution of agency.
    ablation_categorical = ["complaint_type", "borough"]
    ablation_numeric = ["created_hour", "created_weekday"]
    ablation_features = ablation_categorical + ablation_numeric

    ablation_preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                ablation_categorical,
            ),
            (
                "numeric",
                Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]),
                ablation_numeric,
            ),
        ]
    )
    ablation_classifier = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        max_bin=127,
        min_child_samples=100,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=-1,
    )
    ablation_model = Pipeline(
        steps=[
            ("preprocessor", ablation_preprocessor),
            ("classifier", ablation_classifier),
        ]
    )

    print("\nTraining LightGBM WITHOUT agency...")
    ablation_model.fit(
        train[ablation_features], y_train
    )
    ablation_valid_scores = ablation_model.predict_proba(
        valid[ablation_features]
    )[:, 1]
    ablation_test_scores = ablation_model.predict_proba(
        test[ablation_features]
    )[:, 1]
    ablation_threshold = choose_threshold(y_valid, ablation_valid_scores)
    ablation_metrics = score_metrics(
        "LightGBM without agency",
        y_test,
        ablation_test_scores,
        ablation_threshold,
    )

    agency_ablation_df = pd.DataFrame(
        [metric_rows[-1], ablation_metrics]
    )
    agency_ablation_df.to_csv(
        RESULTS / "agency_ablation_metrics.csv", index=False
    )
    print("\nAGENCY ABLATION COMPARISON")
    print(agency_ablation_df.to_string(index=False))

    test_labels = (model_test >= model_threshold).astype(int)
    matrix = confusion_matrix(y_test, test_labels, labels=[0, 1])
    pd.DataFrame(
        matrix,
        index=["actual_on_time_0", "actual_slow_1"],
        columns=["predicted_on_time_0", "predicted_slow_1"],
    ).to_csv(RESULTS / "confusion_matrix.csv")

    # Permutation importance on a deterministic sample to keep runtime bounded.
    sample_n = min(50_000, len(X_test))
    sample = X_test.sample(n=sample_n, random_state=RANDOM_STATE)
    sample_y = y_test.loc[sample.index]
    print(f"\nCalculating permutation importance on {sample_n:,} test rows...")
    importance_result = permutation_importance(
        model,
        sample,
        sample_y,
        scoring="roc_auc",
        n_repeats=3,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    importance = pd.DataFrame(
        {
            "feature": features,
            "auc_drop_mean": importance_result.importances_mean,
            "auc_drop_std": importance_result.importances_std,
        }
    ).sort_values("auc_drop_mean", ascending=False)
    importance.to_csv(RESULTS / "feature_importance.csv", index=False)

    plt.figure(figsize=(8, 4.8))
    top = importance.sort_values("auc_drop_mean").tail(10)
    plt.barh(top["feature"], top["auc_drop_mean"])
    plt.xlabel("Decrease in ROC-AUC after permutation")
    plt.title("Creation-time feature importance (test sample)")
    plt.tight_layout()
    plt.savefig(RESULTS / "feature_importance.png", dpi=160)
    plt.close()

    joblib.dump(model, MODELS / "slow_72h_model.joblib")

    summary = {
        "target": "Not closed within 72 hours of creation",
        "target_column": "slow_72h",
        "target_definition": (
            "1 = closure timestamp is more than 72 hours after creation, "
            "or no closure timestamp exists and status is not Closed; "
            "0 = closed within 72 hours; NULL outcomes are excluded."
        ),
        "interpretation_note": (
            "Operational outcome from the downloaded 2025 cohort; not an "
            "official NYC SLA breach and not a live real-time prediction."
        ),
        "known_outcome_rows": int(len(df)),
        "excluded_unknown_outcome_rows": int(unknown_outcome_rows),
        "train_rows": int(len(train)),
        "validation_rows": int(len(valid)),
        "test_rows": int(len(test)),
        "train_end_exclusive": str(train_end.date()),
        "validation_end_exclusive": str(valid_end.date()),
        "test_start": str(valid_end.date()),
        "train_slow_rate": float(y_train.mean()),
        "validation_slow_rate": float(y_valid.mean()),
        "test_slow_rate": float(y_test.mean()),
        "features": features,
        "thresholds_selected_on_validation": {
            "majority_class_baseline": majority_threshold,
            "complaint_type_rate_baseline": type_threshold,
            "lightgbm": model_threshold,
        },
        "test_metrics": metric_rows,
        "agency_ablation": {
            "comparison_file": "results/agency_ablation_metrics.csv",
            "full_model": metric_rows[-1],
            "without_agency": ablation_metrics,
            "interpretation": (
                "Both models use identical chronological train/validation/test "
                "dates. The without-agency model excludes only the agency feature."
            ),
        },
        "permutation_importance_sample_rows": int(sample_n),
    }
    (REPORTS / "model_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print(
        "\nSaved results/model_metrics.csv, results/confusion_matrix.csv, "
        "results/feature_importance.csv/png, "
        "results/agency_ablation_metrics.csv, reports/model_summary.json, "
        "models/slow_72h_model.joblib"
    )


if __name__ == "__main__":
    main()

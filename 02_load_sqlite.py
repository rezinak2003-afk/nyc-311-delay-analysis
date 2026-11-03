"""Chunked, auditable loader for the NYC 311 2025 extract.

Run from the project root:
    python 02_load_sqlite.py

Source CSV is retained unchanged. Invalid chronology is flagged, not deleted.
72-hour target:
  0 = closed within 72 hours
  1 = valid closure took >72 hours OR no closure timestamp and status is not Closed
  NULL = invalid chronology/creation date OR status says Closed but closed_date is missing
The target is an operational outcome, not an official NYC SLA breach.
"""
from __future__ import annotations
import sqlite3
import sys
from pathlib import Path
import pandas as pd

CSV_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/raw_311_2025.csv")
DB_PATH = Path("data/nyc311.db")
CHUNK_SIZE = 100_000

KEEP = [
    "unique_key", "created_date", "closed_date", "due_date",
    "resolution_action_updated_date", "agency", "complaint_type",
    "descriptor", "location_type", "borough", "status", "incident_zip",
]


def prepare_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    chunk = chunk.rename(columns={"problem": "complaint_type",
                                  "problem_detail": "descriptor"})
    for col in KEEP:
        if col not in chunk.columns:
            chunk[col] = pd.NA
    df = chunk[KEEP].copy()

    for col in ["created_date", "closed_date", "due_date",
                "resolution_action_updated_date"]:
        df[col] = pd.to_datetime(df[col], errors="coerce")

    df["created_date_invalid"] = df["created_date"].isna().astype("int8")
    df["closure_order_invalid"] = (
        df["created_date"].notna() & df["closed_date"].notna()
        & (df["closed_date"] < df["created_date"])
    ).astype("int8")
    df["status_closed_missing_closed_date"] = (
        df["status"].astype("string").str.strip().str.casefold().eq("closed")
        & df["closed_date"].isna()
    ).fillna(False).astype("int8")

    valid_closure = (
        df["closed_date"].notna() & df["created_date"].notna()
        & (df["closure_order_invalid"] == 0)
    )
    df["resolution_hours"] = float("nan")
    df.loc[valid_closure, "resolution_hours"] = (
        (df.loc[valid_closure, "closed_date"] -
         df.loc[valid_closure, "created_date"]).dt.total_seconds() / 3600
    )

    df["is_closed"] = df["closed_date"].notna().astype("int8")
    df["has_due_date"] = df["due_date"].notna().astype("int8")

    # Known outcome only. A Closed status without closed_date is contradictory;
    # do not silently label it slow. A non-Closed status without close_date is open.
    status_is_closed = (
        df["status"].astype("string").str.strip().str.casefold().eq("closed")
        .fillna(False)
    )
    valid_target = (
        df["created_date"].notna()
        & (df["closure_order_invalid"] == 0)
        & ~((df["closed_date"].isna()) & status_is_closed)
    )
    closed_within_72 = (
        df["closed_date"].notna()
        & (df["closed_date"] <= df["created_date"] + pd.Timedelta(hours=72))
    )
    df["slow_72h"] = float("nan")
    df.loc[valid_target, "slow_72h"] = (
        ~closed_within_72.loc[valid_target]
    ).astype("int8")

    due_comparable = (
        df["closed_date"].notna() & df["due_date"].notna()
        & (df["closure_order_invalid"] == 0)
    )
    df["closed_after_due"] = float("nan")
    df.loc[due_comparable, "closed_after_due"] = (
        df.loc[due_comparable, "closed_date"] >
        df.loc[due_comparable, "due_date"]
    ).astype("int8")

    df["borough"] = df["borough"].fillna("Unspecified")
    df["agency"] = df["agency"].fillna("Unspecified")
    df["complaint_type"] = df["complaint_type"].fillna("Unspecified")
    df["created_hour"] = df["created_date"].dt.hour.astype("float64")
    df["created_weekday"] = df["created_date"].dt.weekday.astype("float64")
    df["created_month"] = df["created_date"].dt.strftime("%Y-%m")
    df["created_day"] = df["created_date"].dt.strftime("%Y-%m-%d")

    for col in ["created_date", "closed_date", "due_date",
                "resolution_action_updated_date"]:
        df[col] = df[col].dt.strftime("%Y-%m-%d %H:%M:%S")
    return df


def main() -> None:
    if not CSV_PATH.exists():
        raise SystemExit(f"Input CSV not found: {CSV_PATH.resolve()}")
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    if DB_PATH.exists():
        DB_PATH.unlink()

    total_in = total_loaded = invalid_created = invalid_order = 0
    status_conflicts = duplicates_in_file = 0
    seen_keys: set[str] = set()

    print(f"Input: {CSV_PATH.resolve()}")
    print(f"Database: {DB_PATH.resolve()}")
    print(f"Reading in chunks of {CHUNK_SIZE:,} rows...")

    with sqlite3.connect(DB_PATH) as con:
        for number, raw in enumerate(
            pd.read_csv(CSV_PATH, chunksize=CHUNK_SIZE, low_memory=False), start=1
        ):
            total_in += len(raw)
            prepared = prepare_chunk(raw)
            invalid_created += int(prepared["created_date_invalid"].sum())
            invalid_order += int(prepared["closure_order_invalid"].sum())
            status_conflicts += int(prepared["status_closed_missing_closed_date"].sum())

            keys = prepared["unique_key"].astype("string")
            duplicate_mask = keys.isna() | keys.isin(seen_keys) | keys.duplicated()
            duplicates_in_file += int(duplicate_mask.sum())
            prepared = prepared.loc[~duplicate_mask].copy()
            seen_keys.update(keys.loc[~duplicate_mask].astype(str).tolist())

            prepared.to_sql(
                "requests", con,
                if_exists="append" if number > 1 else "replace",
                index=False, chunksize=20_000
            )
            total_loaded += len(prepared)
            print(f"Chunk {number}: read {len(raw):,}, loaded {len(prepared):,}; "
                  f"total loaded {total_loaded:,}")

        for index_sql in [
            "CREATE INDEX IF NOT EXISTS ix_requests_agency ON requests(agency)",
            "CREATE INDEX IF NOT EXISTS ix_requests_type ON requests(complaint_type)",
            "CREATE INDEX IF NOT EXISTS ix_requests_borough ON requests(borough)",
            "CREATE INDEX IF NOT EXISTS ix_requests_created ON requests(created_date)",
            "CREATE INDEX IF NOT EXISTS ix_requests_month ON requests(created_month)",
        ]:
            con.execute(index_sql)
        con.commit()

        check = con.execute("""
            SELECT COUNT(*), COUNT(DISTINCT unique_key),
                   MIN(created_date), MAX(created_date),
                   SUM(closure_order_invalid),
                   SUM(status_closed_missing_closed_date),
                   COUNT(slow_72h), SUM(slow_72h)
            FROM requests
        """).fetchone()

    print("\nLOAD COMPLETE")
    print(f"Rows read: {total_in:,}")
    print(f"Rows loaded: {total_loaded:,}")
    print(f"Duplicate/missing IDs skipped: {duplicates_in_file:,}")
    print(f"Invalid creation dates: {invalid_created:,}")
    print(f"Closed before created (flagged, retained): {invalid_order:,}")
    print(f"Status Closed but closed_date missing (flagged, target unknown): {status_conflicts:,}")
    print(f"Database rows: {check[0]:,}")
    print(f"Distinct IDs: {check[1]:,}")
    print(f"Created date range: {check[2]} to {check[3]}")
    print(f"Known 72h labels: {check[6]:,}")
    print(f"Slow 72h labels: {int(check[7] or 0):,}")


if __name__ == "__main__":
    main()

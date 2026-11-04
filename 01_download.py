"""Download NYC 311 requests by creation date, one day at a time.

Default period: 2025-01-01 through 2025-12-31 (END is exclusive).
Run from the project root:
    python 01_download.py
Optional:
    python 01_download.py --start 2025-01-01 --end 2026-01-01
    python 01_download.py --refresh-existing

Daily files are written atomically. Existing files are checked for parseability,
required columns, duplicate IDs, and date-range validity before being reused.
Set SOCRATA_APP_TOKEN in the environment if you have one.
"""
from __future__ import annotations

import argparse
import io
import json
import hashlib
import os
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

BASE = "https://data.cityofnewyork.us/resource/erm2-nwe9.csv"
FIELD_OPTIONS = [
    ["unique_key"],
    ["created_date"],
    ["closed_date"],
    ["due_date"],
    ["resolution_action_updated_date"],
    ["agency"],
    ["complaint_type", "problem"],
    ["descriptor", "problem_detail"],
    ["location_type"],
    ["borough"],
    ["status"],
    ["incident_zip"],
    ["latitude"],
    ["longitude"],
]
PAGE_SIZE = 50000
REQUIRED = {"unique_key", "created_date"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2025-01-01")
    parser.add_argument("--end", default="2026-01-01",
                        help="exclusive end date; default downloads all of 2025")
    parser.add_argument("--out", default="data/raw_311_2025.csv")
    parser.add_argument("--refresh-existing", action="store_true",
                        help="redownload daily files even if they already exist")
    return parser.parse_args()


def request_with_retries(session: requests.Session, params: dict, day: date) -> bytes:
    for attempt in range(6):
        try:
            response = session.get(BASE, params=params, timeout=180)
            response.raise_for_status()
            return response.content
        except requests.RequestException as exc:
            if attempt == 5:
                raise RuntimeError(f"{day}: request failed after 6 attempts: {exc}") from exc
            wait = min(60, 5 * (attempt + 1))
            print(f"{day}: request failed; retrying in {wait}s ({exc})")
            time.sleep(wait)
    raise RuntimeError(f"{day}: unexpected retry exit")


def source_day_count(session: requests.Session, day: date, next_day: date) -> int:
    """Return the source-side row count for one creation-date partition."""
    where = (
        f"created_date >= '{day.isoformat()}T00:00:00' "
        f"AND created_date < '{next_day.isoformat()}T00:00:00'"
    )
    content = request_with_retries(
        session, {"$select": "count(*)", "$where": where}, day
    )
    try:
        count_frame = pd.read_csv(io.BytesIO(content))
        if count_frame.empty or count_frame.shape[1] < 1:
            raise ValueError("empty count response")
        return int(count_frame.iloc[0, 0])
    except Exception as exc:
        raise RuntimeError(f"{day}: could not parse source count response: {exc}") from exc


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_day_file(path: Path, day: date) -> tuple[bool, str]:
    try:
        frame = pd.read_csv(path, low_memory=False)
        if not REQUIRED.issubset(frame.columns):
            return False, "required columns missing"
        created = pd.to_datetime(frame["created_date"], errors="coerce")
        if created.isna().any():
            return False, "invalid created_date values"
        if not frame.empty:
            start = pd.Timestamp(day)
            end = start + pd.Timedelta(days=1)
            if not ((created >= start) & (created < end)).all():
                return False, "contains rows outside requested day"
        if frame["unique_key"].isna().any():
            return False, "missing unique_key values"
        if frame["unique_key"].duplicated().any():
            return False, "duplicate unique_key values"
        return True, f"{len(frame):,} rows validated"
    except Exception as exc:
        return False, f"could not read/validate ({exc})"


def download_day(session: requests.Session, day: date, next_day: date,
                 columns: list[str], out_path: Path) -> int:
    where = (
        f"created_date >= '{day.isoformat()}T00:00:00' "
        f"AND created_date < '{next_day.isoformat()}T00:00:00'"
    )
    offset = 0
    parts: list[pd.DataFrame] = []

    while True:
        params = {
            "$select": ",".join(columns),
            "$where": where,
            "$limit": PAGE_SIZE,
            "$offset": offset,
            "$order": "created_date,unique_key",
        }
        content = request_with_retries(session, params, day)
        if not content.strip():
            break
        page = pd.read_csv(io.BytesIO(content), low_memory=False)
        if page.empty:
            break
        parts.append(page)
        print(f"  page offset {offset:,}: {len(page):,} rows")
        if len(page) < PAGE_SIZE:
            break
        offset += PAGE_SIZE

    daily = (pd.concat(parts, ignore_index=True)
             if parts else pd.DataFrame(columns=columns))

    # Validate the complete day before publishing it as a reusable file.
    if not REQUIRED.issubset(daily.columns):
        raise RuntimeError(f"{day}: download is missing required columns")
    if not daily.empty:
        daily["created_date"] = pd.to_datetime(daily["created_date"], errors="coerce")
        if daily["created_date"].isna().any():
            raise RuntimeError(f"{day}: invalid created_date returned by API")
        start_ts = pd.Timestamp(day)
        end_ts = start_ts + pd.Timedelta(days=1)
        in_day = daily["created_date"].ge(start_ts) & daily["created_date"].lt(end_ts)
        if not in_day.all():
            raise RuntimeError(f"{day}: API returned {int((~in_day).sum())} out-of-day rows")
        if daily["unique_key"].isna().any() or daily["unique_key"].duplicated().any():
            raise RuntimeError(f"{day}: missing or duplicate unique_key values")

    temp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    daily.to_csv(temp_path, index=False)
    temp_path.replace(out_path)
    return len(daily)


def main():
    args = parse_args()
    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    if end <= start:
        raise SystemExit("--end must be after --start")

    out_dir = Path("data/days")
    out_dir.mkdir(parents=True, exist_ok=True)

    headers = {}
    token = os.getenv("SOCRATA_APP_TOKEN")
    if token:
        headers["X-App-Token"] = token
    session = requests.Session()
    session.headers.update(headers)

    print("Checking NYC Open Data dataset schema...")
    probe = session.get(BASE, params={"$limit": 1}, timeout=90)
    probe.raise_for_status()
    available = list(pd.read_csv(io.BytesIO(probe.content), nrows=0).columns)

    columns = [next((candidate for candidate in options if candidate in available), None)
               for options in FIELD_OPTIONS]
    missing = [options[0] for options, selected in zip(FIELD_OPTIONS, columns)
               if selected is None]
    columns = [column for column in columns if column is not None]
    if missing:
        print("WARNING: source columns unavailable:", ", ".join(missing))
    if not REQUIRED.issubset(columns):
        raise SystemExit("Source schema does not contain unique_key and created_date.")

    print(f"Selected {len(columns)} columns: {', '.join(columns)}")
    print(f"Requested period: {start} through {(end - timedelta(days=1))} (inclusive)")
    print(f"Daily files: {out_dir.resolve()}")

    day = start
    total_days = (end - start).days
    completed_rows = 0
    day_number = 0
    run_started_at = datetime.now(timezone.utc).isoformat()
    day_manifest = []

    while day < end:
        day_number += 1
        next_day = min(day + timedelta(days=1), end)
        path = out_dir / f"{day.isoformat()}.csv"

        if path.exists() and not args.refresh_existing:
            valid, message = validate_day_file(path, day)
            if valid:
                cached_count = int(message.split()[0].replace(",", ""))
                expected_count = source_day_count(session, day, next_day)
                if cached_count != expected_count:
                    print(
                        f"[{day_number}/{total_days}] {day}: cached count "
                        f"{cached_count:,} differs from source count {expected_count:,}; "
                        "redownloading"
                    )
                    path.unlink()
                else:
                    print(f"[{day_number}/{total_days}] {day}: existing file OK; "
                          f"{cached_count:,} rows match source count")
                    completed_rows += cached_count
                    day_manifest.append({"date": day.isoformat(), "rows": cached_count,
                                         "origin": "existing_file_source_count_verified"})
                    day = next_day
                    continue
            print(f"[{day_number}/{total_days}] {day}: existing file invalid ({message}); redownloading")
            path.unlink()

        print(f"[{day_number}/{total_days}] Downloading {day}...")
        count = download_day(session, day, next_day, columns, path)
        expected_count = source_day_count(session, day, next_day)
        if count != expected_count:
            path.unlink(missing_ok=True)
            raise RuntimeError(
                f"{day}: downloaded {count:,} rows but NYC Open Data count(*) "
                f"reports {expected_count:,}. Daily file discarded. Retry the run."
            )
        completed_rows += count
        day_manifest.append({"date": day.isoformat(), "rows": count,
                             "origin": "downloaded_and_source_count_verified"})
        print(f"  saved {count:,} rows (source count verified) -> {path}")
        day = next_day

    # Confirm all expected daily files exist and are valid before combining.
    daily_frames = []
    total_rows = 0
    for offset in range(total_days):
        current = start + timedelta(days=offset)
        path = out_dir / f"{current.isoformat()}.csv"
        if not path.exists():
            raise SystemExit(f"Missing daily file: {path}")
        valid, message = validate_day_file(path, current)
        if not valid:
            raise SystemExit(f"Daily file failed final validation: {path}: {message}")
        frame = pd.read_csv(path, low_memory=False)
        daily_frames.append(frame)
        total_rows += len(frame)

    combined = (pd.concat(daily_frames, ignore_index=True)
                if daily_frames else pd.DataFrame(columns=columns))
    if combined["unique_key"].duplicated().any():
        duplicates = int(combined["unique_key"].duplicated().sum())
        raise SystemExit(f"Combined file has {duplicates} duplicate unique_key values.")

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    temp_output = output.with_suffix(output.suffix + ".tmp")
    combined.to_csv(temp_output, index=False)
    temp_output.replace(output)

    print("\nDOWNLOAD COMPLETE")
    print(f"Days checked: {total_days}")
    print(f"Total rows: {total_rows:,}")
    print(f"Unique request IDs: {combined['unique_key'].nunique():,}")
    print(f"Output: {output.resolve()}")
    created = pd.to_datetime(combined["created_date"], errors="coerce")
    completed_at = datetime.now(timezone.utc).isoformat()
    manifest = {
        "dataset": "NYC Open Data 311 Service Requests (erm2-nwe9)",
        "requested_start_date_inclusive": start.isoformat(),
        "requested_end_date_exclusive": end.isoformat(),
        "run_started_at_utc": run_started_at,
        "run_completed_at_utc": completed_at,
        "note": (
            "Run timestamps document when this combined extract was assembled/verified; "
            "they do not reconstruct the original download time of pre-existing cached files."
        ),
        "daily_files": day_manifest,
        "days_checked": total_days,
        "total_rows": int(len(combined)),
        "distinct_request_ids": int(combined["unique_key"].nunique()),
        "created_date_min": None if created.dropna().empty else str(created.min()),
        "created_date_max": None if created.dropna().empty else str(created.max()),
        "combined_csv": str(output),
        "combined_csv_sha256": sha256_file(output),
    }
    manifest_path = Path("reports/data_manifest.json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Manifest: {manifest_path.resolve()}")
    if not combined.empty:
        print(f"Actual created_date range: {created.min()} to {created.max()}")


if __name__ == "__main__":
    main()

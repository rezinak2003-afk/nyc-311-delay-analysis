import sqlite3
import re
from pathlib import Path
import pandas as pd

DB = Path("data/nyc311.db")
SQL = Path("03_analysis.sql")
OUT = Path("results")

def main():
    if not DB.exists():
        raise FileNotFoundError(DB)
    if not SQL.exists():
        raise FileNotFoundError(SQL)

    text = SQL.read_text(encoding="utf-8-sig")
    text = re.sub(r"(?m)--.*$", "", text)

    queries = [q.strip() for q in text.split(";") if q.strip()]
    OUT.mkdir(exist_ok=True)

    with sqlite3.connect(DB) as con:
        con.execute("PRAGMA query_only = ON")

        for i, query in enumerate(queries, 1):
            if not query.upper().lstrip().startswith(("SELECT", "WITH")):
                raise ValueError(f"Unexpected SQL statement {i}")

            df = pd.read_sql_query(query, con)
            filename = OUT / f"query_{i:02d}.csv"
            df.to_csv(filename, index=False)
            print(f"Query {i}: {len(df)} rows exported")

    print(f"\nExported {len(queries)} query outputs to results/")

if __name__ == "__main__":
    main()
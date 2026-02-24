import os
import json
import pandas as pd
import duckdb
import subprocess

DATA_DIR = "../data"
DB_PATH = "../guardian_articles.duckdb"

# Connect to DuckDB
con = duckdb.connect(DB_PATH)


# Run if search_term var is not in the table
# con.execute("""
# ALTER TABLE raw_articles
# ADD COLUMN IF NOT EXISTS search_term TEXT
# """)
con.execute("DROP TABLE IF EXISTS raw_articles")

# Ensure the raw_articles table exists with all needed columns
con.execute("""
CREATE TABLE IF NOT EXISTS raw_articles (
    id TEXT,
    type TEXT,
    sectionId TEXT,
    sectionName TEXT,
    webPublicationDate TIMESTAMP,
    webTitle TEXT,
    webUrl TEXT,
    apiUrl TEXT,
    body TEXT,
    isHosted TEXT,
    pillarId TEXT,
    pillarName TEXT,
    headline TEXT,
    shortUrl TEXT,
    search_term TEXT,
    pull_date TIMESTAMP
)
""")

# List all JSON files in data folder
json_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".json")]
print(f"Found {len(json_files)} JSON files.")

for json_file in json_files:
    file_path = os.path.join(DATA_DIR, json_file)
    print(f"Processing {json_file}...")

    # Load JSON
    with open(file_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    # Flatten 'fields' dict and ensure full HTML body
    records = []
    for item in raw_data:
        fields = item.get("fields", {})
        record = {
            "id": item.get("id"),
            "type": item.get("type"),
            "sectionId": item.get("sectionId"),
            "sectionName": item.get("sectionName"),
            "webPublicationDate": item.get("webPublicationDate"),
            "webTitle": item.get("webTitle"),
            "webUrl": item.get("webUrl"),
            "apiUrl": item.get("apiUrl"),
            "body": fields.get("body") or item.get("body") or "",
            "isHosted": str(item.get("isHosted")),
            "pillarId": item.get("pillarId"),
            "pillarName": item.get("pillarName"),
            "headline": fields.get("headline") or item.get("webTitle"),
            "shortUrl": fields.get("shortUrl") or item.get("webUrl"),
            "search_term": item.get("search_term"),
            "pull_date": pd.Timestamp.now(),
        }
        records.append(record)

    df = pd.DataFrame(records)

    # Keep only columns that exist in the table
    table_columns = [col[0] for col in con.execute("DESCRIBE raw_articles").fetchall()]
    df = df[[c for c in df.columns if c in table_columns]]

    # Remove duplicates based on 'id' and search_term
    existing_pairs = con.execute("SELECT id, search_term FROM raw_articles").fetchall()

    existing_pairs = set(existing_pairs)

    df = df[~df.apply(lambda x: (x["id"], x["search_term"]) in existing_pairs, axis=1)]

    if df.empty:
        print("No new articles to insert.")
        continue

    # Insert into DuckDB
    con.register("temp_df", df)
    con.execute("INSERT INTO raw_articles SELECT * FROM temp_df")
    con.unregister("temp_df")

    print(f"Inserted {len(df)} new articles.")

# running cleaned_articles sql process
subprocess.run(["dbt", "run"], cwd="dbt_guardian")

print("All done!")

import os
import json
import pandas as pd
import duckdb
import subprocess

DATA_DIR = "../data"
DB_PATH = "../guardian_articles.duckdb"

# Connect to DuckDB
con = duckdb.connect(DB_PATH)

# Re-produce the table (maybe edit this to not drop completely)
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
    search_terms TEXT,
    pull_date TIMESTAMP
)
""")

# List all JSON files in data folder
json_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".json")]
print(f"Found {len(json_files)} JSON files.")

for json_file in json_files:
    file_path = os.path.join(DATA_DIR, json_file)
    print(f"Processing {json_file}...")

    # Load the JSON file of raw articles
    with open(file_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

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
            "search_terms": item.get("search_terms"),
            "pull_date": pd.Timestamp.now(),
        }
        records.append(record)

    df = pd.DataFrame(records)

    # Insert into DuckDB
    con.register("temp_df", df)
    con.execute("INSERT INTO raw_articles SELECT * FROM temp_df")
    con.unregister("temp_df")

    print(f"Inserted {len(df)} new articles from {json_file}.")

# running cleaned_articles sql process (if RUN_DBT is TRUE)
RUN_DBT = True

if RUN_DBT:
    subprocess.run(["dbt", "run"], cwd="dbt_guardian")

print("All done!")

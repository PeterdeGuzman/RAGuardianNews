import duckdb
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime

DB_PATH = "../guardian_articles.duckdb"

# Connect to DuckDB
con = duckdb.connect(DB_PATH)

# Read raw_articles table
df = con.execute("SELECT * FROM raw_articles").fetchdf()
print(f"Loaded {len(df)} articles from raw_articles.")


# Function to clean HTML
def clean_html(html_text):
    if not html_text:
        return ""
    return BeautifulSoup(html_text, "html.parser").get_text(separator=" ", strip=True)


# Apply cleaning to the body column
df["clean_body"] = df["body"].apply(clean_html)

# Optionally, you can overwrite body or keep as separate column
# df["body"] = df["clean_body"]

# Create cleaned_articles table
con.execute("""
CREATE TABLE IF NOT EXISTS cleaned_articles AS
SELECT *, clean_body
FROM df
""")

# If you want to replace table instead of appending, you can do:
# con.execute("DROP TABLE IF EXISTS cleaned_articles")
# con.execute("CREATE TABLE cleaned_articles AS SELECT *, clean_body FROM df")

print("Cleaned articles saved to cleaned_articles table!")

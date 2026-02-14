import duckdb
import pandas as pd

# Connect to your database
con = duckdb.connect("../guardian_articles.duckdb")

# See all tables
tables = con.execute("SHOW TABLES").fetchall()
print(tables)

# Quick preview of your articles
df = con.execute("SELECT * FROM raw_articles LIMIT 5").fetchdf()
print(df)

# Example: count articles per section
section_counts = con.execute("""
    SELECT sectionName, COUNT(*) AS num_articles
    FROM raw_articles
    GROUP BY sectionName
    ORDER BY num_articles DESC
""").fetchdf()
print(section_counts)

# Example: search for articles containing "AI" in the body
ai_articles = con.execute("""
    SELECT body
    FROM raw_articles
""").fetchdf()
print(ai_articles)

print("now im gonna look at the cleaned articles table")

cleaned_body = con.execute("""
                           SELECT clean_body
                           FROM cleaned_articles
                           """).fetchdf()
print(cleaned_body)

import duckdb
import pandas as pd
from config import DB_PATH


def get_conn():
    return duckdb.connect(DB_PATH)


def load_articles() -> pd.DataFrame:
    """Load the dbt-produced cleaned_articles table."""
    conn = get_conn()
    df = conn.execute("""
        SELECT id, webTitle, clean_body, webPublicationDate
        FROM cleaned_articles
        ORDER BY webPublicationDate DESC
    """).df()
    conn.close()
    # BERTopic works best on a single concatenated text field
    df["text"] = df["webTitle"].fillna("") + " " + df["clean_body"].fillna("")
    return df


def save_topics(df_topics: pd.DataFrame):
    """Write article→topic assignments back to DuckDB."""
    conn = get_conn()
    conn.execute("DROP TABLE IF EXISTS article_topics")
    conn.execute("""
        CREATE TABLE article_topics AS
        SELECT * FROM df_topics
    """)
    conn.close()

# ner_spacy.py
import duckdb
import spacy
from tqdm import tqdm
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from config import SPACY_MODEL

BATCH_SIZE = 64
DB_PATH = "../guardian_articles.duckdb"

nlp = spacy.load(SPACY_MODEL)

con = duckdb.connect(DB_PATH)

con.execute("""
    CREATE TABLE IF NOT EXISTS article_entities_spacy (
        id VARCHAR,
        entity_text VARCHAR,
        entity_label VARCHAR,
        start_char INTEGER,
        end_char INTEGER
    )
""")

# Track already-processed articles so you can resume if it crashes
processed = (
    con.execute("SELECT DISTINCT id FROM article_entities_spacy").df()["id"].tolist()
)

df = con.execute("""
    SELECT id, text 
    FROM article_topics_labelled
""").df()

df = df[~df["id"].isin(processed)]  # skip already done
print(f"Articles to process: {len(df):,}")

rows = []
texts = df["text"].tolist()
ids = df["id"].tolist()

for doc, article_id in tqdm(
    zip(nlp.pipe(texts, batch_size=BATCH_SIZE), ids), total=len(ids)
):
    for ent in doc.ents:
        rows.append(
            (
                article_id,
                ent.text,
                ent.label_,
                ent.start_char,
                ent.end_char,
            )
        )

    # Checkpoint every 500 articles
    if len(rows) >= 500:
        con.executemany(
            "INSERT INTO article_entities_spacy VALUES (?, ?, ?, ?, ?)", rows
        )
        rows = []

# Insert any remaining
if rows:
    con.executemany("INSERT INTO article_entities_spacy VALUES (?, ?, ?, ?, ?)", rows)

con.close()
print("Done.")

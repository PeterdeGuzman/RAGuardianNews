# ner_bert.py
import duckdb
import torch
from transformers import pipeline
from tqdm import tqdm

BATCH_SIZE = 16  # smaller — BERT is heavier
DB_PATH = "../guardian_articles.duckdb"

# MPS = Apple Silicon GPU — faster than pure CPU
device = 0 if torch.backends.mps.is_available() else -1
print(f"Using {'MPS (Apple Silicon)' if device == 0 else 'CPU'}")

ner_pipeline = pipeline(
    "ner",
    model="dslim/bert-base-NER",
    aggregation_strategy="simple",  # merges subword tokens into full spans
    device=device,
)

con = duckdb.connect(DB_PATH)

con.execute("""
    CREATE TABLE IF NOT EXISTS article_entities_bert (
        id VARCHAR,
        entity_text VARCHAR,
        entity_label VARCHAR,
        score FLOAT,
        start_char INTEGER,
        end_char INTEGER
    )
""")

processed = (
    con.execute("SELECT DISTINCT id FROM article_entities_bert").df()["id"].tolist()
)

df = con.execute("""
    SELECT id, text 
    FROM article_topics_labelled
""").df()

df = df[~df["id"].isin(processed)]
print(f"Articles to process: {len(df):,}")

rows = []
ids = df["id"].tolist()
texts = df["text"].tolist()

for i, (article_id, text) in enumerate(tqdm(zip(ids, texts), total=len(ids))):
    # BERT has a 512 token limit — truncate long articles
    entities = ner_pipeline(text[:3000])

    for ent in entities:
        rows.append(
            (
                article_id,
                ent["word"],
                ent["entity_group"],
                float(ent["score"]),
                ent["start"],
                ent["end"],
            )
        )

    # Checkpoint every 200 articles
    if len(rows) >= 200:
        con.executemany(
            "INSERT INTO article_entities_bert VALUES (?, ?, ?, ?, ?, ?)", rows
        )
        rows = []

if rows:
    con.executemany("INSERT INTO article_entities_bert VALUES (?, ?, ?, ?, ?, ?)", rows)

con.close()
print("Done.")

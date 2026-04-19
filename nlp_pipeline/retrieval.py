import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

import numpy as np
import faiss
import duckdb
from sentence_transformers import SentenceTransformer
from config import EMBEDDINGS_PATH, FAISS_INDEX_PATH, DB_PATH, EMBEDDING_MODEL

# Load these once at a module level so the Streamlit app doesn't reload for every query
_index = None
_id_map = None
_embedder = None


# Function to load all resources for retrieval
def _load_resources():
    global _index, _id_map, _embedder
    if _index is None:
        print("Loading the FAISS index:")
        _index = faiss.read_index(str(FAISS_INDEX_PATH))
        _id_map = np.load(
            FAISS_INDEX_PATH.parent / "faiss_id_map.npy", allow_pickle=True
        )
        _embedder = SentenceTransformer(EMBEDDING_MODEL)


# Function to return top_k most relevant articles for the supplied query.
# Function returns a dict with id, webTitle, webPublicationDate, topic_label_clean, text, and similarity score
# Starting with 5 most relevant articles but will adjust if needed
def retrieve(query: str, top_k: int = 5) -> list[dict]:
    _load_resources()

    # Encode and normalize the query
    query_vec = _embedder.encode([query], normalize_embeddings=True).astype("float32")

    scores, indices = _index.search(query_vec, top_k)

    # Map the FAISS indices to the article IDs
    article_ids = [_id_map[i] for i in indices[0]]

    # Get full article details from DuckDB
    con = duckdb.connect(DB_PATH)
    placeholders = ", ".join(["?" for _ in article_ids])
    rows = con.execute(
        f"""
        SELECT
            id,
            webTitle, webPublicationDate, topic_label_clean, text
        FROM article_topics_labelled
        WHERE id in ({placeholders})               
                       """,
        article_ids,
    ).df()
    con.close()

    # Preserve the FAISS ranking order and append similarity scores to results
    results = []
    for article_id, score in zip(article_ids, scores[0]):
        match = rows[rows["id"] == article_id]
        if not match.empty:
            r = match.iloc[0].to_dict()
            r["score"] = float(score)
            results.append(r)
    return results

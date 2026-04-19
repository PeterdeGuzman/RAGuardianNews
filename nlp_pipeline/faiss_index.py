import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

import numpy as np
import faiss
import duckdb
from config import EMBEDDINGS_PATH, FAISS_INDEX_PATH, DB_PATH


# function to construct FAISS index and map index to article IDs
def build_faiss_index():
    print("Loading embeddings:")
    embeddings = np.load(EMBEDDINGS_PATH).astype("float32")
    n, dim = embeddings.shape
    print(f"The shape of the embeddings: contains {n:,} articles and {dim} dimensions")

    # calculate the inner product index
    # this requires normalized embeddings which were produced by embed.py
    # with normalized vectors, the inner product is equivalent to cosine similarity
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    print(f"FAISS index was built with a total of {index.ntotal:,} vectors.")

    faiss.write_index(index, str(FAISS_INDEX_PATH))
    print(f"FAISS index was saved to {FAISS_INDEX_PATH}")

    # Build and save id lookup table
    # Maps FAISS integer index to article id string
    # building this with article_topics_labelled rather than cleaned_articles
    # because I think it will make downstream topic pulling easier
    # I checked and id order appears identical but if there are issues need to change to same table as the one in embed.py
    con = duckdb.connect(DB_PATH)
    ids = (
        con.execute(
            """
            SELECT id FROM article_topics_labelled ORDER BY webPublicationDate DESC
            """
        )
        .df()["id"]
        .tolist()
    )
    con.close()

    id_map_path = FAISS_INDEX_PATH.parent / "faiss_id_map.npy"
    np.save(id_map_path, np.array(ids))
    print(f"The ID map was saved with {len(ids):,} entries.")


if __name__ == "__main__":
    build_faiss_index()

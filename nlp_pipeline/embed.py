import numpy as np
from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL, EMBEDDINGS_PATH


def get_embeddings(texts: list[str], force_recompute: bool = False) -> np.ndarray:
    """
    Returns shape (N, embedding_dim).
    Caches to disk so BERTopic training and FAISS indexing
    reuse the same vectors without re-encoding.
    """
    if EMBEDDINGS_PATH.exists() and not force_recompute:
        print("Loading cached embeddings...")
        return np.load(EMBEDDINGS_PATH)

    print(f"Encoding {len(texts)} documents with {EMBEDDING_MODEL}...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,  # needed for cosine similarity in FAISS
    )
    np.save(EMBEDDINGS_PATH, embeddings)
    return embeddings

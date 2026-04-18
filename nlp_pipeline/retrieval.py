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

from pathlib import Path

DB_PATH = "../guardian_articles.duckdb"
MODEL_DIR = Path("models/")
MODEL_DIR.mkdir(exist_ok=True)

BERTOPIC_MODEL_PATH = MODEL_DIR / "bertopic_model"
FAISS_INDEX_PATH = MODEL_DIR / "faiss.index"
EMBEDDINGS_PATH = MODEL_DIR / "embeddings.npy"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
SPACY_MODEL = "en_core_web_lg"

# BERTopic parameters
UMAP_N_NEIGHBORS = 15
UMAP_N_COMPONENTS = 5
HDBSCAN_MIN_CLUSTER = 10
MIN_TOPIC_SIZE = 10

from pathlib import Path

DB_PATH = "../guardian_articles.duckdb"
MODEL_DIR = Path("models/")
MODEL_DIR.mkdir(exist_ok=True)


# Version variable so I can iterate on the topic models
MODEL_VERSION = "v1"  # I will adjust this with each iteration

BERTOPIC_MODEL_PATH = MODEL_DIR / f"bertopic_model_{MODEL_VERSION}"
EMBEDDINGS_PATH = MODEL_DIR / "embeddings.npy"
FAISS_INDEX_PATH = MODEL_DIR / "faiss.index"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
SPACY_MODEL = "en_core_web_lg"

# BERTopic parameters
UMAP_N_NEIGHBORS = 15
UMAP_N_COMPONENTS = 5
HDBSCAN_MIN_CLUSTER = 10
MIN_TOPIC_SIZE = 10

# topic model experiments:
EXPERIMENTS = {
    "v1": {
        "min_cluster_size": 10,
        "min_samples": 5,
        "n_neighbors": 15,
        "n_components": 5,
        "method": "eom",
    },
    "v2": {
        "min_cluster_size": 5,
        "min_samples": 3,
        "n_neighbors": 15,
        "n_components": 5,
        "method": "eom",
    },
    "v3": {
        "min_cluster_size": 5,
        "min_samples": 3,
        "n_neighbors": 25,
        "n_components": 10,
        "method": "eom",
    },
    "v4": {
        "min_cluster_size": 5,
        "min_samples": 3,
        "n_neighbors": 25,
        "n_components": 10,
        "method": "leaf",
    },
    "v5": {
        "min_cluster_size": 3,
        "min_samples": 2,
        "n_neighbors": 25,
        "n_components": 10,
        "method": "leaf",
    },
}

EXPERIMENTS_DIR = MODEL_DIR / "experiments"
EXPERIMENTS_DIR.mkdir(exist_ok=True)

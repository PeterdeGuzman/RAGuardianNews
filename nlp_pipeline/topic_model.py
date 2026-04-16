import pandas as pd
import numpy as np
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer  # <-- add this
from config import (
    BERTOPIC_MODEL_PATH,
    UMAP_N_NEIGHBORS,
    UMAP_N_COMPONENTS,
    HDBSCAN_MIN_CLUSTER,
    MIN_TOPIC_SIZE,
    EMBEDDING_MODEL,
)


def build_topic_model() -> BERTopic:
    # Pass the actual model — BERTopic won't re-encode your corpus
    # since you supply embeddings to fit_transform(), but KeyBERTInspired
    # needs it for scoring representative words per topic
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)

    umap_model = UMAP(
        n_neighbors=UMAP_N_NEIGHBORS,
        n_components=UMAP_N_COMPONENTS,
        min_dist=0.0,
        metric="cosine",
        random_state=42,
    )

    hdbscan_model = HDBSCAN(
        min_cluster_size=HDBSCAN_MIN_CLUSTER,
        min_samples=5,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )
    vectorizer = CountVectorizer(
        stop_words="english",
        min_df=5,
        ngram_range=(1, 2),
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b",
    )
    representation_model = KeyBERTInspired()

    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer,
        representation_model=representation_model,
        top_n_words=10,
        min_topic_size=MIN_TOPIC_SIZE,
        nr_topics="auto",
        calculate_probabilities=True,
        verbose=True,
    )
    return topic_model


def build_topic_model_from_cfg(cfg: dict) -> BERTopic:
    """
    Different version of build_topic_model() that uses experiment config dict
    instead of constant vars in config.py
    This runs in the run_experiments.py script but not the run_pipeline.py script
    """
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)

    umap_model = UMAP(
        n_neighbors=cfg["n_neighbors"],
        n_components=cfg["n_components"],
        min_dist=0.0,
        metric="cosine",
        random_state=42,
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=cfg["min_cluster_size"],
        min_samples=cfg["min_samples"],
        metric="euclidean",
        cluster_selection_method=cfg["method"],
        prediction_data=True,
    )
    vectorizer = CountVectorizer(
        stop_words="english",
        min_df=5,
        ngram_range=(1, 2),
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b",
    )

    return BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer,
        representation_model=KeyBERTInspired(),
        top_n_words=10,
        min_topic_size=cfg["min_cluster_size"],
        nr_topics="auto",
        calculate_probabilities=True,
        verbose=False,
    )


def train_and_save(df: pd.DataFrame, embeddings: np.ndarray) -> pd.DataFrame:
    """
    Fit BERTopic, persist the model, return df with topic assignments.
    Passing pre-computed embeddings skips BERTopic's internal encoder.
    """
    topic_model = build_topic_model()
    topics, probs = topic_model.fit_transform(df["text"].tolist(), embeddings)

    topic_model.save(
        str(BERTOPIC_MODEL_PATH.resolve()),
        serialization="safetensors",
        save_ctfidf=True,
        # save_embedding_model=False,
    )

    topic_info = topic_model.get_topic_info()
    print(f"\nFound {len(topic_info) - 1} topics (excluding outliers)\n")
    print(topic_info[["Topic", "Count", "Name"]].head(20).to_string())

    df_out = df[["id", "webTitle", "webPublicationDate", "text", "clean_body"]].copy()
    df_out["topic_id"] = topics
    df_out["topic_prob"] = probs.max(axis=1) if probs.ndim > 1 else probs
    df_out["topic_label"] = df_out["topic_id"].map(
        topic_info.set_index("Topic")["Name"]
    )
    return df_out


def load_model() -> BERTopic:
    return BERTopic.load(str(BERTOPIC_MODEL_PATH.resolve()))


def predict_topics(texts: list[str], embeddings: np.ndarray, topic_model: BERTopic):
    """
    Infer topics for new documents without re-training.
    Uses HDBSCAN's approximate_predict so outliers are handled gracefully.
    """
    topics, probs = topic_model.transform(texts, embeddings)
    return topics, probs

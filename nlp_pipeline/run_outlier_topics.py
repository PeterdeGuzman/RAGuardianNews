import duckdb
import numpy as np
import pandas as pd
from pathlib import Path
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired

from config import DB_PATH, EMBEDDINGS_PATH, MODEL_DIR, EMBEDDING_MODEL

"""
This script runs a fresh BERTopic model fit on the 8,705 uncategorized articles from the v1 topic model.
First, I pull the uncategorized articles, then slice their cached embeddings by index (both tables sorted on web publication date desc)
Then, I run the fresh model on the subcorpus.
"""

# ── tuned for a smaller, more homogeneous corpus ───────────────────────────────
# lower min_cluster_size b/c subcorpus is 1/3 size of main corpus
# leaf method for more granular clusters
OUTLIER_CFG = {
    "min_cluster_size": 5,
    "min_samples": 2,
    "n_neighbors": 15,
    "n_components": 5,
    "method": "leaf",
}


def main():
    conn = duckdb.connect(DB_PATH)

    # get outlier articles with their original row position
    # row_number() gives us the 0-based index into the embeddings array,
    # which was built from the same ORDER BY as load_articles()
    # web publication date DESC
    df_all = conn.execute("""
        SELECT 
            a.id,
            a.webTitle,
            a.webPublicationDate,
            a.clean_body,
            ROW_NUMBER() OVER (ORDER BY a.webPublicationDate DESC) - 1 AS row_idx
        FROM cleaned_articles a
        WHERE a.clean_body IS NOT NULL
    """).df()

    df_outliers = conn.execute("""
        SELECT t.id
        FROM article_topics t
        WHERE t.topic_id = -1
    """).df()

    conn.close()

    # merge to get row indices outlier articles
    df = df_all[df_all["id"].isin(df_outliers["id"])].reset_index(drop=True)
    print(f"Outlier articles: {len(df):,}")

    # slice embeddings by row index
    print("Loading cached embeddings...")
    all_embeddings = np.load(EMBEDDINGS_PATH)
    outlier_embeddings = all_embeddings[df["row_idx"].values]
    print(f"Embedding slice shape: {outlier_embeddings.shape}")

    # build and fit a fresh BERTopic on the outlier articles subcorpus
    print("\nFitting BERTopic on outlier subcorpus...")
    df["text"] = df["webTitle"].fillna("") + " " + df["clean_body"].fillna("")

    umap_model = UMAP(
        n_neighbors=OUTLIER_CFG["n_neighbors"],
        n_components=OUTLIER_CFG["n_components"],
        min_dist=0.0,
        metric="cosine",
        random_state=42,
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=OUTLIER_CFG["min_cluster_size"],
        min_samples=OUTLIER_CFG["min_samples"],
        metric="euclidean",
        cluster_selection_method=OUTLIER_CFG["method"],
        prediction_data=True,
    )
    vectorizer = CountVectorizer(
        stop_words="english",
        min_df=3,  # lower b/c I am using it on the 1/3 size subcorpus
        ngram_range=(1, 2),
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b",
    )

    topic_model = BERTopic(
        embedding_model=SentenceTransformer(EMBEDDING_MODEL),
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer,
        representation_model=KeyBERTInspired(),
        top_n_words=10,
        min_topic_size=OUTLIER_CFG["min_cluster_size"],
        nr_topics="auto",
        calculate_probabilities=True,
        verbose=True,
    )

    topics, probs = topic_model.fit_transform(df["text"].tolist(), outlier_embeddings)

    # perform outlier reduction on the sub-corpus
    # this was also done on the main model ran on the full corpus
    print("\nReducing outliers within subcorpus...")
    topics_reduced = topic_model.reduce_outliers(
        df["text"].tolist(),
        topics,
        embeddings=outlier_embeddings,
        strategy="embeddings",
        threshold=0.5,
    )
    topic_model.update_topics(df["text"].tolist(), topics=topics_reduced)

    # ── save model ─────────────────────────────────────────────────────────────
    model_path = MODEL_DIR / "bertopic_outliers"
    topic_model.save(str(model_path), serialization="safetensors", save_ctfidf=True)
    print(f"\nModel saved → {model_path}")

    # ── build results dataframe ────────────────────────────────────────────────
    topic_info = topic_model.get_topic_info()

    df["outlier_topic_id"] = topics_reduced
    df["outlier_topic_prob"] = (
        probs.max(axis=1) if isinstance(probs, np.ndarray) and probs.ndim > 1 else probs
    )
    df["outlier_topic_label"] = df["outlier_topic_id"].map(
        topic_info.set_index("Topic")["Name"]
    )

    # ── write back to DuckDB ───────────────────────────────────────────────────
    conn = duckdb.connect(DB_PATH)
    conn.execute("CREATE OR REPLACE TABLE article_topics_outliers AS SELECT * FROM df")
    conn.close()
    print("Written → article_topics_outliers")

    # ── summary ────────────────────────────────────────────────────────────────
    total = len(df)
    still_out = (df["outlier_topic_id"] == -1).sum()
    recovered = total - still_out
    n_topics = df[df["outlier_topic_id"] != -1]["outlier_topic_id"].nunique()

    print(f"\n{'═' * 50}")
    print(f"  Subcorpus total:     {total:>6,}")
    print(f"  New topics found:    {n_topics:>6,}")
    print(f"  Recovered articles:  {recovered:>6,}  ({recovered / total * 100:.1f}%)")
    print(f"  Still uncategorised: {still_out:>6,}  ({still_out / total * 100:.1f}%)")
    print(f"{'═' * 50}")

    print(f"\nTop topics in outlier subcorpus:")
    print(
        topic_info[["Topic", "Count", "Name"]]
        .query("Topic != -1")
        .head(25)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()

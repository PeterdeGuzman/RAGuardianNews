import duckdb
import numpy as np
import pandas as pd
from db import load_articles
from embed import get_embeddings
from topic_model import build_topic_model_from_cfg
from config import DB_PATH, EXPERIMENTS, EXPERIMENTS_DIR


def run_experiment(
    version: str, cfg: dict, df: pd.DataFrame, embeddings: np.ndarray
) -> dict:
    print(f"\n{'═' * 55}")
    print(f"  {version}: {cfg}")
    print(f"{'═' * 55}")

    topic_model = build_topic_model_from_cfg(cfg)
    topics, probs = topic_model.fit_transform(df["text"].tolist(), embeddings)

    # post-hoc outlier reduction
    print("  Reducing outliers...")
    topics_reduced = topic_model.reduce_outliers(
        df["text"].tolist(),
        topics,
        embeddings=embeddings,
        strategy="embeddings",
        threshold=0.5,
    )
    topic_model.update_topics(df["text"].tolist(), topics=topics_reduced)

    # save model
    model_path = EXPERIMENTS_DIR / f"bertopic_{version}"
    topic_model.save(str(model_path), serialization="safetensors", save_ctfidf=True)
    print(f"  Saved → {model_path}")

    # build assignments dataframe
    topic_info = topic_model.get_topic_info()
    df_out = df[["id", "webTitle", "webPublicationDate"]].copy()
    df_out["topic_id"] = topics_reduced
    df_out["topic_prob"] = (
        probs.max(axis=1) if isinstance(probs, np.ndarray) and probs.ndim > 1 else probs
    )
    df_out["topic_label"] = df_out["topic_id"].map(
        topic_info.set_index("Topic")["Name"]
    )

    # write to DuckDB as a versioned table
    # don't overwrite article_topics ever
    conn = duckdb.connect(DB_PATH)
    conn.execute(f"DROP TABLE IF EXISTS article_topics_{version}")
    conn.execute(f"CREATE TABLE article_topics_{version} AS SELECT * FROM df_out")
    conn.close()
    print(f"  Written → article_topics_{version}")

    # summary stats
    total = len(df_out)
    outliers = (df_out["topic_id"] == -1).sum()
    n_topics = df_out[df_out["topic_id"] != -1]["topic_id"].nunique()
    outlier_pct = outliers / total * 100

    top5 = (
        df_out[df_out["topic_id"] != -1]
        .groupby(["topic_id", "topic_label"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .head(5)
    )

    print(f"\n  Topics:    {n_topics:,}")
    print(f"  Outliers:  {outliers:,}  ({outlier_pct:.1f}%)")
    print(f"  Coverage:  {total - outliers:,}  ({100 - outlier_pct:.1f}%)")
    print(f"  Top 5:")
    for _, row in top5.iterrows():
        print(
            f"    [{row['topic_id']:>3}] {row['topic_label']:<40} {row['count']:>5,}  ({row['count'] / total * 100:.1f}%)"
        )

    return {
        "version": version,
        "n_topics": n_topics,
        "total": total,
        "outliers": int(outliers),
        "outlier_pct": round(outlier_pct, 2),
        "coverage_pct": round(100 - outlier_pct, 2),
        **{f"param_{k}": v for k, v in cfg.items()},
    }


def print_comparison(summaries: list[dict]):
    df = pd.DataFrame(summaries).sort_values("outlier_pct")
    cols = [
        "version",
        "n_topics",
        "outliers",
        "outlier_pct",
        "coverage_pct",
        "param_min_cluster_size",
        "param_min_samples",
        "param_n_neighbors",
        "param_n_components",
        "param_method",
    ]
    print(f"\n\n{'═' * 75}")
    print("  RESULTS")
    print(f"{'═' * 75}")
    print(df[cols].to_string(index=False))
    best = df.iloc[0]
    print(
        f"\n  Best: {best['version']}  —  {best['coverage_pct']}% coverage,  {best['n_topics']} topics"
    )
    print(f"{'═' * 75}\n")
    return df


def main():
    print("=== Loading articles ===")
    df = load_articles()
    print(f"  {len(df):,} articles")

    print("\n=== Loading embeddings ===")
    embeddings = get_embeddings(df["text"].tolist())
    print(f"  Shape: {embeddings.shape}")

    summaries = []
    for version, cfg in EXPERIMENTS.items():
        summary = run_experiment(version, cfg, df, embeddings)
        summaries.append(summary)

    df_summary = print_comparison(summaries)

    # persist comparison table so you can query it later
    conn = duckdb.connect(DB_PATH)
    conn.execute(
        "CREATE OR REPLACE TABLE experiment_results AS SELECT * FROM df_summary"
    )
    conn.close()
    print("Comparison written → experiment_results")


if __name__ == "__main__":
    main()

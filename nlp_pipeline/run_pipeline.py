from db import load_articles, save_topics
from embed import get_embeddings
from topic_model import train_and_save


def main():
    print("=== Loading articles from DuckDB ===")
    df = load_articles()
    print(f"Loaded {len(df):,} articles")

    print("\n=== Generating embeddings ===")
    embeddings = get_embeddings(df["text"].tolist())

    print("\n=== Training BERTopic ===")
    df_topics = train_and_save(df, embeddings)

    print("\n=== Writing topic assignments back to DuckDB ===")
    save_topics(df_topics)
    print("Done.")


if __name__ == "__main__":
    main()

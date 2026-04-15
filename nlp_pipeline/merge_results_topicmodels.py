import duckdb
import pandas as pd
from config import DB_PATH

conn = duckdb.connect(DB_PATH)


# load tables
df_main = conn.execute(
    """
    SELECT * FROM article_topics
    """
).df()  # LOAD main topic model result table from v1 BERTopic model ran on full corpus

df_outliers = conn.execute(
    """
    SELECT * FROM article_topics_outliers
    """
).df()  # LOAD outliers topic model result table from the BERTopic model ran on the subcorpus of uncategorized articles from initial model run

# The outlier model used 0 as the uncategorized marker instead of -1
# So we will need to remap this before the offset step
df_outliers["outlier_topic_id"] = df_outliers["outlier_topic_id"].replace(0, -1)

# offset outlier topic IDs to avoid topic collision with main model
max_main_topic = df_main[df_main["topic_id"] != -1]["topic_id"].max()
print(f"Max topic ID from main model is: {max_main_topic}")

# offset all topic ids except for the uncategorized topic
df_outliers["outlier_topic_id_offset"] = df_outliers["outlier_topic_id"].apply(
    lambda tid: tid + max_main_topic + 1 if tid != -1 else -1
)

# build lookup table for article_id with the new topic assignments
df_main["id"] = df_main["id"].astype(str)
df_outliers["id"] = df_outliers["id"].astype(str)
outlier_lookup = df_outliers[df_outliers["outlier_topic_id"] != -1].set_index("id")

print(f"\nArticles recovered by outlier model: {len(outlier_lookup):,}")
print(
    f"Articles still -1 after outlier run: {(df_outliers['outlier_topic_id'] == -1).sum():,}"
)


# function to clean labels in the outlier topics tabel (they need to be updated to prevent topic ID collision)
def clean_label(label: str) -> str:
    # remove leading "X_" where X is a number
    return label.split("_", 1)[-1]


# merge this into the main article topics table
def resolve_topic(row):
    """
    For articles that were -1 in the main model,
    check if the outlier model found a topic for them.
    If yes, use that. If no, stay -1.
    """
    if row["topic_id"] != -1:
        # already assigned topic from main model
        # keep it as it is and do not change at this time
        return pd.Series(
            {
                "topic_id": row["topic_id"],
                "topic_prob": row["topic_prob"],
                "topic_label": row["topic_label"],
                "source": "main_model",
            }
        )

    if row["id"] in outlier_lookup.index:
        outlier = outlier_lookup.loc[row["id"]]
        new_id = outlier["outlier_topic_id_offset"]
        raw_label = outlier["outlier_topic_label"]
        cleaned = clean_label(raw_label)
        return pd.Series(
            {
                "topic_id": new_id,
                "topic_prob": outlier["outlier_topic_prob"],
                "topic_label": f"{new_id}_{cleaned}",
                "source": "outlier_model",
            }
        )

    return pd.Series(
        {
            "topic_id": -1,
            "topic_prob": row["topic_prob"],
            "topic_label": "-1_uncategorised",
            "source": "uncategorised",
        }
    )


print("\nMerging topic assignments...")
resolved = df_main.apply(resolve_topic, axis=1)
df_merged = pd.concat(
    [df_main[["id", "webTitle", "webPublicationDate", "clean_body"]], resolved], axis=1
)

# print summary
total = len(df_merged)
from_main = (df_merged["source"] == "main_model").sum()
from_outlier = (df_merged["source"] == "outlier_model").sum()
still_out = (df_merged["source"] == "uncategorised").sum()

print(f"\n{'═' * 50}")
print(f"  Total articles:          {total:>7,}")
print(f"  From main model:         {from_main:>7,}  ({from_main / total * 100:.1f}%)")
print(
    f"  Recovered from outliers: {from_outlier:>7,}  ({from_outlier / total * 100:.1f}%)"
)
print(f"  Still uncategorised:     {still_out:>7,}  ({still_out / total * 100:.1f}%)")
print(
    f"  Total topics:            {df_merged[df_merged['topic_id'] != -1]['topic_id'].nunique():>7,}"
)
print(f"{'═' * 50}")

# checking step
print(
    "Ensuring that the outlier table captured all uncategorized articles from main model:"
)

main_outlier_ids = set(df_main[df_main["topic_id"] == -1]["id"].astype(str))
outlier_table_ids = set(df_outliers["id"].astype(str))

overlap = main_outlier_ids & outlier_table_ids
only_in_main = main_outlier_ids - outlier_table_ids
only_in_outlier_tbl = outlier_table_ids - main_outlier_ids

print(f"Articles that were -1 in main model:        {len(main_outlier_ids):,}")
print(f"Articles in outlier table:                   {len(outlier_table_ids):,}")
print(f"Overlap (should be ~8,705):                  {len(overlap):,}")
print(f"In main -1 but missing from outlier table:   {len(only_in_main):,}")
print(f"In outlier table but not in main -1:         {len(only_in_outlier_tbl):,}")

# ── write back to DuckDB ───────────────────────────────────────────────────────
# don't overwrite article_topics
# write merged as new  table
conn.execute("CREATE OR REPLACE TABLE article_topics_merged AS SELECT * FROM df_merged")

# verify
check = conn.execute("""
    SELECT source, COUNT(*) as count
    FROM article_topics_merged
    GROUP BY source
    ORDER BY count DESC
""").df()
print(f"\nVerification:\n{check.to_string(index=False)}")

# check unique topic_id and topic_label
df_topics = conn.execute(
    """
    SELECT * 
    FROM article_topics_merged
    """
).df()

# checking unique topic ids and topic labels
n_topic_ids = df_topics["topic_id"].nunique()
print("Unique topic_id:", n_topic_ids)

# extract prefix from topic_label
df_topics["label_prefix"] = (
    df_topics["topic_label"].str.extract(r"^(-?\d+)_")[0].astype(int)
)

n_prefixes = df_topics["label_prefix"].nunique()
print("Unique label prefixes:", n_prefixes)

ids_set = set(df_topics["topic_id"].unique())
prefix_set = set(df_topics["label_prefix"].unique())

print("IDs match prefixes:", ids_set == prefix_set)

# checking topic id collision

print(
    "This dataframe will be empty if there is no topic_id mapping to multiple different label strings. Empty dataframe is GOOD."
)
df_topics.groupby("topic_id")["topic_label"].nunique()
bad = (
    df_topics.groupby("topic_id")["topic_label"]
    .nunique()
    .reset_index(name="n_labels")
    .query("n_labels > 1")
)

print(bad)
assert df_topics.groupby("topic_id")["topic_label"].nunique().max() == 1
conn.close()
print("\nWritten → article_topics_merged")

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent / "nlp_pipeline"))

import streamlit as st
import duckdb
from retrieval import retrieve
from rag import rag_query

DB_PATH = "guardian_articles.duckdb"

st.set_page_config(page_title="Guardian News Explorer", page_icon="📰", layout="wide")

st.title("📰 Guardian News Explorer")

tab1, tab2, tab3 = st.tabs(["🔍 Topic Search", "🏷️ NER Search", "🤖 RAG Q&A"])

# ── Tab 1: Topic Search ───────────────────────────────────────────────────────
with tab1:
    st.header("Search by Topic")

    con = duckdb.connect(DB_PATH, read_only=True)
    topics = (
        con.execute("""
        SELECT DISTINCT topic_label_clean 
        FROM article_topics_labelled 
        ORDER BY topic_label_clean
    """)
        .df()["topic_label_clean"]
        .tolist()
    )
    con.close()

    selected_topic = st.selectbox("Select a topic", topics)
    n_results = st.slider("Number of articles", 5, 50, 10)

    if st.button("Search", key="topic_search"):
        con = duckdb.connect(DB_PATH, read_only=True)
        df = con.execute(
            """
            SELECT webTitle, webPublicationDate, topic_label_clean, id
            FROM article_topics_labelled
            WHERE topic_label_clean = ?
            ORDER BY webPublicationDate DESC
            LIMIT ?
        """,
            [selected_topic, n_results],
        ).df()
        con.close()

        st.write(f"**{len(df)} articles** in topic: *{selected_topic}*")
        for _, row in df.iterrows():
            with st.expander(
                f"{row['webTitle']} — {str(row['webPublicationDate'])[:10]}"
            ):
                st.write(f"**Topic:** {row['topic_label_clean']}")
                st.write(f"**ID:** {row['id']}")

# ── Tab 2: NER Search ─────────────────────────────────────────────────────────
with tab2:
    st.header("Search by Named Entity")

    entity_query = st.text_input(
        "Search for a person, place, or organisation", placeholder="e.g. Boris Johnson"
    )
    ner_model = st.radio("NER Model", ["spaCy", "BERT", "Both"], horizontal=True)
    n_ner_results = st.slider("Number of articles", 5, 50, 10, key="ner_slider")

    if st.button("Search", key="ner_search") and entity_query:
        con = duckdb.connect(DB_PATH, read_only=True)

        table_map = {
            "spaCy": ["ner_articles_spacy"],
            "BERT": ["ner_articles_bert"],
            "Both": ["ner_articles_spacy", "ner_articles_bert"],
        }
        tables = table_map[ner_model]

        # Union across selected tables
        union_parts = " UNION ".join(
            [
                f"""
            SELECT DISTINCT e.id, t.webTitle, t.webPublicationDate, t.topic_label_clean
            FROM {table} e
            JOIN article_topics_labelled t USING (id)
            WHERE LOWER(e.entity_text) LIKE LOWER(?)
            AND e.entity_text NOT LIKE '##%'
            """
                for table in tables
            ]
        )

        query = f"""
            SELECT DISTINCT id, webTitle, webPublicationDate, topic_label_clean
            FROM ({union_parts})
            ORDER BY webPublicationDate DESC
            LIMIT ?
        """
        params = [f"%{entity_query}%"] * len(tables) + [n_ner_results]
        df = con.execute(query, params).df()

        # Also get related entities for the found articles
        if not df.empty:
            article_ids = df["id"].tolist()
            placeholders = ", ".join(["?" for _ in article_ids])
            related = con.execute(
                f"""
                SELECT entity_text, COUNT(*) as count
                FROM ner_articles_spacy
                WHERE id IN ({placeholders})
                AND entity_label = 'PERSON'
                AND entity_text NOT LIKE '##%'
                AND LENGTH(entity_text) > 2
                GROUP BY entity_text
                ORDER BY count DESC
                LIMIT 10
            """,
                article_ids,
            ).df()
        con.close()

        st.write(f"**{len(df)} articles** mentioning *{entity_query}*")

        col1, col2 = st.columns([3, 1])
        with col1:
            for _, row in df.iterrows():
                with st.expander(
                    f"{row['webTitle']} — {str(row['webPublicationDate'])[:10]}"
                ):
                    st.write(f"**Topic:** {row['topic_label_clean']}")

        with col2:
            if not df.empty:
                st.write("**Also mentioned:**")
                for _, row in related.iterrows():
                    st.write(f"- {row['entity_text']} ({row['count']})")

# ── Tab 3: RAG Q&A ────────────────────────────────────────────────────────────
with tab3:
    st.header("Ask a Question (RAG)")
    st.caption(
        "Retrieves relevant Guardian articles and generates answers using Gemini and Llama 3.2"
    )

    question = st.text_input(
        "Ask a question about Guardian news",
        placeholder="e.g. What has been the impact of Brexit on UK trade?",
    )
    top_k = st.slider("Number of articles to retrieve", 3, 10, 5)

    if st.button("Ask", key="rag_ask") and question:
        with st.spinner("Retrieving articles and generating answers..."):
            result = rag_query(question, top_k=top_k)

        st.subheader("📄 Retrieved Articles")
        for a in result["articles"]:
            with st.expander(
                f"[{a['score']:.3f}] {a['webTitle']} — {str(a['webPublicationDate'])[:10]}"
            ):
                st.write(f"**Topic:** {a['topic_label_clean']}")
                st.write(a["clean_bodytext"][:500] + "...")

        st.subheader("🤖 Model Responses")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Gemini 2.0 Flash")
            st.write(result["gemini"])

        with col2:
            st.markdown("### Llama 3.2 3B (Ollama)")
            st.write(result["ollama"])

# README for RAG Process

Process:




Related Scripts:
- `nlp_pipeline/faiss_index.py`
    - Builds and saves the FAISS index from the existing embeddings produced by the `embed.py` script
- `nlp_pipeline/RAG/retrieval.py`
    - Loads the FAISS index and retrieves top-k articles for a query
- `nlp_pipeline/RAG/rag.py` 
    - Constructs the dual generation pipeline, provides the same context to both models
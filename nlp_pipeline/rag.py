import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

import os
import requests
from dotenv import load_dotenv
from google import genai
from google.genai import types
from retrieval import retrieve

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

PROMPT_TEMPLATE = """You are a helpful assistant answering questions about Guardian newspaper articles.
    Use ONLY the follow article excerpts to answer the question.
    If the articles do not contain enjough information to answer, say so clearly.

    ARTICLES:
    {context}

    Question: {question}

    ANSWER:"""


# Function to build context from the retrieved articles
def _build_context(articles: list[dict], max_chars_per_article: int = 1000) -> str:
    chunks = []
    for i, a in enumerate(articles, 1):
        chunk = (
            f"[Article {i}] {a['webTitle']} ({a['webPublicationDate'].strftime('%Y-%m-%d')})\n"
            f"Topic: {a['topic_label_clean']}\n"
            f"{a['text'][:max_chars_per_article]}"
        )
        chunks.append(chunk)
    return "\n\n---\n\n".join(chunks)


# Gemini Prompt
# def generate_gemini(prompt: str) -> str:
#     response = client.models.generate_content(
#         model="gemini-2.0-flash",
#         contents=prompt,
#     )
#     return response.text


# Ollama Prompt
def generate_ollama(prompt: str, model: str = "llama3.2:3b") -> str:
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": model, "prompt": prompt, "stream": False},
        timeout=120,
    )
    response.raise_for_status()
    return response.json()["response"]


# helper function that queries both models and returns both generated responses
# retrieves top_k articles
# generates answers from both models
# returns a dict with retrieved articles and responses from both models
def rag_query(question: str, top_k: int = 5) -> str:
    articles = retrieve(question, top_k=top_k)
    context = _build_context(articles)
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)

    print("Querying Llama 3.2 3B...")
    llama_response = generate_ollama(prompt, model="llama3.2:3b")

    print("Querying Mistral 7B...")
    mistral_response = generate_ollama(prompt, model="mistral:7b")

    return {
        "question": question,
        "articles": articles,
        "prompt": prompt,
        "llama": llama_response,
        "mistral": mistral_response,
    }


if __name__ == "__main__":
    result = rag_query("Where has Elon Musk tried to build datacenters for AI and why?")
    print("\n --- Retrieved Articles: ---")
    for a in result["articles"]:
        print(f"   [{a['score']:.3f}] {a['webTitle']}")
        print("\n--- Ollama's Response: ---")
    print(result["llama"])
    print("\n--- Mistral's Response: ---")
    print(result["mistral"])

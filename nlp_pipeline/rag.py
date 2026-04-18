import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

import os
import requests
from dotenv import load_dotenv
import google.generativeai as genai
from retrieval import retrieve

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

PROMPT_TEMPLATE = """You are a helpful assistant answering questions about Guardian newspaper articles.
    Use ONLY the follow article excerpts to answer thq question.
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
            f"[Article {i}] {a['webTitle']} ({a['webPublicationDate'][:10]})\n"
            f"Topic: {a['topic_label_clean']}\n"
            f"{a['text'][:max_chars_per_article]}"
        )
        chunks.append(chunk)
    return "\n\n---\n\n".join(chunks)


# Gemini Prompt
def generate_gemini(prompt: str) -> str:
    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(prompt)
    return response.text


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

    print("Asking Gemini...")
    gemini_response = generate_gemini(prompt)

    print("Now Asking Ollama...")

    ollama_response = generate_ollama(prompt)

    return {
        "question": question,
        "articles": articles,
        "prompt": prompt,
        "gemini": gemini_response,
        "ollama": ollama_response,
    }


if __name__ == "_main__":
    result = rag_query("Where has Elon Musk tried to build datacenters for AI and why?")
    print("\n --- Retrieved Articles: ---")
    for a in result["articles"]:
        print(f"   [{a['score']:.3f}] {a['webTitle']}")
        print("\n--- Gemini's Response: ---")
    print(result["gemini"])
    print("\n--- Ollama's Response: ---")
    print(result["ollama"])

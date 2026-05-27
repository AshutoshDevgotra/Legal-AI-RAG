import os
import time
import requests
from pathlib import Path
from dotenv import load_dotenv
from pinecone import Pinecone
from groq import Groq, RateLimitError, APIStatusError

# Resolve AI-RAG project root
ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

# HuggingFace Inference API for embeddings (no local model needed)
HF_API_URL = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-mpnet-base-v2"
HF_TOKEN = os.getenv("HF_TOKEN", "")
HF_HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}


def get_embedding(text: str) -> list:
    """Get embedding vector via HuggingFace Inference API."""
    response = requests.post(
        HF_API_URL,
        headers=HF_HEADERS,
        json={"inputs": text, "options": {"wait_for_model": True}}
    )
    if response.status_code != 200:
        raise RuntimeError(f"HF Embedding API error: {response.status_code} - {response.text}")
    return response.json()


# Pinecone vector DB
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX"))

# Groq reasoning LLM
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
GROQ_MODEL = "llama-3.1-8b-instant"


def ask_legal_ai(question: str):
    qvec = get_embedding(question)

    results = index.query(
        vector=qvec,
        top_k=3,
        include_metadata=True
    )

    # Safe context building
    context_parts = []
    total_len = 0
    safe_matches = []

    for m in results["matches"]:
        txt = m["metadata"]["text"]
        sec = m["metadata"].get("section", "")
        score = float(m["score"])

        if total_len + len(txt) > 3500:
            break

        context_parts.append(f"[{sec}] {txt}")
        total_len += len(txt)

        # Build clean serializable source object
        safe_matches.append({
            "id": m["id"],
            "score": score,
            "section": sec,
            "text": txt[:500]
        })

    context = "\n".join(context_parts)

    prompt = f"""
You are a legal assistant. Answer strictly from the context below.
Mention proper section numbers.

CONTEXT:
{context}

QUESTION:
{question}
"""

    retries = 3
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": "You are a legal assistant. Answer strictly from the provided context. Mention proper section numbers."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=1024,
            )
            break
        except RateLimitError as e:
            if attempt < retries - 1:
                sleep_time = 2 ** attempt  # 1s, 2s, 4s...
                print(f"Rate limit hit. Retrying in {sleep_time}s...")
                time.sleep(sleep_time)
            else:
                raise e

    return {
        "answer": response.choices[0].message.content,
        "sources": safe_matches
    }

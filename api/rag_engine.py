import os
import time
import requests
from pathlib import Path
from dotenv import load_dotenv
from pinecone import Pinecone
from google import genai
from google.genai import errors as genai_errors

# Resolve AI-RAG project root
ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

# Pinecone vector DB
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX"))

# HuggingFace Inference API — all-mpnet-base-v2 (768-dim, same as ingestion model)
HF_API_KEY = os.getenv("HF_API_KEY")
HF_EMBED_URL = "https://api-inference.huggingface.co/models/sentence-transformers/all-mpnet-base-v2"
HF_HEADERS = {"Authorization": f"Bearer {HF_API_KEY}"}

# ── In-memory embedding cache ─────────────────────────────────────────────────
_embedding_cache: dict[str, list] = {}
CACHE_MAX_SIZE = 256

def get_embedding(text: str) -> list:
    """Get embedding via HuggingFace Inference API (all-mpnet-base-v2, 768-dim).

    No model download — runs entirely as an API call.
    Handles HF free-tier cold starts (503) with a single retry.
    """
    if text in _embedding_cache:
        return _embedding_cache[text]

    max_retries = 4
    base_delay = 3

    for attempt in range(max_retries):
        try:
            resp = requests.post(
                HF_EMBED_URL,
                headers=HF_HEADERS,
                json={"inputs": text},
                timeout=30,
            )

            # 503 = HF model cold-starting on free tier — wait and retry
            if resp.status_code == 503:
                wait = float(resp.json().get("estimated_time", base_delay * (2 ** attempt)))
                print(f"[HF] Model loading, waiting {wait:.0f}s… (attempt {attempt + 1}/{max_retries})")
                time.sleep(wait)
                continue

            resp.raise_for_status()
            embedding = resp.json()
            if isinstance(embedding[0], list):
                embedding = embedding[0]

            if len(_embedding_cache) >= CACHE_MAX_SIZE:
                del _embedding_cache[next(iter(_embedding_cache))]
            _embedding_cache[text] = embedding
            return embedding

        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                sleep_time = base_delay * (2 ** attempt)
                print(f"[HF] Request failed (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {sleep_time}s…")
                time.sleep(sleep_time)
            else:
                raise RuntimeError(f"HF embedding API failed after {max_retries} attempts: {e}") from e




# Gemini LLM — gemini-2.0-flash (fast, generous free tier)
gemini_api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
gemini_client = genai.Client(
    api_key=gemini_api_key,
    http_options={"api_version": "v1"}
)
GEMINI_MODEL = "gemini-2.0-flash"


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

    retries = 5
    for attempt in range(retries):
        try:
            response = gemini_client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config={
                    "system_instruction": "You are a legal assistant. Answer strictly from the provided context. Mention proper section numbers.",
                    "temperature": 0.3,
                    "max_output_tokens": 1024,
                }
            )
            break
        except genai_errors.ClientError as e:
            error_str = str(e).lower()
            is_rate_limit = any(kw in error_str for kw in ["429", "quota", "rate limit", "resource_exhausted"])
            if is_rate_limit and attempt < retries - 1:
                sleep_time = 2 ** attempt  # 1s, 2s, 4s, 8s…
                print(f"[Gemini] Rate limit hit. Retrying in {sleep_time}s…")
                time.sleep(sleep_time)
            else:
                raise e

    return {
        "answer": response.text,
        "sources": safe_matches
    }

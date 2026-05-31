import os
import time
from pathlib import Path
from dotenv import load_dotenv
from pinecone import Pinecone
from google import genai
from google.genai import types, errors as genai_errors

# Resolve AI-RAG project root
ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

# Pinecone vector DB
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX"))

# Gemini client — used for both embeddings and LLM generation
gemini_api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
gemini_client = genai.Client(
    api_key=gemini_api_key,
    http_options={"api_version": "v1"}
)

GEMINI_EMBED_MODEL = "gemini-embedding-001"
GEMINI_LLM_MODEL   = "gemini-3.5-flash"

# ── In-memory embedding cache ─────────────────────────────────────────────────
_embedding_cache: dict[str, list] = {}
CACHE_MAX_SIZE = 256

def get_embedding(text: str) -> list:
    """Get embedding via Gemini gemini-embedding-001 (768-dim).

    Uses output_dimensionality=768 to match the Pinecone index.
    Retries on rate-limit errors with exponential backoff.
    """
    if text in _embedding_cache:
        return _embedding_cache[text]

    max_retries = 5
    for attempt in range(max_retries):
        try:
            result = gemini_client.models.embed_content(
                model=GEMINI_EMBED_MODEL,
                contents=text,
                config=types.EmbedContentConfig(
                    task_type="RETRIEVAL_QUERY",
                    output_dimensionality=768,
                )
            )
            embedding = list(result.embeddings[0].values)

            if len(_embedding_cache) >= CACHE_MAX_SIZE:
                del _embedding_cache[next(iter(_embedding_cache))]
            _embedding_cache[text] = embedding
            return embedding

        except genai_errors.ClientError as e:
            error_str = str(e).lower()
            is_rate_limit = any(k in error_str for k in ["429", "quota", "rate limit", "resource_exhausted"])
            if is_rate_limit and attempt < max_retries - 1:
                sleep_time = 2 ** attempt
                print(f"[Gemini Embed] Rate limit, retrying in {sleep_time}s…")
                time.sleep(sleep_time)
            else:
                raise






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
System: You are a legal assistant. Answer strictly from the context below. Mention proper section numbers.

CONTEXT:
{context}

QUESTION:
{question}
"""

    retries = 5
    for attempt in range(retries):
        try:
            response = gemini_client.models.generate_content(
                model=GEMINI_LLM_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.3,
                    max_output_tokens=1024,
                )
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

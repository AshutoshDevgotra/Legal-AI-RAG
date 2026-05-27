import os
import time
from pathlib import Path
from dotenv import load_dotenv
from pinecone import Pinecone
from groq import Groq, RateLimitError
import google.generativeai as genai

# Resolve AI-RAG project root
ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

# Pinecone vector DB
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX"))

# Gemini for Embeddings
# We use Gemini embeddings here because the Hugging Face Inference API was failing/timing out on Render,
# and we found an existing Pinecone index (nyayadwaar-gemini) built with Gemini embeddings!
gemini_api_key = os.getenv("GEMINI_API_KEY")
if gemini_api_key:
    genai.configure(api_key=gemini_api_key)
else:
    print("Warning: GEMINI_API_KEY not set!")

def get_embedding(text: str) -> list:
    """Get embedding vector using Google Gemini."""
    result = genai.embed_content(
        model="models/embedding-001",
        content=text,
        task_type="retrieval_document"
    )
    return result['embedding']

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

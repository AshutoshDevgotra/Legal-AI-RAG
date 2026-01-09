import os
from pathlib import Path
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

# Resolve AI-RAG project root
ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

# Local embedding engine
embedder = SentenceTransformer("all-mpnet-base-v2")

# Pinecone vector DB
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX"))

# Gemini reasoning LLM
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
llm = genai.GenerativeModel("models/gemini-2.0-flash")


def ask_legal_ai(question: str):
    qvec = embedder.encode(question).tolist()

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

    response = llm.generate_content(prompt)

    return {
        "answer": response.text,
        "sources": safe_matches
    }

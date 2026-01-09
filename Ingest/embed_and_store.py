import os, json
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "../.env"))

model = SentenceTransformer("all-mpnet-base-v2")

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX"))


with open("../data/legal_chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

BATCH = 100
buffer = []

print("Total chunks:", len(chunks))

for i, ch in enumerate(tqdm(chunks)):
    vec = model.encode(ch["text"]).tolist()

    buffer.append((
        f"chunk-{i}",
        vec,
        ch
    ))

    if len(buffer) >= BATCH:
        index.upsert(buffer)
        buffer = []

if buffer:
    index.upsert(buffer)

print("Embedding completed.")

import json
from semantic_chunker import semantic_chunks

print("Starting ingestion...")

with open("../data/legal_raw.json", "r", encoding="utf-8") as f:
    pages = json.load(f)

print("Pages loaded:", len(pages))

chunks = []

for i, p in enumerate(pages):
    if not p["text"]:
        continue

    print("Processing page", i+1)
    new_chunks = semantic_chunks(p["text"])
    print("→", len(new_chunks), "chunks")
    chunks.extend(new_chunks)

with open("../data/legal_chunks.json", "w", encoding="utf-8") as f:
    json.dump(chunks, f, indent=2, ensure_ascii=False)

print("DONE. Total chunks:", len(chunks))

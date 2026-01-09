import pdfplumber
import pandas as pd
import json
import os
from tqdm import tqdm

def pdfs_to_json(folder_path, output_json):
    data = []

    for filename in tqdm(os.listdir(folder_path)):
        if not filename.lower().endswith(".pdf"):
            continue

        pdf_path = os.path.join(folder_path, filename)

        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                tables = page.extract_tables()

                page_data = {
                    "file": filename,
                    "page": page.page_number,
                    "text": text,
                    "tables": []
                }

                for table in tables:
                    if len(table) > 1:
                        df = pd.DataFrame(table[1:], columns=table[0])
                        page_data["tables"].append(df.to_dict())

                data.append(page_data)

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print("JSON built from", len(data), "pages")

# Run
pdfs_to_json("pdfs", "legal_raw.json")

import os
from dotenv import load_dotenv
from pathlib import Path
import google.generativeai as genai

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

model = genai.GenerativeModel("models/gemini-2.0-flash")

resp = model.generate_content("Say hello, what is IPC 302?")
print(resp.text)

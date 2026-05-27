from fastapi import FastAPI
from api.rag_engine import ask_legal_ai
from fastapi.responses import JSONResponse
import traceback

from fastapi.middleware.cors import CORSMiddleware
from groq import RateLimitError

app = FastAPI(title="NyayaSetu AI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"status": "NyayaSetu running"}

@app.post("/ask")
def ask(data: dict):
    try:
        return ask_legal_ai(data["question"])
    except RateLimitError as e:
        return JSONResponse(
            status_code=429,
            content={"error": "Rate Limit Exceeded. Please try again later.", "details": str(e)}
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "trace": traceback.format_exc()}
        )

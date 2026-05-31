from fastapi import FastAPI
from api.rag_engine import ask_legal_ai
from fastapi.responses import JSONResponse
import traceback

from fastapi.middleware.cors import CORSMiddleware

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
    except Exception as e:
        err_str = str(e).lower()
        status = 429 if any(kw in err_str for kw in ["429", "quota", "rate limit", "resource_exhausted", "rate_limit_exceeded"]) else 500
        return JSONResponse(
            status_code=status,
            content={"error": str(e), "trace": traceback.format_exc()}
        )

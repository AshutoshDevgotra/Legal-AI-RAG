from fastapi import FastAPI
from api.rag_engine import ask_legal_ai
from fastapi.responses import JSONResponse
import traceback

app = FastAPI(title="NyayaSetu AI")

@app.get("/")
def root():
    return {"status": "NyayaSetu running"}

@app.post("/ask")
def ask(data: dict):
    try:
        return ask_legal_ai(data["question"])
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "trace": traceback.format_exc()}
        )

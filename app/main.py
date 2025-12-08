
# app/main.py
from fastapi import FastAPI
from pydantic import BaseModel
from .agent import answer_sync

app = FastAPI(title="Agente Gemini + LangChain")

class Query(BaseModel):
    input: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/agent/invoke")
def invoke(q: Query):
    output = answer_sync(q.input)
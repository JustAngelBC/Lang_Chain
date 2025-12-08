
# app/main.py
from fastapi import FastAPI, HTTPException
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
    try:
        output = answer_sync(q.input)
        return {"output": output}
    except Exception as e:
        # Verás el error en Swagger en vez de 200 null
        raise HTTPException(status_code=500, detail=str(e))

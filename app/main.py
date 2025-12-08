
# app/main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from .agent import answer_sync

app = FastAPI(title="Agente Gemini + LangChain")

# Configuración CORS
origins = [
    "https://ui.ponganos10.online",  # tu frontend
    # Si quieres permitir más orígenes, agrégalos aquí
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,          # Lista de orígenes permitidos
    allow_credentials=True,         # Permitir cookies/autenticación
    allow_methods=["*"],            # Permitir todos los métodos (GET, POST, OPTIONS)
    allow_headers=["*"],            # Permitir todos los encabezados
)

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
        raise HTTPException(status_code=500, detail=str(e))
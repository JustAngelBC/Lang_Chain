
# app/main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .agent import answer_sync  # versión con session_id
from .google_oauth import router as oauth_router
from .google_actions import router as actions_router

app = FastAPI(title="Agente Gemini + LangChain")
app.state.google_creds = None  # guardaremos las credenciales aquí

# --- CORS: registrar ANTES de los routers ---
ALLOWED_ORIGINS = [
    "https://ui.ponganos10.online",
    # agrega orígenes de prueba si lo necesitas (localhost, etc.):
    # "http://localhost:5173",
    # "http://localhost:3000",
    # "http://127.0.0.1:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_origin_regex=r"^https://ui\.ponganos10\.online$",
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],   # opcional: si lees headers custom en el front
    max_age=600,            # cache del preflight en el navegador
)

# --- Schemas ---
class Query(BaseModel):
    session_id: str
    input: str

# --- Endpoints ---
@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/agent/invoke")
def invoke(q: Query):
    try:
        output = answer_sync(q.session_id, q.input)
        return {"output": output}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- Routers externos (OAuth + actions) ---
app.include_router(oauth_router)
app.include_router(actions_router)

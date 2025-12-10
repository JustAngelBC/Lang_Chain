
# app/main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from .agent import answer_sync  # usa la versión con session_id
from .google_oauth import router as oauth_router
from .google_actions import router as actions_router

app = FastAPI(title="Agente Gemini + LangChain")
app.state.google_creds = None  # guardaremos las credenciales aquí

origins = ["https://ui.ponganos10.online"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Query(BaseModel):
    session_id: str   # <--- NUEVO
    input: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/agent/invoke")
def invoke(q: Query):
    try:
        output = answer_sync(q.session_id, q.input)  # <--- pasa session_id
        return {"output": output}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# OAuth + acciones
app.include_router(oauth_router)
app.include_router(actions_router)

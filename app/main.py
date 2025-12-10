
# app/main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from .agent import answer_sync
from .google_oauth import router as oauth_router
from .google_actions import router as actions_router
from fastapi import UploadFile, File
from .pdf_ingest import save_pdf_and_text
from .agent import rebuild_index  # lo agregamos en el paso 3.2


app = FastAPI(title="Agente Gemini + LangChain")
app.state.google_creds = None  # guardaremos las credenciales aquí

origins = [ "https://ui.ponganos10.online" ]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Query(BaseModel):
    input: str

@app.get("/health")
def health():
    return {"status": "ok"}

# Tu agente de texto (RAG/LLM) sigue igual
@app.post("/agent/invoke")
def invoke(q: Query):
    try:
        output = answer_sync(q.input)
        return {"output": output}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# OAuth + acciones
app.include_router(oauth_router)
app.include_router(actions_router)


@app.post("/upload/pdf")
async def upload_pdf(file: UploadFile = File(...)):
    content = await file.read()
    info = save_pdf_and_text(content, file.filename)
    # Reindexar tras ingesta
    rebuild_index()
    return {"ok": True, "file": file.filename, "pages": info["pages"]}


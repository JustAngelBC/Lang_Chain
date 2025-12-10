
# app/main.py
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from .agent import answer_sync  # usa la versión con session_id
from .google_oauth import router as oauth_router
from .google_actions import router as actions_router
from .pdf_ingest import save_pdf_and_text
from .agent_tools import pdf_storage  # Almacén compartido para PDFs

app = FastAPI(title="Agente Gemini + LangChain")
app.state.google_creds = None  # guardaremos las credenciales aquí
app.state.pdf_content = None   # contenido del PDF actual en memoria

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

# --- PDF Upload ---
@app.post("/pdf/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """Sube un PDF y extrae su texto para que el agente pueda consultarlo."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Solo se permiten archivos PDF")
    
    try:
        contents = await file.read()
        result = save_pdf_and_text(contents, file.filename)
        
        # Leer el texto extraído y guardarlo en memoria
        with open(result["txt_path"], "r", encoding="utf-8") as f:
            pdf_data = {
                "filename": file.filename,
                "text": f.read(),
                "pages": result["pages"],
            }
            app.state.pdf_content = pdf_data
            # También actualizar el almacén compartido para las tools
            pdf_storage["content"] = pdf_data
        
        return {
            "message": f"PDF '{file.filename}' procesado correctamente",
            "pages": result["pages"],
            "bytes": result["bytes"],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al procesar PDF: {str(e)}")


@app.get("/pdf/status")
def pdf_status():
    """Verifica si hay un PDF cargado."""
    if app.state.pdf_content:
        return {
            "loaded": True,
            "filename": app.state.pdf_content["filename"],
            "pages": app.state.pdf_content["pages"],
        }
    return {"loaded": False}


@app.get("/pdf/content")
def pdf_content():
    """Retorna el contenido del PDF para que el agente lo consulte."""
    if not app.state.pdf_content:
        raise HTTPException(status_code=404, detail="No hay PDF cargado")
    return app.state.pdf_content


# OAuth + acciones
app.include_router(oauth_router)
app.include_router(actions_router)

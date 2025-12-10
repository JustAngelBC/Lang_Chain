
# app/pdf_ingest.py
import os
from typing import Dict
from PyPDF2 import PdfReader

DATA_DIR = os.getenv("DATA_DIR", "data")

def save_pdf_and_text(file_bytes: bytes, filename: str) -> Dict:
    # 1) Guardar PDF
    pdf_path = os.path.join(DATA_DIR, filename)
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(pdf_path, "wb") as f:
        f.write(file_bytes)

    # 2) Extraer texto
    reader = PdfReader(pdf_path)
    pages = len(reader.pages)
    text_parts = []
    for i in range(pages):
        page = reader.pages[i]
        text = page.extract_text() or ""
        text_parts.append(f"[Página {i+1}]\n{text}")

    full_text = "\n\n".join(text_parts).strip()

    # 3) Guardar como .txt para el índice
    txt_name = os.path.splitext(filename)[0] + ".txt"
    txt_path = os.path.join(DATA_DIR, txt_name)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(full_text or "[PDF sin texto extraíble]")

    return {"pdf_path": pdf_path, "txt_path": txt_path, "pages": pages, "bytes": len(file_bytes)}
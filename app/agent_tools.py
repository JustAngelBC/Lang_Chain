# app/agent_tools.py
"""
Tools que Gemini puede invocar automáticamente.
Usan httpx para llamar a los endpoints de google_actions.
"""
import os
import httpx
from pydantic import BaseModel, Field, EmailStr
from langchain_core.tools import StructuredTool
from typing import Optional, List

# URL base del backend (se llama a sí mismo)
API_BASE = os.getenv("API_BASE", "http://localhost:8000")


# ---------- Schemas para validación ----------
class GmailSendArgs(BaseModel):
    to: EmailStr = Field(..., description="Correo del destinatario")
    subject: str = Field(..., description="Asunto del correo")
    body: str = Field(..., description="Contenido del correo en texto plano")
    from_email: Optional[str] = Field(None, description="Alias remitente (opcional)")


class CalendarEventArgs(BaseModel):
    summary: str = Field(..., description="Título del evento")
    start_datetime: str = Field(..., description="Inicio en formato RFC3339, ej: 2025-12-10T10:00:00-07:00")
    end_datetime: str = Field(..., description="Fin en formato RFC3339, ej: 2025-12-10T11:00:00-07:00")
    description: Optional[str] = Field(None, description="Descripción del evento (opcional)")
    location: Optional[str] = Field(None, description="Ubicación del evento (opcional)")
    timezone: Optional[str] = Field("America/Mazatlan", description="Zona horaria, ej: America/Mazatlan")
    attendees: Optional[List[str]] = Field(None, description="Lista de correos de asistentes (opcional)")


# ---------- Implementaciones ----------
def gmail_send_impl(
    to: str,
    subject: str,
    body: str,
    from_email: Optional[str] = None
) -> str:
    """Envía un correo via el endpoint /gmail/send"""
    payload = {"to": to, "subject": subject, "body": body}
    if from_email:
        payload["from_email"] = from_email
    
    try:
        with httpx.Client(timeout=30) as client:
            res = client.post(f"{API_BASE}/gmail/send", json=payload)
            if res.status_code == 401:
                return "❌ Error: No hay credenciales de Google. El usuario debe conectarse primero en /auth/google"
            if res.status_code >= 400:
                return f"❌ Error al enviar correo: {res.text}"
            data = res.json()
            return f"✅ Correo enviado exitosamente a {to}. ID: {data.get('messageId', 'N/A')}"
    except Exception as e:
        return f"❌ Error de conexión: {str(e)}"


def calendar_event_impl(
    summary: str,
    start_datetime: str,
    end_datetime: str,
    description: Optional[str] = None,
    location: Optional[str] = None,
    timezone: Optional[str] = "America/Mazatlan",
    attendees: Optional[List[str]] = None
) -> str:
    """Crea un evento via el endpoint /calendar/event"""
    payload = {
        "summary": summary,
        "start_datetime": start_datetime,
        "end_datetime": end_datetime,
        "timezone": timezone or "America/Mazatlan",
    }
    if description:
        payload["description"] = description
    if location:
        payload["location"] = location
    if attendees:
        payload["attendees"] = attendees
    
    try:
        with httpx.Client(timeout=30) as client:
            res = client.post(f"{API_BASE}/calendar/event", json=payload)
            if res.status_code == 401:
                return "❌ Error: No hay credenciales de Google. El usuario debe conectarse primero en /auth/google"
            if res.status_code >= 400:
                return f"❌ Error al crear evento: {res.text}"
            data = res.json()
            link = data.get("htmlLink", "")
            return f"✅ Evento '{summary}' creado exitosamente. Link: {link}"
    except Exception as e:
        return f"❌ Error de conexión: {str(e)}"


# ---------- Tools para LangChain ----------
gmail_send_tool = StructuredTool.from_function(
    name="gmail_send",
    description=(
        "Envía un correo electrónico usando Gmail. "
        "Usa esta herramienta cuando el usuario pida enviar un email/correo. "
        "Requiere: to (email destino), subject (asunto), body (contenido). "
        "Opcional: from_email (remitente alias)."
    ),
    func=gmail_send_impl,
    args_schema=GmailSendArgs,
)

calendar_create_tool = StructuredTool.from_function(
    name="calendar_create_event",
    description=(
        "Crea un evento en Google Calendar. "
        "Usa esta herramienta cuando el usuario pida crear una cita, reunión o evento. "
        "Requiere: summary (título), start_datetime y end_datetime (formato RFC3339 como 2025-12-10T10:00:00-07:00). "
        "Opcional: description, location, timezone, attendees (lista de emails)."
    ),
    func=calendar_event_impl,
    args_schema=CalendarEventArgs,
)

# Lista de tools disponibles para el agente
TOOLS = [gmail_send_tool, calendar_create_tool]

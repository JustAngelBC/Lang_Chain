
# app/agent_tools.py
import os
import httpx
from pydantic import BaseModel, Field, EmailStr
from langchain.tools import StructuredTool

# Tu backend público; ajústalo si cambia.
API_BASE = os.getenv("API_BASE", "https://asistente.ponganos10.online")


# ---------- Schemas ----------
class GmailMessage(BaseModel):
    to: EmailStr = Field(..., description="Correo del destinatario")
    subject: str = Field(..., description="Asunto del correo")
    body: str = Field(..., description="Contenido del correo en texto plano")
    from_email: EmailStr | None = Field(None, description="Alias remitente (opcional)")

class CalendarEvent(BaseModel):
    summary: str = Field(..., description="Título del evento")
    description: str | None = Field(None, description="Descripción (opcional)")
    location: str | None = Field(None, description="Ubicación (opcional)")
    start_datetime: str = Field(..., description="Inicio RFC3339, ej. 2025-12-10T10:00:00-07:00")
    end_datetime: str = Field(..., description="Fin RFC3339")
    timezone: str | None = Field(None, description="Zona horaria, ej. America/Mazatlan")
    attendees: list[EmailStr] | None = Field(None, description="Lista de asistentes (opcional)")


# ---------- Implementaciones ----------
async def gmail_send_impl(to: str, subject: str, body: str, from_email: str | None = None) -> str:
    payload = {"to": to, "subject": subject, "body": body, "from_email": from_email}
    async with httpx.AsyncClient(timeout=30) as client:
        res = await client.post(f"{API_BASE}/gmail/send", json=payload)
        if res.status_code >= 400:
            raise RuntimeError(f"Gmail error {res.status_code}: {res.text}")
        return "✅ Correo enviado correctamente."

async def calendar_event_impl(
    summary: str,
    description: str | None,
    location: str | None,
    start_datetime: str,
    end_datetime: str,
    timezone: str | None,
    attendees: list[str] | None,
) -> str:
    payload = {
        "summary": summary,
        "description": description,
        "location": location,
        "start_datetime": start_datetime,
        "end_datetime": end_datetime,
        "timezone": timezone or os.getenv("TZ", "America/Mazatlan"),
        "attendees": attendees,
    }
    async with httpx.AsyncClient(timeout=30) as client:
        res = await client.post(f"{API_BASE}/calendar/event", json=payload)
        if res.status_code >= 400:
            raise RuntimeError(f"Calendar error {res.status_code}: {res.text}")
        data = res.json() if res.headers.get("content-type", "").startswith("application/json") else {}
        # Si tu endpoint devuelve htmlLink o id, lo mostramos
        link = data.get("htmlLink") or data.get("id")
        suffix = f" • {link}" if link else ""
        return f"✅ Evento creado: {summary} ({start_datetime} → {end_datetime}){suffix}"


# ---------- Tools LangChain ----------
gmail_send_tool = StructuredTool.from_function(
    name="gmail_send",
    description=(
        "Envía un correo Gmail. Usa esta herramienta cuando el usuario pida enviar un email. "
        "Campos: to (email), subject (texto), body (texto); from_email opcional."
    ),
    args_schema=GmailMessage,
    func=lambda **kwargs: gmail_send_impl(**kwargs),
    coroutine=lambda **kwargs: gmail_send_impl(**kwargs),
)

calendar_create_tool = StructuredTool.from_function(
    name="calendar_create_event",
    description=(
        "Crea un evento en Google Calendar. Requiere summary, start_datetime y end_datetime en RFC3339. "
        "timezone, description, location y attendees son opcionales."
    ),
    args_schema=CalendarEvent,
    func=lambda **kwargs: calendar_event_impl(**kwargs),
    coroutine=lambda **kwargs: calendar_event_impl(**kwargs),
)

TOOLS = [gmail_send_tool, calendar_create_tool]

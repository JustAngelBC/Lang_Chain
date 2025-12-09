
# app/google_actions.py
import base64
import os
from typing import List, Optional
from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel, EmailStr
from googleapiclient.discovery import build

router = APIRouter()

# -------- Gmail --------
class GmailMessage(BaseModel):
    to: EmailStr
    subject: str
    body: str
    from_email: Optional[EmailStr] = None  # opcional: alias remitente

@router.post("/gmail/send")
def gmail_send(msg: GmailMessage, request: Request):
    creds = getattr(request.app.state, "google_creds", None)
    if not creds or not creds.valid:
        raise HTTPException(status_code=401, detail="No hay credenciales Google válidas. Conecta en /auth/google.")

    service = build("gmail", "v1", credentials=creds)
    # Construye MIME simple
    raw_msg = f"From: {msg.from_email or 'me'}\nTo: {msg.to}\nSubject: {msg.subject}\n\n{msg.body}"
    encoded = base64.urlsafe_b64encode(raw_msg.encode("utf-8")).decode("utf-8")
    create_body = {"raw": encoded}

    # users.messages.send (userId='me' indica el usuario autenticado) [8](https://developers.google.com/workspace/gmail/api/reference/rest/v1/users.messages/send)
    sent = service.users().messages().send(userId="me", body=create_body).execute()
    return {"messageId": sent.get("id")}


# -------- Calendar --------
class CalendarEvent(BaseModel):
    summary: str
    description: Optional[str] = None
    location: Optional[str] = None
    start_datetime: str  # RFC3339, ej. "2025-12-10T10:00:00-07:00"
    end_datetime: str    # RFC3339
    timezone: Optional[str] = os.getenv("TZ", "America/Mazatlan")
    attendees: Optional[List[EmailStr]] = None

@router.post("/calendar/event")
def calendar_event(ev: CalendarEvent, request: Request):
    creds = getattr(request.app.state, "google_creds", None)
    if not creds or not creds.valid:
        raise HTTPException(status_code=401, detail="No hay credenciales Google válidas. Conecta en /auth/google.")

    service = build("calendar", "v3", credentials=creds)
    body = {
        "summary": ev.summary,
        "description": ev.description,
        "location": ev.location,
        "start": {"dateTime": ev.start_datetime, "timeZone": ev.timezone},
        "end": {"dateTime": ev.end_datetime, "timeZone": ev.timezone},
        "attendees": [{"email": a} for a in (ev.attendees or [])],
    }
    # events.insert sobre el calendarId 'primary' requiere el scope de Calendar. [2](https://developers.google.com/workspace/calendar/api/guides/create-events)
    created = service.events().insert(calendarId="primary", body=body).execute()
    return {"eventId": created.get("id"), "htmlLink": created.get("htmlLink")}


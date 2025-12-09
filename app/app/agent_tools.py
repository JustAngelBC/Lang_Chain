
# app/agent_tools.py
from langchain.tools import Tool
from fastapi import Request
from .google_actions import gmail_send, calendar_event, GmailMessage, CalendarEvent

def tools_factory(request: Request):
    # enlazar funciones al estado (credenciales)
    def send_mail_nl(to: str, subject: str, body: str, from_email: str = None):
        payload = GmailMessage(to=to, subject=subject, body=body, from_email=from_email)
        return gmail_send(payload, request)

    def create_event_nl(summary: str, start: str, end: str, tz: str = None, description: str = None, location: str = None):
        payload = CalendarEvent(summary=summary, start_datetime=start, end_datetime=end, timezone=tz or "America/Mazatlan",
                                description=description, location=location)
        return calendar_event(payload, request)

    return [
        Tool(name="send_gmail", func=send_mail_nl, description="Enviar un correo Gmail dado: to, subject, body"),
        Tool(name="create_calendar_event", func=create_event_nl, description="Crear evento de Calendar: summary, start RFC3339, end RFC3339"),
    ]

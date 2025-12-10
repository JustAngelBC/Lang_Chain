
# app/google_oauth.py
import os
from fastapi import APIRouter, Request
from fastapi.responses import RedirectResponse
from google_auth_oauthlib.flow import Flow
from google.oauth2.credentials import Credentials

router = APIRouter()

CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
REDIRECT_URI = os.getenv("GOOGLE_OAUTH_REDIRECT_URI")

# Scopes: envío Gmail + gestión de Calendar
SCOPES = [
    "https://www.googleapis.com/auth/gmail.send",   # enviar correo
    "https://www.googleapis.com/auth/calendar",     # crear eventos
]

def _client_config():
    # Formato esperado por google_auth_oauthlib
    return {
        "web": {
            "client_id": CLIENT_ID,
            "project_id": "langchain-assistant",
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "client_secret": CLIENT_SECRET,
            "redirect_uris": [REDIRECT_URI],
            "javascript_origins": [],  # puedes agregar tu frontend
        }
    }

@router.get("/auth/google")
def auth_google():
    flow = Flow.from_client_config(_client_config(), scopes=SCOPES, redirect_uri=REDIRECT_URI)
    auth_url, state = flow.authorization_url(
        access_type="offline",
        include_granted_scopes="true",
        prompt="consent",
    )
    # En una app real, guarda 'state' en sesión/DB
    return RedirectResponse(auth_url)

@router.get("/oauth2/callback")
def oauth2_callback(request: Request):
    # Intercambia el código por tokens
    flow = Flow.from_client_config(_client_config(), scopes=SCOPES, redirect_uri=REDIRECT_URI)
    flow.fetch_token(authorization_response=str(request.url))
    creds = flow.credentials

    # Guarda tokens en memoria (demo) o DB; aquí como atributo global de app
    request.app.state.google_creds = creds
    return RedirectResponse(url="https://ui.ponganos10.online/")

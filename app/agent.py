# app/agent.py
"""
Agente con Gemini + Tools (Gmail, Calendar) + Memoria por sesión.
Usa langgraph para el loop de tool-calling.
"""
import os
from typing import Dict, List

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

from .agent_tools import TOOLS  # Tools de Gmail y Calendar


# ---------- System prompt para el agente ----------
SYSTEM_PROMPT = """Eres un asistente personal inteligente con acceso a Gmail y Google Calendar.

CAPACIDADES:
- Puedes enviar correos electrónicos usando la herramienta 'gmail_send'
- Puedes crear eventos en el calendario usando 'calendar_create_event'

INSTRUCCIONES:
- Si el usuario pide enviar un correo, usa gmail_send con to, subject y body
- Si el usuario pide crear un evento/cita/reunión, usa calendar_create_event
- Para fechas, usa formato RFC3339 (ej: 2025-12-10T10:00:00-07:00)
- Si faltan datos necesarios, pregunta al usuario antes de ejecutar
- Sé amable, conciso y confirma cuando completes una acción
- Si hay error de credenciales, indica que deben conectarse en /auth/google"""


# ---------- Estado del módulo ----------
_agent = None
_memory = MemorySaver()  # Memoria persistente por thread_id


def _get_agent():
    """Lazy init del agente."""
    global _agent
    if _agent is None:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("Falta GOOGLE_API_KEY")
        model_name = os.getenv("MODEL_NAME", "gemini-2.0-flash")

        llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=0.7,
            max_retries=2,
        )

        _agent = create_react_agent(
            model=llm,
            tools=TOOLS,
            prompt=SYSTEM_PROMPT,
            checkpointer=_memory,  # Habilita memoria por sesión
        )
    return _agent


def _extract_response(result: dict) -> str:
    """Extrae el texto de respuesta del resultado del agente."""
    messages = result.get("messages", [])
    # Buscar el último mensaje AI que no sea tool call
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content:
            # Ignorar mensajes que solo tienen tool_calls
            if not msg.tool_calls or msg.content.strip():
                return msg.content.strip()
    return "[Sin contenido del modelo]"


def answer_sync(session_id: str, user_input: str) -> str:
    """
    Invocación síncrona con memoria por sesión.
    El agente puede usar tools automáticamente.
    """
    agent = _get_agent()
    result = agent.invoke(
        {"messages": [HumanMessage(content=user_input)]},
        config={"configurable": {"thread_id": session_id}},
    )
    return _extract_response(result)


async def answer_async(session_id: str, user_input: str) -> str:
    """
    Invocación asíncrona con memoria por sesión.
    """
    agent = _get_agent()
    result = await agent.ainvoke(
        {"messages": [HumanMessage(content=user_input)]},
        config={"configurable": {"thread_id": session_id}},
    )
    return _extract_response(result)
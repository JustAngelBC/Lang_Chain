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

from .agent_tools import TOOLS, pdf_storage  # Tools de Gmail y Calendar + PDF storage


# ---------- System prompt para el agente ----------
def get_system_prompt() -> str:
    """Genera el system prompt con la fecha actual."""
    from datetime import datetime
    now = datetime.now()
    today = now.strftime("%Y-%m-%d")
    day_name = now.strftime("%A")
    
    # Mapeo de días en español
    days_es = {
        "Monday": "Lunes", "Tuesday": "Martes", "Wednesday": "Miércoles",
        "Thursday": "Jueves", "Friday": "Viernes", "Saturday": "Sábado", "Sunday": "Domingo"
    }
    day_es = days_es.get(day_name, day_name)
    
    return f"""Eres un asistente personal inteligente con acceso a Gmail y Google Calendar.

FECHA Y HORA ACTUAL: Hoy es {day_es}, {today}. La zona horaria es America/Mazatlan (UTC-07:00).

CAPACIDADES:
- Puedes enviar correos electrónicos usando la herramienta 'gmail_send'
- Puedes crear eventos en el calendario usando 'calendar_create_event'
- Puedes consultar el contenido de un PDF cargado usando 'pdf_query'

INSTRUCCIONES PARA CORREOS:
- Usa gmail_send con to, subject y body

INSTRUCCIONES PARA EVENTOS:
- Usa calendar_create_event con summary, start_datetime, end_datetime
- IMPORTANTE: Las fechas deben estar en formato RFC3339 (ej: 2025-12-11T10:00:00-07:00)
- Cuando el usuario diga "mañana", calcula la fecha sumando 1 día a {today}
- Cuando diga "pasado mañana", suma 2 días
- Cuando diga "el lunes", calcula la fecha del próximo lunes
- Siempre usa la zona horaria America/Mazatlan (-07:00)
- Si el usuario dice "10:00 AM", conviértelo a "10:00:00-07:00"
- Si dice "11:30 AM", conviértelo a "11:30:00-07:00"

INSTRUCCIONES PARA PDFs:
- Cuando el usuario pregunte sobre un documento, PDF, archivo, o quiera que analices/resumas algo, usa pdf_query
- Si no hay PDF cargado, indica que deben subir uno primero

COMPORTAMIENTO:
- NO preguntes por la fecha si el usuario dice "mañana", "hoy", "pasado mañana", etc. Calcula la fecha tú mismo.
- NO preguntes por la zona horaria, usa America/Mazatlan por defecto.
- Sé proactivo: si tienes suficiente información, ejecuta la acción.
- Confirma brevemente cuando completes una acción.
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
            prompt=get_system_prompt(),  # Prompt dinámico con fecha actual
            checkpointer=_memory,
        )
    return _agent


def _extract_response(result: dict) -> str:
    """Extrae el texto de respuesta del resultado del agente."""
    messages = result.get("messages", [])
    # Buscar el último mensaje AI que tenga contenido de texto
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content:
            content = msg.content
            # msg.content puede ser string o lista de partes
            if isinstance(content, list):
                # Extraer solo las partes de texto
                text_parts = []
                for part in content:
                    if isinstance(part, str):
                        text_parts.append(part)
                    elif isinstance(part, dict) and part.get("type") == "text":
                        text_parts.append(part.get("text", ""))
                content = " ".join(text_parts)
            
            if isinstance(content, str) and content.strip():
                return content.strip()
    return "[Sin contenido del modelo]"


def _build_message_with_context(user_input: str) -> str:
    """Construye el mensaje del usuario, incluyendo contexto del PDF si está cargado."""
    # Si hay un PDF cargado, incluir su contenido en el contexto
    if pdf_storage.get("content"):
        pdf_data = pdf_storage["content"]
        pdf_text = pdf_data.get("text", "")[:12000]  # Limitar a 12k caracteres
        filename = pdf_data.get("filename", "documento.pdf")
        pages = pdf_data.get("pages", 0)
        
        return f"""[CONTEXTO: El usuario tiene cargado el PDF "{filename}" ({pages} páginas). Contenido del documento:]

{pdf_text}

[FIN DEL DOCUMENTO]

Pregunta del usuario: {user_input}"""
    
    return user_input


def answer_sync(session_id: str, user_input: str) -> str:
    """
    Invocación síncrona con memoria por sesión.
    El agente puede usar tools automáticamente.
    """
    agent = _get_agent()
    message_content = _build_message_with_context(user_input)
    result = agent.invoke(
        {"messages": [HumanMessage(content=message_content)]},
        config={"configurable": {"thread_id": session_id}},
    )
    return _extract_response(result)


async def answer_async(session_id: str, user_input: str) -> str:
    """
    Invocación asíncrona con memoria por sesión.
    """
    agent = _get_agent()
    message_content = _build_message_with_context(user_input)
    result = await agent.ainvoke(
        {"messages": [HumanMessage(content=message_content)]},
        config={"configurable": {"thread_id": session_id}},
    )
    return _extract_response(result)
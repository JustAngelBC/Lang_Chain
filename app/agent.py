
# app/agent.py
import os
from typing import Dict
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from .agent_tools import TOOLS  # <-- tools registradas

# ---------- Cadena base (prompt + modelo + parser) ----------
def build_chain():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Falta GOOGLE_API_KEY")
    model_name = os.getenv("MODEL_NAME", "gemini-2.5-flash")

    # Habilita tool-calling
    llm = ChatGoogleGenerativeAI(
        model=model_name,
        temperature=0,
        max_retries=2,  # reintentos leves; no soluciona RPD agotado
    ).bind_tools(TOOLS, tool_choice="auto")

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Eres un asistente útil con herramientas. "
            "Si el usuario pide enviar un correo, usa 'gmail_send' con to, subject, body (y from_email opcional). "
            "Si pide crear un evento, usa 'calendar_create_event' con summary, start_datetime, end_datetime (RFC3339), "
            "y timezone/description/location/attendees opcionales. "
            "Si faltan datos, primero pregúntalos con precisión y luego ejecuta la herramienta."
        ),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ])

    parser = StrOutputParser()  # salida texto plano
    return prompt | llm | parser


# ---------- Estado módulo ----------
_base_chain = None
_session_store: Dict[str, InMemoryChatMessageHistory] = {}

def _get_base_chain():
    global _base_chain
    if _base_chain is None:
        _base_chain = build_chain()
    return _base_chain

def _get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in _session_store:
        _session_store[session_id] = InMemoryChatMessageHistory()
    return _session_store[session_id]

def _get_agent_with_memory():
    base = _get_base_chain()
    # Tu entorno usa el parámetro get_session_history (no get_message_history)
    return RunnableWithMessageHistory(
        runnable=base,
        get_session_history=_get_session_history,
        input_messages_key="input",
        history_messages_key="history",
    )

def answer_sync(session_id: str, user_input: str) -> str:
    agent = _get_agent_with_memory()
    text = agent.invoke(
        {"input": user_input},
        config={"configurable": {"session_id": session_id}},
    )
    return text.strip() if isinstance(text, str) and text.strip() else "[Sin contenido del modelo]"

async def answer_async(session_id: str, user_input: str) -> str:
    agent = _get_agent_with_memory()
    text = await agent.ainvoke(
        {"input": user_input},
        config={"configurable": {"session_id": session_id}},
    )
    return text.strip() if isinstance(text, str) and text.strip() else "[Sin contenido del modelo]"

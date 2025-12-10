
# app/agent.py
"""
Agente con Gemini + Tools (Gmail, Calendar) + Memoria por sesión.
"""
import os
from typing import Dict

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.agents import AgentExecutor, create_tool_calling_agent

from .agent_tools import TOOLS  # Tools de Gmail y Calendar


# ---------- Construir el Agente con Tools ----------
def build_agent():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Falta GOOGLE_API_KEY")
    model_name = os.getenv("MODEL_NAME", "gemini-2.0-flash")

    # LLM con soporte para tool-calling
    llm = ChatGoogleGenerativeAI(
        model=model_name,
        temperature=0.7,
        max_retries=2,
    )

    # Prompt del agente
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """Eres un asistente personal inteligente con acceso a Gmail y Google Calendar.

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
        ),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    # Crear el agente con tool-calling
    agent = create_tool_calling_agent(llm, TOOLS, prompt)
    
    # Executor que maneja el loop de tools
    executor = AgentExecutor(
        agent=agent,
        tools=TOOLS,
        verbose=True,  # Para debug, puedes poner False en producción
        handle_parsing_errors=True,
        max_iterations=5,
    )
    
    return executor


# ---------- Estado del módulo ----------
_agent_executor = None
_session_store: Dict[str, InMemoryChatMessageHistory] = {}


def _get_agent():
    """Lazy init del agente."""
    global _agent_executor
    if _agent_executor is None:
        _agent_executor = build_agent()
    return _agent_executor


def _get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    """Obtiene/crea el historial de chat para una sesión."""
    if session_id not in _session_store:
        _session_store[session_id] = InMemoryChatMessageHistory()
    return _session_store[session_id]


def _get_agent_with_memory():
    """Envuelve el agente con memoria de conversación."""
    agent = _get_agent()
    return RunnableWithMessageHistory(
        runnable=agent,
        get_session_history=_get_session_history,
        input_messages_key="input",
        history_messages_key="history",
    )


def answer_sync(session_id: str, user_input: str) -> str:
    """
    Invocación síncrona con memoria por sesión.
    El agente puede usar tools automáticamente.
    """
    agent = _get_agent_with_memory()
    result = agent.invoke(
        {"input": user_input},
        config={"configurable": {"session_id": session_id}},
    )
    # AgentExecutor retorna dict con 'output'
    if isinstance(result, dict):
        output = result.get("output", "")
    else:
        output = str(result)
    return output.strip() if output.strip() else "[Sin contenido del modelo]"


async def answer_async(session_id: str, user_input: str) -> str:
    """
    Invocación asíncrona con memoria por sesión.
    """
    agent = _get_agent_with_memory()
    result = await agent.ainvoke(
        {"input": user_input},
        config={"configurable": {"session_id": session_id}},
    )
    if isinstance(result, dict):
        output = result.get("output", "")
    else:
        output = str(result)
    return output.strip() if output.strip() else "[Sin contenido del modelo]"
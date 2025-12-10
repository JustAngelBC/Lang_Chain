
# app/agent.py
import os
from typing import Dict

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory


# Cadena base (prompt + modelo + parser) SIN estado
def build_chain():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Falta GOOGLE_API_KEY")
    model_name = os.getenv("MODEL_NAME", "gemini-2.5-flash")

    llm = ChatGoogleGenerativeAI(model=model_name, temperature=0)

    # Prompt con espacio para historial
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Eres un asistente útil. Responde breve y claro."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ])

    parser = StrOutputParser()  # fuerza salida a texto plano

    # Devuelve una Runnable que toma {"input": "..."} y retorna str
    return prompt | llm | parser


# --- Estado en este módulo ---
_base_chain = None
_session_store: Dict[str, InMemoryChatMessageHistory] = {}  # simple store en memoria


def _get_base_chain():
    """Lazy init de la cadena base."""
    global _base_chain
    if _base_chain is None:
        _base_chain = build_chain()
    return _base_chain


def _get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    """Obtiene/crea el historial de chat para una sesión."""
    if session_id not in _session_store:
        _session_store[session_id] = InMemoryChatMessageHistory()
    return _session_store[session_id]



def _get_agent_with_memory():
    base = _get_base_chain()
    return RunnableWithMessageHistory(
        runnable=base,
        # 👇 nombre que tu versión espera
        get_session_history=_get_session_history,
        input_messages_key="input",
        history_messages_key="history",
    )



def answer_sync(session_id: str, user_input: str) -> str:
    """
    Invocación síncrona con memoria por sesión.
    Retorna texto plano (gracias a StrOutputParser).
    """
    agent = _get_agent_with_memory()
    text = agent.invoke(
        {"input": user_input},
        config={"configurable": {"session_id": session_id}},
    )
    return text.strip() if isinstance(text, str) and text.strip() else "[Sin contenido del modelo]"


async def answer_async(session_id: str, user_input: str) -> str:
    """
    Invocación asíncrona con memoria por sesión.
    """
    agent = _get_agent_with_memory()
    text = await agent.ainvoke(
        {"input": user_input},
        config={"configurable": {"session_id": session_id}},
    )
    return text.strip() if isinstance(text, str) and text.strip() else "[Sin contenido del modelo]"
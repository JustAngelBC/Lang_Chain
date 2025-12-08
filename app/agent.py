
# app/agent.py
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

def build_chain():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Falta GOOGLE_API_KEY")
    model_name = os.getenv("MODEL_NAME", "gemini-2.5-flash")  # modelo vigente
    llm = ChatGoogleGenerativeAI(model=model_name, temperature=0)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Eres un asistente útil. Responde breve y claro."),
        ("human", "{input}")
    ])

    parser = StrOutputParser()  # fuerza a texto plano
    return prompt | llm | parser

_chain = None

def answer_sync(user_input: str) -> str:
    global _chain
    if _chain is None:
        _chain = build_chain()
    text = _chain.invoke({"input": user_input})
    if not text or not text.strip():
        return "[Sin contenido del modelo]"  # fallback visible
    return text.strip()


# app/agent.py
import os
from langchain_google_genai import ChatGoogleGenerativeAI

def build_model():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Falta GOOGLE_API_KEY en variables de entorno.")
    # Modelos comunes: "gemini-1.5-flash" (rápido) o "gemini-1.5-pro" (más capaz)
    model_name = os.getenv("MODEL_NAME", "gemini-1.5-flash")
    return ChatGoogleGenerativeAI(model=model_name, temperature=0)

def answer_sync(user_input: str) -> str:
    llm = build_model()
    resp = llm.invoke(user_input)
    # Unifica salida a string
    return str(resp)
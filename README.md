
# Agente Gemini + LangChain

API en FastAPI con LangChain usando modelos de Gemini.

## Endpoints
- `GET /health` -> estado
- `POST /agent/invoke` -> cuerpo: `{ "input": "..." }`

## Variables de entorno
- `GOOGLE_API_KEY` (obligatoria)
- `MODEL_NAME` (opcional, por defecto `gemini-1.5-flash`)

## Docker local
```bash
docker build -t agente-gemini:local .
docker run -e GOOGLE_API_KEY=TU_CLAVE -p 8000:8000 agente-gemini:local
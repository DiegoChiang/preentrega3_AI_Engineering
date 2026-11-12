# Sistema de recuperación semántica local

Sistema RAG asíncrono que indexa documentos `.txt` y `.md` en ChromaDB, recupera los fragmentos
más relevantes y genera respuestas sustentadas únicamente en el contexto recuperado.

## Componentes

### Ingesta

- Lectura recursiva de documentos desde `data/`.
- `RecursiveCharacterTextSplitter` con fragmentos de 500 unidades léxicas y 50 de solapamiento.
- Identificadores reproducibles mediante SHA-256.
- Embeddings de OpenAI y persistencia local en `vectorstore/`.
- Manifiesto para omitir documentos sin cambios y retirar fragmentos obsoletos.
- Reconstrucción controlada cuando cambia el modelo de embeddings o el chunking.

### Recuperación y generación

- El mismo modelo de embeddings se utiliza para indexar y consultar.
- Recuperación de cuatro fragmentos por defecto, restringida al rango de 3 a 5.
- Filtro de relevancia con similitud coseno.
- Prompt que responde exclusivamente con el contexto disponible.
- Cadena LCEL con `PydanticOutputParser` y ejecución asíncrona.
- Citas validadas contra los identificadores realmente recuperados.
- Respuesta `No lo sé con la información disponible.` cuando no existe respaldo documental.

## Contrato de salida

```json
{
  "answer": "Las claves de idempotencia se conservan durante 24 horas.",
  "grounded": true,
  "confidence": 0.97,
  "cited_chunk_ids": ["identificador-del-fragmento"],
  "sources": [
    {
      "source": "redis_idempotencia.txt",
      "chunk_id": "identificador-del-fragmento",
      "excerpt": "El clúster Redis conserva claves...",
      "score": 0.44
    }
  ]
}
```

## Instalación

```bash
python -m venv .venv
python -m pip install -e ".[dev]"
```

En PowerShell:

```powershell
Copy-Item .env.example .env
```

Configurar las claves y modelos en `.env`. OpenAI se utiliza para embeddings; el modelo de
generación puede ser OpenAI o Anthropic mediante `LLM_PROVIDER`.

`.env` y `vectorstore/` están excluidos del control de versiones.

## Ejecución

Indexar los documentos:

```bash
python main.py ingest
```

Realizar una consulta:

```bash
python main.py query "¿Durante cuánto tiempo se conservan las claves de idempotencia?"
```

Demostración completa:

```bash
python examples/rag_demo.py
```

## Pruebas de evidencia

Pregunta respaldada:

```bash
python main.py query "¿Durante cuánto tiempo se conservan las claves de idempotencia?"
```

El resultado menciona 24 horas y referencia `redis_idempotencia.txt`.

Pregunta sin respaldo:

```bash
python main.py query "¿Cuál es la contraseña del servidor de producción?"
```

El resultado debe indicar que la información no está disponible, con `grounded=false`, confianza
cero y sin fuentes.

## Función pública

```python
from incident_intelligence.rag import get_rag_response

answer = await get_rag_response("¿Cuánto duran las claves de idempotencia?")
```

## Verificación

```bash
pytest
ruff check .
python -m compileall src examples main.py
```

Las pruebas cubren fragmentación, solapamiento, persistencia, ingesta idempotente, recuperación,
parsing Pydantic, citas, respuesta respaldada y pregunta trampa.

## Estructura

```text
.
├── data/                         # Documentos de ejemplo
├── docs/rag_local.md             # Diseño y operación
├── examples/rag_demo.py          # Demostración asíncrona
├── src/incident_intelligence/
│   ├── cli/rag_local.py
│   ├── integrations/langchain.py
│   ├── rag/
│   │   ├── chunking.py
│   │   ├── documents.py
│   │   ├── ingestion.py
│   │   ├── repository.py
│   │   ├── generation.py
│   │   ├── schemas.py
│   │   └── service.py
│   └── config.py
├── tests/
├── .env.example
├── main.py
└── pyproject.toml
```

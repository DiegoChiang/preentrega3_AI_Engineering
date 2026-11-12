# Incident Intelligence System

Sistema asíncrono en Python para consultar modelos de OpenAI y Anthropic, extraer información
técnica con salida validada y responder preguntas sobre una base de conocimiento local.

## Capacidades

### Cliente LLM común

- Interfaz única para OpenAI y Anthropic.
- Selección del proveedor por argumento o variable `LLM_PROVIDER`.
- Mensajes, configuración, uso de tokens y respuestas modelados con Pydantic.
- Generación completa y streaming asíncrono con `async`/`await`.
- Reintentos con backoff exponencial y propagación segura de cancelaciones.
- Errores de configuración y de proveedor diferenciados.

### Extracción técnica estructurada

- Identificación de tecnologías mencionadas en un texto.
- Criticidad normalizada: `baja`, `media` o `alta`.
- Resumen técnico breve y validado.
- Composición LCEL `prompt | modelo estructurado`.
- Ejecución con `ainvoke`, reintentos y registros operativos.

Ejemplo de salida:

```json
{
  "tecnologias": ["Kubernetes", "PostgreSQL"],
  "nivel_de_criticidad": "alta",
  "resumen_tecnico": "La API presenta errores generalizados por agotamiento del pool de PostgreSQL."
}
```

### RAG local

- Carga recursiva de archivos `.txt` y `.md`.
- Fragmentos de 500 unidades léxicas con 50 de solapamiento, contadas localmente y sin red.
- Identificadores de chunks reproducibles mediante SHA-256.
- Persistencia local en Chroma e ingesta idempotente mediante manifiesto.
- Mismo modelo de embeddings en indexación y consulta.
- Recuperación asíncrona de los cuatro fragmentos más relevantes por defecto.
- Filtrado de fragmentos con relevancia inferior a `0.30` y similitud coseno.
- Respuesta Pydantic con estado de respaldo, confianza, fuentes, fragmentos y puntajes.
- Respuesta explícita `No lo sé con la información disponible.` cuando no hay evidencia.

## Requisitos

- Python 3.12 o superior.
- Una clave de OpenAI para embeddings y para los flujos configurados con OpenAI.
- Una clave de Anthropic cuando `LLM_PROVIDER=anthropic`.

## Instalación

```bash
python -m venv .venv
```

Activación en PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Instalación del proyecto y herramientas de desarrollo:

```bash
python -m pip install -e ".[dev]"
```

Configuración local:

```powershell
Copy-Item .env.example .env
```

Completar en `.env` únicamente las claves necesarias. El archivo `.env` está excluido del control
de versiones.

## Ejecución

### Consulta normal

```bash
llm-demo "Explica qué es un circuit breaker" --provider openai
```

### Streaming

```bash
llm-demo "Resume el patrón retry con backoff" --provider anthropic --stream
```

### Respuesta completa y streaming

```bash
llm-demo "¿Qué es entropía?" --provider openai --both
```

### Extracción desde texto

```bash
extract-entities --text "El deployment de Kubernetes devuelve 503 porque Redis agotó la memoria."
```

### Extracción desde archivo

```bash
extract-entities --file data/runbook_api_pagos.md
```

### Demostración asíncrona de extracción

```bash
python examples/extraction_demo.py
```

### Ingesta local

```bash
rag-local ingest
```

### Consulta con recuperación

```bash
rag-local query "¿Cuándo debe escalarse un incidente de PostgreSQL?"
```

También se puede usar el punto de entrada general:

```bash
python main.py llm "¿Qué es idempotencia?" --provider openai
python main.py extract --text "PostgreSQL rechaza todas las conexiones."
python main.py rag query "¿Cuál es el TTL de las claves de idempotencia?"
```

## Pruebas de evidencia

Consulta respaldada por los documentos:

```bash
rag-local query "¿Durante cuánto tiempo se conservan las claves de idempotencia?"
```

La respuesta esperada menciona 24 horas e incluye `redis_idempotencia.txt` entre las fuentes.

Consulta trampa sin respaldo documental:

```bash
rag-local query "¿Cuál es la contraseña del servidor de producción?"
```

La respuesta debe indicar que la información no está disponible, con `grounded=false` y sin
presentar una contraseña.

## Verificación

```bash
pytest
ruff check .
python -m compileall src main.py
```

Las pruebas usan dobles asíncronos para proveedores, cadenas y repositorios. No consumen APIs ni
requieren credenciales.

## Estructura

```text
.
├── data/                         # Base documental de ejemplo
├── docs/                         # Arquitectura y guías operativas
├── examples/extraction_demo.py   # Demostración asíncrona de extracción
├── src/incident_intelligence/
│   ├── cli/                      # Comandos ejecutables
│   ├── extraction/               # Esquema, cadena LCEL y servicio
│   ├── integrations/             # Modelos y embeddings
│   ├── llm/                      # Abstracción y adaptadores LLM
│   └── rag/                      # Ingesta, Chroma, retrieval y generación
├── tests/                        # Pruebas unitarias y de integración local
├── .env.example
├── main.py
└── pyproject.toml
```

## Decisiones de seguridad y operación

- Las claves se leen desde variables de entorno y no se imprimen en logs.
- Un stream solo se reintenta antes de emitir el primer token para evitar respuestas duplicadas.
- La ingesta elimina chunks obsoletos cuando cambia o desaparece un documento.
- Los resultados estructurados rechazan campos inesperados.
- Las cancelaciones asíncronas no se convierten en reintentos.

Detalles del cliente asíncrono en [docs/llm_client.md](docs/llm_client.md), del pipeline de
extracción en [docs/extraction_pipeline.md](docs/extraction_pipeline.md) y del flujo completo en
[docs/architecture.md](docs/architecture.md).

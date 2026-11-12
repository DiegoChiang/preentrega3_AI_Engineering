# Pipeline de procesamiento validado

Pipeline asíncrono para extraer entidades técnicas desde logs, descripciones de arquitectura o
reportes de incidentes. La salida se valida con Pydantic y conserva un contrato JSON estable.

## Salida estructurada

```json
{
  "tecnologias": ["FastAPI", "Redis", "PostgreSQL"],
  "nivel_de_criticidad": "alta",
  "resumen_tecnico": "La API presenta errores por agotamiento del pool de PostgreSQL."
}
```

El esquema acepta únicamente:

- `tecnologias`: lista no vacía de tecnologías sin duplicados.
- `nivel_de_criticidad`: `baja`, `media` o `alta`.
- `resumen_tecnico`: resumen de entre 10 y 2000 caracteres.

## Pipeline

La composición LCEL conecta:

```text
ChatPromptTemplate
    | modelo.with_structured_output(TechnicalEntityExtraction, include_raw=True)
    | validación Pydantic
```

La función `process_text(text: str)` ejecuta la cadena con `.ainvoke()`. La validación detecta
salidas truncadas, JSON inválido y campos incompletos. `.with_retry()` realiza al menos dos
intentos cuando la salida no satisface el contrato. Los logs registran el inicio, la validación,
los reintentos y la finalización sin exponer claves.

## Instalación

```bash
python -m venv .venv
python -m pip install -e ".[dev]"
```

En PowerShell:

```powershell
Copy-Item .env.example .env
```

Configurar un proveedor:

```dotenv
LLM_PROVIDER=openai
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
OPENAI_MODEL=gpt-5.4-mini
ANTHROPIC_MODEL=claude-sonnet-4-6
```

`.env` está excluido del control de versiones.

## Ejecución

Desde texto:

```bash
python main.py --text "La API devuelve 503 porque PostgreSQL agotó el pool de conexiones."
```

Desde un archivo UTF-8:

```bash
python main.py --file ejemplo.txt
```

Demostración asíncrona:

```bash
python examples/extraction_demo.py
```

## Verificación

```bash
pytest
ruff check .
python -m compileall src examples main.py
```

Las pruebas usan modelos simulados y cubren el esquema, la composición LCEL, `.ainvoke()`, logs,
truncamiento, errores de parseo y reintentos.

## Estructura

```text
.
├── docs/extraction_pipeline.md
├── examples/extraction_demo.py
├── src/incident_intelligence/
│   ├── cli/extract_entities.py
│   ├── extraction/
│   │   ├── schemas.py
│   │   ├── chain.py
│   │   └── service.py
│   ├── integrations/langchain.py
│   ├── config.py
│   └── errors.py
├── tests/
├── .env.example
├── main.py
└── pyproject.toml
```

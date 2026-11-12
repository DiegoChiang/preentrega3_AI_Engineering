# Pipeline de extracción técnica

## Objetivo

El pipeline recibe texto sin estructura —por ejemplo, un log, una descripción de arquitectura o
el reporte de un incidente— y devuelve un objeto validado con tecnologías, nivel de criticidad y
resumen técnico.

## Contrato de salida

`TechnicalEntityExtraction` define tres campos:

| Campo JSON | Tipo | Restricciones |
| --- | --- | --- |
| `tecnologias` | lista de strings | Entre 1 y 30 elementos no vacíos; elimina duplicados sin alterar el orden. |
| `nivel_de_criticidad` | enum | `baja`, `media` o `alta`. |
| `resumen_tecnico` | string | Entre 10 y 2000 caracteres después de eliminar espacios externos. |

Los campos desconocidos se rechazan. Los nombres internos en Python permanecen disponibles como
`technologies`, `criticality` y `technical_summary`.

Ejemplo:

```json
{
  "tecnologias": ["Kubernetes", "PostgreSQL"],
  "nivel_de_criticidad": "alta",
  "resumen_tecnico": "La API de pagos presenta errores 503 por agotamiento del pool de PostgreSQL."
}
```

## Composición LCEL

La cadena se construye con tres etapas:

```text
ChatPromptTemplate
    | modelo.with_structured_output(TechnicalEntityExtraction, include_raw=True)
    | validación de respuesta
```

El prompt separa las instrucciones del texto de entrada y utiliza la variable `{text}`. La salida
incluye el mensaje original del proveedor, el objeto interpretado y cualquier error de parseo.

La validación posterior comprueba:

- el motivo de finalización informado por el proveedor;
- la presencia de una salida estructurada;
- errores de parseo o campos incompletos;
- el contrato Pydantic completo.

Los motivos `length`, `max_tokens` y `model_context_window_exceeded` se consideran truncamientos.
Un error de validación se convierte en `StructuredOutputError` dentro de la composición. La
política `.with_retry()` vuelve a ejecutar toda la cadena y realiza al menos dos intentos.

## Ejecución asíncrona

La función pública recibe únicamente el texto y una configuración opcional:

```python
from incident_intelligence.extraction.chain import process_text

result = await process_text("Redis agotó la memoria y la API devuelve errores 503.")
```

`EntityExtractionService` limpia la entrada, ejecuta `.ainvoke({"text": ...})`, valida el resultado
y registra el inicio, los fallos de validación y la finalización. Los logs incluyen longitudes,
conteos y tipos de error; no incluyen claves ni el contenido completo de entrada.

## Ejecución

Desde texto:

```powershell
extract-entities --text "PostgreSQL agotó el pool y la API devuelve errores 503."
```

Desde archivo:

```powershell
extract-entities --file ejemplo.txt
```

Demostración asíncrona:

```powershell
python examples/extraction_demo.py
```

El proveedor se selecciona mediante `LLM_PROVIDER=openai` o `LLM_PROVIDER=anthropic`.

## Verificación local

```powershell
pytest tests/test_schemas.py tests/test_extraction_chain.py tests/test_extraction_service.py
ruff check src tests examples
python -m compileall src examples main.py
```

Las pruebas usan modelos y cadenas simulados. Cubren normalización, aliases JSON, entradas vacías,
salidas truncadas, JSON inválido, reintento automático, ejecución con `ainvoke` y logs operativos.

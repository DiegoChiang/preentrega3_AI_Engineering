# Recuperación semántica local

## Objetivo

El sistema responde consultas operativas a partir de una base documental local. La generación se
limita a los fragmentos recuperados y devuelve una respuesta explícita de desconocimiento cuando
no encuentra evidencia suficiente.

## Flujo

1. El cargador recorre `data/` y acepta archivos `.txt` y `.md`.
2. `RecursiveCharacterTextSplitter` divide cada documento en fragmentos de hasta 500 unidades
   léxicas con 50 unidades de solapamiento.
3. OpenAI genera los embeddings y Chroma los persiste en `vectorstore/`.
4. La consulta se convierte en un embedding con el mismo modelo usado durante la indexación.
5. Chroma recupera entre tres y cinco candidatos; el valor predeterminado es cuatro.
6. Los candidatos con similitud inferior a `0.30` se descartan.
7. Una cadena LCEL entrega el contexto al modelo y `PydanticOutputParser` valida la salida.
8. Las referencias se aceptan únicamente si corresponden a identificadores recuperados.

## Ingesta y persistencia

```bash
rag-local ingest
```

El manifiesto `vectorstore/manifest.json` conserva la huella SHA-256 de cada documento y sus
identificadores de fragmento. Los documentos sin cambios no vuelven a indexarse. Un cambio del
modelo de embeddings o de los parámetros de fragmentación modifica la firma del índice y provoca
una reconstrucción controlada. Los fragmentos asociados a archivos eliminados también se borran.

La carpeta `vectorstore/` está excluida del repositorio porque Chroma puede reconstruirla desde los
documentos versionados.

## Contrato asíncrono

La función pública requiere solamente la consulta; la configuración es opcional:

```python
from incident_intelligence.rag import get_rag_response

answer = await get_rag_response("¿Cuánto duran las claves de idempotencia?")
```

El resultado contiene:

- `answer`: texto de respuesta.
- `grounded`: indica si existe respaldo documental.
- `confidence`: confianza normalizada entre 0 y 1.
- `cited_chunk_ids`: fragmentos concretos usados por el modelo.
- `sources`: archivo, fragmento, extracto y puntaje de similitud de cada referencia válida.

Una respuesta marcada como respaldada necesita al menos una cita válida. Una respuesta no
respaldada no puede incluir citas ni fuentes.

## Consultas de comprobación

Pregunta con respuesta presente:

```bash
rag-local query "¿Durante cuánto tiempo se conservan las claves de idempotencia?"
```

El resultado menciona 24 horas y referencia `redis_idempotencia.txt`.

Pregunta sin respuesta presente:

```bash
rag-local query "¿Cuál es la contraseña del servidor de producción?"
```

El resultado es `No lo sé con la información disponible.`, con `grounded` en `false`, confianza
cero y listas vacías de citas y fuentes.

## Seguridad

Las claves se leen desde `.env`, que está excluido de Git. `.env.example` contiene únicamente los
nombres de variables. Los prompts tratan el contexto como datos e ignoran instrucciones incluidas
dentro de los documentos.

## Pruebas

```bash
pytest tests/test_documents_and_ingestion.py \
       tests/test_chroma_repository.py \
       tests/test_rag_generation.py \
       tests/test_rag_service.py
```

Las pruebas verifican fragmentación y solapamiento, ingesta idempotente, reindexación por cambio de
configuración, persistencia tras reabrir Chroma, recuperación limitada, parsing Pydantic, selección
de referencias, rechazo de citas inventadas y los casos respaldado y desconocido.

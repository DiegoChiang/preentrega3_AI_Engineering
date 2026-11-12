"""Demostración asíncrona de ingesta y dos consultas RAG."""

import asyncio
import json

from incident_intelligence.config import Settings
from incident_intelligence.rag.ingestion import RAGIngestor
from incident_intelligence.rag.repository import build_chroma_repository
from incident_intelligence.rag.service import get_rag_response


async def main() -> None:
    settings = Settings()
    repository = build_chroma_repository(settings)
    ingestor = RAGIngestor(
        repository,
        data_dir=settings.data_dir,
        manifest_path=settings.chroma_persist_dir / "manifest.json",
        chunk_size=settings.rag_chunk_size,
        chunk_overlap=settings.rag_chunk_overlap,
        index_signature=(
            f"{settings.openai_embedding_model}:"
            f"{settings.rag_chunk_size}:{settings.rag_chunk_overlap}"
        ),
    )
    ingestion = await ingestor.ingest()
    print(json.dumps(ingestion, ensure_ascii=False, indent=2))

    questions = [
        "¿Durante cuánto tiempo se conservan las claves de idempotencia?",
        "¿Cuál es la contraseña del servidor de producción?",
    ]
    for question in questions:
        answer = await get_rag_response(question, settings)
        print(json.dumps(answer.model_dump(mode="json"), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())

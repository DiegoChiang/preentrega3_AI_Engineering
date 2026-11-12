"""Ingesta y consulta del sistema RAG local."""

import argparse
import asyncio
import json

from incident_intelligence.config import Settings
from incident_intelligence.rag.ingestion import RAGIngestor
from incident_intelligence.rag.repository import ChromaRepository, build_chroma_repository
from incident_intelligence.rag.service import get_rag_response


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Gestiona la base de conocimiento local")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("ingest", help="Indexa archivos .txt y .md")
    query_parser = subparsers.add_parser("query", help="Consulta la base vectorial")
    query_parser.add_argument("question")
    return parser


def build_repository(settings: Settings) -> ChromaRepository:
    return build_chroma_repository(settings)


async def run(command: str, question: str | None = None) -> None:
    settings = Settings()

    if command == "ingest":
        repository = build_repository(settings)
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
        result = await ingestor.ingest()
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    if question is None:
        raise ValueError("La consulta requiere una pregunta")
    result = await get_rag_response(question, settings)
    print(json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2))


def main() -> None:
    args = build_parser().parse_args()
    asyncio.run(run(args.command, getattr(args, "question", None)))


if __name__ == "__main__":
    main()

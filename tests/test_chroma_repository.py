from pathlib import Path

import pytest

from incident_intelligence.rag.repository import ChromaRepository
from incident_intelligence.rag.schemas import DocumentChunk


class KeywordEmbeddings:
    @staticmethod
    def _vector(text: str) -> list[float]:
        lowered = text.casefold()
        return [
            float(lowered.count("redis")) * 0.25,
            float(lowered.count("postgresql")) * 0.25,
            float(lowered.count("kubernetes")) * 0.25,
            1.0,
        ]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


@pytest.mark.asyncio
async def test_chroma_add_search_and_delete(tmp_path: Path) -> None:
    repository = ChromaRepository(
        collection_name="test-knowledge",
        persist_directory=str(tmp_path / "chroma"),
        embedding_function=KeywordEmbeddings(),
    )
    chunks = [
        DocumentChunk(
            id="redis-chunk",
            content="Redis conserva claves de idempotencia.",
            source="redis.txt",
            chunk_index=0,
        ),
        DocumentChunk(
            id="postgres-chunk",
            content="PostgreSQL conserva los pagos confirmados.",
            source="postgres.md",
            chunk_index=0,
        ),
    ]

    await repository.add(chunks)
    reopened_repository = ChromaRepository(
        collection_name="test-knowledge",
        persist_directory=str(tmp_path / "chroma"),
        embedding_function=KeywordEmbeddings(),
    )
    results = await reopened_repository.search("Redis", k=1)

    assert len(results) == 1
    assert results[0].id == "redis-chunk"
    assert results[0].source == "redis.txt"

    await reopened_repository.delete(["redis-chunk"])
    remaining = await reopened_repository.search("Redis", k=2)
    assert all(chunk.id != "redis-chunk" for chunk in remaining)

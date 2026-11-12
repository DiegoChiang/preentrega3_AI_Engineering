import pytest
from pydantic import ValidationError

from incident_intelligence.config import Settings
from incident_intelligence.rag.schemas import RAGDraft, RetrievedChunk
from incident_intelligence.rag.service import RAGService, get_rag_response


class FakeRepository:
    def __init__(self, chunks: list[RetrievedChunk]) -> None:
        self.chunks = chunks
        self.last_query: str | None = None
        self.last_k: int | None = None

    async def add(self, chunks: list[object]) -> None:
        return None

    async def delete(self, ids: list[str]) -> None:
        return None

    async def search(self, query: str, *, k: int) -> list[RetrievedChunk]:
        self.last_query = query
        self.last_k = k
        return self.chunks[:k]


class FakeGenerator:
    def __init__(self, draft: RAGDraft) -> None:
        self.draft = draft
        self.context: str | None = None

    async def generate(self, *, query: str, context: str) -> RAGDraft:
        self.context = context
        return self.draft


@pytest.mark.asyncio
async def test_supported_question_includes_sources() -> None:
    repository = FakeRepository(
        [
            RetrievedChunk(
                id="chunk-redis",
                content="Las claves de idempotencia se conservan durante 24 horas.",
                source="redis_idempotencia.txt",
                chunk_index=0,
                score=0.93,
            )
        ]
    )
    generator = FakeGenerator(
        RAGDraft(
            answer="Las claves de idempotencia se conservan durante 24 horas.",
            grounded=True,
            confidence=0.95,
            cited_chunk_ids=["chunk-redis"],
        )
    )
    service = RAGService(repository, generator, top_k=4)

    result = await service.get_rag_response("¿Cuál es el TTL de las claves?")

    assert result.grounded is True
    assert result.sources[0].source == "redis_idempotencia.txt"
    assert result.sources[0].chunk_id == "chunk-redis"
    assert "24 horas" in result.sources[0].excerpt
    assert generator.context is not None and "redis_idempotencia.txt" in generator.context
    assert repository.last_k == 4


@pytest.mark.asyncio
async def test_question_without_retrieval_returns_explicit_unknown() -> None:
    repository = FakeRepository([])
    generator = FakeGenerator(
        RAGDraft(answer="No debe ejecutarse", grounded=False, confidence=0)
    )
    service = RAGService(repository, generator)

    result = await service.get_rag_response("¿Cuál es la contraseña de producción?")

    assert result.answer == "No lo sé con la información disponible."
    assert result.grounded is False
    assert result.sources == []
    assert generator.context is None


@pytest.mark.asyncio
async def test_only_model_citations_become_sources() -> None:
    repository = FakeRepository(
        [
            RetrievedChunk(
                id="chunk-redis",
                content="Las claves de idempotencia se conservan durante 24 horas.",
                source="redis_idempotencia.txt",
                chunk_index=0,
                score=0.93,
            ),
            RetrievedChunk(
                id="chunk-postgres",
                content="Los respaldos de PostgreSQL se verifican cada semana.",
                source="postgresql_recuperacion.md",
                chunk_index=0,
                score=0.81,
            ),
        ]
    )
    generator = FakeGenerator(
        RAGDraft(
            answer="Las claves se conservan durante 24 horas.",
            grounded=True,
            confidence=0.9,
            cited_chunk_ids=["chunk-redis"],
        )
    )

    result = await RAGService(repository, generator).get_rag_response(
        "¿Cuánto duran las claves de idempotencia?"
    )

    assert [source.chunk_id for source in result.sources] == ["chunk-redis"]


@pytest.mark.asyncio
async def test_unknown_citation_is_rejected_as_ungrounded() -> None:
    repository = FakeRepository(
        [
            RetrievedChunk(
                id="chunk-redis",
                content="Las claves de idempotencia se conservan durante 24 horas.",
                source="redis_idempotencia.txt",
                chunk_index=0,
                score=0.93,
            )
        ]
    )
    generator = FakeGenerator(
        RAGDraft(
            answer="Respuesta sin fuente recuperada.",
            grounded=True,
            confidence=0.9,
            cited_chunk_ids=["chunk-inventado"],
        )
    )

    result = await RAGService(repository, generator).get_rag_response("Consulta")

    assert result.answer == "No lo sé con la información disponible."
    assert result.grounded is False
    assert result.sources == []


def test_rag_draft_enforces_grounding_contract() -> None:
    with pytest.raises(ValidationError):
        RAGDraft(answer="Respuesta sin cita", grounded=True, confidence=0.8)

    with pytest.raises(ValidationError):
        RAGDraft(
            answer="No lo sé con la información disponible.",
            grounded=False,
            confidence=0,
            cited_chunk_ids=["chunk-invalido"],
        )


@pytest.mark.parametrize("top_k", [2, 6])
def test_service_rejects_excessive_retrieval(top_k: int) -> None:
    generator = FakeGenerator(
        RAGDraft(
            answer="No lo sé con la información disponible.",
            grounded=False,
            confidence=0,
        )
    )
    with pytest.raises(ValueError, match="top_k"):
        RAGService(FakeRepository([]), generator, top_k=top_k)


@pytest.mark.asyncio
async def test_public_function_only_requires_query_and_optional_settings(monkeypatch) -> None:
    repository = FakeRepository(
        [
            RetrievedChunk(
                id="chunk-redis",
                content="Las claves se conservan durante 24 horas.",
                source="redis_idempotencia.txt",
                chunk_index=0,
                score=0.9,
            )
        ]
    )
    generator = FakeGenerator(
        RAGDraft(
            answer="Las claves se conservan durante 24 horas.",
            grounded=True,
            confidence=0.9,
            cited_chunk_ids=["chunk-redis"],
        )
    )
    monkeypatch.setattr(
        "incident_intelligence.rag.service.build_chroma_repository",
        lambda settings: repository,
    )
    monkeypatch.setattr(
        "incident_intelligence.rag.service.build_answer_generator",
        lambda settings: generator,
    )

    result = await get_rag_response(
        "¿Cuánto duran las claves?", Settings(_env_file=None)
    )

    assert result.grounded is True
    assert result.cited_chunk_ids == ["chunk-redis"]


@pytest.mark.asyncio
async def test_ungrounded_generation_hides_retrieved_sources() -> None:
    repository = FakeRepository(
        [
            RetrievedChunk(
                id="chunk-api",
                content="La API usa Kubernetes.",
                source="runbook_api_pagos.md",
                chunk_index=0,
                score=0.5,
            )
        ]
    )
    generator = FakeGenerator(
        RAGDraft(
            answer="No lo sé con la información disponible.",
            grounded=False,
            confidence=0.05,
        )
    )

    result = await RAGService(repository, generator).get_rag_response(
        "¿Cuál es la contraseña de producción?"
    )

    assert result.grounded is False
    assert result.sources == []

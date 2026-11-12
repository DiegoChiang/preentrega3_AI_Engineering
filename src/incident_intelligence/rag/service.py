"""Flujo asíncrono end-to-end de recuperación y generación."""

from incident_intelligence.config import Settings
from incident_intelligence.rag.generation import AnswerGenerator, build_answer_generator
from incident_intelligence.rag.repository import VectorRepository, build_chroma_repository
from incident_intelligence.rag.schemas import RAGAnswer, SourceReference

UNKNOWN_ANSWER = "No lo sé con la información disponible."


class RAGService:
    def __init__(
        self,
        repository: VectorRepository,
        generator: AnswerGenerator,
        *,
        top_k: int = 4,
        min_relevance_score: float = 0.3,
    ) -> None:
        if not 3 <= top_k <= 5:
            raise ValueError("top_k debe estar entre 3 y 5")
        if not 0 <= min_relevance_score <= 1:
            raise ValueError("min_relevance_score debe estar entre 0 y 1")
        self.repository = repository
        self.generator = generator
        self.top_k = top_k
        self.min_relevance_score = min_relevance_score

    async def get_rag_response(self, query: str) -> RAGAnswer:
        clean = query.strip()
        if not clean:
            raise ValueError("La consulta no puede estar vacía")

        chunks = await self.repository.search(clean, k=self.top_k)
        chunks = [
            chunk
            for chunk in chunks
            if chunk.score is None or chunk.score >= self.min_relevance_score
        ]
        if not chunks:
            return RAGAnswer(
                answer=UNKNOWN_ANSWER,
                grounded=False,
                confidence=0,
                cited_chunk_ids=[],
                sources=[],
            )

        context = "\n\n".join(
            f"[SOURCE={chunk.source} | FRAGMENT_ID={chunk.id}]\n{chunk.content}"
            for chunk in chunks
        )
        draft = await self.generator.generate(query=clean, context=context)
        sources: list[SourceReference] = []
        if draft.grounded:
            chunks_by_id = {chunk.id: chunk for chunk in chunks}
            sources = [
                SourceReference(
                    source=chunk.source,
                    chunk_id=chunk.id,
                    excerpt=chunk.content[:500],
                    score=chunk.score,
                )
                for chunk_id in draft.cited_chunk_ids
                if (chunk := chunks_by_id.get(chunk_id)) is not None
            ]
            if not sources:
                return RAGAnswer(
                    answer=UNKNOWN_ANSWER,
                    grounded=False,
                    confidence=0,
                    cited_chunk_ids=[],
                    sources=[],
                )
        payload = draft.model_dump()
        payload["cited_chunk_ids"] = [source.chunk_id for source in sources]
        return RAGAnswer(**payload, sources=sources)


async def get_rag_response(
    query: str,
    settings: Settings | None = None,
) -> RAGAnswer:
    """Responde una consulta usando la base local y el proveedor configurado."""

    active_settings = settings or Settings()
    return await RAGService(
        build_chroma_repository(active_settings),
        build_answer_generator(active_settings),
        top_k=active_settings.rag_top_k,
        min_relevance_score=active_settings.rag_min_relevance_score,
    ).get_rag_response(query)


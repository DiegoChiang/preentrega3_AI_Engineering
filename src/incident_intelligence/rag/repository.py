"""Puerto de almacenamiento vectorial y adaptador ChromaDB."""

import asyncio
from typing import Any, Protocol

from incident_intelligence.config import Settings
from incident_intelligence.integrations.langchain import build_openai_embeddings
from incident_intelligence.rag.schemas import DocumentChunk, RetrievedChunk


def _cosine_relevance(distance: float) -> float:
    return max(0.0, min(1.0, 1.0 - distance))


def _normalized_l2_relevance(distance: float) -> float:
    """Convierte distancia L2 cuadrática de embeddings normalizados en similitud coseno."""

    return max(0.0, min(1.0, 1.0 - (distance / 2.0)))


class VectorRepository(Protocol):
    async def add(self, chunks: list[DocumentChunk]) -> None: ...

    async def delete(self, ids: list[str]) -> None: ...

    async def search(self, query: str, *, k: int) -> list[RetrievedChunk]: ...


class ChromaRepository:
    """Persistencia local con Chroma y un único modelo de embeddings."""

    def __init__(
        self,
        *,
        collection_name: str,
        persist_directory: str,
        embedding_function: Any,
    ) -> None:
        try:
            from langchain_chroma import Chroma
        except ImportError as exc:  # pragma: no cover - depende del entorno
            raise RuntimeError("Instala la dependencia 'langchain-chroma'") from exc
        self._store = Chroma(
            collection_name=collection_name,
            persist_directory=persist_directory,
            embedding_function=embedding_function,
            collection_metadata={"hnsw:space": "cosine"},
            relevance_score_fn=_cosine_relevance,
        )
        metadata = self._store._collection.metadata or {}
        self._distance_metric = str(metadata.get("hnsw:space", "l2"))

    async def add(self, chunks: list[DocumentChunk]) -> None:
        if not chunks:
            return
        try:
            from langchain_core.documents import Document
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Instala la dependencia 'langchain-core'") from exc
        documents = [
            Document(
                page_content=chunk.content,
                metadata={**chunk.metadata, "chunk_id": chunk.id, "source": chunk.source},
            )
            for chunk in chunks
        ]
        await self._store.aadd_documents(documents=documents, ids=[chunk.id for chunk in chunks])

    async def delete(self, ids: list[str]) -> None:
        if ids:
            await asyncio.to_thread(self._store.delete, ids=ids)

    async def search(self, query: str, *, k: int) -> list[RetrievedChunk]:
        rows = await self._store.asimilarity_search_with_score(query, k=k)
        results: list[RetrievedChunk] = []
        for document, distance in rows:
            metadata = dict(document.metadata)
            score = (
                _cosine_relevance(float(distance))
                if self._distance_metric == "cosine"
                else _normalized_l2_relevance(float(distance))
            )
            results.append(
                RetrievedChunk(
                    id=str(metadata.pop("chunk_id", "")),
                    content=document.page_content,
                    source=str(metadata.pop("source", "desconocido")),
                    chunk_index=int(metadata.pop("chunk_index", 0)),
                    metadata=metadata,
                    score=score,
                )
            )
        return results


def build_chroma_repository(settings: Settings) -> ChromaRepository:
    """Construye el repositorio usado tanto para indexar como para consultar."""

    return ChromaRepository(
        collection_name=settings.chroma_collection,
        persist_directory=str(settings.chroma_persist_dir),
        embedding_function=build_openai_embeddings(settings),
    )


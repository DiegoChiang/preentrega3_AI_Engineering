from pathlib import Path

import pytest

from incident_intelligence.rag.chunking import count_lexical_tokens, split_documents
from incident_intelligence.rag.documents import load_knowledge_documents
from incident_intelligence.rag.ingestion import RAGIngestor
from incident_intelligence.rag.schemas import DocumentChunk, KnowledgeDocument, RetrievedChunk


class MemoryRepository:
    def __init__(self) -> None:
        self.items: dict[str, DocumentChunk] = {}

    async def add(self, chunks: list[DocumentChunk]) -> None:
        self.items.update({chunk.id: chunk for chunk in chunks})

    async def delete(self, ids: list[str]) -> None:
        for item_id in ids:
            self.items.pop(item_id, None)

    async def search(self, query: str, *, k: int) -> list[RetrievedChunk]:
        return []


def test_loader_accepts_txt_and_markdown_only(tmp_path: Path) -> None:
    (tmp_path / "one.md").write_text("Contenido Markdown", encoding="utf-8")
    (tmp_path / "two.txt").write_text("Contenido de texto", encoding="utf-8")
    (tmp_path / "ignored.json").write_text("{}", encoding="utf-8")

    documents = load_knowledge_documents(tmp_path)

    assert [document.source for document in documents] == ["one.md", "two.txt"]


def test_splitter_uses_stable_ids_and_overlap() -> None:
    document = KnowledgeDocument(
        content=" ".join(f"palabra{index:04d}" for index in range(1400)),
        source="large.md",
    )

    first = split_documents([document], chunk_size=500, chunk_overlap=50)
    second = split_documents([document], chunk_size=500, chunk_overlap=50)

    assert len(first) >= 2
    assert [chunk.id for chunk in first] == [chunk.id for chunk in second]
    assert all(chunk.source == "large.md" for chunk in first)
    assert all(count_lexical_tokens(chunk.content) <= 500 for chunk in first)
    assert first[0].content.split()[-50:] == first[1].content.split()[:50]


@pytest.mark.asyncio
async def test_ingestion_is_idempotent(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "knowledge.md").write_text(
        "Redis conserva las claves de idempotencia durante 24 horas.", encoding="utf-8"
    )
    repository = MemoryRepository()
    ingestor = RAGIngestor(
        repository,
        data_dir=data_dir,
        manifest_path=tmp_path / "vectorstore" / "manifest.json",
        chunk_size=500,
        chunk_overlap=50,
    )

    first = await ingestor.ingest()
    second = await ingestor.ingest()

    assert first == {"documents": 1, "chunks_added": 1, "documents_skipped": 0}
    assert second == {"documents": 1, "chunks_added": 0, "documents_skipped": 1}
    assert len(repository.items) == 1


@pytest.mark.asyncio
async def test_ingestion_reindexes_when_embedding_or_chunking_changes(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "knowledge.md").write_text(
        "Redis conserva las claves de idempotencia durante 24 horas.", encoding="utf-8"
    )
    repository = MemoryRepository()
    manifest_path = tmp_path / "vectorstore" / "manifest.json"

    first = await RAGIngestor(
        repository,
        data_dir=data_dir,
        manifest_path=manifest_path,
        chunk_size=500,
        chunk_overlap=50,
        index_signature="embedding-a:500:50",
    ).ingest()
    second = await RAGIngestor(
        repository,
        data_dir=data_dir,
        manifest_path=manifest_path,
        chunk_size=500,
        chunk_overlap=50,
        index_signature="embedding-b:500:50",
    ).ingest()

    assert first["chunks_added"] == 1
    assert second == {"documents": 1, "chunks_added": 1, "documents_skipped": 0}
    assert len(repository.items) == 1

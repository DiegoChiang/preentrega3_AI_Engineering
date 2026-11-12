"""Ingesta idempotente de documentos en la base vectorial."""

import asyncio
import hashlib
import json
from pathlib import Path
from typing import Any

from incident_intelligence.rag.chunking import split_documents
from incident_intelligence.rag.documents import load_knowledge_documents
from incident_intelligence.rag.repository import VectorRepository


class RAGIngestor:
    def __init__(
        self,
        repository: VectorRepository,
        *,
        data_dir: Path,
        manifest_path: Path,
        chunk_size: int,
        chunk_overlap: int,
        index_signature: str | None = None,
    ) -> None:
        if chunk_size < 500:
            raise ValueError("chunk_size debe ser al menos 500")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap debe estar entre 0 y chunk_size")
        self.repository = repository
        self.data_dir = data_dir
        self.manifest_path = manifest_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.index_signature = index_signature or f"lexical:{chunk_size}:{chunk_overlap}"

    def _read_manifest(self) -> dict[str, Any]:
        if not self.manifest_path.exists():
            return {"sources": {}}
        try:
            return json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {"sources": {}}

    def _write_manifest(self, manifest: dict[str, Any]) -> None:
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.manifest_path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temporary.replace(self.manifest_path)

    async def ingest(self) -> dict[str, int]:
        documents = await asyncio.to_thread(load_knowledge_documents, self.data_dir)
        manifest = self._read_manifest()
        previous_sources: dict[str, Any] = manifest.get("sources", {})
        if manifest.get("index_signature") != self.index_signature:
            for previous in previous_sources.values():
                await self.repository.delete(
                    [str(value) for value in previous.get("chunk_ids", [])]
                )
            previous_sources = {}
        next_sources: dict[str, Any] = {}
        added = 0
        skipped = 0

        for document in documents:
            fingerprint = hashlib.sha256(document.content.encode("utf-8")).hexdigest()
            previous = previous_sources.get(document.source, {})
            if previous.get("fingerprint") == fingerprint:
                next_sources[document.source] = previous
                skipped += 1
                continue

            chunks = await asyncio.to_thread(
                split_documents,
                [document],
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            )
            old_ids = [str(value) for value in previous.get("chunk_ids", [])]
            await self.repository.delete(old_ids)
            await self.repository.add(chunks)
            next_sources[document.source] = {
                "fingerprint": fingerprint,
                "chunk_ids": [chunk.id for chunk in chunks],
            }
            added += len(chunks)

        removed_sources = set(previous_sources) - set(next_sources)
        for source in removed_sources:
            await self.repository.delete(
                [str(value) for value in previous_sources[source].get("chunk_ids", [])]
            )

        await asyncio.to_thread(
            self._write_manifest,
            {"index_signature": self.index_signature, "sources": next_sources},
        )
        return {"documents": len(documents), "chunks_added": added, "documents_skipped": skipped}


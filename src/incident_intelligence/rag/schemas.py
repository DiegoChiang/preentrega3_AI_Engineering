"""Contratos del flujo RAG."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class KnowledgeDocument(BaseModel):
    model_config = ConfigDict(extra="forbid")

    content: str = Field(min_length=1)
    source: str = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentChunk(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    content: str = Field(min_length=1)
    source: str
    chunk_index: int = Field(ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievedChunk(DocumentChunk):
    score: float | None = None


class SourceReference(BaseModel):
    source: str
    chunk_id: str
    excerpt: str = Field(max_length=500)
    score: float | None = None


class RAGDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: str = Field(min_length=1, max_length=8000)
    grounded: bool
    confidence: float = Field(ge=0, le=1)
    cited_chunk_ids: list[str] = Field(default_factory=list, max_length=5)

    @field_validator("cited_chunk_ids")
    @classmethod
    def normalize_citations(cls, values: list[str]) -> list[str]:
        normalized: list[str] = []
        seen: set[str] = set()
        for value in values:
            clean = value.strip()
            if clean and clean not in seen:
                normalized.append(clean)
                seen.add(clean)
        return normalized

    @model_validator(mode="after")
    def validate_grounding_contract(self) -> "RAGDraft":
        if self.grounded and not self.cited_chunk_ids:
            raise ValueError("Una respuesta respaldada debe citar al menos un fragmento")
        if not self.grounded and self.cited_chunk_ids:
            raise ValueError("Una respuesta no respaldada no puede citar fragmentos")
        return self


class RAGAnswer(RAGDraft):
    sources: list[SourceReference] = Field(default_factory=list)


"""Configuración central cargada desde variables de entorno."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Parámetros de proveedores, generación y recuperación local."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    llm_provider: Literal["openai", "anthropic"] = "openai"
    openai_api_key: SecretStr | None = None
    anthropic_api_key: SecretStr | None = None
    openai_model: str = "gpt-5.4-mini"
    anthropic_model: str = "claude-sonnet-4-6"
    openai_embedding_model: str = "text-embedding-3-small"

    llm_temperature: float = Field(default=0.2, ge=0, le=2)
    llm_max_tokens: int = Field(default=1200, ge=1, le=128_000)
    llm_timeout_seconds: float = Field(default=45, gt=0, le=600)
    llm_max_retries: int = Field(default=3, ge=1, le=10)

    data_dir: Path = Path("data")
    chroma_persist_dir: Path = Path("vectorstore")
    chroma_collection: str = "incident-knowledge"
    rag_top_k: int = Field(default=4, ge=3, le=5)
    rag_min_relevance_score: float = Field(default=0.3, ge=0, le=1)
    rag_chunk_size: int = Field(default=500, ge=500, le=8_000)
    rag_chunk_overlap: int = Field(default=50, ge=0, le=2_000)

    @field_validator("rag_chunk_overlap")
    @classmethod
    def validate_overlap(cls, value: int, info: object) -> int:
        data = getattr(info, "data", {})
        chunk_size = data.get("rag_chunk_size")
        if chunk_size is not None and value >= chunk_size:
            raise ValueError("rag_chunk_overlap debe ser menor que rag_chunk_size")
        return value

    def require_provider_key(self, provider: str | None = None) -> str:
        """Devuelve la clave del proveedor seleccionado o informa su ausencia."""

        selected = provider or self.llm_provider
        secret = self.openai_api_key if selected == "openai" else self.anthropic_api_key
        if secret is None or not secret.get_secret_value().strip():
            variable = "OPENAI_API_KEY" if selected == "openai" else "ANTHROPIC_API_KEY"
            raise ValueError(f"Falta la variable de entorno {variable}")
        return secret.get_secret_value()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


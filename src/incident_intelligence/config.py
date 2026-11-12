"""Configuración central cargada desde variables de entorno."""

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Parámetros del modelo usado por el pipeline de extracción."""

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

    llm_temperature: float = Field(default=0.2, ge=0, le=2)
    llm_max_tokens: int = Field(default=1200, ge=1, le=128_000)
    llm_timeout_seconds: float = Field(default=45, gt=0, le=600)
    llm_max_retries: int = Field(default=3, ge=1, le=10)

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


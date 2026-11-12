"""Construcción de modelos LangChain a partir de la configuración central."""

from typing import Any

from incident_intelligence.certificates import configure_system_certificate_store
from incident_intelligence.config import Settings


def build_chat_model(settings: Settings) -> Any:
    """Construye el modelo de chat configurado para cadenas estructuradas."""

    configure_system_certificate_store()
    if settings.llm_provider == "openai":
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:  # pragma: no cover - depende del entorno
            raise RuntimeError("Instala la dependencia 'langchain-openai'") from exc
        kwargs: dict[str, Any] = {
            "api_key": settings.require_provider_key("openai"),
            "model": settings.openai_model,
            "max_tokens": settings.llm_max_tokens,
            "timeout": settings.llm_timeout_seconds,
            "max_retries": 0,
        }
        if not settings.openai_model.startswith(("gpt-5", "o")):
            kwargs["temperature"] = settings.llm_temperature
        return ChatOpenAI(
            **kwargs,
        )

    try:
        from langchain_anthropic import ChatAnthropic
    except ImportError as exc:  # pragma: no cover - depende del entorno
        raise RuntimeError("Instala la dependencia 'langchain-anthropic'") from exc

    kwargs: dict[str, Any] = {
        "api_key": settings.require_provider_key("anthropic"),
        "model": settings.anthropic_model,
        "max_tokens": settings.llm_max_tokens,
        "timeout": settings.llm_timeout_seconds,
        "max_retries": 0,
    }
    return ChatAnthropic(**kwargs)


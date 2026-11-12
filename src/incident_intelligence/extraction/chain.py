"""Cadena LCEL para extraer entidades técnicas."""

import logging
from collections.abc import Mapping
from typing import Any

from pydantic import ValidationError

from incident_intelligence.config import Settings
from incident_intelligence.errors import StructuredOutputError
from incident_intelligence.extraction.schemas import TechnicalEntityExtraction
from incident_intelligence.integrations.langchain import build_chat_model

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """Eres un analista de incidentes de software.
Extrae únicamente información respaldada por el texto recibido.
Identifica tecnologías concretas, asigna criticidad baja, media o alta y redacta un resumen
técnico breve. La criticidad alta corresponde a indisponibilidad, pérdida de datos, acceso no
autorizado o impacto generalizado; media a degradación parcial; baja a consultas o problemas
sin impacto operativo. No inventes tecnologías ausentes. Si el texto es ambiguo, conserva solo
los datos explícitos y evita completar información por inferencia. Devuelve los campos
tecnologias, nivel_de_criticidad y resumen_tecnico definidos por el contrato estructurado."""

TRUNCATED_FINISH_REASONS = {
    "length",
    "max_tokens",
    "model_context_window_exceeded",
}


def _finish_reason(raw: Any) -> str | None:
    containers: list[Mapping[str, Any]] = []
    if isinstance(raw, Mapping):
        containers.append(raw)
    for attribute in ("response_metadata", "additional_kwargs"):
        value = getattr(raw, attribute, None)
        if isinstance(value, Mapping):
            containers.append(value)

    for metadata in containers:
        for key in ("finish_reason", "stop_reason"):
            reason = metadata.get(key)
            if reason is not None:
                return str(reason).strip().casefold()
    return None


def validate_structured_response(value: Any) -> TechnicalEntityExtraction:
    """Valida la envoltura de salida y detecta respuestas truncadas."""

    if isinstance(value, TechnicalEntityExtraction):
        return value

    parsed: Any = value
    if isinstance(value, Mapping) and {"raw", "parsed", "parsing_error"} & value.keys():
        reason = _finish_reason(value.get("raw"))
        if reason in TRUNCATED_FINISH_REASONS:
            logger.warning("Salida estructurada truncada", extra={"finish_reason": reason})
            raise StructuredOutputError("El modelo agotó el límite antes de completar la salida")

        parsing_error = value.get("parsing_error")
        if parsing_error is not None:
            logger.warning(
                "Fallo de validación de salida estructurada",
                extra={"error_type": type(parsing_error).__name__},
            )
            raise StructuredOutputError("El modelo devolvió una salida incompleta o mal formada")
        parsed = value.get("parsed")

    if parsed is None:
        logger.warning("El modelo no devolvió una salida estructurada")
        raise StructuredOutputError("El modelo no devolvió una salida estructurada")

    try:
        return TechnicalEntityExtraction.model_validate(parsed)
    except ValidationError as exc:
        logger.warning(
            "La salida no cumple el contrato técnico",
            extra={"validation_error_count": exc.error_count()},
        )
        raise StructuredOutputError("La salida no cumple el contrato técnico") from exc


def build_extraction_chain(settings: Settings) -> Any:
    """Compone prompt, modelo estructurado y política de reintentos."""

    try:
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.runnables import RunnableLambda
    except ImportError as exc:  # pragma: no cover - depende del entorno
        raise RuntimeError("Instala la dependencia 'langchain-core'") from exc

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", "Texto técnico:\n{text}"),
        ]
    )
    model = build_chat_model(settings)
    structured_model = model.with_structured_output(
        TechnicalEntityExtraction,
        include_raw=True,
    )
    chain = prompt | structured_model | RunnableLambda(validate_structured_response)
    return chain.with_retry(
        retry_if_exception_type=(Exception,),
        wait_exponential_jitter=True,
        stop_after_attempt=max(2, settings.llm_max_retries),
    )


async def process_text(
    text: str,
    settings: Settings | None = None,
) -> TechnicalEntityExtraction:
    """Construye y ejecuta el pipeline mediante ``ainvoke``."""

    from incident_intelligence.extraction.service import EntityExtractionService

    active_settings = settings or Settings()
    chain = build_extraction_chain(active_settings)
    return await EntityExtractionService(chain).process_text(text)


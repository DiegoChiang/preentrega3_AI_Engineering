"""Servicio asíncrono del pipeline de extracción."""

import logging
from typing import Any, Protocol

from pydantic import ValidationError

from incident_intelligence.errors import StructuredOutputError
from incident_intelligence.extraction.schemas import TechnicalEntityExtraction

logger = logging.getLogger(__name__)


class AsyncRunnable(Protocol):
    async def ainvoke(self, value: dict[str, Any]) -> Any: ...


class EntityExtractionService:
    def __init__(self, chain: AsyncRunnable) -> None:
        self._chain = chain

    async def process_text(self, text: str) -> TechnicalEntityExtraction:
        clean = text.strip()
        if not clean:
            raise ValueError("El texto de entrada no puede estar vacío")

        logger.info("Iniciando extracción técnica", extra={"input_length": len(clean)})
        try:
            result = await self._chain.ainvoke({"text": clean})
            if isinstance(result, TechnicalEntityExtraction):
                parsed = result
            else:
                parsed = TechnicalEntityExtraction.model_validate(result)
        except ValidationError as exc:
            logger.warning("La salida no cumple el esquema", exc_info=exc)
            raise StructuredOutputError("La salida no cumple el esquema técnico") from exc
        except StructuredOutputError:
            raise
        except Exception as exc:
            logger.exception("Falló el pipeline de extracción")
            raise StructuredOutputError(f"No se pudo extraer información técnica: {exc}") from exc

        logger.info(
            "Extracción técnica completada",
            extra={"technology_count": len(parsed.technologies), "criticality": parsed.criticality},
        )
        return parsed


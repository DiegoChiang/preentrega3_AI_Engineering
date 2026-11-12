import logging

import pytest

from incident_intelligence.errors import StructuredOutputError
from incident_intelligence.extraction.schemas import Criticality
from incident_intelligence.extraction.service import EntityExtractionService


class FakeChain:
    def __init__(self, result: object) -> None:
        self.result = result
        self.received: dict[str, str] | None = None

    async def ainvoke(self, value: dict[str, str]) -> object:
        self.received = value
        return self.result


class FailingChain:
    def __init__(self, error: Exception) -> None:
        self.error = error

    async def ainvoke(self, value: dict[str, str]) -> object:
        raise self.error


@pytest.mark.asyncio
async def test_process_text_returns_validated_result(caplog: pytest.LogCaptureFixture) -> None:
    chain = FakeChain(
        {
            "technologies": ["Kubernetes", "Redis"],
            "criticality": "alta",
            "technical_summary": "Redis agotó memoria y la API devuelve errores 503.",
        }
    )
    service = EntityExtractionService(chain)

    with caplog.at_level(logging.INFO):
        result = await service.process_text("  Redis agotó memoria en Kubernetes.  ")

    assert chain.received == {"text": "Redis agotó memoria en Kubernetes."}
    assert result.criticality is Criticality.HIGH
    assert result.technologies == ["Kubernetes", "Redis"]
    assert "Iniciando extracción técnica" in caplog.text
    assert "Extracción técnica completada" in caplog.text


@pytest.mark.asyncio
async def test_process_text_rejects_invalid_structured_output() -> None:
    service = EntityExtractionService(FakeChain({"technologies": []}))

    with pytest.raises(StructuredOutputError):
        await service.process_text("Fallo de Redis")


@pytest.mark.asyncio
async def test_process_text_rejects_empty_input() -> None:
    service = EntityExtractionService(FakeChain({}))

    with pytest.raises(ValueError, match="vacío"):
        await service.process_text("   ")


@pytest.mark.asyncio
async def test_process_text_preserves_structured_output_error() -> None:
    expected = StructuredOutputError("salida truncada")
    service = EntityExtractionService(FailingChain(expected))

    with pytest.raises(StructuredOutputError) as raised:
        await service.process_text("Fallo de Redis")

    assert raised.value is expected


@pytest.mark.asyncio
async def test_process_text_converts_provider_failure_to_domain_error() -> None:
    service = EntityExtractionService(FailingChain(ConnectionError("sin conexión")))

    with pytest.raises(StructuredOutputError, match="No se pudo extraer"):
        await service.process_text("Fallo de Redis")

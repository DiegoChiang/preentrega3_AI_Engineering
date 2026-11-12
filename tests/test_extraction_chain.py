from types import SimpleNamespace
from typing import Any

import pytest
from langchain_core.runnables import RunnableLambda

import incident_intelligence.extraction.chain as chain_module
from incident_intelligence.config import Settings
from incident_intelligence.errors import StructuredOutputError
from incident_intelligence.extraction.schemas import Criticality, TechnicalEntityExtraction


class FakeChatModel:
    def __init__(self, outputs: list[Any]) -> None:
        self.outputs = iter(outputs)
        self.calls = 0
        self.prompts: list[Any] = []
        self.schema: type[TechnicalEntityExtraction] | None = None
        self.include_raw: bool | None = None

    def with_structured_output(
        self,
        schema: type[TechnicalEntityExtraction],
        *,
        include_raw: bool,
    ) -> RunnableLambda:
        self.schema = schema
        self.include_raw = include_raw

        async def invoke(prompt: Any) -> Any:
            self.calls += 1
            self.prompts.append(prompt)
            return next(self.outputs)

        return RunnableLambda(invoke)


def raw_message(finish_reason: str = "stop") -> SimpleNamespace:
    return SimpleNamespace(response_metadata={"finish_reason": finish_reason})


def valid_extraction() -> TechnicalEntityExtraction:
    return TechnicalEntityExtraction(
        technologies=["Kubernetes", "PostgreSQL"],
        criticality=Criticality.HIGH,
        technical_summary="La API devuelve errores 503 por agotamiento del pool.",
    )


def test_validation_rejects_truncated_model_output() -> None:
    value = {
        "raw": raw_message("length"),
        "parsed": None,
        "parsing_error": None,
    }

    with pytest.raises(StructuredOutputError, match="agotó el límite"):
        chain_module.validate_structured_response(value)


def test_validation_rejects_malformed_structured_output() -> None:
    value = {
        "raw": raw_message(),
        "parsed": None,
        "parsing_error": ValueError("JSON incompleto"),
    }

    with pytest.raises(StructuredOutputError, match="mal formada"):
        chain_module.validate_structured_response(value)


def test_validation_rejects_missing_and_invalid_parsed_output() -> None:
    with pytest.raises(StructuredOutputError, match="no devolvió"):
        chain_module.validate_structured_response(
            {"raw": raw_message(), "parsed": None, "parsing_error": None}
        )

    with pytest.raises(StructuredOutputError, match="contrato técnico"):
        chain_module.validate_structured_response(
            {
                "tecnologias": [],
                "nivel_de_criticidad": "desconocida",
                "resumen_tecnico": "corto",
            }
        )


def test_validation_accepts_an_already_valid_model() -> None:
    result = valid_extraction()

    assert chain_module.validate_structured_response(result) is result


@pytest.mark.asyncio
async def test_chain_uses_lcel_and_retries_invalid_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = FakeChatModel(
        [
            {
                "raw": raw_message(),
                "parsed": None,
                "parsing_error": ValueError("JSON incompleto"),
            },
            {
                "raw": raw_message(),
                "parsed": valid_extraction(),
                "parsing_error": None,
            },
        ]
    )
    monkeypatch.setattr(chain_module, "build_chat_model", lambda settings: model)
    settings = Settings(_env_file=None, llm_max_retries=1)
    chain = chain_module.build_extraction_chain(settings)

    result = await chain.ainvoke(
        {"text": "Kubernetes devuelve 503 porque PostgreSQL agotó el pool."}
    )

    assert result == valid_extraction()
    assert model.calls == 2
    assert model.schema is TechnicalEntityExtraction
    assert model.include_raw is True
    messages = model.prompts[0].to_messages()
    assert "analista de incidentes" in messages[0].content
    assert "Kubernetes devuelve 503" in messages[1].content


@pytest.mark.asyncio
async def test_public_process_text_builds_and_invokes_chain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = valid_extraction()

    class Chain:
        async def ainvoke(self, value: dict[str, str]) -> TechnicalEntityExtraction:
            assert value == {"text": "Redis agotó la memoria."}
            return expected

    settings = Settings(_env_file=None)
    monkeypatch.setattr(
        chain_module,
        "build_extraction_chain",
        lambda received: Chain() if received is settings else None,
    )

    result = await chain_module.process_text(" Redis agotó la memoria. ", settings)

    assert result is expected

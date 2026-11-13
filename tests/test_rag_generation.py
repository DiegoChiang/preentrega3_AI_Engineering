import json

import pytest
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from incident_intelligence.config import Settings
from incident_intelligence.rag.generation import (
    UNKNOWN_ANSWER,
    LangChainAnswerGenerator,
    build_answer_generator,
)


@pytest.mark.asyncio
async def test_lcel_chain_parses_pydantic_output_and_includes_truth_filter(
    monkeypatch,
) -> None:
    received_prompts = []

    def fake_model(prompt):
        received_prompts.append(prompt)
        return AIMessage(
            content=json.dumps(
                {
                    "answer": "Las claves se conservan durante 24 horas.",
                    "grounded": True,
                    "confidence": 0.94,
                    "cited_chunk_ids": ["chunk-redis"],
                }
            )
        )

    monkeypatch.setattr(
        "incident_intelligence.rag.generation.build_chat_model",
        lambda settings: RunnableLambda(fake_model),
    )
    generator = build_answer_generator(Settings(_env_file=None, llm_max_retries=2))

    result = await generator.generate(
        query="¿Cuánto duran las claves?",
        context="[SOURCE=redis.txt | FRAGMENT_ID=chunk-redis]\nDuran 24 horas.",
    )

    prompt_text = received_prompts[0].to_string()
    assert result.cited_chunk_ids == ["chunk-redis"]
    assert result.grounded is True
    assert "No lo sé con la información disponible." in prompt_text
    assert "FRAGMENT_ID=chunk-redis" in prompt_text


@pytest.mark.asyncio
async def test_malformed_structured_output_returns_safe_unknown_answer() -> None:
    class MalformedChain:
        async def ainvoke(self, _payload):
            raise OutputParserException("La salida no contiene JSON válido")

    generator = LangChainAnswerGenerator(MalformedChain())

    result = await generator.generate(query="¿Qué ocurrió?", context="Sin evidencia")

    assert result.answer == UNKNOWN_ANSWER
    assert result.grounded is False
    assert result.confidence == 0
    assert result.cited_chunk_ids == []


def test_anthropic_builder_omits_temperature(monkeypatch) -> None:
    from incident_intelligence.integrations import langchain as langchain_module

    captured = {}

    class FakeChatAnthropic:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr("langchain_anthropic.ChatAnthropic", FakeChatAnthropic)
    settings = Settings(
        _env_file=None,
        llm_provider="anthropic",
        anthropic_api_key="test-key",
        llm_temperature=0.9,
    )

    langchain_module.build_chat_model(settings)

    assert captured["model"] == settings.anthropic_model
    assert "temperature" not in captured

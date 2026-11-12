import json

import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from incident_intelligence.config import Settings
from incident_intelligence.rag.generation import build_answer_generator


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

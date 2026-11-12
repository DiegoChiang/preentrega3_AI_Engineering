import pytest
from pydantic import ValidationError

from incident_intelligence.extraction.schemas import Criticality, TechnicalEntityExtraction


def test_extraction_accepts_spanish_aliases_and_serializes_them() -> None:
    result = TechnicalEntityExtraction.model_validate(
        {
            "tecnologias": [" Kubernetes ", "kubernetes", "PostgreSQL"],
            "nivel_de_criticidad": "alta",
            "resumen_tecnico": "  La API dejó de responder por agotamiento del pool.  ",
        }
    )

    assert result.technologies == ["Kubernetes", "PostgreSQL"]
    assert result.technical_summary == "La API dejó de responder por agotamiento del pool."
    assert result.model_dump(mode="json", by_alias=True) == {
        "tecnologias": ["Kubernetes", "PostgreSQL"],
        "nivel_de_criticidad": "alta",
        "resumen_tecnico": "La API dejó de responder por agotamiento del pool.",
    }


def test_extraction_rejects_blank_summary() -> None:
    with pytest.raises(ValidationError):
        TechnicalEntityExtraction(
            technologies=["Redis"],
            criticality=Criticality.MEDIUM,
            technical_summary="          ",
        )


def test_extraction_rejects_technologies_that_normalize_to_empty() -> None:
    with pytest.raises(ValidationError):
        TechnicalEntityExtraction(
            technologies=[" ", "  "],
            criticality=Criticality.LOW,
            technical_summary="No se identificó impacto operativo.",
        )


def test_extraction_normalizes_and_deduplicates_technologies() -> None:
    result = TechnicalEntityExtraction(
        technologies=[" PostgreSQL ", "postgresql", "Redis"],
        criticality=Criticality.HIGH,
        technical_summary="La base de datos dejó de aceptar conexiones.",
    )

    assert result.technologies == ["PostgreSQL", "Redis"]
    assert result.criticality is Criticality.HIGH


def test_extraction_forbids_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        TechnicalEntityExtraction.model_validate(
            {
                "technologies": ["Redis"],
                "criticality": "media",
                "technical_summary": "Redis presenta una degradación parcial.",
                "unsupported": True,
            }
        )

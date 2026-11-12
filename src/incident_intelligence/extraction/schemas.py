"""Contrato de salida del pipeline técnico."""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Criticality(StrEnum):
    LOW = "baja"
    MEDIUM = "media"
    HIGH = "alta"


class TechnicalEntityExtraction(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    technologies: list[str] = Field(
        alias="tecnologias",
        min_length=1,
        max_length=30,
        description="Tecnologías mencionadas explícitamente en el texto de entrada.",
    )
    criticality: Criticality = Field(
        alias="nivel_de_criticidad",
        description="Impacto operativo clasificado como baja, media o alta.",
    )
    technical_summary: str = Field(
        alias="resumen_tecnico",
        min_length=10,
        max_length=2000,
        description="Resumen técnico sustentado únicamente por el texto de entrada.",
    )

    @field_validator("technologies")
    @classmethod
    def normalize_technologies(cls, values: list[str]) -> list[str]:
        normalized: list[str] = []
        seen: set[str] = set()
        for value in values:
            clean = value.strip()
            if clean and clean.casefold() not in seen:
                normalized.append(clean)
                seen.add(clean.casefold())
        if not normalized:
            raise ValueError("Debe existir al menos una tecnología")
        return normalized

    @field_validator("technical_summary")
    @classmethod
    def normalize_summary(cls, value: str) -> str:
        clean = value.strip()
        if len(clean) < 10:
            raise ValueError("El resumen técnico debe contener al menos 10 caracteres")
        return clean


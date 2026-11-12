"""Errores controlados del dominio."""


class IncidentIntelligenceError(Exception):
    """Error base del sistema."""


class StructuredOutputError(IncidentIntelligenceError):
    """La respuesta no cumple el contrato de datos."""


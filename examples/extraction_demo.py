"""Ejecuta el pipeline asíncrono con una descripción técnica de ejemplo."""

import asyncio
import json
import logging

from incident_intelligence.extraction.chain import process_text

EXAMPLE_TEXT = """
El deployment de la API de pagos en Kubernetes devuelve errores 503 para todos los clientes.
Los logs muestran que PostgreSQL agotó el pool de conexiones después de un aumento de tráfico.
"""


async def run() -> None:
    result = await process_text(EXAMPLE_TEXT)
    print(
        json.dumps(
            result.model_dump(mode="json", by_alias=True),
            ensure_ascii=False,
            indent=2,
        )
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    asyncio.run(run())


if __name__ == "__main__":
    main()

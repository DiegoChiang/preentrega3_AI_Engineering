"""Extracción técnica estructurada desde texto o archivo."""

import argparse
import asyncio
import json
import logging
from pathlib import Path

from incident_intelligence.extraction.chain import process_text


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extrae tecnologías, criticidad y resumen")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--text", help="Texto técnico para analizar")
    source.add_argument("--file", type=Path, help="Archivo UTF-8 para analizar")
    return parser


async def run(text: str) -> None:
    result = await process_text(text)
    print(
        json.dumps(
            result.model_dump(mode="json", by_alias=True),
            ensure_ascii=False,
            indent=2,
        )
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = build_parser().parse_args()
    text = args.text if args.text is not None else args.file.read_text(encoding="utf-8")
    asyncio.run(run(text))


if __name__ == "__main__":
    main()

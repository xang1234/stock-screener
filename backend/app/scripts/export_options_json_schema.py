"""Export the Pydantic-owned Options Command Center wire contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from app.schemas.options_analytics import (
    OptionsCommandCenterResponse,
    OptionsSymbolDetailResponse,
    StaticOptionsManifest,
)

OPTIONS_WIRE_SCHEMA_VERSION = "options-wire-schema-v1"


def build_options_schema() -> dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "schema_version": OPTIONS_WIRE_SCHEMA_VERSION,
        "models": {
            "manifest": StaticOptionsManifest.model_json_schema(),
            "command_center": OptionsCommandCenterResponse.model_json_schema(),
            "symbol_detail": OptionsSymbolDetailResponse.model_json_schema(),
        },
    }


def default_output_path() -> Path:
    return (
        Path(__file__).resolve().parents[3]
        / "frontend"
        / "src"
        / "features"
        / "options"
        / "optionsSchema.json"
    )


def write_options_schema(path: Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(build_options_schema(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=default_output_path())
    args = parser.parse_args()
    write_options_schema(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

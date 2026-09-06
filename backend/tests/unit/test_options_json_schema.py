import json
from pathlib import Path

from app.scripts.export_options_json_schema import build_options_schema


def test_committed_options_schema_matches_pydantic_models() -> None:
    schema_path = (
        Path(__file__).resolve().parents[3]
        / "frontend"
        / "src"
        / "features"
        / "options"
        / "optionsSchema.json"
    )

    assert json.loads(schema_path.read_text(encoding="utf-8")) == build_options_schema()

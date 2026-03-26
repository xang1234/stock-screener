from types import SimpleNamespace

from app.schemas.theme import ContentSourceResponse


def test_content_source_response_parses_legacy_json_string_pipelines():
    source = SimpleNamespace(
        id=1,
        name="Legacy source",
        source_type="twitter",
        url="https://x.com/example",
        priority=50,
        fetch_interval_minutes=60,
        is_active=True,
        pipelines='["fundamental"]',
        last_fetched_at=None,
        total_items_fetched=0,
        created_at=None,
        updated_at=None,
    )

    result = ContentSourceResponse.model_validate(source)

    assert result.pipelines == ["fundamental"]

from app.config.settings import Settings


def test_zai_api_keys_list_prefers_multi_key_field() -> None:
    settings = Settings(zai_api_keys=" key-a, key-b ", zai_api_key="single-key")

    assert settings.zai_api_keys_list == ["key-a", "key-b"]


def test_zai_api_keys_list_falls_back_to_single_key() -> None:
    settings = Settings(zai_api_key="single-key")

    assert settings.zai_api_keys_list == ["single-key"]


def test_universe_source_timeout_seconds_must_be_positive() -> None:
    try:
        Settings(universe_source_timeout_seconds=0)
    except ValueError as exc:
        assert "universe_source_timeout_seconds must be > 0" in str(exc)
    else:
        raise AssertionError("Expected invalid universe_source_timeout_seconds to fail")

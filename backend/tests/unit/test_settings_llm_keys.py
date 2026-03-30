from app.config.settings import Settings


def test_zai_api_keys_list_prefers_multi_key_field() -> None:
    settings = Settings(zai_api_keys=" key-a, key-b ", zai_api_key="single-key")

    assert settings.zai_api_keys_list == ["key-a", "key-b"]


def test_zai_api_keys_list_falls_back_to_single_key() -> None:
    settings = Settings(zai_api_key="single-key")

    assert settings.zai_api_keys_list == ["single-key"]

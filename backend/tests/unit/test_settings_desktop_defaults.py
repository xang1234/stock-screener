from app.config.settings import Settings


def test_desktop_mode_skips_heavy_bootstrap_defaults(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))

    settings = Settings(desktop_mode=True)

    assert settings.desktop_bootstrap_refresh_universe is False
    assert settings.desktop_bootstrap_fundamentals_limit == 0


def test_desktop_mode_preserves_explicit_bootstrap_overrides(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    monkeypatch.setenv("DESKTOP_BOOTSTRAP_REFRESH_UNIVERSE", "true")
    monkeypatch.setenv("DESKTOP_BOOTSTRAP_FUNDAMENTALS_LIMIT", "25")

    settings = Settings(desktop_mode=True)

    assert settings.desktop_bootstrap_refresh_universe is True
    assert settings.desktop_bootstrap_fundamentals_limit == 25

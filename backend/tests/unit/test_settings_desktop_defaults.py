import importlib

from app.config.settings import Settings

settings_module = importlib.import_module("app.config.settings")


def test_desktop_mode_skips_heavy_bootstrap_defaults(tmp_path, monkeypatch):
    settings = Settings(desktop_mode=True, desktop_data_dir=str(tmp_path / "desktop-data"))

    assert settings.desktop_bootstrap_refresh_universe is False
    assert settings.desktop_bootstrap_fundamentals_limit == 0


def test_desktop_mode_preserves_explicit_bootstrap_overrides(tmp_path, monkeypatch):
    monkeypatch.setenv("DESKTOP_BOOTSTRAP_REFRESH_UNIVERSE", "true")
    monkeypatch.setenv("DESKTOP_BOOTSTRAP_FUNDAMENTALS_LIMIT", "25")

    settings = Settings(desktop_mode=True, desktop_data_dir=str(tmp_path / "desktop-data"))

    assert settings.desktop_bootstrap_refresh_universe is True
    assert settings.desktop_bootstrap_fundamentals_limit == 25


def test_default_desktop_data_dir_uses_user_data_path_when_env_vars_are_missing(tmp_path, monkeypatch):
    monkeypatch.delenv("LOCALAPPDATA", raising=False)
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(settings_module.sys, "platform", "linux")

    data_dir = settings_module._get_default_desktop_data_dir()

    assert data_dir == tmp_path / ".local" / "share" / "StockScanner"

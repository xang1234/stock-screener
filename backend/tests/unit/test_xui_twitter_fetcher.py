"""Unit tests for xui-backed twitter ingestion adapter."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from types import SimpleNamespace

import pytest

from app.config import settings
from app.models.theme import ContentSource
from app.services import xui_twitter_fetcher as xui_fetcher_mod
from app.services.xui_twitter_fetcher import XUITwitterFetcher


@dataclass(frozen=True)
class _FakeSourceRef:
    source_id: str
    kind: str
    value: str
    enabled: bool = True
    label: str | None = None


class _FakeSourceKind:
    LIST = "list"
    USER = "user"


def _set_xui_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "xui_enabled", True)
    monkeypatch.setattr(settings, "xui_profile", "default")
    monkeypatch.setattr(settings, "xui_config_path", "/tmp/xui/config.toml")
    monkeypatch.setattr(settings, "xui_limit_per_source", 50)
    monkeypatch.setattr(settings, "xui_new_only", True)
    monkeypatch.setattr(settings, "xui_checkpoint_mode", "auto")


def test_ensure_xui_module_path_skips_when_repo_root_not_found(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(xui_fetcher_mod, "_repo_root_from_here", lambda: None)
    injected_paths: list[str] = []
    monkeypatch.setattr(xui_fetcher_mod.sys, "path", injected_paths)

    xui_fetcher_mod._ensure_xui_module_path()

    assert injected_paths == []


def test_fetch_user_source_maps_tweet_items(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_xui_defaults(monkeypatch)

    source = ContentSource(name="@alice", source_type="twitter", url="https://twitter.com/alice")
    created_at = datetime(2026, 3, 1, 0, 0, tzinfo=timezone.utc)

    def parse_list_id(_raw: str) -> str:
        raise ValueError("not a list")

    def parse_handle(raw: str) -> str:
        assert raw == "https://x.com/alice"
        return "alice"

    bindings = SimpleNamespace(
        parse_list_id=parse_list_id,
        parse_handle=parse_handle,
        source_ref_cls=_FakeSourceRef,
        source_kind_enum=_FakeSourceKind,
        load_runtime_config=lambda _path: SimpleNamespace(sources=()),
        run_configured_read=lambda config, **_kwargs: SimpleNamespace(
            failed=0,
            outcomes=(SimpleNamespace(ok=True, error=None),),
            items=(
                SimpleNamespace(
                    tweet_id="100",
                    text="hello world",
                    author_handle="@alice",
                    created_at=created_at,
                ),
            ),
        ),
        probe_auth_status=lambda **_kwargs: SimpleNamespace(
            authenticated=True,
            status_code="authenticated",
            message="ok",
        ),
    )
    monkeypatch.setattr("app.services.xui_twitter_fetcher._load_xui_bindings", lambda: bindings)

    fetcher = XUITwitterFetcher()
    result = fetcher.fetch(source, since=None)

    assert len(result) == 1
    assert result[0]["external_id"] == hashlib.md5("twitter:100".encode("utf-8")).hexdigest()
    assert result[0]["content"] == "hello world"
    assert result[0]["url"] == "https://x.com/alice/status/100"
    assert result[0]["author"] == "@alice"
    assert result[0]["published_at"] == created_at


def test_fetch_list_source_uses_web_status_url_when_author_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_xui_defaults(monkeypatch)

    source = ContentSource(
        name="Tech List",
        source_type="twitter",
        url="https://x.com/i/lists/84839422",
    )
    created_at = datetime(2026, 3, 1, 0, 0, tzinfo=timezone.utc)

    bindings = SimpleNamespace(
        parse_list_id=lambda raw: "84839422",
        parse_handle=lambda _raw: (_ for _ in ()).throw(ValueError("not user")),
        source_ref_cls=_FakeSourceRef,
        source_kind_enum=_FakeSourceKind,
        load_runtime_config=lambda _path: SimpleNamespace(sources=()),
        run_configured_read=lambda config, **_kwargs: SimpleNamespace(
            failed=0,
            outcomes=(SimpleNamespace(ok=True, error=None),),
            items=(
                SimpleNamespace(
                    tweet_id="200",
                    text="list item",
                    author_handle=None,
                    created_at=created_at,
                ),
            ),
        ),
        probe_auth_status=lambda **_kwargs: SimpleNamespace(
            authenticated=True,
            status_code="authenticated",
            message="ok",
        ),
    )
    monkeypatch.setattr("app.services.xui_twitter_fetcher._load_xui_bindings", lambda: bindings)

    fetcher = XUITwitterFetcher()
    result = fetcher.fetch(source, since=None)

    assert len(result) == 1
    assert result[0]["url"] == "https://x.com/i/web/status/200"
    assert result[0]["author"] == "Tech List"


def test_fetch_raises_for_unparseable_locator(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_xui_defaults(monkeypatch)

    source = ContentSource(name="broken", source_type="twitter", url="not a twitter source")
    bindings = SimpleNamespace(
        parse_list_id=lambda _raw: (_ for _ in ()).throw(ValueError("bad list")),
        parse_handle=lambda _raw: (_ for _ in ()).throw(ValueError("bad user")),
        source_ref_cls=_FakeSourceRef,
        source_kind_enum=_FakeSourceKind,
        load_runtime_config=lambda _path: SimpleNamespace(sources=()),
        run_configured_read=lambda _config, **_kwargs: None,
        probe_auth_status=lambda **_kwargs: SimpleNamespace(
            authenticated=True,
            status_code="authenticated",
            message="ok",
        ),
    )
    monkeypatch.setattr("app.services.xui_twitter_fetcher._load_xui_bindings", lambda: bindings)

    fetcher = XUITwitterFetcher()
    with pytest.raises(RuntimeError, match="Unable to parse twitter source locator"):
        fetcher.fetch(source, since=None)


def test_fetch_raises_actionable_error_when_auth_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_xui_defaults(monkeypatch)

    source = ContentSource(name="@alice", source_type="twitter", url="@alice")
    bindings = SimpleNamespace(
        parse_list_id=lambda _raw: (_ for _ in ()).throw(ValueError("bad list")),
        parse_handle=lambda _raw: "alice",
        source_ref_cls=_FakeSourceRef,
        source_kind_enum=_FakeSourceKind,
        load_runtime_config=lambda _path: SimpleNamespace(sources=()),
        run_configured_read=lambda _config, **_kwargs: None,
        probe_auth_status=lambda **_kwargs: SimpleNamespace(
            authenticated=False,
            status_code="missing_storage_state",
            message="No storage state found.",
        ),
    )
    monkeypatch.setattr("app.services.xui_twitter_fetcher._load_xui_bindings", lambda: bindings)

    fetcher = XUITwitterFetcher()
    with pytest.raises(RuntimeError, match="xui auth login --profile default --path /tmp/xui/config.toml"):
        fetcher.fetch(source, since=None)


def test_fetch_raises_when_xui_read_outcome_is_failed(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_xui_defaults(monkeypatch)

    source = ContentSource(name="@alice", source_type="twitter", url="@alice")
    bindings = SimpleNamespace(
        parse_list_id=lambda _raw: (_ for _ in ()).throw(ValueError("bad list")),
        parse_handle=lambda _raw: "alice",
        source_ref_cls=_FakeSourceRef,
        source_kind_enum=_FakeSourceKind,
        load_runtime_config=lambda _path: SimpleNamespace(sources=()),
        run_configured_read=lambda _config, **_kwargs: SimpleNamespace(
            failed=1,
            outcomes=(SimpleNamespace(ok=False, error="selector mismatch"),),
            items=(),
        ),
        probe_auth_status=lambda **_kwargs: SimpleNamespace(
            authenticated=True,
            status_code="authenticated",
            message="ok",
        ),
    )
    monkeypatch.setattr("app.services.xui_twitter_fetcher._load_xui_bindings", lambda: bindings)

    fetcher = XUITwitterFetcher()
    with pytest.raises(RuntimeError, match="selector mismatch"):
        fetcher.fetch(source, since=None)


def test_fetch_applies_since_filter(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_xui_defaults(monkeypatch)

    source = ContentSource(name="@alice", source_type="twitter", url="@alice")
    old = datetime(2026, 3, 1, 0, 0, tzinfo=timezone.utc)
    new = datetime(2026, 3, 1, 1, 0, tzinfo=timezone.utc)

    bindings = SimpleNamespace(
        parse_list_id=lambda _raw: (_ for _ in ()).throw(ValueError("bad list")),
        parse_handle=lambda _raw: "alice",
        source_ref_cls=_FakeSourceRef,
        source_kind_enum=_FakeSourceKind,
        load_runtime_config=lambda _path: SimpleNamespace(sources=()),
        run_configured_read=lambda _config, **_kwargs: SimpleNamespace(
            failed=0,
            outcomes=(SimpleNamespace(ok=True, error=None),),
            items=(
                SimpleNamespace(tweet_id="10", text="old", author_handle="@alice", created_at=old),
                SimpleNamespace(tweet_id="11", text="new", author_handle="@alice", created_at=new),
            ),
        ),
        probe_auth_status=lambda **_kwargs: SimpleNamespace(
            authenticated=True,
            status_code="authenticated",
            message="ok",
        ),
    )
    monkeypatch.setattr("app.services.xui_twitter_fetcher._load_xui_bindings", lambda: bindings)

    fetcher = XUITwitterFetcher()
    result = fetcher.fetch(source, since=datetime(2026, 3, 1, 0, 30, tzinfo=timezone.utc))

    assert len(result) == 1
    assert result[0]["content"] == "new"

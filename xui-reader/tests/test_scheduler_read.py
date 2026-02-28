"""Multi-source read orchestration behavior."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from xui_reader.config import RuntimeConfig
from xui_reader.errors import CollectError, SchedulerError
from xui_reader.models import SourceKind, SourceRef, TweetItem
from xui_reader.scheduler.read import collect_source_items, run_configured_read, run_multi_source_read


def test_run_multi_source_read_isolates_source_failures_and_merges_successes() -> None:
    sources = (
        SourceRef(source_id="list:1", kind=SourceKind.LIST, value="1", enabled=True),
        SourceRef(source_id="user:a", kind=SourceKind.USER, value="alice", enabled=True),
    )

    def read_source(source: SourceRef) -> tuple[TweetItem, ...]:
        if source.source_id == "user:a":
            raise RuntimeError("collector failed")
        return (
            _tweet("100", source.source_id, "2026-03-01T00:00:00+00:00"),
            _tweet("101", source.source_id, "2026-03-02T00:00:00+00:00"),
        )

    result = run_multi_source_read(sources, read_source)

    assert result.succeeded == 1
    assert result.failed == 1
    assert [item.tweet_id for item in result.items] == ["101", "100"]
    assert result.outcomes[0].ok is True
    assert result.outcomes[1].ok is False
    assert "collector failed" in str(result.outcomes[1].error)


def test_run_configured_read_rejects_configs_without_enabled_sources() -> None:
    config = RuntimeConfig(
        sources=(
            SourceRef(source_id="list:1", kind=SourceKind.LIST, value="1", enabled=False),
            SourceRef(source_id="user:a", kind=SourceKind.USER, value="alice", enabled=False),
        )
    )

    with pytest.raises(SchedulerError, match="No enabled sources configured"):
        run_configured_read(config, limit=10, read_source=lambda _source: ())


def test_collect_source_items_raises_when_extractor_returns_empty(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    state_path = tmp_path / "storage_state.json"
    state_path.write_text("{}", encoding="utf-8")
    source = SourceRef(source_id="list:1", kind=SourceKind.LIST, value="1", enabled=True)
    config = RuntimeConfig(sources=(source,))

    fallback_item = _tweet("fallback", source.source_id, "2026-03-01T00:00:00+00:00")

    class FakeSession:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            return None

        def __enter__(self) -> object:
            return object()

        def __exit__(self, *_args: object) -> None:
            return None

    class FakeCollector:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            return None

        def collect(self, _source: SourceRef, *, limit: int) -> SimpleNamespace:
            assert limit == 3
            return SimpleNamespace(items=(fallback_item,), dom_snapshots=("<article />",))

    class FakeExtractor:
        def extract(self, _payload: dict[str, object]) -> tuple[TweetItem, ...]:
            return ()

    monkeypatch.setattr("xui_reader.scheduler.read.storage_state_path", lambda *_args, **_kwargs: state_path)
    monkeypatch.setattr("xui_reader.scheduler.read.PlaywrightBrowserSession", FakeSession)
    monkeypatch.setattr("xui_reader.scheduler.read.TimelineCollector", FakeCollector)
    monkeypatch.setattr("xui_reader.scheduler.read.PrimaryFallbackTweetExtractor", lambda: FakeExtractor())

    with pytest.raises(CollectError, match="Extractor produced no items"):
        collect_source_items(config, source, limit=3)


def _tweet(tweet_id: str, source_id: str, created_at: str) -> TweetItem:
    return TweetItem(
        tweet_id=tweet_id,
        created_at=datetime.fromisoformat(created_at).astimezone(timezone.utc),
        author_handle="@a",
        text=tweet_id,
        source_id=source_id,
        is_reply=False,
        is_repost=False,
        is_pinned=False,
        has_quote=False,
        quote_tweet_id=None,
    )

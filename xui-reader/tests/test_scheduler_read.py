"""Multi-source read orchestration behavior."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from xui_reader.config import RuntimeConfig
from xui_reader.diagnostics.events import JsonlEventLogger
from xui_reader.errors import CollectError, SchedulerError
from xui_reader.models import SourceKind, SourceRef, TweetItem
from xui_reader.scheduler.read import (
    SourceReadResult,
    collect_source_items,
    run_configured_read,
    run_multi_source_read,
)


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
            return SimpleNamespace(
                items=(fallback_item,),
                dom_snapshots=("<article />",),
                stats=SimpleNamespace(scroll_rounds=0, observed_ids=1),
            )

    class FakeExtractor:
        def extract(self, _payload: dict[str, object]) -> tuple[TweetItem, ...]:
            return ()

    monkeypatch.setattr("xui_reader.scheduler.read.storage_state_path", lambda *_args, **_kwargs: state_path)
    monkeypatch.setattr("xui_reader.scheduler.read.PlaywrightBrowserSession", FakeSession)
    monkeypatch.setattr("xui_reader.scheduler.read.TimelineCollector", FakeCollector)
    monkeypatch.setattr(
        "xui_reader.scheduler.read.PrimaryFallbackTweetExtractor",
        lambda **_kwargs: FakeExtractor(),
    )

    with pytest.raises(CollectError, match="Extractor produced no items"):
        collect_source_items(config, source, limit=3)


def test_run_configured_read_writes_failure_artifacts_with_run_linkage(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    source = SourceRef(source_id="list:1", kind=SourceKind.LIST, value="1", enabled=True)
    config = RuntimeConfig(sources=(source,))

    def failing_read(_source: SourceRef) -> tuple[TweetItem, ...]:
        raise RuntimeError("selector mismatch")

    result = run_configured_read(
        config,
        config_path=config_path,
        read_source=failing_read,
        run_id="run-123",
        enable_debug_artifacts=True,
    )

    assert result.failed == 1
    outcome = result.outcomes[0]
    assert outcome.html_artifact_path is not None
    assert outcome.selector_report_path is not None
    assert "run-123" in outcome.html_artifact_path
    assert "run-123" in outcome.selector_report_path

    selector_payload = json.loads(Path(outcome.selector_report_path).read_text(encoding="utf-8"))
    assert selector_payload["run_id"] == "run-123"
    assert selector_payload["source_id"] == "list:1"
    assert selector_payload["error"] == "selector mismatch"


def test_run_configured_read_emits_jsonl_event_with_counters(tmp_path: Path) -> None:
    source = SourceRef(source_id="list:1", kind=SourceKind.LIST, value="1", enabled=True)
    config = RuntimeConfig(sources=(source,))
    logger = JsonlEventLogger(tmp_path / "events.jsonl")

    def read_source(_source: SourceRef) -> SourceReadResult:
        return SourceReadResult(
            items=(_tweet("100", "list:1", "2026-03-01T00:00:00+00:00"),),
            page_loads=1,
            scroll_rounds=2,
            observed_ids=3,
            dom_snapshots=1,
        )

    result = run_configured_read(
        config,
        read_source=read_source,
        run_id="read-1",
        event_logger=logger,
    )
    assert result.total_page_loads == 1
    assert result.total_scroll_rounds == 2
    assert result.total_observed_ids == 3

    lines = (tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    event = json.loads(lines[0])
    assert event["event_type"] == "read_run"
    assert event["run_id"] == "read-1"
    assert event["payload"]["page_loads"] == 1
    assert event["payload"]["scroll_rounds"] == 2
    assert event["payload"]["seen_items"] == 3


def test_run_configured_read_new_only_uses_source_checkpoint(tmp_path: Path) -> None:
    source = SourceRef(source_id="list:1", kind=SourceKind.LIST, value="1", enabled=True)
    config = RuntimeConfig(sources=(source,))
    config_path = tmp_path / "config.toml"

    def read_source(_source: SourceRef) -> SourceReadResult:
        return SourceReadResult(
            items=(
                _tweet("100", "list:1", "2026-03-02T00:00:00+00:00"),
                _tweet("99", "list:1", "2026-03-01T00:00:00+00:00"),
            ),
            observed_ids=2,
        )

    first = run_configured_read(
        config,
        config_path=config_path,
        new_only=True,
        read_source=read_source,
    )
    second = run_configured_read(
        config,
        config_path=config_path,
        new_only=True,
        read_source=read_source,
    )

    assert [item.tweet_id for item in first.items] == ["100", "99"]
    assert second.items == ()


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

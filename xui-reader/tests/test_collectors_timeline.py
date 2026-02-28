"""Timeline collector navigation, tab fallback, and bounded loop behavior."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from xui_reader.collectors.timeline import (
    ScrollBounds,
    TimelineCollector,
    canonical_list_url,
    canonical_user_url,
    select_user_tab,
)
from xui_reader.config import RuntimeConfig
from xui_reader.errors import CollectError
from xui_reader.models import SourceKind, SourceRef


class FakeElement:
    def __init__(self, *, visible: bool = True) -> None:
        self._visible = visible
        self.clicks = 0

    def is_visible(self) -> bool:
        return self._visible

    def click(self, timeout: int | None = None) -> None:
        _ = timeout
        self.clicks += 1


@dataclass
class FakeRouteCall:
    pattern: str
    handler: object


class FakePage:
    def __init__(
        self,
        *,
        contents: list[str],
        selectors: dict[str, FakeElement | None] | None = None,
    ) -> None:
        self._contents = contents
        self._selectors = selectors or {}
        self._index = 0
        self._url = "about:blank"
        self.goto_calls: list[str] = []
        self.route_calls: list[FakeRouteCall] = []
        self.scroll_calls = 0

    @property
    def url(self) -> str:
        return self._url

    def goto(self, url: str, **kwargs: object) -> None:
        _ = kwargs
        self._url = url
        self.goto_calls.append(url)

    def content(self) -> str:
        if not self._contents:
            return ""
        if self._index < len(self._contents):
            return self._contents[self._index]
        return self._contents[-1]

    def evaluate(self, expression: str) -> None:
        _ = expression
        self.scroll_calls += 1
        self._index += 1

    def route(self, pattern: str, handler: object) -> None:
        self.route_calls.append(FakeRouteCall(pattern=pattern, handler=handler))

    def query_selector(self, selector: str) -> FakeElement | None:
        return self._selectors.get(selector)

    def title(self) -> str:
        return "Home / X"

    def inner_text(self, selector: str, timeout: int | None = None) -> str:
        _ = (selector, timeout)
        return ""


class FakeSession:
    def __init__(self, page: FakePage) -> None:
        self.page = page
        self.new_page_calls = 0

    def new_page(self) -> FakePage:
        self.new_page_calls += 1
        return self.page


def test_select_user_tab_prefers_href_selector_before_text_fallback() -> None:
    href = FakeElement()
    text = FakeElement()
    page = FakePage(
        contents=[""],
        selectors={
            'nav a[href="/alice/with_replies"]': href,
            'nav a:has-text("Replies")': text,
        },
    )

    result = select_user_tab(page, handle="alice", tab="replies")

    assert result.selected is True
    assert result.strategy == "href"
    assert result.selector == 'nav a[href="/alice/with_replies"]'
    assert href.clicks == 1
    assert text.clicks == 0


def test_select_user_tab_falls_back_to_text_when_href_missing() -> None:
    text = FakeElement()
    page = FakePage(
        contents=[""],
        selectors={'nav a:has-text("Media")': text},
    )

    result = select_user_tab(page, handle="alice", tab="media")

    assert result.selected is True
    assert result.strategy == "text"
    assert result.selector == 'nav a:has-text("Media")'
    assert text.clicks == 1


def test_select_user_tab_returns_actionable_diagnostics_on_failure() -> None:
    page = FakePage(contents=[""])
    result = select_user_tab(page, handle="alice", tab="replies")

    assert result.selected is False
    assert "Could not select user tab 'replies'" in result.diagnostics
    assert "Tried selectors:" in result.diagnostics


def test_timeline_collector_list_collection_stops_on_stagnation() -> None:
    page = FakePage(
        contents=[
            '<a href="/alice/status/10">one</a><a href="/alice/status/11">two</a>',
            '<a href="/alice/status/10">one</a><a href="/alice/status/11">two</a>',
            '<a href="/alice/status/10">one</a><a href="/alice/status/11">two</a>',
        ],
    )
    session = FakeSession(page)
    route_calls = 0
    overlay_calls = 0

    def fake_route(page_obj: object, config: object) -> None:
        nonlocal route_calls
        _ = (page_obj, config)
        route_calls += 1

    def fake_overlay(page_obj: object, max_clicks: int = 3) -> None:
        nonlocal overlay_calls
        _ = (page_obj, max_clicks)
        overlay_calls += 1

    collector = TimelineCollector(
        RuntimeConfig(),
        session,
        bounds=ScrollBounds(max_scrolls=10, stagnation_rounds=2),
        route_configurer=fake_route,
        overlay_dismisser=fake_overlay,
    )
    source = SourceRef(source_id="list:84839422", kind=SourceKind.LIST, value="84839422")

    result = collector.collect(source)

    assert route_calls == 1
    assert overlay_calls == 3
    assert page.goto_calls == ["https://x.com/i/lists/84839422"]
    assert result.stats is not None
    assert result.stats.stop_reason == "stagnation"
    assert result.stats.scroll_rounds == 2
    assert result.stats.observed_ids == 2
    assert result.stats.dom_snapshots == 3
    assert [item.tweet_id for item in result.items] == ["10", "11"]


def test_timeline_collector_user_collection_selects_tab_and_stops_on_max_scrolls() -> None:
    media_tab = FakeElement()
    page = FakePage(
        contents=[
            '<a href="/alice/status/1">one</a>',
            '<a href="/alice/status/2">two</a>',
        ],
        selectors={'nav a[href="/alice/media"]': media_tab},
    )
    session = FakeSession(page)
    collector = TimelineCollector(
        RuntimeConfig(),
        session,
        bounds=ScrollBounds(max_scrolls=1, stagnation_rounds=4),
        user_tab="media",
    )
    source = SourceRef(source_id="user:alice", kind=SourceKind.USER, value="@alice")

    result = collector.collect(source)

    assert page.goto_calls == ["https://x.com/alice"]
    assert media_tab.clicks == 1
    assert result.stats is not None
    assert result.stats.stop_reason == "max_scrolls"
    assert result.stats.scroll_rounds == 1
    assert [item.tweet_id for item in result.items] == ["1", "2"]


def test_timeline_collector_raises_when_user_tab_cannot_be_selected() -> None:
    collector = TimelineCollector(
        RuntimeConfig(),
        FakeSession(FakePage(contents=["<div>no tabs</div>"])),
        user_tab="replies",
    )
    source = SourceRef(source_id="user:alice", kind=SourceKind.USER, value="alice")

    with pytest.raises(CollectError, match="Could not select user tab"):
        collector.collect(source)


def test_canonical_url_helpers_validate_inputs() -> None:
    assert canonical_list_url("https://x.com/i/lists/84839422") == "https://x.com/i/lists/84839422"
    assert canonical_user_url("@alice") == "https://x.com/alice"

    with pytest.raises(CollectError, match="Could not parse list id"):
        canonical_list_url("https://x.com/i/lists/not-a-number")
    with pytest.raises(CollectError, match="Invalid handle"):
        canonical_user_url("https://x.com/not-valid-handle/")

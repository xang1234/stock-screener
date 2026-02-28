"""Primary/fallback tweet extraction behavior."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from xui_reader.errors import ExtractError
from xui_reader.extract.tweets import PrimaryFallbackTweetExtractor


PRIMARY_HTML = """
<section>
  <article data-testid="tweet">
    <a href="/alice/status/111">status</a>
    <time datetime="2026-02-28T10:20:30Z"></time>
    <div data-testid="tweetText">Alpha <b>tweet</b> text</div>
    <div data-testid="reply">Replying to @bob</div>
    <div data-testid="socialContext">Alice reposted</div>
    <svg aria-label="Pinned Tweet"></svg>
    <article data-testid="tweet">
      <a href="/charlie/status/222">quoted</a>
      <div data-testid="tweetText">Quoted text</div>
    </article>
  </article>
</section>
"""

FALLBACK_HTML = """
<div>
  <a href="/bob/status/333">status only</a>
  <time datetime="2026-02-28T11:22:33Z"></time>
  <div data-testid="tweetText">Fallback path text</div>
  <div data-testid="reply">Replying to @alice</div>
  <div data-testid="socialContext">Bob reposted</div>
  <svg aria-label="Pinned Tweet"></svg>
  <a href="/dana/status/444">quoted link</a>
</div>
"""


def test_extract_primary_article_first_with_metadata() -> None:
    extractor = PrimaryFallbackTweetExtractor()

    items = extractor.extract({"html": PRIMARY_HTML, "source_id": "list:demo"})

    assert len(items) == 1
    item = items[0]
    assert item.tweet_id == "111"
    assert item.author_handle == "@alice"
    assert item.created_at == datetime(2026, 2, 28, 10, 20, 30, tzinfo=timezone.utc)
    assert item.text == "Alpha tweet text"
    assert item.source_id == "list:demo"
    assert item.is_reply is True
    assert item.is_repost is True
    assert item.is_pinned is True
    assert item.has_quote is True
    assert item.quote_tweet_id == "222"


def test_extract_fallback_link_first_with_metadata_consistency() -> None:
    extractor = PrimaryFallbackTweetExtractor()

    items = extractor.extract({"html": FALLBACK_HTML, "source_id": "user:bob"})

    assert len(items) >= 1
    item = items[0]
    assert item.tweet_id == "333"
    assert item.author_handle == "@bob"
    assert item.created_at == datetime(2026, 2, 28, 11, 22, 33, tzinfo=timezone.utc)
    assert item.text == "Fallback path text"
    assert item.is_reply is True
    assert item.is_repost is True
    assert item.is_pinned is True
    assert item.has_quote is True
    assert item.quote_tweet_id == "444"


def test_extract_returns_explicit_nulls_when_fields_missing() -> None:
    extractor = PrimaryFallbackTweetExtractor()

    items = extractor.extract({"html": '<a href="/eve/status/555">minimal</a>', "source_id": "src"})

    assert len(items) == 1
    item = items[0]
    assert item.tweet_id == "555"
    assert item.author_handle == "@eve"
    assert item.created_at is None
    assert item.text is None
    assert item.is_reply is False
    assert item.is_repost is False
    assert item.is_pinned is False
    assert item.has_quote is False
    assert item.quote_tweet_id is None


def test_extract_honors_selector_override_for_text() -> None:
    html = """
<article data-testid="tweet">
  <a href="/alice/status/777">status</a>
  <span data-custom="txt">Override selector text</span>
</article>
"""
    extractor = PrimaryFallbackTweetExtractor(
        override_data={"tweet.text": 'span[data-custom="txt"]'}
    )

    items = extractor.extract({"html": html, "source_id": "src"})
    assert len(items) == 1
    assert items[0].text == "Override selector text"


def test_extract_dedupes_tweet_ids_across_dom_snapshots() -> None:
    extractor = PrimaryFallbackTweetExtractor()
    payload = {
        "source_id": "src",
        "dom_snapshots": [
            '<a href="/alice/status/888">a</a>',
            '<a href="/alice/status/888">duplicate</a><a href="/bob/status/999">b</a>',
        ],
    }

    items = extractor.extract(payload)

    assert [item.tweet_id for item in items] == ["888", "999"]


def test_extract_uses_html_when_dom_snapshots_empty() -> None:
    extractor = PrimaryFallbackTweetExtractor()
    payload = {
        "source_id": "src",
        "dom_snapshots": [],
        "html": '<a href="/alice/status/123">x</a>',
    }

    items = extractor.extract(payload)

    assert [item.tweet_id for item in items] == ["123"]


def test_extract_fallback_does_not_leak_fields_between_status_links() -> None:
    extractor = PrimaryFallbackTweetExtractor()
    html = """
<div>
  <a href="/alice/status/111">one</a>
  <time datetime="2026-01-01T00:00:00Z"></time>
  <div data-testid="tweetText">FIRST</div>
  <a href="/bob/status/222">two</a>
  <time datetime="2026-01-02T00:00:00Z"></time>
  <div data-testid="tweetText">SECOND</div>
</div>
"""

    items = extractor.extract({"html": html, "source_id": "src"})

    assert [item.tweet_id for item in items] == ["111", "222"]
    assert items[0].text == "FIRST"
    assert items[1].text == "SECOND"
    assert items[0].created_at == datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    assert items[1].created_at == datetime(2026, 1, 2, 0, 0, 0, tzinfo=timezone.utc)


def test_extract_honors_selector_override_for_time() -> None:
    extractor = PrimaryFallbackTweetExtractor(
        override_data={"tweet.time": 'time[data-role="preferred"]'}
    )
    html = """
<article data-testid="tweet">
  <a href="/alice/status/321">status</a>
  <time datetime="2024-01-01T00:00:00Z"></time>
  <time data-role="preferred" datetime="2026-01-01T00:00:00Z"></time>
</article>
"""

    items = extractor.extract({"html": html, "source_id": "src"})

    assert len(items) == 1
    assert items[0].created_at == datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)


def test_extract_applies_bounded_expansion_from_payload() -> None:
    extractor = PrimaryFallbackTweetExtractor(max_expansions=2)
    payload = {
        "source_id": "src",
        "html": '<a href="/alice/status/101">x</a><div data-testid="tweetText">Needs more...</div>',
        "expanded_text_by_id": {"101": "Expanded\u3000text"},
        "max_expansions": 1,
    }

    items = extractor.extract(payload)

    assert len(items) == 1
    assert items[0].tweet_id == "101"
    assert items[0].text == "Expanded text"


def test_extract_warns_on_invalid_max_expansions_override() -> None:
    extractor = PrimaryFallbackTweetExtractor()
    payload = {
        "source_id": "src",
        "html": '<a href="/alice/status/202">ok</a>',
        "max_expansions": -3,
    }

    result = extractor.extract_with_warnings(payload)

    assert [item.tweet_id for item in result.items] == ["202"]
    assert any("max_expansions" in warning for warning in result.warnings)


def test_extract_rejects_unsupported_payload_types() -> None:
    extractor = PrimaryFallbackTweetExtractor()

    with pytest.raises(ExtractError, match="Unsupported extractor payload type"):
        extractor.extract(123)

    with pytest.raises(ExtractError, match="must contain `html` string or `dom_snapshots`"):
        extractor.extract({"source_id": "src"})

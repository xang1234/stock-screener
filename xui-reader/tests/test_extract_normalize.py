"""Normalization and bounded-expansion behavior."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from xui_reader.extract.normalize import TweetNormalizer
from xui_reader.models import TweetItem

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "normalization_edge_cases.json"


def test_normalizer_edge_case_fixture() -> None:
    payload = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    normalizer = TweetNormalizer()

    for case in payload["cases"]:
        items = tuple(_item_from_dict(entry) for entry in case["input"])
        result = normalizer.normalize(
            items,
            expanded_text_by_id=case.get("expanded_text_by_id"),
            max_expansions=int(case.get("max_expansions", 0)),
        )
        expected = case["expected"]
        by_id = {item.tweet_id: item for item in result.items}

        assert [item.tweet_id for item in result.items] == expected["tweet_ids"]
        for tweet_id in expected["tweet_ids"]:
            item = by_id[tweet_id]
            assert item.text == expected["texts"][tweet_id]
            assert item.author_handle == expected["author_handles"][tweet_id]
            assert item.quote_tweet_id == expected["quote_ids"][tweet_id]
            assert item.is_reply is expected["is_reply"][tweet_id]
            assert item.is_repost is expected["is_repost"][tweet_id]
            assert item.is_pinned is expected["is_pinned"][tweet_id]
            assert item.has_quote is expected["has_quote"][tweet_id]

            raw_created_at = expected["created_at"][tweet_id]
            if raw_created_at is None:
                assert item.created_at is None
            else:
                assert item.created_at is not None
                assert item.created_at.isoformat() == raw_created_at

        assert result.dropped_invalid_ids == expected["dropped_invalid_ids"]
        assert result.deduped_count == expected["deduped_count"]
        assert result.expansions_applied == expected["expansions_applied"]
        assert result.warnings


def _item_from_dict(payload: dict[str, Any]) -> TweetItem:
    raw_created_at = payload.get("created_at")
    created_at = datetime.fromisoformat(raw_created_at) if isinstance(raw_created_at, str) else None
    return TweetItem(
        tweet_id=str(payload.get("tweet_id", "")),
        created_at=created_at,
        author_handle=payload.get("author_handle"),
        text=payload.get("text"),
        source_id=str(payload.get("source_id", "unknown")),
        is_reply=payload.get("is_reply"),
        is_repost=payload.get("is_repost"),
        is_pinned=payload.get("is_pinned"),
        has_quote=payload.get("has_quote"),
        quote_tweet_id=payload.get("quote_tweet_id"),
    )


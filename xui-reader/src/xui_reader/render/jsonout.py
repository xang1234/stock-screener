"""JSON and JSONL tweet rendering."""

from __future__ import annotations

import json

from xui_reader.models import TweetItem


def render_json(items: tuple[TweetItem, ...]) -> str:
    return json.dumps([tweet_item_to_dict(item) for item in items], indent=2, sort_keys=True)


def render_jsonl(items: tuple[TweetItem, ...]) -> str:
    return "\n".join(json.dumps(tweet_item_to_dict(item), sort_keys=True) for item in items)


def tweet_item_to_dict(item: TweetItem) -> dict[str, object]:
    return {
        "tweet_id": item.tweet_id,
        "created_at": item.created_at.isoformat() if item.created_at else None,
        "author_handle": item.author_handle,
        "text": item.text,
        "source_id": item.source_id,
        "is_reply": item.is_reply,
        "is_repost": item.is_repost,
        "is_pinned": item.is_pinned,
        "has_quote": item.has_quote,
        "quote_tweet_id": item.quote_tweet_id,
    }

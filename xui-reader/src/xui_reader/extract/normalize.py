"""Tweet normalization helpers for deterministic downstream behavior."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
import re
import unicodedata

from xui_reader.models import TweetItem

_STATUS_PATH_ID_RE = re.compile(r"/status/(\d+)")
_DIGITS_ONLY_RE = re.compile(r"^\d+$")
_ZERO_WIDTH_RE = re.compile(r"[\u200b\u200c\u200d\ufeff]")
_TRUNCATED_TEXT_RE = re.compile(r"(?:\.\.\.|…)\s*$")


@dataclass(frozen=True)
class TweetNormalizationResult:
    items: tuple[TweetItem, ...]
    warnings: tuple[str, ...] = ()
    dropped_invalid_ids: int = 0
    deduped_count: int = 0
    expansions_applied: int = 0


class TweetNormalizer:
    """Normalize ids/text/timestamps and enforce stable in-run dedupe."""

    def normalize(
        self,
        items: Sequence[TweetItem],
        *,
        expanded_text_by_id: Mapping[str, str] | None = None,
        max_expansions: int = 0,
    ) -> TweetNormalizationResult:
        if max_expansions < 0:
            raise ValueError("max_expansions must be >= 0.")

        expansion_map = {
            normalized_id: normalized_text
            for raw_id, raw_text in (expanded_text_by_id or {}).items()
            for normalized_id, normalized_text in [(_normalize_tweet_id(raw_id), _normalize_text(raw_text))]
            if normalized_id is not None and normalized_text is not None
        }

        warnings: list[str] = []
        normalized_items: list[TweetItem] = []
        seen_ids: set[str] = set()
        dropped_invalid_ids = 0
        deduped_count = 0
        expansions_applied = 0

        for item in items:
            normalized_id = _normalize_tweet_id(item.tweet_id)
            if normalized_id is None:
                dropped_invalid_ids += 1
                warnings.append(f"Dropped item with invalid tweet_id '{item.tweet_id}'.")
                continue

            if normalized_id in seen_ids:
                deduped_count += 1
                continue
            seen_ids.add(normalized_id)

            normalized_text = _normalize_text(item.text)
            if (
                max_expansions > 0
                and expansions_applied < max_expansions
                and _is_truncated_text(normalized_text)
            ):
                expanded_text = expansion_map.get(normalized_id)
                if expanded_text:
                    normalized_text = expanded_text
                    expansions_applied += 1

            normalized_quote_id = _normalize_tweet_id(item.quote_tweet_id)
            has_quote = bool(item.has_quote) or normalized_quote_id is not None
            if normalized_quote_id == normalized_id:
                normalized_quote_id = None
                has_quote = False

            normalized_items.append(
                TweetItem(
                    tweet_id=normalized_id,
                    created_at=_normalize_datetime(item.created_at),
                    author_handle=_normalize_author_handle(item.author_handle),
                    text=normalized_text,
                    source_id=_normalize_source_id(item.source_id),
                    is_reply=bool(item.is_reply),
                    is_repost=bool(item.is_repost),
                    is_pinned=bool(item.is_pinned),
                    has_quote=has_quote,
                    quote_tweet_id=normalized_quote_id,
                )
            )

        return TweetNormalizationResult(
            items=tuple(normalized_items),
            warnings=tuple(warnings),
            dropped_invalid_ids=dropped_invalid_ids,
            deduped_count=deduped_count,
            expansions_applied=expansions_applied,
        )


def _normalize_tweet_id(raw: object) -> str | None:
    if raw is None:
        return None
    value = str(raw).strip()
    if not value:
        return None
    if _DIGITS_ONLY_RE.fullmatch(value):
        return value
    status_match = _STATUS_PATH_ID_RE.search(value)
    if status_match is not None:
        return status_match.group(1)
    return None


def _normalize_author_handle(raw: str | None) -> str | None:
    if raw is None:
        return None
    value = raw.strip()
    if not value:
        return None
    stripped = value.lstrip("@")
    return f"@{stripped}" if stripped else None


def _normalize_text(raw: str | None) -> str | None:
    if raw is None:
        return None
    normalized = unicodedata.normalize("NFKC", raw)
    without_zero_width = _ZERO_WIDTH_RE.sub("", normalized)
    collapsed = " ".join(without_zero_width.split())
    return collapsed or None


def _normalize_datetime(raw: datetime | None) -> datetime | None:
    if raw is None:
        return None
    if raw.tzinfo is None:
        return raw.replace(tzinfo=timezone.utc)
    return raw.astimezone(timezone.utc)


def _normalize_source_id(raw: str) -> str:
    value = str(raw).strip()
    return value if value else "unknown"


def _is_truncated_text(text: str | None) -> bool:
    if text is None:
        return False
    return _TRUNCATED_TEXT_RE.search(text) is not None


"""Primary/fallback tweet extraction with selector-pack support."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from html import unescape
import re
from typing import Any

from xui_reader.errors import ExtractError
from xui_reader.extract.normalize import TweetNormalizer
from xui_reader.extract.selectors import (
    SelectorPack,
    SelectorPackResolution,
    resolve_selector_pack,
)
from xui_reader.models import TweetItem

_STATUS_LINK_RE = re.compile(
    r"""href\s*=\s*["'](?:https?://(?:x\.com|twitter\.com))?/([A-Za-z0-9_]{1,32})/status/(\d+)[^"']*["']""",
    re.IGNORECASE,
)
_STATUS_LINK_ID_ONLY_RE = re.compile(
    r"""href\s*=\s*["'](?:https?://(?:x\.com|twitter\.com))?(?:/i/web)?/status/(\d+)[^"']*["']""",
    re.IGNORECASE,
)
_TIME_DATETIME_ATTR_RE = re.compile(r"""\bdatetime\s*=\s*["']([^"']+)["']""", re.IGNORECASE)
_TAG_STRIP_RE = re.compile(r"<[^>]+>")
_ARTICLE_TAG_RE = re.compile(r"</?article\b[^>]*>", re.IGNORECASE)
_REPLY_FALLBACK_RE = re.compile(r"replying to", re.IGNORECASE)
_REPOST_FALLBACK_RE = re.compile(r"reposted", re.IGNORECASE)
_PINNED_FALLBACK_RE = re.compile(r"pinned", re.IGNORECASE)


@dataclass(frozen=True)
class TweetExtractionResult:
    items: tuple[TweetItem, ...]
    warnings: tuple[str, ...] = ()


class PrimaryFallbackTweetExtractor:
    """Extract tweet records via article-first then link-first fallback strategy."""

    def __init__(
        self,
        *,
        override_path: str | None = None,
        override_data: Mapping[str, Any] | None = None,
        selector_resolution: SelectorPackResolution | None = None,
        max_expansions: int = 0,
        normalizer: TweetNormalizer | None = None,
    ) -> None:
        if selector_resolution is not None:
            self._selector_resolution = selector_resolution
        else:
            self._selector_resolution = resolve_selector_pack(
                override_path=override_path,
                override_data=override_data,
            )
        if max_expansions < 0:
            raise ExtractError("max_expansions must be >= 0.")
        self._max_expansions = max_expansions
        self._normalizer = normalizer or TweetNormalizer()

    @property
    def selector_resolution(self) -> SelectorPackResolution:
        return self._selector_resolution

    def extract(self, raw_payload: Any) -> tuple[TweetItem, ...]:
        return self.extract_with_warnings(raw_payload).items

    def extract_with_warnings(self, raw_payload: Any) -> TweetExtractionResult:
        html_docs, source_id = _coerce_payload(raw_payload)
        selectors = self._selector_resolution.selectors
        warnings = list(self._selector_resolution.warnings)
        expanded_text_by_id, payload_max_expansions, payload_warnings = _coerce_expansion_payload(raw_payload)
        warnings.extend(payload_warnings)

        ordered_items: list[TweetItem] = []
        seen_ids: set[str] = set()
        for html in html_docs:
            primary_items = _extract_primary_items(html, source_id, selectors)
            for item in primary_items:
                if item.tweet_id in seen_ids:
                    continue
                ordered_items.append(item)
                seen_ids.add(item.tweet_id)

            if primary_items:
                continue

            for item in _extract_fallback_items(html, source_id, selectors):
                if item.tweet_id in seen_ids:
                    continue
                ordered_items.append(item)
                seen_ids.add(item.tweet_id)

        effective_max_expansions = (
            payload_max_expansions if payload_max_expansions is not None else self._max_expansions
        )
        normalized = self._normalizer.normalize(
            ordered_items,
            expanded_text_by_id=expanded_text_by_id,
            max_expansions=effective_max_expansions,
        )
        warnings.extend(normalized.warnings)
        return TweetExtractionResult(items=normalized.items, warnings=tuple(warnings))


def _coerce_payload(raw_payload: Any) -> tuple[tuple[str, ...], str]:
    source_id = "unknown"
    html_docs: list[str] = []

    if isinstance(raw_payload, str):
        return (raw_payload,), source_id

    if isinstance(raw_payload, Mapping):
        source_id = _coerce_source_id(raw_payload.get("source_id", "unknown"))
        html_value = raw_payload.get("html")
        dom_snapshots = raw_payload.get("dom_snapshots")
        if isinstance(dom_snapshots, Sequence) and not isinstance(dom_snapshots, str):
            html_docs.extend(str(entry) for entry in dom_snapshots if str(entry))
            if not html_docs and isinstance(html_value, str):
                html_docs.append(html_value)
        elif isinstance(html_value, str):
            html_docs.append(html_value)

        if not html_docs:
            raise ExtractError(
                "Extractor payload mapping must contain `html` string or `dom_snapshots` sequence."
            )
        return tuple(html_docs), source_id

    if isinstance(raw_payload, Sequence) and not isinstance(raw_payload, str):
        html_docs.extend(str(entry) for entry in raw_payload if str(entry))
        if html_docs:
            return tuple(html_docs), source_id

    raise ExtractError("Unsupported extractor payload type; expected html string or mapping payload.")


def _coerce_source_id(raw: Any) -> str:
    value = str(raw).strip()
    return value if value else "unknown"


def _coerce_expansion_payload(raw_payload: Any) -> tuple[dict[str, str] | None, int | None, tuple[str, ...]]:
    if not isinstance(raw_payload, Mapping):
        return None, None, ()

    warnings: list[str] = []
    expansion_map: dict[str, str] = {}
    raw_map = raw_payload.get("expanded_text_by_id")
    if raw_map is not None:
        if isinstance(raw_map, Mapping):
            for raw_id, raw_text in raw_map.items():
                expansion_map[str(raw_id)] = str(raw_text)
        else:
            warnings.append(
                "Payload key 'expanded_text_by_id' must be a mapping; ignoring expansion hints."
            )

    max_expansions: int | None = None
    raw_max_expansions = raw_payload.get("max_expansions")
    if raw_max_expansions is not None:
        if isinstance(raw_max_expansions, bool):
            warnings.append("Payload key 'max_expansions' must be an integer >= 0; ignoring override.")
        else:
            try:
                candidate = int(raw_max_expansions)
            except (TypeError, ValueError):
                warnings.append("Payload key 'max_expansions' must be an integer >= 0; ignoring override.")
            else:
                if candidate < 0:
                    warnings.append("Payload key 'max_expansions' must be an integer >= 0; ignoring override.")
                else:
                    max_expansions = candidate

    return (expansion_map or None), max_expansions, tuple(warnings)


def _extract_primary_items(html: str, source_id: str, selectors: SelectorPack) -> tuple[TweetItem, ...]:
    items: list[TweetItem] = []
    for article_html in _top_level_article_blocks(html, selectors.get("tweet.article", ())):
        candidates = _status_candidates(article_html)
        if not candidates:
            continue
        tweet_id, handle = candidates[0]
        quote_tweet_id = next((candidate_id for candidate_id, _ in candidates[1:] if candidate_id != tweet_id), None)
        items.append(
            _build_item(
                tweet_id=tweet_id,
                source_id=source_id,
                handle=handle,
                text=_extract_text(article_html, selectors.get("tweet.text", ())),
                created_at=_extract_datetime(article_html, selectors.get("tweet.time", ())),
                is_reply=_has_indicator(
                    article_html,
                    selectors.get("tweet.reply_badge", ()),
                    fallback_regex=_REPLY_FALLBACK_RE,
                ),
                is_repost=_has_indicator(
                    article_html,
                    selectors.get("tweet.repost_badge", ()),
                    fallback_regex=_REPOST_FALLBACK_RE,
                ),
                is_pinned=_has_indicator(
                    article_html,
                    selectors.get("tweet.pinned_badge", ()),
                    fallback_regex=_PINNED_FALLBACK_RE,
                ),
                has_quote=quote_tweet_id is not None
                or _has_quote_container(article_html, selectors.get("tweet.quote_container", ())),
                quote_tweet_id=quote_tweet_id,
            )
        )
    return tuple(items)


def _extract_fallback_items(html: str, source_id: str, selectors: SelectorPack) -> tuple[TweetItem, ...]:
    items: list[TweetItem] = []
    candidates = _status_candidates(html)
    for tweet_id, handle in candidates:
        context = _context_around_status(html, tweet_id)
        context_candidates = _status_candidates(context)
        quote_tweet_id = next(
            (candidate_id for candidate_id, _ in context_candidates if candidate_id != tweet_id),
            None,
        )
        items.append(
            _build_item(
                tweet_id=tweet_id,
                source_id=source_id,
                handle=handle,
                text=_extract_text(context, selectors.get("tweet.text", ())),
                created_at=_extract_datetime(context, selectors.get("tweet.time", ())),
                is_reply=_has_indicator(
                    context,
                    selectors.get("tweet.reply_badge", ()),
                    fallback_regex=_REPLY_FALLBACK_RE,
                ),
                is_repost=_has_indicator(
                    context,
                    selectors.get("tweet.repost_badge", ()),
                    fallback_regex=_REPOST_FALLBACK_RE,
                ),
                is_pinned=_has_indicator(
                    context,
                    selectors.get("tweet.pinned_badge", ()),
                    fallback_regex=_PINNED_FALLBACK_RE,
                ),
                has_quote=quote_tweet_id is not None
                or _has_quote_container(context, selectors.get("tweet.quote_container", ())),
                quote_tweet_id=quote_tweet_id,
            )
        )
    return tuple(items)


def _build_item(
    *,
    tweet_id: str,
    source_id: str,
    handle: str | None,
    text: str | None,
    created_at: datetime | None,
    is_reply: bool,
    is_repost: bool,
    is_pinned: bool,
    has_quote: bool,
    quote_tweet_id: str | None,
) -> TweetItem:
    normalized_handle = f"@{handle}" if handle else None
    return TweetItem(
        tweet_id=tweet_id,
        created_at=created_at,
        author_handle=normalized_handle,
        text=text,
        source_id=source_id,
        is_reply=is_reply,
        is_repost=is_repost,
        is_pinned=is_pinned,
        has_quote=has_quote,
        quote_tweet_id=quote_tweet_id,
    )


def _top_level_article_blocks(html: str, selectors: tuple[str, ...]) -> tuple[str, ...]:
    if not selectors:
        return ()

    blocks: list[str] = []
    stack: list[tuple[int, int]] = []
    for match in _ARTICLE_TAG_RE.finditer(html):
        token = match.group(0).lower()
        if token.startswith("</"):
            if not stack:
                continue
            start, open_end = stack.pop()
            if stack:
                continue
            snippet = html[start : match.end()]
            open_tag = html[start:open_end]
            if _matches_any_article_selector(open_tag, selectors):
                blocks.append(snippet)
            continue
        stack.append((match.start(), match.end()))
    return tuple(blocks)


def _matches_any_article_selector(open_tag: str, selectors: tuple[str, ...]) -> bool:
    lowered = open_tag.lower()
    for selector in selectors:
        sel = selector.strip().lower()
        if not sel:
            continue
        if sel == "article":
            return True
        if "data-testid" in sel and "tweet" in sel:
            if 'data-testid="tweet"' in lowered or "data-testid='tweet'" in lowered:
                return True
    return False


def _status_candidates(fragment: str) -> list[tuple[str, str | None]]:
    ordered: list[tuple[str, str | None]] = []
    seen_ids: set[str] = set()
    for handle, tweet_id in _STATUS_LINK_RE.findall(fragment):
        if tweet_id in seen_ids:
            continue
        ordered.append((tweet_id, handle))
        seen_ids.add(tweet_id)
    for tweet_id in _STATUS_LINK_ID_ONLY_RE.findall(fragment):
        if tweet_id in seen_ids:
            continue
        ordered.append((tweet_id, None))
        seen_ids.add(tweet_id)
    return ordered


def _extract_text(fragment: str, selectors: tuple[str, ...]) -> str | None:
    for selector in selectors:
        text = _extract_inner_text_by_selector(fragment, selector)
        if text:
            return text
    return None


def _extract_datetime(fragment: str, selectors: tuple[str, ...]) -> datetime | None:
    if not selectors:
        return None
    for selector in selectors:
        parsed = _extract_datetime_by_selector(fragment, selector)
        if parsed is not None:
            return parsed
    return None


def _extract_datetime_by_selector(fragment: str, selector: str) -> datetime | None:
    tag = _selector_tag(selector)
    if tag != "time":
        return None
    open_tag = _extract_open_tag_by_selector(fragment, selector)
    if open_tag is None:
        return None
    match = _TIME_DATETIME_ATTR_RE.search(open_tag)
    if match is None:
        return None
    return _parse_datetime(match.group(1))


def _parse_datetime(raw: str) -> datetime | None:
    value = raw.strip()
    if not value:
        return None
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        return datetime.fromisoformat(normalized)
    except ValueError:
        return None


def _has_indicator(fragment: str, selectors: tuple[str, ...], *, fallback_regex: re.Pattern[str]) -> bool:
    for selector in selectors:
        if _selector_matches_fragment(fragment, selector):
            return True
    return fallback_regex.search(fragment) is not None


def _has_quote_container(fragment: str, selectors: tuple[str, ...]) -> bool:
    for selector in selectors:
        normalized = selector.replace(" ", "")
        if normalized == 'article[data-testid="tweet"]article':
            if fragment.lower().count("<article") > 1:
                return True
        elif _selector_matches_fragment(fragment, selector):
            return True
    return False


def _context_around_status(html: str, tweet_id: str) -> str:
    match = re.search(rf"/status/{re.escape(tweet_id)}", html)
    if match is None:
        return html
    start = match.start()
    end = min(len(html), match.end() + 600)
    return html[start:end]


def _extract_inner_text_by_selector(fragment: str, selector: str) -> str | None:
    tag = _selector_tag(selector)
    if tag is None:
        return None
    match = _selector_tag_block_match(fragment, selector)
    return _clean_text(match.group(1)) if match else None


def _extract_open_tag_by_selector(fragment: str, selector: str) -> str | None:
    tag = _selector_tag(selector)
    if tag is None:
        return None
    match = _selector_open_tag_match(fragment, selector)
    return match.group(0) if match else None


def _selector_tag_block_match(fragment: str, selector: str) -> re.Match[str] | None:
    tag = _selector_tag(selector)
    if tag is None:
        return None
    attr_match = _selector_attribute(selector)
    class_match = _selector_class(selector)

    if class_match is not None:
        class_pattern = re.compile(
            rf"""<{tag}\b[^>]*\bclass\s*=\s*["'][^"']*\b{re.escape(class_match)}\b[^"']*["'][^>]*>(.*?)</{tag}>""",
            re.IGNORECASE | re.DOTALL,
        )
        return class_pattern.search(fragment)

    if attr_match is not None:
        attr_name, contains, attr_value = attr_match
        if contains:
            pattern = re.compile(
                rf"""<{tag}\b[^>]*\b{re.escape(attr_name)}\s*=\s*["'][^"']*{re.escape(attr_value)}[^"']*["'][^>]*>(.*?)</{tag}>""",
                re.IGNORECASE | re.DOTALL,
            )
        else:
            pattern = re.compile(
                rf"""<{tag}\b[^>]*\b{re.escape(attr_name)}\s*=\s*["']{re.escape(attr_value)}["'][^>]*>(.*?)</{tag}>""",
                re.IGNORECASE | re.DOTALL,
            )
        return pattern.search(fragment)

    return re.search(
        rf"""<{tag}\b[^>]*>(.*?)</{tag}>""",
        fragment,
        flags=re.IGNORECASE | re.DOTALL,
    )


def _selector_open_tag_match(fragment: str, selector: str) -> re.Match[str] | None:
    tag = _selector_tag(selector)
    if tag is None:
        return None
    attr_match = _selector_attribute(selector)
    class_match = _selector_class(selector)

    if class_match is not None:
        class_pattern = re.compile(
            rf"""<{tag}\b[^>]*\bclass\s*=\s*["'][^"']*\b{re.escape(class_match)}\b[^"']*["'][^>]*>""",
            re.IGNORECASE,
        )
        return class_pattern.search(fragment)

    if attr_match is not None:
        attr_name, contains, attr_value = attr_match
        if contains:
            pattern = re.compile(
                rf"""<{tag}\b[^>]*\b{re.escape(attr_name)}\s*=\s*["'][^"']*{re.escape(attr_value)}[^"']*["'][^>]*>""",
                re.IGNORECASE,
            )
        else:
            pattern = re.compile(
                rf"""<{tag}\b[^>]*\b{re.escape(attr_name)}\s*=\s*["']{re.escape(attr_value)}["'][^>]*>""",
                re.IGNORECASE,
            )
        return pattern.search(fragment)

    return re.search(rf"<{tag}\b[^>]*>", fragment, flags=re.IGNORECASE)


def _selector_matches_fragment(fragment: str, selector: str) -> bool:
    tag = _selector_tag(selector)
    if tag is None:
        return False
    attr_match = _selector_attribute(selector)
    class_match = _selector_class(selector)

    if class_match is not None:
        return bool(
            re.search(
                rf"""<{tag}\b[^>]*\bclass\s*=\s*["'][^"']*\b{re.escape(class_match)}\b[^"']*["'][^>]*>""",
                fragment,
                flags=re.IGNORECASE,
            )
        )

    if attr_match is not None:
        attr_name, contains, attr_value = attr_match
        if contains:
            return bool(
                re.search(
                    rf"""<{tag}\b[^>]*\b{re.escape(attr_name)}\s*=\s*["'][^"']*{re.escape(attr_value)}[^"']*["'][^>]*>""",
                    fragment,
                    flags=re.IGNORECASE,
                )
            )
        return bool(
            re.search(
                rf"""<{tag}\b[^>]*\b{re.escape(attr_name)}\s*=\s*["']{re.escape(attr_value)}["'][^>]*>""",
                fragment,
                flags=re.IGNORECASE,
            )
        )

    return bool(re.search(rf"<{tag}\b", fragment, flags=re.IGNORECASE))


def _selector_tag(selector: str) -> str | None:
    normalized = _last_selector_token(selector)
    match = re.match(r"([a-zA-Z][a-zA-Z0-9]*)", normalized)
    return match.group(1).lower() if match else None


def _selector_attribute(selector: str) -> tuple[str, bool, str] | None:
    normalized = _last_selector_token(selector)
    match = re.search(r"""\[([A-Za-z0-9_-]+)(\*)?=["']([^"']+)["']\]""", normalized)
    if match is None:
        return None
    attr_name = match.group(1)
    contains = bool(match.group(2))
    attr_value = match.group(3)
    return attr_name, contains, attr_value


def _selector_class(selector: str) -> str | None:
    normalized = _last_selector_token(selector)
    match = re.search(r"\.([A-Za-z0-9_-]+)", normalized)
    return match.group(1) if match else None


def _last_selector_token(selector: str) -> str:
    token = selector.strip()
    if " " in token:
        token = token.split()[-1]
    if ":" in token:
        token = token.split(":", 1)[0]
    return token


def _clean_text(raw_html: str) -> str:
    without_tags = _TAG_STRIP_RE.sub(" ", raw_html)
    collapsed = " ".join(unescape(without_tags).split())
    return collapsed

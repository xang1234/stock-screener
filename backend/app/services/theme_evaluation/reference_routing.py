"""Structural routing and destination identity for collected references."""

import re
from dataclasses import dataclass
from typing import Iterable, Literal
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from .records import http_url


ReferenceKind = Literal[
    "article_candidate", "linked_x_post", "same_x_post", "not_article"
]

_X_HOSTS = {"x.com", "www.x.com", "mobile.x.com", "twitter.com", "www.twitter.com"}
_TRACKING_PARAMETERS = {
    "dclid",
    "fbclid",
    "gclid",
    "mc_cid",
    "mc_eid",
    "ref_src",
    "ref_url",
}
_GENERIC_PAGE_NAMES = {"", "home", "index.html", "index.htm", "default.aspx"}


def normalize_post_id(value: str) -> str:
    candidate = value.removeprefix("post:")
    if not candidate.isdigit():
        raise ValueError("invalid_x_post_id")
    return candidate


def x_post_id(url: str) -> str | None:
    http_url(url)
    parts = urlsplit(url)
    if (parts.hostname or "").lower() not in _X_HOSTS:
        return None
    match = re.search(r"/(?:i/web/|[^/]+/)?status(?:es)?/(\d+)(?:/|$)", parts.path)
    return match.group(1) if match else None


def classify_reference(url: str, *, parent_post_id: str) -> ReferenceKind:
    """Classify URL structure without judging its investment relevance."""
    http_url(url)
    parts = urlsplit(url)
    host = (parts.hostname or "").lower()
    post_id = x_post_id(url)
    if host in _X_HOSTS:
        if post_id is None:
            return (
                "article_candidate"
                if re.fullmatch(r"/i/article/\d+/?", parts.path)
                else "not_article"
            )
        return (
            "same_x_post"
            if post_id == normalize_post_id(parent_post_id)
            else "linked_x_post"
        )
    path_parts = [part.lower() for part in parts.path.split("/") if part]
    if not path_parts or (
        host.startswith("investor.")
        and len(path_parts) <= 2
        and path_parts[-1] in _GENERIC_PAGE_NAMES
    ):
        return "not_article"
    if path_parts[-1] == "search" or "search" in path_parts and not parts.query:
        return "not_article"
    return "article_candidate"


def normalize_destination(url: str) -> str:
    """Normalize transport noise while retaining potentially identifying queries."""
    http_url(url)
    parts = urlsplit(url)
    scheme = parts.scheme.lower()
    host = (parts.hostname or "").lower()
    port = parts.port
    if (scheme, port) in {("http", 80), ("https", 443)}:
        port = None
    netloc = host if port is None else f"{host}:{port}"
    query = urlencode(
        [
            (name, value)
            for name, value in parse_qsl(parts.query, keep_blank_values=True)
            if not name.lower().startswith("utm_")
            and name.lower() not in _TRACKING_PARAMETERS
        ],
        doseq=True,
    )
    path = parts.path or "/"
    return urlunsplit((scheme, netloc, path, query, ""))


def _origin(url):
    parts = urlsplit(url)
    return parts.scheme, parts.netloc


def _identity_tokens(url):
    parts = urlsplit(url)
    values = []
    for segment in parts.path.split("/"):
        lowered = segment.lower().strip()
        if lowered and lowered not in _GENERIC_PAGE_NAMES and (
            any(character.isdigit() for character in lowered) or len(lowered) >= 8
        ):
            values.append(lowered)
    for name, value in parse_qsl(parts.query, keep_blank_values=True):
        if name.lower() in {"id", "article", "article_id", "no", "p", "story", "v"}:
            values.append(f"{name.lower()}={value}")
    return set(values)


def destination_identity(final_url: str, canonical_url: str | None = None) -> str:
    """Use a canonical alias only when it cannot cross origin or article identity."""
    final = normalize_destination(final_url)
    if not canonical_url:
        return final
    canonical = normalize_destination(canonical_url)
    if _origin(final) != _origin(canonical):
        return final
    if final == canonical:
        return canonical
    final_tokens = _identity_tokens(final)
    canonical_tokens = _identity_tokens(canonical)
    if final_tokens and canonical_tokens and not final_tokens.intersection(canonical_tokens):
        return final
    return canonical


@dataclass(frozen=True)
class ReferenceRoute:
    reference_id: str
    source_post_id: str
    original_url: str
    final_url: str
    classification: ReferenceKind
    target_post_id: str | None = None
    canonical_url: str | None = None
    source_depth: int = 0


def route_reference(
    *,
    reference_id: str,
    source_post_id: str,
    original_url: str,
    final_url: str,
    canonical_url: str | None = None,
    source_depth: int = 0,
) -> ReferenceRoute:
    if source_depth < 0:
        raise ValueError("invalid_reference_depth")
    classification = classify_reference(final_url, parent_post_id=source_post_id)
    return ReferenceRoute(
        reference_id=reference_id,
        source_post_id=source_post_id,
        original_url=original_url,
        final_url=final_url,
        classification=classification,
        target_post_id=x_post_id(final_url),
        canonical_url=canonical_url,
        source_depth=source_depth,
    )


@dataclass(frozen=True)
class DestinationGroup:
    identity_url: str
    references: tuple[ReferenceRoute, ...]


def group_article_destinations(
    routes: Iterable[ReferenceRoute],
) -> list[DestinationGroup]:
    groups = {}
    for route in routes:
        if route.classification != "article_candidate":
            continue
        identity = destination_identity(route.final_url, route.canonical_url)
        groups.setdefault(identity, []).append(route)
    return [
        DestinationGroup(identity, tuple(references))
        for identity, references in groups.items()
    ]


LinkedDisposition = Literal[
    "queued", "duplicate", "already_in_base", "one_hop_limit", "limit_exceeded"
]


@dataclass(frozen=True)
class LinkedPostEntry:
    reference_id: str
    source_post_id: str
    target_post_id: str
    url: str
    disposition: LinkedDisposition


@dataclass(frozen=True)
class LinkedPostManifest:
    post_ids: tuple[str, ...]
    entries: tuple[LinkedPostEntry, ...]
    max_posts: int
    max_hops: int = 1


def build_linked_post_manifest(
    routes: Iterable[ReferenceRoute], *, base_post_ids: set[str], max_posts: int = 20
) -> LinkedPostManifest:
    if not 0 <= max_posts <= 20:
        raise ValueError("invalid_linked_post_limit")
    base = {normalize_post_id(value) for value in base_post_ids}
    queued = []
    seen = set()
    entries = []
    for route in routes:
        if route.classification != "linked_x_post" or route.target_post_id is None:
            continue
        target = normalize_post_id(route.target_post_id)
        if route.source_depth >= 1:
            disposition = "one_hop_limit"
        elif target in base:
            disposition = "already_in_base"
        elif target in seen:
            disposition = "duplicate"
        elif len(queued) >= max_posts:
            disposition = "limit_exceeded"
        else:
            disposition = "queued"
            seen.add(target)
            queued.append(target)
        entries.append(
            LinkedPostEntry(
                reference_id=route.reference_id,
                source_post_id=route.source_post_id,
                target_post_id=target,
                url=normalize_destination(route.final_url),
                disposition=disposition,
            )
        )
    return LinkedPostManifest(tuple(queued), tuple(entries), max_posts)

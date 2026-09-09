"""Conservative article recovery: readable text is not proof of completeness."""

import json
import re
from datetime import datetime, timezone
from typing import Literal
from urllib.parse import urljoin, urlsplit

from bs4 import BeautifulSoup
from pydantic import AwareDatetime, Field, model_serializer

from .bundle import sha256
from .records import SHA, URL, Record


class ArticleRecovery(Record):
    title: str
    text: str
    final_url: URL
    canonical_url: URL | None = None
    method: Literal["json_ld", "semantic_html", "browser_import", "unavailable"]
    capture_status: Literal["full", "partial"] = "partial"
    warnings: list[str] = Field(default_factory=list)
    response_sha256: SHA
    retrieved_at: AwareDatetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    match_basis: str | None = None
    body_sha256: SHA | None = None
    completeness_basis: str | None = None

    @model_serializer(mode="wrap")
    def preserve_legacy_shape(self, handler):
        result = handler(self)
        # Reconstructing an article-v1 result must not add v2 provenance and
        # thereby change its immutable serialized identity.
        for name in ("body_sha256", "completeness_basis"):
            if name not in self.model_fields_set:
                result.pop(name, None)
        return result


def _articles(value):
    if isinstance(value, list):
        for child in value:
            yield from _articles(child)
    elif isinstance(value, dict):
        types = value.get("@type", [])
        types = [types] if isinstance(types, str) else types
        if any(t in ("Article", "NewsArticle", "BlogPosting", "Report") for t in types):
            yield value
        if "@graph" in value:
            yield from _articles(value["@graph"])


def _origin(url):
    parts = urlsplit(url)
    port = parts.port
    if port in (80, 443) and (
        (parts.scheme.lower() == "http" and port == 80)
        or (parts.scheme.lower() == "https" and port == 443)
    ):
        port = None
    return parts.scheme.lower(), parts.hostname.lower() if parts.hostname else "", port


def _canonical_url(soup, final_url, warnings):
    canonical = soup.find("link", rel=lambda value: value and "canonical" in value)
    if not canonical or not canonical.get("href"):
        return None
    candidate = urljoin(final_url, canonical["href"])
    try:
        if _origin(candidate) != _origin(final_url):
            warnings.append("untrusted_canonical_url")
            return None
        return candidate
    except ValueError:
        warnings.append("invalid_canonical_url")
        return None


def _text(value):
    return re.sub(r"\s+", " ", value).strip()


_BLOCKS = "p, h1, h2, h3, h4, h5, h6, li, blockquote, pre"
_CONTROLS = (
    "script, style, noscript, template, nav, footer, form, aside, button, "
    "select, input, textarea, svg, [role='toolbar'], [role='navigation'], "
    "[role='menu'], [aria-hidden='true']"
)


def _region_text(region):
    clean = BeautifulSoup(str(region), "html.parser")
    for element in clean.select(_CONTROLS):
        element.decompose()
    blocks = []
    for element in clean.select(_BLOCKS):
        if element.select_one(_BLOCKS):
            continue
        value = _text(element.get_text(" ", strip=True))
        if value:
            blocks.append(value)
    if blocks:
        return "\n\n".join(blocks)
    return _text(clean.get_text(" ", strip=True))


def _publisher_regions(soup, final_url):
    host = (urlsplit(final_url).hostname or "").lower()
    if host == "v.daum.net":
        return soup.select(".article_view[data-translation-body]")
    if host == "investor.oracle.com":
        return soup.select("#_ctrl0_ctl42_divBody.evergreen-news-body")
    return []


def _body_regions(soup, final_url):
    publisher = _publisher_regions(soup, final_url)
    if publisher:
        return publisher
    for selector in (
        "[itemprop='articleBody']",
        "[data-testid='article-body']",
        ".article-body",
        ".article__body",
        ".story-body",
        ".entry-content",
        ".post-content",
    ):
        regions = soup.select(selector)
        if regions:
            return regions
    return soup.find_all("article") or soup.find_all("main")


def _looks_render_dependent(soup, final_url):
    host = (urlsplit(final_url).hostname or "").lower()
    if host in {"biz.chosun.com", "zdnet.co.kr"}:
        return True
    scripts = " ".join(script.get_text(" ", strip=True) for script in soup.find_all("script"))
    return bool(
        re.search(
            r"loadarticle|hydrate|__next_data__|webpack|javascript",
            scripts,
            flags=re.IGNORECASE,
        )
    )


def parse_article(raw: bytes, final_url: str) -> ArticleRecovery:
    soup = BeautifulSoup(raw, "html.parser")
    title = soup.title.get_text(" ", strip=True) if soup.title else ""
    visible = soup.get_text(" ", strip=True)
    warnings = []
    canonical_url = _canonical_url(soup, final_url, warnings)

    if any(
        marker in (title + " " + visible).lower()
        for marker in (
            "verify you are human",
            "checking your browser",
            "just a moment...",
            "enable javascript and cookies",
        )
    ):
        warnings.extend(["access_restricted", "access_interstitial"])
        return ArticleRecovery(
            title=title,
            text="",
            final_url=final_url,
            canonical_url=canonical_url,
            method="unavailable",
            warnings=warnings,
            response_sha256=sha256(raw),
            body_sha256=sha256(b""),
        )

    articles = []
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            articles.extend(_articles(json.loads(script.get_text())))
        except (ValueError, TypeError):
            warnings.append("invalid_json_ld")
    if any(a.get("isAccessibleForFree") in (False, "false") for a in articles) or any(
        marker in visible.lower()
        for marker in (
            "subscribe to continue",
            "sign in to continue",
            "subscriber-only",
        )
    ):
        warnings.append("access_restricted")

    bodies = list(
        dict.fromkeys(
            _text(article["articleBody"])
            for article in articles
            if isinstance(article.get("articleBody"), str)
            and article["articleBody"].strip()
        )
    )
    method = "unavailable"
    text = ""
    if len(bodies) > 1:
        warnings.append("body_ambiguous")
    elif bodies:
        text, method = bodies[0], "json_ld"
        if not title:
            headlines = [
                _text(article["headline"])
                for article in articles
                if isinstance(article.get("headline"), str)
                and article["headline"].strip()
            ]
            if len(set(headlines)) == 1:
                title = headlines[0]
    else:
        region_texts = list(
            dict.fromkeys(
                value
                for value in (_region_text(region) for region in _body_regions(soup, final_url))
                if value
            )
        )
        if len(region_texts) > 1:
            warnings.append("body_ambiguous")
        elif region_texts:
            text, method = region_texts[0], "semantic_html"
        else:
            warnings.append("body_missing")
            if _looks_render_dependent(soup, final_url):
                warnings.append("render_required")

    if text:
        warnings.append("completeness_unverified")
        if len(text) < 500:
            warnings.append("short_article_or_excerpt")
    return ArticleRecovery(
        title=title,
        text=text,
        final_url=final_url,
        canonical_url=canonical_url,
        method=method,
        warnings=list(dict.fromkeys(warnings)),
        response_sha256=sha256(raw),
        body_sha256=sha256(text.encode()),
    )

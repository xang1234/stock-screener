"""Conservative article recovery: readable text is not proof of completeness."""

import json
from datetime import datetime, timezone
from typing import Literal
from urllib.parse import urljoin

from bs4 import BeautifulSoup
from pydantic import AwareDatetime, Field

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


def parse_article(raw: bytes, final_url: str) -> ArticleRecovery:
    soup = BeautifulSoup(raw, "html.parser")
    title = soup.title.get_text(" ", strip=True) if soup.title else ""
    visible = soup.get_text(" ", strip=True)
    canonical = soup.find("link", rel="canonical")
    canonical_url = (
        urljoin(final_url, canonical["href"])
        if canonical and canonical.get("href")
        else None
    )
    try:
        result = ArticleRecovery(
            title=title,
            text="",
            final_url=final_url,
            canonical_url=canonical_url,
            method="unavailable",
            response_sha256=sha256(raw),
        )
    except ValueError:
        result = ArticleRecovery(
            title=title,
            text="",
            final_url=final_url,
            method="unavailable",
            response_sha256=sha256(raw),
            warnings=["invalid_canonical_url"],
        )
    if any(
        marker in (title + " " + visible).lower()
        for marker in (
            "verify you are human",
            "checking your browser",
            "just a moment...",
            "enable javascript and cookies",
        )
    ):
        result.warnings.append("access_interstitial")
        return result
    articles = []
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            articles.extend(_articles(json.loads(script.get_text())))
        except (ValueError, TypeError):
            result.warnings.append("invalid_json_ld")
    if any(a.get("isAccessibleForFree") in (False, "false") for a in articles) or any(
        marker in visible.lower()
        for marker in (
            "subscribe to continue",
            "sign in to continue",
            "subscriber-only",
        )
    ):
        result.warnings.append("access_restricted")
    bodies = list(
        dict.fromkeys(
            a["articleBody"]
            for a in articles
            if isinstance(a.get("articleBody"), str) and a["articleBody"].strip()
        )
    )
    if len(bodies) > 1:
        result.warnings.append("ambiguous_article_bodies")
        return result
    if bodies:
        result.text, result.method = bodies[0], "json_ld"
    else:
        regions = soup.find_all("article") or soup.find_all("main")
        if len(regions) != 1:
            result.warnings.append("article_body_missing_or_ambiguous")
            return result
        region = regions[0]
        for element in region.select("script, style, nav, footer, form, aside"):
            element.decompose()
        result.text = region.get_text("\n\n", strip=True)
        result.method = "semantic_html"
    result.warnings.append("completeness_unverified")
    if len(result.text) < 500:
        result.warnings.append("short_article_or_excerpt")
    return result

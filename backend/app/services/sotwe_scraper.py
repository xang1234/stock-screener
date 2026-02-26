"""
Sotwe Scraper - Fetches tweets from sotwe.com

Sotwe renders Twitter profiles via SSR and embeds tweet data in
window.__NUXT__. This scraper fetches the page with curl_cffi
(to bypass Cloudflare), extracts the NUXT payload, evaluates it
with QuickJS, and parses the resulting JSON.

Requires: curl_cffi, quickjs.
"""
import json
import logging
import random
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from curl_cffi import requests as cffi_requests

from ..config import settings

logger = logging.getLogger(__name__)

BASE_URL = "https://www.sotwe.com"

NUXT_PATTERN = re.compile(r"window\.__NUXT__=(.+?);\s*</script>")


class SotweScraperError(Exception):
    """Error during Sotwe scraping."""
    pass


@dataclass
class SotweTweet:
    """Tweet data from Sotwe."""

    id: str
    text: str
    created_at: datetime
    username: str
    author_name: str
    like_count: int
    retweet_count: int
    reply_count: int
    view_count: int | None
    bookmark_count: int | None
    quote_count: int | None


class SotweScraper:
    """Scrapes tweet data from sotwe.com user pages."""

    def __init__(self, timeout: int | None = None):
        self.timeout = timeout or settings.sotwe_request_timeout

    def fetch_user_tweets(self, username: str) -> list[SotweTweet]:
        """Fetch recent tweets for a Twitter username.

        Args:
            username: Twitter handle without the '@' prefix.

        Returns:
            List of SotweTweet objects.

        Raises:
            SotweScraperError: On network or parsing failures.
        """
        html = self._fetch_page(username)
        nuxt_data = self._extract_nuxt_data(html)
        tweets = self._parse_tweets(nuxt_data, username)
        logger.info(f"Sotwe: fetched {len(tweets)} tweets for @{username}")
        return tweets

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fetch_page(self, username: str) -> str:
        """Fetch the Sotwe profile page HTML with retry on transient errors."""
        url = f"{BASE_URL}/{username}"
        max_retries = 3
        retryable_codes = {429, 500, 502, 503}

        for attempt in range(max_retries):
            try:
                resp = cffi_requests.get(
                    url,
                    impersonate="chrome",
                    timeout=self.timeout,
                )
                if resp.status_code in retryable_codes and attempt < max_retries - 1:
                    wait = (2 ** (attempt + 2)) + random.uniform(1, 3)
                    logger.warning(
                        f"Sotwe returned {resp.status_code} for @{username}, "
                        f"retrying in {wait:.1f}s (attempt {attempt + 1}/{max_retries})"
                    )
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                return resp.text
            except Exception as exc:
                if attempt < max_retries - 1:
                    wait = (2 ** (attempt + 2)) + random.uniform(1, 3)
                    logger.warning(
                        f"Sotwe request failed for @{username}: {exc}, "
                        f"retrying in {wait:.1f}s (attempt {attempt + 1}/{max_retries})"
                    )
                    time.sleep(wait)
                    continue
                raise SotweScraperError(
                    f"Failed to fetch Sotwe page for @{username} after {max_retries} attempts: {exc}"
                ) from exc

    def _extract_nuxt_data(self, html: str) -> dict[str, Any]:
        """Extract and evaluate the __NUXT__ payload from HTML."""
        match = NUXT_PATTERN.search(html)
        if not match:
            raise SotweScraperError("Could not find __NUXT__ payload in page HTML")
        nuxt_js = match.group(1)
        return self._evaluate_nuxt_js(nuxt_js)

    def _evaluate_nuxt_js(self, nuxt_code: str) -> dict[str, Any]:
        """Evaluate minified NUXT JS payload using QuickJS engine."""
        import quickjs

        try:
            ctx = quickjs.Context()
            ctx.set_time_limit(10)
            result = ctx.eval(f"JSON.stringify({nuxt_code})")
            if not isinstance(result, str):
                raise SotweScraperError(
                    "NUXT payload did not evaluate to a JSON-serializable value"
                )
            return json.loads(result)
        except quickjs.JSException as exc:
            raise SotweScraperError(
                f"QuickJS evaluation failed: {exc}"
            ) from exc
        except json.JSONDecodeError as exc:
            raise SotweScraperError(
                f"Failed to parse JS output as JSON: {exc}"
            ) from exc

    def _parse_tweets(
        self, nuxt_data: dict[str, Any], username: str
    ) -> list[SotweTweet]:
        """Navigate the NUXT state tree and parse tweet entries."""
        state = nuxt_data.get("state", {})
        timeline_data = (
            state.get("user", {}).get("timeline", {}).get("data", {})
        )

        # Try exact username key, then lowercase, then first available list
        entries = timeline_data.get(username)
        if entries is None:
            entries = timeline_data.get(username.lower())
        if entries is None:
            # Fall back to the first list value in timeline data
            for val in timeline_data.values():
                if isinstance(val, list):
                    entries = val
                    break

        if not entries or not isinstance(entries, list):
            logger.warning(f"Sotwe: no timeline entries found for @{username}")
            return []

        logger.debug(f"Sotwe: {len(entries)} candidate entries for @{username}")

        tweets: list[SotweTweet] = []
        for entry in entries:
            tweet = self._parse_single_tweet(entry)
            if tweet is not None:
                tweets.append(tweet)

        return tweets

    def _parse_single_tweet(self, data: dict[str, Any]) -> SotweTweet | None:
        """Parse a single tweet dict into a SotweTweet, or None on failure."""
        tweet_id = data.get("id")
        text = data.get("text")
        if not tweet_id or not text:
            return None

        # createdAt is Unix milliseconds
        created_ms = data.get("createdAt", 0)
        try:
            created_at = datetime.fromtimestamp(created_ms / 1000, tz=timezone.utc)
        except (ValueError, TypeError, OSError):
            created_at = datetime.now(tz=timezone.utc)

        user_info = data.get("user", {})

        return SotweTweet(
            id=str(tweet_id),
            text=text,
            created_at=created_at,
            username=user_info.get("username", ""),
            author_name=user_info.get("fullname", ""),
            like_count=data.get("favoriteCount", 0),
            retweet_count=data.get("retweetCount", 0),
            reply_count=data.get("replyCount", 0),
            view_count=data.get("viewCount"),
            bookmark_count=data.get("bookmarkCount"),
            quote_count=data.get("quoteCount"),
        )

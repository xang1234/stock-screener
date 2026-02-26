"""
Content Ingestion Service for Theme Discovery

Fetches content from multiple sources:
- Substack RSS feeds
- Twitter/X API or Nitter scraping
- News APIs (Benzinga, Seeking Alpha)
- Reddit API
"""
import logging
import hashlib
import time
from datetime import datetime, timedelta
from typing import Optional
from abc import ABC, abstractmethod

import feedparser
import requests
from sqlalchemy.orm import Session

from ..models.theme import ContentSource, ContentItem
from ..config import settings

logger = logging.getLogger(__name__)


class BaseContentFetcher(ABC):
    """Base class for content fetchers"""

    @abstractmethod
    def fetch(self, source: ContentSource, since: Optional[datetime] = None) -> list[dict]:
        """Fetch content from source, returns list of content items"""
        pass

    def generate_external_id(self, source_type: str, unique_string: str) -> str:
        """Generate a unique external ID for deduplication"""
        return hashlib.md5(f"{source_type}:{unique_string}".encode()).hexdigest()


class RSSFetcher(BaseContentFetcher):
    """Fetches content from RSS/Atom feeds (Substack, blogs, etc.)"""

    def fetch(self, source: ContentSource, since: Optional[datetime] = None) -> list[dict]:
        """Fetch RSS feed items"""
        items = []
        try:
            feed = feedparser.parse(source.url)

            for entry in feed.entries:
                # Parse published date
                published_at = None
                if hasattr(entry, "published_parsed") and entry.published_parsed:
                    published_at = datetime(*entry.published_parsed[:6])
                elif hasattr(entry, "updated_parsed") and entry.updated_parsed:
                    published_at = datetime(*entry.updated_parsed[:6])

                # Skip if older than since date
                if since and published_at and published_at < since:
                    continue

                # Extract content
                content = ""
                if hasattr(entry, "content") and entry.content:
                    content = entry.content[0].get("value", "")
                elif hasattr(entry, "summary"):
                    content = entry.summary
                elif hasattr(entry, "description"):
                    content = entry.description

                # Generate unique ID
                external_id = self.generate_external_id(
                    source.source_type,
                    entry.get("id", entry.get("link", entry.get("title", "")))
                )

                items.append({
                    "external_id": external_id,
                    "title": entry.get("title", ""),
                    "content": content,
                    "url": entry.get("link", ""),
                    "author": entry.get("author", source.name),
                    "published_at": published_at,
                })

            logger.info(f"Fetched {len(items)} items from RSS feed: {source.name}")

        except Exception as e:
            logger.error(f"Error fetching RSS feed {source.name}: {e}")

        return items


class TwitterFetcher(BaseContentFetcher):
    """
    Fetches content from Twitter/X

    Options:
    1. Official Twitter API v2 (requires paid access)
    2. Nitter instances (free but less reliable)
    3. RSS bridges like RSSHub
    """

    def __init__(self):
        self.api_bearer_token = getattr(settings, "twitter_bearer_token", None)
        self.sotwe_scraper = None
        if not self.api_bearer_token and getattr(settings, "sotwe_enabled", True):
            try:
                from .sotwe_scraper import SotweScraper
                self.sotwe_scraper = SotweScraper()
            except ImportError:
                logger.warning("curl_cffi not installed; Sotwe scraper unavailable")

    def fetch(self, source: ContentSource, since: Optional[datetime] = None) -> list[dict]:
        """Fetch tweets from user timeline.

        Fallback chain: Twitter API v2 -> Sotwe -> Nitter RSS.
        """
        if self.api_bearer_token:
            return self._fetch_via_api(source, since)

        if self.sotwe_scraper:
            items = self._fetch_via_sotwe(source, since)
            if items:
                return items
            logger.info(f"Sotwe returned no items for {source.name}, falling back to Nitter")

        return self._fetch_via_nitter_rss(source, since)

    def _fetch_via_api(self, source: ContentSource, since: Optional[datetime] = None) -> list[dict]:
        """Fetch via official Twitter API v2"""
        items = []
        try:
            # Extract username from URL (format: twitter.com/username or @username)
            username = source.url.split("/")[-1].replace("@", "")

            # First get user ID
            user_response = requests.get(
                f"https://api.twitter.com/2/users/by/username/{username}",
                headers={"Authorization": f"Bearer {self.api_bearer_token}"}
            )
            user_data = user_response.json()
            if "data" not in user_data:
                logger.error(f"Twitter user not found: {username}")
                return items

            user_id = user_data["data"]["id"]

            # Fetch recent tweets
            params = {
                "max_results": 100,
                "tweet.fields": "created_at,text,author_id",
            }
            if since:
                params["start_time"] = since.isoformat() + "Z"

            response = requests.get(
                f"https://api.twitter.com/2/users/{user_id}/tweets",
                headers={"Authorization": f"Bearer {self.api_bearer_token}"},
                params=params
            )
            tweets = response.json()

            for tweet in tweets.get("data", []):
                published_at = datetime.fromisoformat(tweet["created_at"].replace("Z", "+00:00"))
                external_id = self.generate_external_id("twitter", tweet["id"])

                items.append({
                    "external_id": external_id,
                    "title": "",  # Tweets don't have titles
                    "content": tweet["text"],
                    "url": f"https://twitter.com/{username}/status/{tweet['id']}",
                    "author": f"@{username}",
                    "published_at": published_at,
                })

            logger.info(f"Fetched {len(items)} tweets from @{username}")

        except Exception as e:
            logger.error(f"Error fetching Twitter API for {source.name}: {e}")

        return items

    def _fetch_via_sotwe(self, source: ContentSource, since: Optional[datetime] = None) -> list[dict]:
        """Fetch via Sotwe scraper (free fallback before Nitter)

        Note: We intentionally don't filter by `since` date here because:
        1. Sotwe only returns ~20-25 tweets anyway (small dataset)
        2. Database deduplication in fetch_source() is more reliable
        3. The `since` filter can cause issues if last_fetched_at was set
           but no data was actually stored (e.g., wrong source_type)
        """
        items = []
        try:
            username = source.url.split("/")[-1].replace("@", "")
            tweets = self.sotwe_scraper.fetch_user_tweets(username)

            for tweet in tweets:
                external_id = self.generate_external_id("twitter", tweet.id)

                items.append({
                    "external_id": external_id,
                    "title": "",
                    "content": tweet.text,
                    "url": f"https://twitter.com/{username}/status/{tweet.id}",
                    "author": f"@{username}",
                    "published_at": tweet.created_at,
                })

            logger.info(f"Fetched {len(items)} tweets via Sotwe for @{username}")

        except Exception as e:
            logger.error(f"Error fetching Sotwe for {source.name}: {e}")

        return items

    def _fetch_via_nitter_rss(self, source: ContentSource, since: Optional[datetime] = None) -> list[dict]:
        """
        Fetch via Nitter RSS (fallback when no API key)
        Uses public Nitter instances that provide RSS feeds

        Note: We don't filter by `since` date - database deduplication handles it.
        """
        items = []
        try:
            # Extract username
            username = source.url.split("/")[-1].replace("@", "")

            # List of Nitter instances that provide RSS
            nitter_instances = [
                "nitter.net",
                "nitter.privacydev.net",
                "nitter.poast.org",
            ]

            feed = None
            for instance in nitter_instances:
                try:
                    rss_url = f"https://{instance}/{username}/rss"
                    feed = feedparser.parse(rss_url)
                    if feed.entries:
                        break
                except Exception:
                    continue

            if not feed or not feed.entries:
                logger.warning(f"Could not fetch Nitter RSS for @{username}")
                return items

            for entry in feed.entries:
                published_at = None
                if hasattr(entry, "published_parsed") and entry.published_parsed:
                    published_at = datetime(*entry.published_parsed[:6])

                external_id = self.generate_external_id("twitter", entry.get("id", entry.get("link", "")))

                items.append({
                    "external_id": external_id,
                    "title": "",
                    "content": entry.get("title", "") + " " + entry.get("description", ""),
                    "url": entry.get("link", "").replace("nitter.net", "twitter.com"),
                    "author": f"@{username}",
                    "published_at": published_at,
                })

            logger.info(f"Fetched {len(items)} items via Nitter RSS for @{username}")

        except Exception as e:
            logger.error(f"Error fetching Nitter RSS for {source.name}: {e}")

        return items


class NewsFetcher(BaseContentFetcher):
    """
    Fetches financial news from various APIs

    Supported sources:
    - Benzinga (requires API key)
    - Seeking Alpha RSS
    - MarketWatch RSS
    - Yahoo Finance RSS
    """

    def __init__(self):
        self.benzinga_api_key = getattr(settings, "benzinga_api_key", None)

    def fetch(self, source: ContentSource, since: Optional[datetime] = None) -> list[dict]:
        """Fetch news based on source URL pattern"""
        url = source.url.lower()

        if "benzinga" in url and self.benzinga_api_key:
            return self._fetch_benzinga(source, since)
        elif "seekingalpha" in url or "seeking-alpha" in url:
            return self._fetch_seekingalpha_rss(source, since)
        else:
            # Default to RSS parsing
            rss_fetcher = RSSFetcher()
            return rss_fetcher.fetch(source, since)

    def _fetch_benzinga(self, source: ContentSource, since: Optional[datetime] = None) -> list[dict]:
        """Fetch from Benzinga API"""
        items = []
        try:
            params = {
                "token": self.benzinga_api_key,
                "pageSize": 50,
                "displayOutput": "full",
            }
            if since:
                params["dateFrom"] = since.strftime("%Y-%m-%d")

            response = requests.get(
                "https://api.benzinga.com/api/v2/news",
                params=params
            )
            news_items = response.json()

            for article in news_items:
                published_at = datetime.fromisoformat(article.get("created", "").replace("Z", "+00:00"))
                external_id = self.generate_external_id("benzinga", str(article.get("id", "")))

                items.append({
                    "external_id": external_id,
                    "title": article.get("title", ""),
                    "content": article.get("body", article.get("teaser", "")),
                    "url": article.get("url", ""),
                    "author": article.get("author", "Benzinga"),
                    "published_at": published_at,
                })

            logger.info(f"Fetched {len(items)} articles from Benzinga")

        except Exception as e:
            logger.error(f"Error fetching Benzinga: {e}")

        return items

    def _fetch_seekingalpha_rss(self, source: ContentSource, since: Optional[datetime] = None) -> list[dict]:
        """Fetch from Seeking Alpha RSS feed"""
        rss_fetcher = RSSFetcher()
        return rss_fetcher.fetch(source, since)


class RedditFetcher(BaseContentFetcher):
    """Fetches content from Reddit via JSON API (no auth needed for public)"""

    def fetch(self, source: ContentSource, since: Optional[datetime] = None) -> list[dict]:
        """Fetch posts from subreddit"""
        items = []
        try:
            # Extract subreddit from URL
            # Format: reddit.com/r/wallstreetbets or just /r/wallstreetbets
            parts = source.url.replace("https://", "").replace("http://", "").split("/")
            subreddit = None
            for i, part in enumerate(parts):
                if part == "r" and i + 1 < len(parts):
                    subreddit = parts[i + 1]
                    break

            if not subreddit:
                logger.error(f"Could not extract subreddit from URL: {source.url}")
                return items

            # Fetch via Reddit JSON API (public, no auth needed)
            response = requests.get(
                f"https://www.reddit.com/r/{subreddit}/hot.json",
                headers={"User-Agent": "StockScanner/1.0"},
                params={"limit": 50}
            )
            data = response.json()

            for post in data.get("data", {}).get("children", []):
                post_data = post.get("data", {})

                # Convert Unix timestamp
                created_utc = post_data.get("created_utc", 0)
                published_at = datetime.utcfromtimestamp(created_utc)

                if since and published_at < since:
                    continue

                external_id = self.generate_external_id("reddit", post_data.get("id", ""))

                # Combine title and selftext for content
                content = post_data.get("title", "")
                if post_data.get("selftext"):
                    content += "\n\n" + post_data.get("selftext", "")

                items.append({
                    "external_id": external_id,
                    "title": post_data.get("title", ""),
                    "content": content,
                    "url": f"https://reddit.com{post_data.get('permalink', '')}",
                    "author": post_data.get("author", ""),
                    "published_at": published_at,
                })

            logger.info(f"Fetched {len(items)} posts from r/{subreddit}")

        except Exception as e:
            logger.error(f"Error fetching Reddit {source.name}: {e}")

        return items


class ContentIngestionService:
    """Main service for ingesting content from all sources"""

    def __init__(self, db: Session):
        self.db = db
        self.fetchers = {
            "substack": RSSFetcher(),
            "rss": RSSFetcher(),
            "twitter": TwitterFetcher(),
            "news": NewsFetcher(),
            "reddit": RedditFetcher(),
        }

    def get_fetcher(self, source_type: str) -> BaseContentFetcher:
        """Get appropriate fetcher for source type"""
        return self.fetchers.get(source_type, RSSFetcher())

    def fetch_source(self, source: ContentSource, lookback_days: int | None = None) -> int:
        """Fetch new content from a single source, returns count of new items.

        Args:
            source: The content source to fetch from.
            lookback_days: If set, override the since date to fetch articles
                from the last N days. Used for backfilling gaps. When backfilling,
                last_fetched_at is NOT updated so normal runs resume correctly.
        """
        fetcher = self.get_fetcher(source.source_type)

        # Determine since date
        if lookback_days:
            since = datetime.utcnow() - timedelta(days=lookback_days)
        else:
            since = source.last_fetched_at
            if not since:
                since = datetime.utcnow() - timedelta(days=30)  # Default 30-day lookback for new sources

        # Fetch items
        items = fetcher.fetch(source, since)

        # Store new items - track seen external_ids to avoid duplicates within batch
        new_count = 0
        seen_ids = set()
        source_id = source.id
        source_type = source.source_type
        source_name = source.name

        for item_data in items:
            external_id = item_data["external_id"]

            # Skip if we've already seen this in this batch
            if external_id in seen_ids:
                continue
            seen_ids.add(external_id)

            # Check if already exists in database
            existing = self.db.query(ContentItem).filter(
                ContentItem.source_type == source_type,
                ContentItem.external_id == external_id
            ).first()

            if not existing:
                content_item = ContentItem(
                    source_id=source_id,
                    source_type=source_type,
                    source_name=source_name,
                    external_id=external_id,
                    title=item_data["title"],
                    content=item_data["content"],
                    url=item_data["url"],
                    author=item_data["author"],
                    published_at=item_data["published_at"],
                    is_processed=False,
                )
                self.db.add(content_item)
                new_count += 1

        # Commit all new items
        if new_count > 0:
            self.db.commit()

        # Update source metadata (refresh source first in case session was affected)
        source = self.db.query(ContentSource).filter(ContentSource.id == source_id).first()
        # Only update last_fetched_at during normal runs, not backfills.
        # Backfill leaves last_fetched_at unchanged so the next normal run
        # resumes from the correct point.
        if not lookback_days:
            source.last_fetched_at = datetime.utcnow()
        source.total_items_fetched = (source.total_items_fetched or 0) + new_count
        self.db.commit()

        logger.info(f"Ingested {new_count} new items from {source.name}")
        return new_count

    def fetch_all_active_sources(self, lookback_days: int | None = None) -> dict:
        """Fetch from all active sources, returns summary.

        Args:
            lookback_days: If set, re-fetch articles from the last N days
                (backfill mode). Deduplication ensures no duplicates.
        """
        sources = self.db.query(ContentSource).filter(
            ContentSource.is_active == True
        ).order_by(ContentSource.priority.desc()).all()

        results = {
            "total_sources": len(sources),
            "total_new_items": 0,
            "sources_fetched": [],
            "errors": [],
        }

        for source in sources:
            try:
                new_count = self.fetch_source(source, lookback_days=lookback_days)
                results["total_new_items"] += new_count
                results["sources_fetched"].append({
                    "name": source.name,
                    "type": source.source_type,
                    "new_items": new_count,
                })
            except Exception as e:
                # Rollback the failed transaction before continuing
                self.db.rollback()
                logger.error(f"Error fetching {source.name}: {e}")
                results["errors"].append({
                    "name": source.name,
                    "error": str(e),
                })

            # Rate-limit Sotwe/Twitter requests to avoid Cloudflare 503s
            if source.source_type == "twitter":
                time.sleep(settings.sotwe_request_delay)

        return results

    def add_source(
        self,
        name: str,
        source_type: str,
        url: str,
        priority: int = 50,
        fetch_interval_minutes: int = 60,
    ) -> ContentSource:
        """Add a new content source"""
        source = ContentSource(
            name=name,
            source_type=source_type,
            url=url,
            priority=priority,
            fetch_interval_minutes=fetch_interval_minutes,
            is_active=True,
        )
        self.db.add(source)
        self.db.commit()
        self.db.refresh(source)
        return source

    def get_unprocessed_items(self, limit: int = 100) -> list[ContentItem]:
        """Get content items that haven't been processed by LLM yet"""
        return self.db.query(ContentItem).filter(
            ContentItem.is_processed == False
        ).order_by(
            ContentItem.published_at.desc()
        ).limit(limit).all()

    def mark_processed(self, content_item_id: int, error: Optional[str] = None):
        """Mark a content item as processed"""
        item = self.db.query(ContentItem).filter(ContentItem.id == content_item_id).first()
        if item:
            item.is_processed = True
            item.processed_at = datetime.utcnow()
            if error:
                item.extraction_error = error
            self.db.commit()


# Default sources to seed the system
DEFAULT_SOURCES = [
    # Financial Substacks
    {"name": "Doomberg", "type": "substack", "url": "https://doomberg.substack.com/feed", "priority": 80},
    {"name": "SentimenTrader", "type": "substack", "url": "https://sentimentrader.substack.com/feed", "priority": 75},

    # Twitter/X Financial Accounts (examples - user should customize)
    {"name": "@Jesse_Livermore", "type": "twitter", "url": "https://twitter.com/Jesse_Livermore", "priority": 70},
    {"name": "@markminervini", "type": "twitter", "url": "https://twitter.com/markminervini", "priority": 70},

    # Reddit
    {"name": "r/wallstreetbets", "type": "reddit", "url": "https://reddit.com/r/wallstreetbets", "priority": 40},
    {"name": "r/stocks", "type": "reddit", "url": "https://reddit.com/r/stocks", "priority": 50},

    # News RSS
    {"name": "Yahoo Finance", "type": "news", "url": "https://finance.yahoo.com/rss/topstories", "priority": 60},
    {"name": "MarketWatch", "type": "news", "url": "https://feeds.marketwatch.com/marketwatch/topstories/", "priority": 60},
]


def seed_default_sources(db: Session):
    """Seed the database with default content sources"""
    service = ContentIngestionService(db)

    for source_config in DEFAULT_SOURCES:
        # Check if already exists
        existing = db.query(ContentSource).filter(
            ContentSource.source_type == source_config["type"],
            ContentSource.url == source_config["url"]
        ).first()

        if not existing:
            service.add_source(
                name=source_config["name"],
                source_type=source_config["type"],
                url=source_config["url"],
                priority=source_config["priority"],
            )
            logger.info(f"Added default source: {source_config['name']}")

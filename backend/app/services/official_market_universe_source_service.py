"""Fetch and parse official exchange universe snapshots for HK, IN, JP, KR, TW, CN, CA, and DE."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from email.utils import parsedate_to_datetime
import io
import json
import logging
import math
from pathlib import Path
import re
import time
from typing import Any, Iterable
from zoneinfo import ZoneInfo

from bs4 import BeautifulSoup
import pandas as pd
import requests
import urllib3

from ..config import settings
from .de_universe_source_signing import build_signed_headers

logger = logging.getLogger(__name__)


_HK_SOURCE_NAME = "hkex_official"
_IN_SOURCE_NAME = "in_reference_bundle"
_NSE_SOURCE_NAME = "nse_official"
_BSE_SOURCE_NAME = "bse_official"
_JP_SOURCE_NAME = "jpx_official"
_KR_SOURCE_NAME = "krx_official"
_TW_SOURCE_NAME = "tw_reference_bundle"
_CN_SOURCE_NAME = "cn_akshare_eastmoney"
_CA_SOURCE_NAME = "tmx_official"
_DE_SOURCE_NAME = "dbg_official"
# Source identifier emitted on snapshot rows when the live API is unreachable
# and the bundled DE seed CSV is used instead. Distinct from `_DE_SOURCE_NAME`
# so downstream coverage gates can tell live from fallback.
_DE_FALLBACK_SOURCE_NAME = "de_manual_csv"
# Boerse Frankfurt's public equity-search endpoint pages results in fixed-size
# windows. 100 keeps each request small enough to avoid timeouts while bounding
# the total round-trip count to ~10 for the ~1,000-symbol Xetra/Frankfurt list.
_BOERSE_FRANKFURT_PAGE_SIZE = 100
# Hard cap on pagination loops so a misbehaving upstream that never decrements
# `recordsTotal` cannot spin forever inside the GitHub Actions job.
_BOERSE_FRANKFURT_MAX_PAGES = 50
# Bail out of the live fetch after this many consecutive page failures so a
# salt rotation / DNS outage does not waste minutes retrying every offset
# before falling back to the CSV.
_BOERSE_FRANKFURT_MAX_CONSECUTIVE_PAGE_FAILURES = 3
# Yahoo-incompatible local codes are dropped from live results. The DE adapter
# enforces ``[A-Z0-9]{1,8}`` and downstream yfinance only accepts well-formed
# tickers — there is no value in publishing a row whose symbol cannot be
# resolved to a market data feed.
_DE_LIVE_TICKER_RE = re.compile(r"^[A-Z0-9]{1,8}$")
# Single tokens (matched as whole words after splitting the instrument-type
# string on non-alphanumeric chars). Plurals are listed explicitly so that
# "Notes" matches but "Footnotes" does not.
_CA_EXCLUDED_INSTRUMENT_TOKENS: frozenset[str] = frozenset(
    {
        "etf",
        "etn",
        "bond",
        "bonds",
        "debenture",
        "debentures",
        "note",
        "notes",
        "right",
        "rights",
        "warrant",
        "warrants",
        "nvcc",
        "convertible",
        "convertibles",
    }
)
# Multi-word phrases (matched as case-insensitive substrings; specific enough
# that false positives are unlikely).
_CA_EXCLUDED_INSTRUMENT_PHRASES: frozenset[str] = frozenset(
    {
        "exchange-traded fund",
        "exchange traded fund",
        "exchange-traded note",
        "exchange traded note",
        "mutual fund",
        "closed-end fund",
        "closed end fund",
        "investment fund",
        "subscription receipt",
        "structured product",
    }
)
_CA_INSTRUMENT_TOKEN_SPLIT = re.compile(r"[^a-z0-9]+")
_KR_VALIDATED_BASELINE = {
    "source_url": "https://global.krx.co.kr/main/main.jspx?bld=listing_list",
    "validated_at": "2026-04-29",
    "kospi": 839,
    "kosdaq": 1819,
    "konex": 110,
    "total": 2768,
    "kospi_kosdaq_target": 2658,
}
_KR_VALIDATED_BASELINE_TOLERANCE = 0.02
_CN_VALIDATED_BASELINE = {
    "source_url": "https://en.people.cn/n3/2026/0326/c90000-20440055.html",
    "validated_at": "2026-04-30",
    "as_of": "2026-02-28",
    "sse": 2310,
    "szse": 2887,
    "bse": 295,
    "total": 5492,
    "notes": "Xinhua/China Association for Public Companies end-February 2026 domestic listed company count.",
}
_CN_VALIDATED_BASELINE_TOLERANCE = 0.02

_HK_EQUITY_CATEGORY = "equity"
_HK_EQUITY_SUBCATEGORY_TOKEN = "equity securities"
_JP_ALLOWED_MARKET_SECTIONS = frozenset(
    {
        "プライム（内国株式）",
        "スタンダード（内国株式）",
        "グロース（内国株式）",
    }
)
_TW_UPDATED_AT_RE = re.compile(r"Date\s+Stock\s+Updated:\s*(\d{4}/\d{2}/\d{2})", re.IGNORECASE)
_TW_CODE_NAME_RE = re.compile(r"^([0-9A-Z]{3,6}[A-Z]?)\s+(.+?)$")
_HTTP_GET_MAX_ATTEMPTS = 3
_NSE_ARCHIVE_SOURCE_URL = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
_NSE_SOURCE_HEADERS = {
    "Accept": "text/csv,*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/",
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/123.0.0.0 Safari/537.36"
    ),
}
_BSE_SOURCE_HEADERS = {
    "Accept": "application/json,*/*",
    "Referer": "https://www.bseindia.com/corporates/List_Scrips.html",
}


@dataclass(frozen=True)
class OfficialMarketUniverseSnapshot:
    """Canonical upstream snapshot ready for stock_universe ingest."""

    market: str
    source_name: str
    snapshot_id: str
    snapshot_as_of: str
    source_metadata: dict[str, Any]
    rows: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class _FetchedSource:
    url: str
    content: bytes
    fetched_at: str
    last_modified: str | None
    tls_verification_disabled: bool


class OfficialMarketUniverseSourceService:
    """Fetch official exchange listings and normalize them into ingest rows."""

    def __init__(
        self,
        *,
        timeout_seconds: int | None = None,
        user_agent: str | None = None,
        kr_provider: Any | None = None,
        cn_provider: Any | None = None,
        market_calendar: Any | None = None,
    ) -> None:
        self._explicit_timeout_seconds = timeout_seconds
        self._timeout_seconds = int(
            timeout_seconds or settings.universe_source_timeout_seconds
        )
        self._user_agent = user_agent or settings.universe_source_user_agent
        self._kr_provider = kr_provider
        self._cn_provider = cn_provider
        self._market_calendar = market_calendar

    def fetch_market_snapshot(self, market: str) -> OfficialMarketUniverseSnapshot:
        normalized_market = str(market or "").strip().upper()
        if normalized_market == "HK":
            return self.fetch_hk_snapshot()
        if normalized_market == "IN":
            return self.fetch_in_snapshot()
        if normalized_market == "JP":
            return self.fetch_jp_snapshot()
        if normalized_market == "KR":
            return self.fetch_kr_snapshot()
        if normalized_market == "TW":
            return self.fetch_tw_snapshot()
        if normalized_market == "CN":
            return self.fetch_cn_snapshot()
        if normalized_market == "CA":
            return self.fetch_ca_snapshot()
        if normalized_market == "DE":
            return self.fetch_de_snapshot()
        raise ValueError(f"Official universe refresh is unsupported for market {market!r}")

    def fetch_hk_snapshot(self) -> OfficialMarketUniverseSnapshot:
        fetched = self._http_get(settings.hk_universe_source_url)
        fallback_as_of = self._date_from_http_header(fetched.last_modified) or self._utc_today()
        rows = self.parse_hk_rows(fetched.content)
        snapshot_as_of = fallback_as_of.isoformat()
        source_metadata = {
            "source_urls": [settings.hk_universe_source_url],
            "fetched_at": fetched.fetched_at,
            "http_last_modified": fetched.last_modified,
            "tls_verification_disabled": fetched.tls_verification_disabled,
            "filters": {
                "category_equals": "Equity",
                "sub_category_contains": "Equity Securities",
            },
            "sheet_name": "ListOfSecurities",
        }
        return OfficialMarketUniverseSnapshot(
            market="HK",
            source_name=_HK_SOURCE_NAME,
            snapshot_id=f"hkex-listofsecurities-{snapshot_as_of}",
            snapshot_as_of=snapshot_as_of,
            source_metadata=source_metadata,
            rows=tuple(rows),
        )

    def fetch_jp_snapshot(self) -> OfficialMarketUniverseSnapshot:
        fetched = self._http_get(settings.jp_universe_source_url)
        frame = self._read_excel_bytes(fetched.content, engine="xlrd")
        rows, snapshot_date = self._parse_jp_frame(frame)
        snapshot_as_of = snapshot_date.isoformat()
        source_metadata = {
            "source_urls": [settings.jp_universe_source_url],
            "fetched_at": fetched.fetched_at,
            "http_last_modified": fetched.last_modified,
            "tls_verification_disabled": fetched.tls_verification_disabled,
            "filters": {
                "market_product_category_in": sorted(_JP_ALLOWED_MARKET_SECTIONS),
            },
        }
        return OfficialMarketUniverseSnapshot(
            market="JP",
            source_name=_JP_SOURCE_NAME,
            snapshot_id=f"jpx-data-j-{snapshot_as_of}",
            snapshot_as_of=snapshot_as_of,
            source_metadata=source_metadata,
            rows=tuple(rows),
        )

    def fetch_kr_snapshot(self) -> OfficialMarketUniverseSnapshot:
        provider = self._get_kr_provider()
        requested_listing_as_of = self._kr_listing_as_of()
        rows = list(
            provider.listing_rows(boards=("KOSPI", "KOSDAQ"), as_of=requested_listing_as_of)
        )
        historical_listing_empty = False
        krx_listing_mode = "last_completed_trading_day"
        listing_as_of = requested_listing_as_of
        if self._kr_kospi_kosdaq_row_count(rows) == 0:
            historical_listing_empty = True
            logger.warning(
                "KR official universe fetch returned no rows for historical as_of=%s; "
                "retrying current KRX listing finder.",
                requested_listing_as_of.isoformat(),
            )
            rows = list(provider.listing_rows(boards=("KOSPI", "KOSDAQ"), as_of=None))
            if self._kr_kospi_kosdaq_row_count(rows) > 0:
                krx_listing_mode = "current_listing_fallback"
                listing_as_of = requested_listing_as_of
        if self._kr_kospi_kosdaq_row_count(rows) == 0:
            raise ValueError("KR official universe fetch returned no KOSPI/KOSDAQ rows")

        rows = self._kr_supported_board_rows(rows)
        rows = self._enrich_kr_rows_with_taxonomy(rows)
        row_counts = {
            "kospi": sum(1 for row in rows if str(row.get("exchange") or "").upper() == "KOSPI"),
            "kosdaq": sum(1 for row in rows if str(row.get("exchange") or "").upper() == "KOSDAQ"),
        }
        count_breaches = self._kr_baseline_count_breaches(row_counts)
        if count_breaches:
            logger.warning(
                "KR official universe fetch count outside static KRX baseline: %s",
                "; ".join(
                    f"{breach['board']} actual={breach['actual']} expected={breach['expected']} "
                    f"range=[{breach['min']},{breach['max']}]"
                    for breach in count_breaches
                ),
            )

        snapshot_as_of = listing_as_of.isoformat()
        source_metadata = {
            "source_urls": [_KR_VALIDATED_BASELINE["source_url"]],
            "fetched_at": datetime.now(UTC).isoformat(),
            "listing_as_of": snapshot_as_of,
            "requested_listing_as_of": requested_listing_as_of.isoformat(),
            "krx_listing_mode": krx_listing_mode,
            "historical_listing_empty": historical_listing_empty,
            "filters": {
                "boards": ["KOSPI", "KOSDAQ"],
                "excluded_products": ["ETF", "ETN", "ELW", "funds", "REITs"],
            },
            "excluded_boards": ["KONEX"],
            "row_counts": row_counts,
            "source_count": len(rows),
            "validated_krx_baseline": dict(_KR_VALIDATED_BASELINE),
            "validated_krx_baseline_tolerance": _KR_VALIDATED_BASELINE_TOLERANCE,
            "validated_krx_baseline_breaches": count_breaches,
            "krx_baseline_status": (
                "outside_static_baseline" if count_breaches else "within_static_baseline"
            ),
        }
        return OfficialMarketUniverseSnapshot(
            market="KR",
            source_name=_KR_SOURCE_NAME,
            snapshot_id=f"krx-listings-{snapshot_as_of}",
            snapshot_as_of=snapshot_as_of,
            source_metadata=source_metadata,
            rows=tuple(rows),
        )

    def fetch_cn_snapshot(self) -> OfficialMarketUniverseSnapshot:
        provider = self._get_cn_provider()
        rows = list(provider.listing_rows(as_of=self._utc_today()))
        if not rows:
            raise ValueError("CN official universe fetch returned no A-share rows")

        rows = self._enrich_cn_rows_with_taxonomy(rows)
        row_counts = {
            "sse": sum(1 for row in rows if str(row.get("exchange") or "").upper() == "SSE"),
            "szse": sum(1 for row in rows if str(row.get("exchange") or "").upper() == "SZSE"),
            "bse": sum(1 for row in rows if str(row.get("exchange") or "").upper() in {"BJSE", "BSE"}),
        }
        count_breaches = self._cn_baseline_count_breaches(row_counts)

        snapshot_as_of = self._utc_today().isoformat()
        source_metadata = {
            "source_urls": [
                _CN_VALIDATED_BASELINE["source_url"],
                "https://akshare.akfamily.xyz/data/stock/stock.html",
            ],
            "fetched_at": datetime.now(UTC).isoformat(),
            "snapshot_as_of": snapshot_as_of,
            "filters": {
                "market": "mainland_a_shares",
                "exchanges": ["SSE", "SZSE", "BJSE"],
                "excluded_products": [
                    "B-shares",
                    "ETFs",
                    "funds",
                    "bonds",
                    "convertibles",
                    "indices",
                    "NEEQ-only securities",
                ],
            },
            "row_counts": row_counts,
            "source_count": len(rows),
            "validated_cn_baseline": dict(_CN_VALIDATED_BASELINE),
            "validated_cn_baseline_tolerance": _CN_VALIDATED_BASELINE_TOLERANCE,
            "validated_cn_baseline_breaches": count_breaches,
            "cn_baseline_status": (
                "outside_static_baseline" if count_breaches else "within_static_baseline"
            ),
        }
        return OfficialMarketUniverseSnapshot(
            market="CN",
            source_name=_CN_SOURCE_NAME,
            snapshot_id=f"cn-a-share-{snapshot_as_of}",
            snapshot_as_of=snapshot_as_of,
            source_metadata=source_metadata,
            rows=tuple(rows),
        )

    @staticmethod
    def _seoul_today() -> date:
        return datetime.now(ZoneInfo("Asia/Seoul")).date()

    def _kr_listing_as_of(self) -> date:
        try:
            listing_day = self._get_market_calendar().last_completed_trading_day("KR")
            if isinstance(listing_day, datetime):
                return listing_day.date()
            if isinstance(listing_day, date):
                return listing_day
            raise ValueError(f"KR calendar returned invalid date: {listing_day!r}")
        except (AttributeError, ImportError, RuntimeError, TypeError, ValueError) as exc:
            logger.warning(
                "KR calendar lookup unavailable; using Seoul weekday fallback: %s",
                exc,
            )
            return self._previous_seoul_business_day(self._seoul_today())

    def _get_market_calendar(self):
        if self._market_calendar is None:
            from .market_calendar_service import MarketCalendarService

            self._market_calendar = MarketCalendarService()
        return self._market_calendar

    @staticmethod
    def _previous_seoul_business_day(day: date) -> date:
        if day.weekday() == 0:
            return day - timedelta(days=3)
        if day.weekday() == 6:
            return day - timedelta(days=2)
        return day - timedelta(days=1)

    def fetch_nse_snapshot(self) -> OfficialMarketUniverseSnapshot:
        source_urls = [settings.nse_universe_source_url]
        try:
            fetched = self._http_get(
                settings.nse_universe_source_url,
                extra_headers=_NSE_SOURCE_HEADERS,
            )
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
            if settings.nse_universe_source_url == _NSE_ARCHIVE_SOURCE_URL:
                raise
            source_urls.append(_NSE_ARCHIVE_SOURCE_URL)
            fetched = self._http_get(
                _NSE_ARCHIVE_SOURCE_URL,
                extra_headers=_NSE_SOURCE_HEADERS,
            )
        fallback_as_of = self._date_from_http_header(fetched.last_modified) or self._utc_today()
        rows = self.parse_nse_rows(fetched.content)
        snapshot_as_of = fallback_as_of.isoformat()
        return OfficialMarketUniverseSnapshot(
            market="IN",
            source_name=_NSE_SOURCE_NAME,
            snapshot_id=f"nse-equity-{snapshot_as_of}",
            snapshot_as_of=snapshot_as_of,
            source_metadata={
                "source_urls": source_urls,
                "fetched_at": fetched.fetched_at,
                "http_last_modified": fetched.last_modified,
                "tls_verification_disabled": fetched.tls_verification_disabled,
                "filters": {"series_equals": "EQ"},
            },
            rows=tuple(rows),
        )

    def _get_kr_provider(self):
        if self._kr_provider is None:
            from .kr_market_data_service import KrxMarketDataService

            self._kr_provider = KrxMarketDataService()
        return self._kr_provider

    def _get_cn_provider(self):
        if self._cn_provider is None:
            from .cn_market_data_service import CnMarketDataService

            self._cn_provider = CnMarketDataService(
                timeout_seconds=self._explicit_timeout_seconds,
            )
        return self._cn_provider

    @staticmethod
    def _kr_expected_range(expected: int) -> tuple[int, int]:
        tolerance = _KR_VALIDATED_BASELINE_TOLERANCE
        return (
            math.ceil(expected * (1.0 - tolerance)),
            math.floor(expected * (1.0 + tolerance)),
        )

    @staticmethod
    def _kr_kospi_kosdaq_row_count(rows: Iterable[dict[str, Any]]) -> int:
        return sum(
            1
            for row in rows
            if OfficialMarketUniverseSourceService._is_kr_supported_board(row)
        )

    @staticmethod
    def _kr_supported_board_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            row
            for row in rows
            if OfficialMarketUniverseSourceService._is_kr_supported_board(row)
        ]

    @staticmethod
    def _is_kr_supported_board(row: dict[str, Any]) -> bool:
        return str(row.get("exchange") or "").strip().upper() in {"KOSPI", "KOSDAQ"}

    @classmethod
    def _kr_baseline_count_breaches(cls, row_counts: dict[str, int]) -> list[dict[str, Any]]:
        breaches: list[dict[str, Any]] = []
        for board in ("kospi", "kosdaq"):
            expected = int(_KR_VALIDATED_BASELINE[board])
            minimum, maximum = cls._kr_expected_range(expected)
            actual = int(row_counts.get(board) or 0)
            if actual < minimum or actual > maximum:
                breaches.append(
                    {
                        "board": board,
                        "actual": actual,
                        "expected": expected,
                        "min": minimum,
                        "max": maximum,
                    }
                )
        return breaches

    @staticmethod
    def _cn_expected_range(expected: int) -> tuple[int, int]:
        tolerance = _CN_VALIDATED_BASELINE_TOLERANCE
        return (
            math.ceil(expected * (1.0 - tolerance)),
            math.floor(expected * (1.0 + tolerance)),
        )

    @classmethod
    def _cn_baseline_count_breaches(cls, row_counts: dict[str, int]) -> list[dict[str, Any]]:
        breaches: list[dict[str, Any]] = []
        for exchange in ("sse", "szse", "bse"):
            expected = int(_CN_VALIDATED_BASELINE[exchange])
            minimum, maximum = cls._cn_expected_range(expected)
            actual = int(row_counts.get(exchange) or 0)
            if actual < minimum or actual > maximum:
                breaches.append(
                    {
                        "exchange": exchange,
                        "actual": actual,
                        "expected": expected,
                        "min": minimum,
                        "max": maximum,
                    }
                )
        expected_total = int(_CN_VALIDATED_BASELINE["total"])
        minimum, maximum = cls._cn_expected_range(expected_total)
        actual_total = sum(int(row_counts.get(exchange) or 0) for exchange in ("sse", "szse", "bse"))
        if actual_total < minimum or actual_total > maximum:
            breaches.append(
                {
                    "exchange": "total",
                    "actual": actual_total,
                    "expected": expected_total,
                    "min": minimum,
                    "max": maximum,
                }
            )
        return breaches

    @staticmethod
    def _enrich_kr_rows_with_taxonomy(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Attach committed KR taxonomy fields where the KRX listing row is sparse."""
        try:
            from .market_taxonomy_service import TaxonomyLoadError, get_market_taxonomy_service
        except ImportError as exc:  # pragma: no cover - taxonomy availability is launch-gated separately
            logger.warning("KR taxonomy enrichment unavailable: %s", exc)
            return rows

        try:
            taxonomy = get_market_taxonomy_service()
        except TaxonomyLoadError as exc:  # pragma: no cover - taxonomy availability is launch-gated separately
            logger.warning("KR taxonomy enrichment unavailable: %s", exc)
            return rows

        enriched_rows: list[dict[str, Any]] = []
        for row in rows:
            enriched = dict(row)
            try:
                entry = taxonomy.get(
                    enriched.get("symbol") or enriched.get("local_code"),
                    market="KR",
                    exchange=enriched.get("exchange"),
                )
            except TaxonomyLoadError as exc:
                logger.warning("KR taxonomy enrichment unavailable: %s", exc)
                return rows
            if entry is not None:
                if not str(enriched.get("sector") or "").strip() and entry.sector:
                    enriched["sector"] = entry.sector
                if not str(enriched.get("industry_group") or "").strip() and entry.industry_group:
                    enriched["industry_group"] = entry.industry_group
                if not str(enriched.get("industry") or "").strip() and entry.industry:
                    enriched["industry"] = entry.industry
                if not str(enriched.get("sub_industry") or "").strip() and entry.sub_industry:
                    enriched["sub_industry"] = entry.sub_industry
            enriched_rows.append(enriched)
        return enriched_rows

    @staticmethod
    def _enrich_cn_rows_with_taxonomy(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Attach committed CN taxonomy fields where the listing row is sparse."""
        try:
            from .market_taxonomy_service import TaxonomyLoadError, get_market_taxonomy_service
        except ImportError as exc:  # pragma: no cover - taxonomy availability is launch-gated separately
            logger.warning("CN taxonomy enrichment unavailable: %s", exc)
            return rows

        try:
            taxonomy = get_market_taxonomy_service()
        except TaxonomyLoadError as exc:  # pragma: no cover - taxonomy availability is launch-gated separately
            logger.warning("CN taxonomy enrichment unavailable: %s", exc)
            return rows

        enriched_rows: list[dict[str, Any]] = []
        for row in rows:
            enriched = dict(row)
            try:
                entry = taxonomy.get(
                    enriched.get("symbol") or enriched.get("local_code"),
                    market="CN",
                    exchange=enriched.get("exchange"),
                )
            except TaxonomyLoadError as exc:
                logger.warning("CN taxonomy enrichment unavailable: %s", exc)
                return rows
            if entry is not None:
                if not str(enriched.get("sector") or "").strip() and entry.sector:
                    enriched["sector"] = entry.sector
                if not str(enriched.get("industry_group") or "").strip() and entry.industry_group:
                    enriched["industry_group"] = entry.industry_group
                if not str(enriched.get("industry") or "").strip() and entry.industry:
                    enriched["industry"] = entry.industry
                if not str(enriched.get("sub_industry") or "").strip() and entry.sub_industry:
                    enriched["sub_industry"] = entry.sub_industry
            enriched_rows.append(enriched)
        return enriched_rows

    def fetch_bse_snapshot(self) -> OfficialMarketUniverseSnapshot:
        fetched = self._http_get(
            settings.bse_universe_source_url,
            extra_headers=_BSE_SOURCE_HEADERS,
        )
        fallback_as_of = self._date_from_http_header(fetched.last_modified) or self._utc_today()
        rows = self.parse_bse_rows(fetched.content)
        snapshot_as_of = fallback_as_of.isoformat()
        return OfficialMarketUniverseSnapshot(
            market="IN",
            source_name=_BSE_SOURCE_NAME,
            snapshot_id=f"bse-equity-{snapshot_as_of}",
            snapshot_as_of=snapshot_as_of,
            source_metadata={
                "source_urls": [settings.bse_universe_source_url],
                "fetched_at": fetched.fetched_at,
                "http_last_modified": fetched.last_modified,
                "tls_verification_disabled": fetched.tls_verification_disabled,
                "filters": {"segment_equals": "Equity", "status_equals": "Active"},
            },
            rows=tuple(rows),
        )

    def fetch_in_snapshot(self) -> OfficialMarketUniverseSnapshot:
        nse_snapshot = self.fetch_nse_snapshot()
        bse_snapshot = self.fetch_bse_snapshot()

        nse_by_isin = {
            str(row.get("isin") or "").strip().upper(): row
            for row in nse_snapshot.rows
            if str(row.get("isin") or "").strip()
        }
        bse_by_isin = {
            str(row.get("isin") or "").strip().upper(): row
            for row in bse_snapshot.rows
            if str(row.get("isin") or "").strip()
        }

        overlap_isins = sorted(set(nse_by_isin) & set(bse_by_isin))
        combined_rows = list(sorted(nse_snapshot.rows, key=lambda row: str(row.get("symbol") or "")))
        for isin, row in sorted(bse_by_isin.items()):
            if isin in nse_by_isin:
                continue
            combined_rows.append(row)

        snapshot_as_of = max(nse_snapshot.snapshot_as_of, bse_snapshot.snapshot_as_of)
        return OfficialMarketUniverseSnapshot(
            market="IN",
            source_name=_IN_SOURCE_NAME,
            snapshot_id=f"in-reference-bundle-{snapshot_as_of}",
            snapshot_as_of=snapshot_as_of,
            source_metadata={
                "source_urls": [
                    settings.nse_universe_source_url,
                    settings.bse_universe_source_url,
                ],
                "nse_snapshot_id": nse_snapshot.snapshot_id,
                "bse_snapshot_id": bse_snapshot.snapshot_id,
                "nse_count": len(nse_snapshot.rows),
                "bse_count": len(bse_snapshot.rows),
                "overlap_isin_count": len(overlap_isins),
                "combined_count": len(combined_rows),
            },
            rows=tuple(combined_rows),
        )

    def fetch_tw_snapshot(self) -> OfficialMarketUniverseSnapshot:
        twse_fetched = self._http_get(
            settings.tw_universe_source_twse_url,
            allow_insecure_fallback=settings.tw_universe_allow_insecure_fallback,
        )
        tpex_fetched = self._http_get(
            settings.tw_universe_source_tpex_url,
            allow_insecure_fallback=settings.tw_universe_allow_insecure_fallback,
        )

        twse_html = twse_fetched.content.decode("cp950", errors="replace")
        tpex_html = tpex_fetched.content.decode("cp950", errors="replace")
        twse_date = self._parse_tw_updated_at(twse_html)
        tpex_date = self._parse_tw_updated_at(tpex_html)
        if twse_date != tpex_date:
            raise ValueError(
                "TW reference bundle date mismatch between TWSE and TPEx "
                f"({twse_date.isoformat()} vs {tpex_date.isoformat()})"
            )

        twse_rows = self.parse_tw_rows(twse_html, exchange="TWSE")
        tpex_rows = self.parse_tw_rows(tpex_html, exchange="TPEX")
        snapshot_as_of = twse_date.isoformat()
        source_metadata = {
            "source_urls": [
                settings.tw_universe_source_twse_url,
                settings.tw_universe_source_tpex_url,
            ],
            "fetched_at": max(twse_fetched.fetched_at, tpex_fetched.fetched_at),
            "http_last_modified": {
                "twse": twse_fetched.last_modified,
                "tpex": tpex_fetched.last_modified,
            },
            "tls_verification_disabled": {
                "twse": twse_fetched.tls_verification_disabled,
                "tpex": tpex_fetched.tls_verification_disabled,
            },
            "filters": {
                "sections": ["Stocks"],
                "exchanges": ["TWSE", "TPEX"],
            },
            "row_counts": {
                "twse": len(twse_rows),
                "tpex": len(tpex_rows),
            },
        }
        return OfficialMarketUniverseSnapshot(
            market="TW",
            source_name=_TW_SOURCE_NAME,
            snapshot_id=f"tw-reference-bundle-{snapshot_as_of}",
            snapshot_as_of=snapshot_as_of,
            source_metadata=source_metadata,
            rows=tuple([*twse_rows, *tpex_rows]),
        )

    def fetch_ca_snapshot(self) -> OfficialMarketUniverseSnapshot:
        """Fetch combined TSX + TSXV issuer listings from TMX.

        TMX publishes per-board JSON issuer directories. Each endpoint returns
        a payload like ``{"length": N, "results": [{"symbol": "...",
        "name": "...", "instrumentType": "..."}]}``. We combine both boards,
        exclude non-equity instrument types (ETFs, debentures, warrants, etc.),
        and tag each row with its exchange so the ingestion adapter can pick
        the correct ``.TO`` / ``.V`` suffix.

        The configured URLs may contain a ``{initial}`` placeholder, in which
        case we iterate A-Z and concatenate the results. This is needed for
        TMX's letter-bucketed directory; a single letter typically returns
        ~50-200 issuers so all 26 fetches together cover the universe. Per-
        letter failures are logged and skipped — the snapshot still succeeds
        as long as at least one letter returned rows on each board.
        """
        tsx_rows, tsx_metadata = self._fetch_ca_board("TSX", settings.ca_universe_source_tsx_url)
        tsxv_rows, tsxv_metadata = self._fetch_ca_board("TSXV", settings.ca_universe_source_tsxv_url)

        # Both boards must contribute rows. A full TSX or TSXV outage (or
        # parse failure on every letter) should surface as a hard error so
        # operators can retry, rather than silently publishing a half-empty
        # CA snapshot that bypasses reconciliation safety checks downstream.
        empty_boards = [
            board for board, rows in (("TSX", tsx_rows), ("TSXV", tsxv_rows)) if not rows
        ]
        if empty_boards:
            raise ValueError(
                "CA official universe fetch returned no equity rows for "
                f"{', '.join(empty_boards)}; refusing to publish a partial snapshot."
            )

        snapshot_as_of = (
            self._date_from_http_header(tsx_metadata.get("http_last_modified"))
            or self._date_from_http_header(tsxv_metadata.get("http_last_modified"))
            or self._utc_today()
        ).isoformat()
        fetched_ats = [
            metadata["fetched_at"]
            for metadata in (tsx_metadata, tsxv_metadata)
            if metadata.get("fetched_at")
        ]
        source_metadata = {
            "source_urls": [
                settings.ca_universe_source_tsx_url,
                settings.ca_universe_source_tsxv_url,
            ],
            "fetched_at": max(fetched_ats) if fetched_ats else datetime.now(UTC).isoformat(),
            "http_last_modified": {
                "tsx": tsx_metadata.get("http_last_modified"),
                "tsxv": tsxv_metadata.get("http_last_modified"),
            },
            "tls_verification_disabled": {
                "tsx": tsx_metadata.get("tls_verification_disabled"),
                "tsxv": tsxv_metadata.get("tls_verification_disabled"),
            },
            "fetch_mode": {
                "tsx": tsx_metadata.get("fetch_mode"),
                "tsxv": tsxv_metadata.get("fetch_mode"),
            },
            "fetch_attempts": {
                "tsx": tsx_metadata.get("attempts"),
                "tsxv": tsxv_metadata.get("attempts"),
            },
            "fetch_errors": {
                "tsx": tsx_metadata.get("errors"),
                "tsxv": tsxv_metadata.get("errors"),
            },
            "filters": {
                "exchanges": ["TSX", "TSXV"],
                "excluded_instrument_tokens": sorted(_CA_EXCLUDED_INSTRUMENT_TOKENS),
                "excluded_instrument_phrases": sorted(_CA_EXCLUDED_INSTRUMENT_PHRASES),
            },
            "row_counts": {
                "tsx": len(tsx_rows),
                "tsxv": len(tsxv_rows),
            },
        }
        return OfficialMarketUniverseSnapshot(
            market="CA",
            source_name=_CA_SOURCE_NAME,
            snapshot_id=f"tmx-issuer-directory-{snapshot_as_of}",
            snapshot_as_of=snapshot_as_of,
            source_metadata=source_metadata,
            rows=tuple([*tsx_rows, *tsxv_rows]),
        )

    def _fetch_ca_board(
        self,
        exchange: str,
        url_template: str,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Fetch one TMX board (TSX or TSXV) with optional A-Z pagination.

        Returns ``(rows, metadata)`` where metadata describes how the fetch
        was performed (single vs paginated, attempt count, last_modified,
        per-letter errors). Symbols are deduplicated across letter buckets
        on ``(exchange, symbol)``.
        """
        templated = "{initial}" in url_template
        attempts: list[str] = []
        errors: dict[str, str] = {}
        last_modified: str | None = None
        tls_verification_disabled = False
        fetched_ats: list[str] = []
        rows_by_symbol: dict[str, dict[str, Any]] = {}

        if templated:
            letters = [chr(code) for code in range(ord("A"), ord("Z") + 1)]
            for letter in letters:
                url = url_template.format(initial=letter)
                attempts.append(letter)
                try:
                    fetched = self._http_get(
                        url,
                        allow_insecure_fallback=settings.ca_universe_allow_insecure_fallback,
                    )
                except requests.exceptions.RequestException as exc:
                    errors[letter] = str(exc)
                    logger.warning(
                        "TMX directory fetch failed for %s/%s: %s", exchange, letter, exc
                    )
                    continue
                fetched_ats.append(fetched.fetched_at)
                if fetched.last_modified:
                    last_modified = fetched.last_modified
                if fetched.tls_verification_disabled:
                    tls_verification_disabled = True
                try:
                    bucket_rows = self.parse_ca_rows(fetched.content, exchange=exchange)
                except ValueError as exc:
                    errors[letter] = f"parse failed: {exc}"
                    logger.warning(
                        "TMX directory parse failed for %s/%s: %s", exchange, letter, exc
                    )
                    continue
                for row in bucket_rows:
                    rows_by_symbol.setdefault(row["symbol"], row)
        else:
            attempts.append("single")
            fetched = self._http_get(
                url_template,
                allow_insecure_fallback=settings.ca_universe_allow_insecure_fallback,
            )
            fetched_ats.append(fetched.fetched_at)
            last_modified = fetched.last_modified
            tls_verification_disabled = fetched.tls_verification_disabled
            for row in self.parse_ca_rows(fetched.content, exchange=exchange):
                rows_by_symbol.setdefault(row["symbol"], row)

        rows = sorted(rows_by_symbol.values(), key=lambda row: row["symbol"])
        metadata: dict[str, Any] = {
            "fetch_mode": "letter_buckets" if templated else "single_url",
            "attempts": attempts,
            "errors": errors,
            "fetched_at": max(fetched_ats) if fetched_ats else None,
            "http_last_modified": last_modified,
            "tls_verification_disabled": tls_verification_disabled,
        }
        return rows, metadata

    @classmethod
    def parse_ca_rows(cls, content: bytes, *, exchange: str) -> list[dict[str, Any]]:
        """Parse a TMX issuer-directory JSON payload into ingestion rows."""
        if exchange not in ("TSX", "TSXV"):
            raise ValueError(f"Unsupported CA exchange '{exchange}'")

        try:
            payload = json.loads(content.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"Invalid TMX directory payload for {exchange}: {exc}"
            ) from exc

        if isinstance(payload, dict):
            results = payload.get("results") or payload.get("companies") or []
        elif isinstance(payload, list):
            results = payload
        else:
            raise ValueError(f"Unexpected TMX payload shape for {exchange}: {type(payload).__name__}")

        rows: list[dict[str, Any]] = []
        for entry in results:
            if not isinstance(entry, dict):
                continue
            raw_symbol = (
                entry.get("symbol")
                or entry.get("ticker")
                or entry.get("rootTicker")
                or ""
            )
            symbol = str(raw_symbol).strip().upper().replace(" ", "")
            if not symbol:
                continue

            instrument_type = str(
                entry.get("instrumentType")
                or entry.get("type")
                or entry.get("securityType")
                or ""
            ).strip().lower()
            if cls._ca_is_excluded_instrument(instrument_type):
                continue

            name = str(entry.get("name") or entry.get("issuerName") or entry.get("companyName") or "").strip()
            sector = str(entry.get("sector") or entry.get("gicsSector") or "").strip()
            industry = str(entry.get("industry") or entry.get("gicsIndustry") or "").strip()

            rows.append(
                {
                    "symbol": symbol,
                    "name": name,
                    "exchange": exchange,
                    "sector": sector,
                    "industry": industry,
                    "market_cap": entry.get("marketCap"),
                    "instrument_type": instrument_type,
                }
            )

        return rows

    @staticmethod
    def _ca_is_excluded_instrument(instrument_type: str) -> bool:
        """Return True for non-equity TMX instrument types.

        Multi-word phrases (e.g. "Closed-End Fund") match as substrings.
        Single-word tokens (e.g. "ETF", "Notes") match only as whole words —
        so "Common Shares" passes but "Notes 6.5%" is excluded.
        """
        if not instrument_type:
            return False
        normalized = instrument_type.lower().strip()
        for phrase in _CA_EXCLUDED_INSTRUMENT_PHRASES:
            if phrase in normalized:
                return True
        tokens = (token for token in _CA_INSTRUMENT_TOKEN_SPLIT.split(normalized) if token)
        return any(token in _CA_EXCLUDED_INSTRUMENT_TOKENS for token in tokens)

    def fetch_de_snapshot(self) -> OfficialMarketUniverseSnapshot:
        """Fetch the DE equity universe from Boerse Frankfurt with CSV fallback.

        Live path (best-effort, see caveats below): paginate the public
        ``/v1/search/equity_search`` endpoint with the signed headers from
        ``de_universe_source_signing.build_signed_headers``, derive a
        Yahoo-compatible ticker via ``_derive_de_ticker``, dedupe by ISIN, and
        prefer Xetra over Frankfurt-floor listings.

        Fallback path: when the live request fails (DNS/timeout/HTTP error,
        signature rotation, every page errored, no row resolved to a usable
        ticker, the live snapshot dropped curated baseline symbols, or fewer
        than ``settings.de_live_min_resolution_ratio`` of upstream rows
        resolved) load the bundled DAX-40 CSV at
        ``settings.de_universe_fallback_csv_path``. The snapshot is still
        published, but ``source_metadata['fetch_mode'] == 'csv_fallback'`` and
        ``source_name == 'de_manual_csv'`` so the coverage gate / downstream
        operators can see the snapshot is incomplete. The two safety guards
        (CSV-superset and minimum-resolution-ratio) prevent
        ``ingest_de_snapshot_rows`` from deactivating known DAX names when
        the live API returns partial data or our resolution heuristics
        regress for a subset of the universe.

        ## Live-path caveats

        Two assumptions in the live path are not validated against a real
        endpoint from this codebase and may regress without notice:

        1. The ``X-Security`` signing scheme + public salt in
           ``de_universe_source_signing`` is reverse-engineered from the
           boerse-frankfurt.de SPA. If Deutsche Boerse rotates the salt or
           changes the algorithm, every signed request returns 401/403 and
           the CSV fallback takes over.
        2. The Boerse Frankfurt search payload exposes ``isin``, ``wkn``,
           ``slug``, etc. but does not directly carry the Yahoo Finance
           ticker. ``_derive_de_ticker`` first checks for explicit
           ``tickerSymbol``/``ticker`` fields, then consults the bundled
           CSV's ISIN-to-ticker map. Rows that fall through both are
           dropped so the live path never publishes a Yahoo-incompatible
           symbol like ``716460.DE`` (the WKN, not the ticker). Until the
           bundled CSV ships verified ISIN→ticker pairs for the wider
           Xetra/Frankfurt universe, the live path will resolve only rows
           that the API tags with an explicit ticker field, and the CSV
           fallback is responsible for the rest.

        Together these mean operators should treat the live path as
        opportunistic and expect the CSV fallback to be the contract until
        somebody cuts a release after verifying the live API end-to-end.
        """
        fetch_mode = "live_http"
        fetch_errors: dict[str, str] = {}
        fetched_ats: list[str] = []
        last_modified: str | None = None
        tls_verification_disabled = False
        page_attempts: list[int] = []
        page_errors: dict[str, str] = {}
        ticker_resolution: dict[str, int] = {}

        try:
            live_meta = self._fetch_de_live()
        except (requests.exceptions.RequestException, ValueError) as exc:
            live_error = str(exc)
            logger.warning(
                "Live DE universe fetch failed (%s); falling back to bundled CSV at %s",
                live_error,
                settings.de_universe_fallback_csv_path,
            )
            fetch_errors["live_http"] = live_error
            rows = self._load_de_csv_fallback()
            if not rows:
                raise ValueError(
                    "DE official universe fetch failed and no fallback CSV rows are available "
                    f"(live error: {live_error})"
                ) from exc
            fetch_mode = "csv_fallback"
            fetched_ats = [datetime.now(UTC).isoformat()]
        else:
            rows = live_meta["rows"]
            page_attempts = live_meta.get("page_attempts", [])
            page_errors = live_meta.get("page_errors", {})
            fetched_ats = live_meta.get("fetched_ats", [])
            last_modified = live_meta.get("http_last_modified")
            tls_verification_disabled = bool(live_meta.get("tls_verification_disabled"))
            ticker_resolution = live_meta.get("ticker_resolution", {})
            if page_errors:
                fetch_errors.update({f"page_{k}": v for k, v in page_errors.items()})

        if not rows:
            raise ValueError("DE official universe fetch returned no rows")

        snapshot_as_of = (
            self._date_from_http_header(last_modified) or self._utc_today()
        ).isoformat()
        source_name = (
            _DE_FALLBACK_SOURCE_NAME if fetch_mode == "csv_fallback" else _DE_SOURCE_NAME
        )
        source_metadata: dict[str, Any] = {
            "source_urls": [settings.de_universe_source_url],
            "fetched_at": max(fetched_ats) if fetched_ats else datetime.now(UTC).isoformat(),
            "http_last_modified": last_modified,
            "tls_verification_disabled": tls_verification_disabled,
            "fetch_mode": fetch_mode,
            "page_attempts": page_attempts,
            "fetch_errors": fetch_errors,
            "row_counts": {
                "xetra": sum(1 for row in rows if row.get("exchange") == "XETR"),
                "frankfurt": sum(1 for row in rows if row.get("exchange") == "XFRA"),
                "total": len(rows),
            },
        }
        if fetch_mode == "live_http" and ticker_resolution:
            source_metadata["ticker_resolution"] = ticker_resolution
        if fetch_mode == "csv_fallback":
            source_metadata["fallback_csv_path"] = settings.de_universe_fallback_csv_path

        snapshot_id_prefix = "dbg-equity" if fetch_mode == "live_http" else "de-csv-fallback"
        return OfficialMarketUniverseSnapshot(
            market="DE",
            source_name=source_name,
            snapshot_id=f"{snapshot_id_prefix}-{snapshot_as_of}",
            snapshot_as_of=snapshot_as_of,
            source_metadata=source_metadata,
            rows=tuple(rows),
        )

    def _fetch_de_live(self) -> dict[str, Any]:
        """Paginate the Boerse Frankfurt search API tolerating per-page failures.

        Mirrors ``_fetch_ca_board``'s resilience pattern: a transient HTTP or
        parse error on any single page is logged and recorded but does not
        abort the whole fetch. The live path only fails (and triggers the
        CSV fallback) when *every* page errored or no row resolved to a
        Yahoo-compatible ticker.
        """
        base_url = settings.de_universe_source_url
        isin_to_ticker = self._load_de_isin_to_ticker_map()
        rows_by_isin: dict[str, dict[str, Any]] = {}
        page_attempts: list[int] = []
        page_errors: dict[str, str] = {}
        fetched_ats: list[str] = []
        last_modified: str | None = None
        tls_verification_disabled = False
        ticker_resolution: dict[str, int] = {
            "explicit_field": 0,
            "isin_map": 0,
            "unresolved": 0,
        }
        consecutive_failures = 0

        for page_index in range(_BOERSE_FRANKFURT_MAX_PAGES):
            offset = page_index * _BOERSE_FRANKFURT_PAGE_SIZE
            url = (
                f"{base_url}?lang=en&offset={offset}"
                f"&limit={_BOERSE_FRANKFURT_PAGE_SIZE}&sort=NAME&sortOrder=ASC"
            )
            page_attempts.append(offset)
            try:
                fetched = self._http_get(
                    url,
                    allow_insecure_fallback=settings.de_universe_allow_insecure_fallback,
                    extra_headers=build_signed_headers(url),
                )
            except requests.exceptions.RequestException as exc:
                page_errors[str(offset)] = str(exc)
                logger.warning(
                    "DE live fetch failed for offset %s: %s", offset, exc
                )
                consecutive_failures += 1
                if consecutive_failures >= _BOERSE_FRANKFURT_MAX_CONSECUTIVE_PAGE_FAILURES:
                    logger.warning(
                        "DE live fetch hit %s consecutive page failures; aborting",
                        consecutive_failures,
                    )
                    break
                continue

            fetched_ats.append(fetched.fetched_at)
            if fetched.last_modified:
                last_modified = fetched.last_modified
            if fetched.tls_verification_disabled:
                tls_verification_disabled = True

            try:
                page_rows, total_count, page_resolution, raw_row_count = (
                    self._parse_de_results(
                        fetched.content, isin_to_ticker=isin_to_ticker
                    )
                )
            except ValueError as exc:
                page_errors[str(offset)] = f"parse failed: {exc}"
                logger.warning(
                    "DE live parse failed for offset %s: %s", offset, exc
                )
                consecutive_failures += 1
                if consecutive_failures >= _BOERSE_FRANKFURT_MAX_CONSECUTIVE_PAGE_FAILURES:
                    logger.warning(
                        "DE live fetch hit %s consecutive page failures; aborting",
                        consecutive_failures,
                    )
                    break
                continue

            consecutive_failures = 0
            for bucket, count in page_resolution.items():
                ticker_resolution[bucket] = ticker_resolution.get(bucket, 0) + count

            for row in page_rows:
                isin = row.get("isin")
                if not isin:
                    continue
                # Prefer Xetra over Frankfurt-floor for the same ISIN: Xetra is
                # the primary venue and yields the .DE suffix the rest of the
                # platform expects. If we already have a row but the new one is
                # XETR and the existing one is XFRA, swap. Otherwise keep the
                # first hit.
                existing = rows_by_isin.get(isin)
                if existing is None:
                    rows_by_isin[isin] = row
                elif existing.get("exchange") != "XETR" and row.get("exchange") == "XETR":
                    rows_by_isin[isin] = row

            # Pagination break checks use the *raw* upstream row count, not the
            # post-filter ``page_rows`` length. Otherwise a full page that
            # happened to contain many unresolved-ticker rows would look like a
            # partial page and stop pagination early.
            if raw_row_count == 0:
                break
            if total_count and (offset + raw_row_count) >= total_count:
                break
            if raw_row_count < _BOERSE_FRANKFURT_PAGE_SIZE:
                break

        successful_pages = len(page_attempts) - len(page_errors)
        if successful_pages <= 0:
            raise ValueError(
                f"DE live universe fetch failed for every page ({len(page_errors)} attempts)"
            )
        if not rows_by_isin:
            raise ValueError(
                "DE live universe fetch returned no rows whose tickers could be resolved"
            )

        # Safety guards against silent universe shrinkage. The reconciler in
        # ``ingest_de_snapshot_rows`` deactivates rows that are absent from the
        # snapshot, so a partially-resolved live fetch (e.g. when only a few
        # rows carry an explicit ticker field and the CSV-derived ISIN map is
        # sparse) would silently disable known DAX names. Raising here forces
        # ``fetch_de_snapshot`` to fall back to the curated CSV, which is the
        # safer floor.
        live_symbols = {row["symbol"] for row in rows_by_isin.values()}
        csv_symbols = {row["symbol"] for row in self._load_de_csv_fallback()}
        missing_csv_symbols = csv_symbols - live_symbols
        if missing_csv_symbols:
            sample = sorted(missing_csv_symbols)[:5]
            raise ValueError(
                f"DE live universe fetch missing {len(missing_csv_symbols)} curated "
                f"baseline symbols (e.g. {sample}); refusing to publish — would "
                "deactivate known names"
            )

        min_ratio = float(getattr(settings, "de_live_min_resolution_ratio", 0.0))
        # ``total_count`` is the most recent value reported by the API. We
        # guarded above that at least one page succeeded, so it has been
        # assigned at least once when we reach this point.
        if min_ratio > 0 and total_count > 0:
            resolved_ratio = len(rows_by_isin) / total_count
            if resolved_ratio < min_ratio:
                raise ValueError(
                    f"DE live universe resolved only {len(rows_by_isin)}/{total_count} "
                    f"rows ({resolved_ratio:.1%}, below {min_ratio:.0%} threshold); "
                    "refusing to publish a partial bundle"
                )

        rows = sorted(rows_by_isin.values(), key=lambda row: row["symbol"])
        return {
            "rows": rows,
            "page_attempts": page_attempts,
            "page_errors": page_errors,
            "fetched_ats": fetched_ats,
            "http_last_modified": last_modified,
            "tls_verification_disabled": tls_verification_disabled,
            "ticker_resolution": ticker_resolution,
        }

    @classmethod
    def _parse_de_results(
        cls,
        content: bytes,
        *,
        isin_to_ticker: dict[str, str] | None = None,
    ) -> tuple[list[dict[str, Any]], int, dict[str, int], int]:
        """Parse a Boerse Frankfurt search-API page into ingestion row dicts.

        Returns ``(rows, total_count, ticker_resolution, raw_row_count)``.
        ``rows`` is the post-filtering list (only entries with a resolvable
        Yahoo-compatible ticker). ``raw_row_count`` is the size of the upstream
        payload's data array before any filtering — the caller uses this for
        partial-page detection so a page full of unresolved tickers doesn't
        prematurely stop pagination. ``ticker_resolution`` is a per-bucket
        counter recording how each resolved row was matched.
        """
        try:
            payload = json.loads(content.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid Boerse Frankfurt response: {exc}") from exc

        if not isinstance(payload, dict):
            raise ValueError(
                f"Unexpected Boerse Frankfurt payload shape: {type(payload).__name__}"
            )

        raw_rows = payload.get("data") or payload.get("results") or []
        if not isinstance(raw_rows, list):
            raise ValueError(
                "Boerse Frankfurt 'data' field is not a list "
                f"(got {type(raw_rows).__name__})"
            )
        raw_row_count = len(raw_rows)

        total_count = 0
        for total_key in ("totalCount", "recordsTotal", "total"):
            value = payload.get(total_key)
            if isinstance(value, (int, float)) and value > 0:
                total_count = int(value)
                break

        isin_map = isin_to_ticker or {}
        rows: list[dict[str, Any]] = []
        resolution = {
            "explicit_field": 0,
            "isin_map": 0,
            "unresolved": 0,
        }
        for entry in raw_rows:
            if not isinstance(entry, dict):
                continue
            isin = str(entry.get("isin") or "").strip().upper()
            wkn = str(entry.get("wkn") or "").strip().upper()
            # Only ISIN is required: it keys the dedupe map and is what the
            # downstream reconciler matches on. WKN is carried through as
            # diagnostic context but isn't used for ticker resolution
            # (``_derive_de_ticker`` keys off ``tickerSymbol``/``ticker`` or
            # the ISIN map), so a row with ``isin + tickerSymbol`` but no WKN
            # is still publishable. Dropping those would shrink the live
            # universe unnecessarily.
            if not isin:
                continue

            name_field = entry.get("name")
            if isinstance(name_field, dict):
                name = str(
                    name_field.get("originalValue")
                    or name_field.get("translation")
                    or name_field.get("display")
                    or ""
                ).strip()
            else:
                name = str(name_field or "").strip()

            is_xetra = bool(
                entry.get("isXetraTradable")
                or entry.get("isXetraEquity")
                or entry.get("xetraTradable")
            )
            exchange = "XETR" if is_xetra else "XFRA"
            symbol_suffix = ".DE" if exchange == "XETR" else ".F"

            ticker, source = cls._derive_de_ticker(entry, isin=isin, isin_to_ticker=isin_map)
            if ticker is None:
                resolution["unresolved"] += 1
                logger.debug(
                    "Dropping DE live row %s (wkn=%s): no Yahoo-compatible ticker",
                    isin,
                    wkn,
                )
                continue
            resolution[source] += 1

            rows.append(
                {
                    "symbol": f"{ticker}{symbol_suffix}",
                    "name": name,
                    "exchange": exchange,
                    "sector": str(entry.get("sector") or "").strip(),
                    "industry": str(entry.get("industry") or "").strip(),
                    "market_cap": entry.get("marketCap"),
                    "isin": isin,
                    "wkn": wkn,
                }
            )

        return rows, total_count, resolution, raw_row_count

    @classmethod
    def _derive_de_ticker(
        cls,
        entry: dict[str, Any],
        *,
        isin: str,
        isin_to_ticker: dict[str, str],
    ) -> tuple[str | None, str]:
        """Resolve a Yahoo-compatible ticker from a Boerse Frankfurt entry.

        Resolution order: explicit ``tickerSymbol`` / ``ticker`` field →
        bundled-CSV ISIN map. Returns ``(ticker, "explicit_field" | "isin_map")``
        or ``(None, "unresolved")`` if neither path yields a value matching
        the DE adapter's local-code pattern.

        Fields that look like tickers but are not are intentionally NOT
        consulted: ``shortName`` is a descriptive company label (``Adidas``,
        ``SAP SE``) that may accidentally match the ``[A-Z0-9]{1,8}`` regex
        even though it is not a real exchange ticker. Letting such values
        through would publish plausible-but-wrong symbols and inflate the
        resolved-row count past the safety guards in ``_fetch_de_live``.
        WKN is never used either: Yahoo Finance keys German equities by
        ticker (``SAP``), not by WKN (``716460``). Slug-based derivation was
        tried earlier in the PR and removed for the same reason — slugs
        encode the company name, not the ticker. Until the bundled ISIN map
        is populated with verified ISIN→ticker pairs for the wider Xetra /
        Frankfurt universe, only rows that the API labels with an explicit
        ticker field can resolve via the live path; everything else is
        dropped and the CSV fallback covers the gap.
        """
        for field in ("tickerSymbol", "ticker"):
            candidate = str(entry.get(field) or "").strip().upper()
            if candidate and _DE_LIVE_TICKER_RE.fullmatch(candidate):
                return candidate, "explicit_field"

        if isin and isin in isin_to_ticker:
            return isin_to_ticker[isin], "isin_map"

        return None, "unresolved"

    @classmethod
    def _load_de_isin_to_ticker_map(cls) -> dict[str, str]:
        """Build an ISIN→ticker map from the bundled DE seed CSV.

        The CSV ships with the curated DAX-40 (and any future MDAX/SDAX
        additions). Live-fetch rows whose ISIN is in this map can publish a
        ticker that's known to round-trip through yfinance, even when the
        live API doesn't expose a usable ticker field. Returns an empty map
        if the CSV is missing — callers tolerate that and fall through to
        slug-based derivation.
        """
        csv_path = Path(settings.de_universe_fallback_csv_path)
        if not csv_path.exists():
            return {}

        frame = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
        if "isin" not in frame.columns or "symbol" not in frame.columns:
            return {}

        isin_to_ticker: dict[str, str] = {}
        for record in frame.to_dict("records"):
            isin = str(record.get("isin") or "").strip().upper()
            symbol = str(record.get("symbol") or "").strip().upper()
            if not isin or not symbol:
                continue
            ticker = symbol.split(".", 1)[0]
            if _DE_LIVE_TICKER_RE.fullmatch(ticker):
                isin_to_ticker.setdefault(isin, ticker)
        return isin_to_ticker

    @classmethod
    def _load_de_csv_fallback(cls) -> list[dict[str, Any]]:
        """Read the bundled DE seed CSV into the same row schema as the live path."""
        csv_path = Path(settings.de_universe_fallback_csv_path)
        if not csv_path.exists():
            return []

        frame = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
        rows: list[dict[str, Any]] = []
        for record in frame.to_dict("records"):
            symbol = str(record.get("symbol") or "").strip()
            name = str(record.get("name") or "").strip()
            exchange = str(record.get("exchange") or "XETR").strip().upper()
            if not symbol or not name:
                continue
            rows.append(
                {
                    "symbol": symbol,
                    "name": name,
                    "exchange": exchange,
                    "sector": "",
                    "industry": "",
                    "market_cap": None,
                }
            )
        return sorted(rows, key=lambda row: row["symbol"])

    def parse_hk_rows(self, content: bytes) -> list[dict[str, Any]]:
        frame = pd.read_excel(
            io.BytesIO(content),
            sheet_name="ListOfSecurities",
            header=2,
        )
        normalized = self._normalize_columns(frame)
        self._require_columns(
            normalized.columns,
            required=("stock_code", "name_of_securities", "category"),
            market="HK",
        )
        category = normalized["category"].astype(str).str.strip().str.lower()
        filtered = normalized[category == _HK_EQUITY_CATEGORY].copy()
        if "sub_category" in filtered.columns:
            sub_category = filtered["sub_category"].astype(str).str.strip().str.lower()
            filtered = filtered[
                sub_category.eq("")
                | sub_category.eq("nan")
                | sub_category.str.contains(_HK_EQUITY_SUBCATEGORY_TOKEN, na=False)
            ].copy()

        rows: list[dict[str, Any]] = []
        for row in filtered.to_dict("records"):
            local_code = self._normalize_digits(row.get("stock_code"), pad_to=4, max_digits=5)
            name = str(row.get("name_of_securities") or "").strip()
            if not local_code or not name:
                continue
            rows.append(
                {
                    "symbol": f"{local_code}.HK",
                    "name": name,
                    "exchange": "XHKG",
                    "sector": "",
                    "industry": "",
                    "market_cap": None,
                }
            )

        if not rows:
            raise ValueError("HK official universe parse returned no equity rows")
        return rows

    def parse_jp_rows(self, content: bytes) -> OfficialMarketUniverseSnapshot:
        frame = self._read_excel_bytes(content, engine="xlrd")
        rows, snapshot_date = self._parse_jp_frame(frame)
        return OfficialMarketUniverseSnapshot(
            market="JP",
            source_name=_JP_SOURCE_NAME,
            snapshot_id=f"jpx-data-j-{snapshot_date.isoformat()}",
            snapshot_as_of=snapshot_date.isoformat(),
            source_metadata={},
            rows=tuple(rows),
        )

    def parse_nse_rows(self, content: bytes) -> list[dict[str, Any]]:
        frame = pd.read_csv(io.BytesIO(content), dtype=str, keep_default_na=False)
        normalized = self._normalize_columns(frame)
        self._require_columns(
            normalized.columns,
            required=("symbol", "name_of_company", "series", "isin_number"),
            market="IN/NSE",
        )
        series = normalized["series"].astype(str).str.strip().str.upper()
        filtered = normalized[series == "EQ"].copy()

        rows: list[dict[str, Any]] = []
        for row in filtered.to_dict("records"):
            local_code = self._collapse_whitespace(str(row.get("symbol") or "")).upper()
            name = str(row.get("name_of_company") or "").strip()
            isin = str(row.get("isin_number") or "").strip().upper()
            if not local_code or not name or not isin:
                continue
            rows.append(
                {
                    "symbol": f"{local_code}.NS",
                    "name": name,
                    "exchange": "XNSE",
                    "sector": "",
                    "industry": "",
                    "market_cap": None,
                    "isin": isin,
                }
            )

        if not rows:
            raise ValueError("IN/NSE official universe parse returned no EQ rows")
        return sorted(rows, key=lambda row: row["symbol"])

    def parse_bse_rows(self, content: bytes) -> list[dict[str, Any]]:
        payload = json.loads(content.decode("utf-8"))
        if not isinstance(payload, list):
            raise ValueError("IN/BSE official universe response must be a JSON list")

        rows: list[dict[str, Any]] = []
        for row in payload:
            if not isinstance(row, dict):
                continue
            local_code = self._normalize_digits(row.get("SCRIP_CD"), pad_to=6, max_digits=6)
            if not local_code:
                continue
            status = str(row.get("Status") or "").strip().lower()
            segment = str(row.get("Segment") or "").strip().lower()
            if status != "active":
                continue
            if segment != "equity":
                continue

            name = self._collapse_whitespace(
                str(row.get("Issuer_Name") or row.get("Scrip_Name") or "")
            )
            isin = str(row.get("ISIN_NUMBER") or "").strip().upper()
            market_cap = self._parse_optional_float(row.get("Mktcap"))
            industry = self._collapse_whitespace(str(row.get("INDUSTRY") or ""))
            if not name or not isin:
                continue

            rows.append(
                {
                    "symbol": f"{local_code}.BO",
                    "name": name,
                    "exchange": "XBOM",
                    "sector": industry,
                    "industry": industry,
                    "market_cap": market_cap,
                    "isin": isin,
                    "security_id": self._collapse_whitespace(str(row.get("scrip_id") or "")).upper(),
                }
            )

        if not rows:
            raise ValueError("IN/BSE official universe parse returned no active equity rows")
        return sorted(rows, key=lambda row: row["symbol"])

    def parse_tw_rows(self, html: str, *, exchange: str) -> list[dict[str, Any]]:
        soup = BeautifulSoup(html, "html.parser")
        current_section = ""
        rows: list[dict[str, Any]] = []

        for tr in soup.find_all("tr"):
            cells = tr.find_all("td")
            if not cells:
                continue

            texts = [self._collapse_whitespace(td.get_text(" ", strip=True)) for td in cells]
            if len(cells) == 1 and texts:
                current_section = texts[0]
                continue

            if not self._is_tw_stock_section(current_section):
                continue

            first_cell = texts[0] if texts else ""
            if not first_cell or first_cell.lower().startswith("code"):
                continue

            parsed_identity = self._parse_tw_code_name(first_cell)
            if parsed_identity is None:
                continue
            code, name = parsed_identity

            industry = ""
            if len(texts) >= 5:
                industry = texts[4]
            elif len(texts) >= 4:
                industry = texts[3]

            rows.append(
                {
                    "symbol": f"{code}.TW" if exchange == "TWSE" else f"{code}.TWO",
                    "name": name,
                    "exchange": exchange,
                    "sector": industry,
                    "industry": industry,
                    "market_cap": None,
                }
            )

        if not rows:
            raise ValueError(f"TW official universe parse returned no stock rows for {exchange}")
        return rows

    def _parse_jp_frame(self, frame: pd.DataFrame) -> tuple[list[dict[str, Any]], date]:
        normalized = self._normalize_columns(frame)
        self._require_columns(
            normalized.columns,
            required=("日付", "コード", "銘柄名", "市場・商品区分"),
            market="JP",
        )
        category = normalized["市場・商品区分"].astype(str).str.strip()
        filtered = normalized[category.isin(_JP_ALLOWED_MARKET_SECTIONS)].copy()

        dates = {
            self._coerce_date(value)
            for value in filtered["日付"].tolist()
            if self._coerce_date(value) is not None
        }
        if not dates:
            raise ValueError("JP official universe parse could not determine snapshot date")
        if len(dates) != 1:
            raise ValueError(f"JP official universe parse saw multiple snapshot dates: {sorted(d.isoformat() for d in dates)}")
        snapshot_date = next(iter(dates))

        rows: list[dict[str, Any]] = []
        for row in filtered.to_dict("records"):
            code = self._normalize_digits(row.get("コード"), pad_to=4, max_digits=5)
            name = str(row.get("銘柄名") or "").strip()
            if not code or not name:
                continue
            industry = str(row.get("33業種区分") or "").strip()
            sector = str(row.get("17業種区分") or "").strip()
            rows.append(
                {
                    "symbol": f"{code}.T",
                    "name": name,
                    "exchange": "XTKS",
                    "sector": sector,
                    "industry": industry,
                    "market_cap": None,
                }
            )

        if not rows:
            raise ValueError("JP official universe parse returned no domestic equity rows")
        return rows, snapshot_date

    def _http_get(
        self,
        url: str,
        *,
        allow_insecure_fallback: bool = False,
        extra_headers: dict[str, str] | None = None,
    ) -> _FetchedSource:
        headers = {"User-Agent": self._user_agent}
        if extra_headers:
            headers.update(extra_headers)
        fetched_at = datetime.now(UTC).isoformat()
        try:
            return self._http_get_with_retries(
                url,
                headers=headers,
                fetched_at=fetched_at,
                verify_tls=True,
            )
        except requests.exceptions.SSLError:
            if not allow_insecure_fallback:
                raise
            logger.warning(
                "Retrying official universe fetch with TLS verification disabled for %s",
                url,
            )
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
            return self._http_get_with_retries(
                url,
                headers=headers,
                fetched_at=fetched_at,
                verify_tls=False,
            )

    def _http_get_with_retries(
        self,
        url: str,
        *,
        headers: dict[str, str],
        fetched_at: str,
        verify_tls: bool,
    ) -> _FetchedSource:
        request_kwargs: dict[str, Any] = {
            "headers": headers,
            "timeout": self._timeout_seconds,
        }
        if not verify_tls:
            request_kwargs["verify"] = False

        for attempt in range(1, _HTTP_GET_MAX_ATTEMPTS + 1):
            try:
                response = requests.get(url, **request_kwargs)
                response.raise_for_status()
                return _FetchedSource(
                    url=response.url or url,
                    content=response.content,
                    fetched_at=fetched_at,
                    last_modified=response.headers.get("Last-Modified"),
                    tls_verification_disabled=not verify_tls,
                )
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
                if attempt >= _HTTP_GET_MAX_ATTEMPTS:
                    raise
                backoff_seconds = float(attempt)
                logger.warning(
                    "Retrying official universe fetch for %s after attempt %s/%s",
                    url,
                    attempt,
                    _HTTP_GET_MAX_ATTEMPTS,
                )
                time.sleep(backoff_seconds)

        raise RuntimeError(f"Unreachable retry loop for official universe fetch {url}")

    @staticmethod
    def _read_excel_bytes(content: bytes, *, engine: str | None = None) -> pd.DataFrame:
        kwargs: dict[str, Any] = {}
        if engine:
            kwargs["engine"] = engine
        return pd.read_excel(io.BytesIO(content), **kwargs)

    @staticmethod
    def _normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
        renamed = frame.copy()
        renamed.columns = [OfficialMarketUniverseSourceService._normalize_column_name(col) for col in renamed.columns]
        return renamed

    @staticmethod
    def _normalize_column_name(value: Any) -> str:
        text = OfficialMarketUniverseSourceService._collapse_whitespace(str(value or ""))
        text = text.replace("/", "_").replace("-", "_")
        text = re.sub(r"[^0-9A-Za-z\u3040-\u30ff\u3400-\u9fff_]+", "_", text)
        text = re.sub(r"_+", "_", text).strip("_")
        return text.lower()

    @staticmethod
    def _require_columns(columns: Iterable[str], *, required: tuple[str, ...], market: str) -> None:
        present = set(columns)
        missing = [column for column in required if column not in present]
        if missing:
            raise ValueError(
                f"{market} official universe source missing required columns: {', '.join(missing)}"
            )

    @staticmethod
    def _normalize_digits(value: Any, *, pad_to: int, max_digits: int) -> str:
        raw = str(value or "").strip()
        if not raw:
            return ""
        match = re.search(r"\d+", raw)
        if match is None:
            return ""
        digits = match.group(0).lstrip("0")
        if not digits:
            return ""
        if len(digits) > max_digits:
            raise ValueError(f"Unexpected local code length for {raw!r}")
        if len(digits) < pad_to:
            return digits.zfill(pad_to)
        return digits

    @staticmethod
    def _parse_optional_float(value: Any) -> float | None:
        raw = str(value or "").strip().replace(",", "")
        if not raw:
            return None
        try:
            return float(raw)
        except ValueError:
            return None

    @staticmethod
    def _coerce_date(value: Any) -> date | None:
        if value is None or value == "":
            return None
        if isinstance(value, datetime):
            return value.date()
        if isinstance(value, date):
            return value
        parsed = pd.to_datetime(value, errors="coerce")
        if pd.isna(parsed):
            return None
        return parsed.date()

    @staticmethod
    def _utc_today() -> date:
        return datetime.now(UTC).date()

    @staticmethod
    def _date_from_http_header(header_value: str | None) -> date | None:
        if not header_value:
            return None
        try:
            return parsedate_to_datetime(header_value).astimezone(UTC).date()
        except Exception:
            return None

    @staticmethod
    def _collapse_whitespace(value: str) -> str:
        return " ".join(
            str(value or "").replace("\u3000", " ").replace("\xa0", " ").split()
        )

    @staticmethod
    def _parse_tw_updated_at(html: str) -> date:
        match = _TW_UPDATED_AT_RE.search(html)
        if match is None:
            raise ValueError("TW official universe source missing 'Date Stock Updated' banner")
        return datetime.strptime(match.group(1), "%Y/%m/%d").date()

    @staticmethod
    def _is_tw_stock_section(section_label: str) -> bool:
        normalized = OfficialMarketUniverseSourceService._collapse_whitespace(section_label).lower()
        return normalized in {"stocks", "stock"} or normalized.startswith("stocks ")

    @staticmethod
    def _parse_tw_code_name(cell_text: str) -> tuple[str, str] | None:
        match = _TW_CODE_NAME_RE.match(
            OfficialMarketUniverseSourceService._collapse_whitespace(cell_text)
        )
        if match is None:
            return None
        return match.group(1), match.group(2)

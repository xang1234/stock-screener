"""Fetch and parse official exchange universe snapshots for HK, IN, JP, KR, TW, and CN."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from email.utils import parsedate_to_datetime
import io
import json
import logging
import math
import re
import time
from typing import Any, Iterable
from zoneinfo import ZoneInfo

from bs4 import BeautifulSoup
import pandas as pd
import requests
import urllib3

from ..config import settings

logger = logging.getLogger(__name__)


_HK_SOURCE_NAME = "hkex_official"
_IN_SOURCE_NAME = "in_reference_bundle"
_NSE_SOURCE_NAME = "nse_official"
_BSE_SOURCE_NAME = "bse_official"
_JP_SOURCE_NAME = "jpx_official"
_KR_SOURCE_NAME = "krx_official"
_TW_SOURCE_NAME = "tw_reference_bundle"
_CN_SOURCE_NAME = "cn_akshare_eastmoney"
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

            self._cn_provider = CnMarketDataService(timeout_seconds=self._timeout_seconds)
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
            if str(row.get("exchange") or "").strip().upper() in {"KOSPI", "KOSDAQ"}
        )

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

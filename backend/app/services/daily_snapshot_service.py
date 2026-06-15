"""Aggregated Daily Snapshot payload for server mode.

Mirrors the static-site ``home.json`` bundle so the Daily Snapshot tab renders
from a single request (key-market cards, top scan candidates, leaders in
leading groups, top groups, freshness dates) instead of ~14 round-trips.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from time import monotonic
from typing import Any, Callable

from sqlalchemy.orm import Session

from app.domain.common.query import FilterSpec, PageSpec, QuerySpec, SortOrder, SortSpec
from app.domain.markets.catalog import get_market_catalog
from app.domain.scanning.default_filters import resolve_default_scan_filters
from app.models.market_breadth import MarketBreadth
from app.models.scan_result import Scan
from app.schemas.scanning import ScanResultItem
from app.services.key_market_history import build_key_market_entries
from app.services.redis_pool import get_redis_client
from app.use_cases.scanning.get_scan_results import GetScanResultsQuery

logger = logging.getLogger(__name__)

DAILY_SNAPSHOT_SCHEMA_VERSION = 1
DAILY_SNAPSHOT_CACHE_TTL_SECONDS = 600
DAILY_SNAPSHOT_TOP_RESULTS = 20
LEADERS_MAX_GROUP_RANK = 40
LEADERS_MIN_RS_RATING = 80
TOP_GROUPS_LIMIT = 10
DAILY_SNAPSHOT_MEMORY_CACHE_MAX_ENTRIES = 32


@dataclass(frozen=True)
class _DailySnapshotMemoryCacheEntry:
    payload_json: str
    expires_at: float


_daily_snapshot_memory_cache: dict[str, _DailySnapshotMemoryCacheEntry] = {}


def daily_snapshot_cache_key(market: str, scan_id: str | None) -> str:
    """Cache key scoped to the scan run, so a new run invalidates immediately."""
    return (
        f"daily_snapshot:v{DAILY_SNAPSHOT_SCHEMA_VERSION}"
        f":{market.upper()}:{scan_id or 'no-scan'}"
    )


def daily_snapshot_etag(payload_json: str) -> str:
    return 'W/"{}"'.format(hashlib.sha1(payload_json.encode("utf-8")).hexdigest())


def _prune_daily_snapshot_memory_cache(now: float) -> None:
    expired_keys = [
        key
        for key, entry in _daily_snapshot_memory_cache.items()
        if entry.expires_at <= now
    ]
    for key in expired_keys:
        _daily_snapshot_memory_cache.pop(key, None)


def get_daily_snapshot_memory_cache(cache_key: str, *, now: float | None = None) -> str | None:
    now = monotonic() if now is None else now
    entry = _daily_snapshot_memory_cache.get(cache_key)
    if entry is None:
        return None
    if entry.expires_at <= now:
        _daily_snapshot_memory_cache.pop(cache_key, None)
        return None
    return entry.payload_json


def set_daily_snapshot_memory_cache(
    cache_key: str,
    payload_json: str,
    *,
    ttl_seconds: int = DAILY_SNAPSHOT_CACHE_TTL_SECONDS,
    now: float | None = None,
) -> None:
    now = monotonic() if now is None else now
    _prune_daily_snapshot_memory_cache(now)
    if (
        cache_key not in _daily_snapshot_memory_cache
        and len(_daily_snapshot_memory_cache) >= DAILY_SNAPSHOT_MEMORY_CACHE_MAX_ENTRIES
    ):
        _daily_snapshot_memory_cache.pop(next(iter(_daily_snapshot_memory_cache)), None)
    _daily_snapshot_memory_cache[cache_key] = _DailySnapshotMemoryCacheEntry(
        payload_json=payload_json,
        expires_at=now + ttl_seconds,
    )


def _clear_daily_snapshot_memory_cache() -> None:
    _daily_snapshot_memory_cache.clear()


def _decode_cached_payload(cached: Any) -> str | None:
    if not cached:
        return None
    return cached.decode("utf-8") if isinstance(cached, bytes) else cached


def _read_daily_snapshot_redis_cache(
    cache_key: str,
    *,
    redis: Any,
    now: float | None = None,
) -> str | None:
    try:
        payload_json = _decode_cached_payload(redis.get(cache_key))
    except Exception as exc:  # noqa: BLE001 - cache read failures should degrade to rebuilds.
        logger.warning("Daily snapshot cache read failed: %s", exc)
        return None
    if payload_json is not None:
        try:
            ttl_seconds = redis.ttl(cache_key)
        except Exception as exc:  # noqa: BLE001 - TTL failures should not block Redis hits.
            logger.warning("Daily snapshot cache TTL read failed: %s", exc)
        else:
            if not isinstance(ttl_seconds, int) or ttl_seconds <= 0:
                logger.warning(
                    "Daily snapshot cache entry has no positive TTL; refreshing: %s",
                    ttl_seconds,
                )
                return None
            set_daily_snapshot_memory_cache(
                cache_key,
                payload_json,
                ttl_seconds=ttl_seconds,
                now=now,
            )
    return payload_json


def _write_daily_snapshot_redis_cache(
    cache_key: str,
    payload_json: str,
    *,
    redis: Any | None,
) -> None:
    if redis is None:
        return
    try:
        redis.setex(cache_key, DAILY_SNAPSHOT_CACHE_TTL_SECONDS, payload_json)
    except Exception as exc:  # noqa: BLE001 - cache write failures should not fail requests.
        logger.warning("Daily snapshot cache write failed: %s", exc)


def get_or_build_daily_snapshot_payload(
    cache_key: str,
    build_payload_json: Callable[[], str],
    *,
    redis_client_factory: Callable[[], Any | None] = get_redis_client,
    now: float | None = None,
) -> str:
    payload_json = get_daily_snapshot_memory_cache(cache_key, now=now)
    if payload_json is not None:
        return payload_json

    try:
        redis = redis_client_factory()
    except Exception as exc:  # noqa: BLE001 - Redis outages should fall back to rebuilding.
        logger.warning("Daily snapshot cache client acquisition failed: %s", exc)
        redis = None
    if redis is not None:
        payload_json = _read_daily_snapshot_redis_cache(cache_key, redis=redis, now=now)
        if payload_json is not None:
            return payload_json

    payload_json = build_payload_json()
    _write_daily_snapshot_redis_cache(cache_key, payload_json, redis=redis)
    set_daily_snapshot_memory_cache(cache_key, payload_json, now=now)
    return payload_json


def latest_completed_scan(db: Session, market: str) -> Scan | None:
    return (
        db.query(Scan)
        .filter(
            Scan.status == "completed",
            Scan.universe_market == market.upper(),
        )
        .order_by(Scan.completed_at.desc().nullslast(), Scan.id.desc())
        .first()
    )


def _scan_freshness(scan: Scan | None) -> dict[str, Any]:
    if scan is None:
        return {
            "scan_id": None,
            "scan_as_of_date": None,
            "scan_published_at": None,
        }
    run = scan.feature_run
    as_of = run.as_of_date.isoformat() if run is not None and run.as_of_date else None
    if as_of is None and scan.completed_at is not None:
        as_of = scan.completed_at.date().isoformat()
    published_at = None
    if run is not None and run.published_at is not None:
        published_at = run.published_at.isoformat()
    elif scan.completed_at is not None:
        published_at = scan.completed_at.isoformat()
    return {
        "scan_id": scan.scan_id,
        "scan_as_of_date": as_of,
        "scan_published_at": published_at,
    }


def _query_scan_rows(
    *,
    uow: Any,
    use_case: Any,
    scan_id: str,
    filters: FilterSpec,
) -> list[dict[str, Any]]:
    query = GetScanResultsQuery(
        scan_id=scan_id,
        query_spec=QuerySpec(
            filters=filters,
            sort=SortSpec(field="composite_score", order=SortOrder.DESC),
            page=PageSpec(page=1, per_page=DAILY_SNAPSHOT_TOP_RESULTS),
        ),
        include_sparklines=True,
        include_setup_payload=False,
    )
    result = use_case.execute(uow, query)
    return [
        ScanResultItem.from_domain(item, include_setup_payload=False).model_dump(mode="json")
        for item in result.page.items
    ]


def _build_top_groups(db: Session, market: str) -> tuple[list[dict[str, Any]], str | None]:
    from app.wiring.bootstrap import get_group_rank_service

    if not get_market_catalog().get(market).capabilities.group_rankings:
        return [], None
    rankings = get_group_rank_service().get_current_rankings(
        db, limit=TOP_GROUPS_LIMIT, market=market
    )
    if not rankings:
        return [], None
    groups_date = rankings[0].get("date")
    keep = (
        "industry_group",
        "rank",
        "rank_change_1w",
        "rank_change_1m",
        "top_symbol",
        "top_symbol_name",
        "top_rs_rating",
    )
    return [{key: row.get(key) for key in keep} for row in rankings], groups_date


def _latest_breadth_date(db: Session, market: str) -> str | None:
    latest = (
        db.query(MarketBreadth.date)
        .filter(MarketBreadth.market == market)
        .order_by(MarketBreadth.date.desc())
        .first()
    )
    if latest is None or latest[0] is None:
        return None
    value = latest[0]
    return value.isoformat() if hasattr(value, "isoformat") else str(value)


def build_daily_snapshot_payload(
    db: Session,
    *,
    market: str,
    market_display_name: str,
    scan: Scan | None,
    uow: Any,
    scan_results_use_case: Any,
) -> dict[str, Any]:
    """Assemble the full Daily Snapshot payload for one market.

    ``scan`` is the latest completed scan for the market (resolved by the
    caller, which also keys the response cache on it).
    """
    normalized = market.upper()
    min_volume = resolve_default_scan_filters(normalized).get("minVolume")

    top_candidates: list[dict[str, Any]] = []
    leaders: list[dict[str, Any]] = []
    if scan is not None:
        candidate_filters = FilterSpec()
        candidate_filters.add_range("volume", min_volume, None)
        top_candidates = _query_scan_rows(
            uow=uow,
            use_case=scan_results_use_case,
            scan_id=scan.scan_id,
            filters=candidate_filters,
        )

        leader_filters = FilterSpec()
        leader_filters.add_range("volume", min_volume, None)
        leader_filters.add_range("rs_rating", LEADERS_MIN_RS_RATING, None)
        leader_filters.add_range("ibd_group_rank", None, LEADERS_MAX_GROUP_RANK)
        leaders = _query_scan_rows(
            uow=uow,
            use_case=scan_results_use_case,
            scan_id=scan.scan_id,
            filters=leader_filters,
        )

    top_groups, groups_date = _build_top_groups(db, normalized)
    freshness = _scan_freshness(scan)
    freshness["breadth_latest_date"] = _latest_breadth_date(db, normalized)
    freshness["groups_latest_date"] = groups_date

    return {
        "schema_version": DAILY_SNAPSHOT_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "market": normalized,
        "market_display_name": market_display_name,
        "scan_id": freshness["scan_id"],
        "freshness": freshness,
        "key_markets": build_key_market_entries(db, normalized),
        "top_candidates": {
            "min_dollar_volume": min_volume,
            "rows": top_candidates,
        },
        "leaders": {
            "criteria": {
                "max_group_rank": LEADERS_MAX_GROUP_RANK,
                "min_rs_rating": LEADERS_MIN_RS_RATING,
                "min_dollar_volume": min_volume,
            },
            "rows": leaders,
        },
        "top_groups": top_groups,
    }

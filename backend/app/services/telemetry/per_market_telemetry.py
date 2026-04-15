"""Per-market telemetry service (bead asia.10.1).

Two-tier emission path:

1. **Redis hot-path** (live counters/gauges, 15d TTL):
   - Gauges: ``telemetry:gauge:{metric_key}:{market}`` → JSON value
   - Counters: ``telemetry:counter:{metric_key}:{market}:{dim}:{day}`` → INCR

2. **PostgreSQL append-only event log** (durable, queryable, 15d retention):
   - ``market_telemetry_events`` table — one row per emission

Reads prefer Redis for "right now" values and PG (via the daily view) for
trend/audit. Both sides degrade independently — a Redis outage doesn't break
emission, and a PG outage doesn't break Redis reads.

Design choices:
- **Inline emission only** (per spec choice 3a): no Celery beat polling. The
  weekly_full_refresh task piggybacks the 15d cleanup so retention happens
  without a dedicated periodic job.
- **Append-only PG**: avoids read-modify-write races on counter increments;
  daily aggregation is a SQL view.
- **Best-effort writes**: telemetry must never break the request path. Both
  Redis and PG errors are logged at debug and swallowed.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional

from ...tasks.market_queues import SHARED_SENTINEL, SUPPORTED_MARKETS, market_suffix, normalize_market
from .schema import MetricKey

logger = logging.getLogger(__name__)


# 15-day retention applied uniformly to Redis TTL and PG cleanup.
_RETENTION_DAYS = 15
_REDIS_TTL_S = _RETENTION_DAYS * 86400

# Redis key namespaces — single source of truth so a future namespace change
# only edits these two constants instead of three call sites.
_GAUGE_NS = "telemetry:gauge"
_COUNTER_NS = "telemetry:counter"

# Bound counter-key cardinality: language strings come from upstream content
# and can be malformed. Anything not matching this gets bucketed as "unknown".
_LANG_MAX_LEN = 16


def _sanitize_language(value: Optional[str]) -> str:
    """Clamp a BCP-47 language tag to a safe Redis dimension.

    Bucket unknown / weird tags under "unknown" so an LLM-generated language
    string can't blow up the key space (mirrors the universe_compat_metrics
    sanitizer pattern).
    """
    if not value:
        return "unknown"
    cleaned = value.strip().lower()
    if not cleaned or len(cleaned) > _LANG_MAX_LEN:
        return "unknown"
    if any(c.isspace() or c == ":" for c in cleaned):
        return "unknown"
    return cleaned


def _gauge_key(metric_key: str, market: Optional[str]) -> str:
    return f"{_GAUGE_NS}:{metric_key}:{market_suffix(market)}"


def _counter_key(metric_key: str, market: Optional[str], dim: str, day: str) -> str:
    return f"{_COUNTER_NS}:{metric_key}:{market_suffix(market)}:{dim}:{day}"


def _today_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d")


class PerMarketTelemetry:
    """Emit + read per-market telemetry across Redis hot-path + PG event log.

    All public ``record_*`` methods return None and never raise. Reads return
    None / empty dict on failure rather than raising — telemetry is advisory,
    not a hard contract.
    """

    def __init__(
        self,
        redis_client_factory: Optional[Callable[[], Any]] = None,
        session_factory: Optional[Callable[[], Any]] = None,
    ):
        self._redis_factory = redis_client_factory
        self._session_factory = session_factory

    # ------------------------------------------------------------------
    # Lazy dependency lookup (kept lazy so import doesn't pull DB/Redis)
    # ------------------------------------------------------------------
    def _redis(self):
        if self._redis_factory is not None:
            try:
                return self._redis_factory()
            except Exception:
                return None
        try:
            from ..redis_pool import get_redis_client
            return get_redis_client()
        except Exception:
            return None

    def _session(self):
        if self._session_factory is not None:
            return self._session_factory()
        from ...database import SessionLocal
        return SessionLocal()

    # ------------------------------------------------------------------
    # Generic emission primitives
    # ------------------------------------------------------------------
    def _emit_pg(
        self, *, market: str, metric_key: str, payload: Dict[str, Any]
    ) -> None:
        """Insert a single event row. Best-effort: errors are swallowed."""
        try:
            db = self._session()
        except Exception as exc:
            logger.debug("telemetry: session unavailable (%s); skipping PG emit", exc)
            return
        try:
            from ...models.market_telemetry import MarketTelemetryEvent

            event = MarketTelemetryEvent(
                market=market,
                metric_key=metric_key,
                schema_version=int(payload.get("schema_version", 1)),
                payload=payload,
                recorded_at=datetime.now(timezone.utc),
            )
            db.add(event)
            db.commit()
        except Exception as exc:
            logger.debug("telemetry: PG emit failed for %s/%s: %s", market, metric_key, exc)
            try:
                db.rollback()
            except Exception:
                pass
        finally:
            try:
                db.close()
            except Exception:
                pass

    def _set_gauge(self, metric_key: str, market: str, payload: Dict[str, Any]) -> None:
        client = self._redis()
        if client is None:
            return
        try:
            client.set(_gauge_key(metric_key, market), json.dumps(payload), ex=_REDIS_TTL_S)
        except Exception as exc:
            logger.debug("telemetry: Redis gauge set failed (%s)", exc)

    def _incr_counter(self, metric_key: str, market: str, dim: str, by: int = 1) -> None:
        client = self._redis()
        if client is None:
            return
        try:
            key = _counter_key(metric_key, market, dim, _today_str())
            # Non-transactional pipeline matches the canonical best-effort
            # counter pattern in universe_compat_metrics; faster, no MULTI/EXEC.
            pipe = client.pipeline(transaction=False)
            pipe.incr(key, amount=by)
            pipe.expire(key, _REDIS_TTL_S)
            pipe.execute()
        except Exception as exc:
            logger.debug("telemetry: Redis counter incr failed (%s)", exc)

    # ------------------------------------------------------------------
    # Public emit API — one per metric category
    # ------------------------------------------------------------------
    def record_freshness(
        self, market: Optional[str], *, source: str, symbols_refreshed: int
    ) -> None:
        """Record that a market's data was just refreshed. Lag is derived at read time."""
        from .schema import freshness_lag_payload

        m = normalize_market(market)
        payload = freshness_lag_payload(
            last_refresh_at_epoch=time.time(),
            source=source,
            symbols_refreshed=symbols_refreshed,
        )
        self._set_gauge(MetricKey.FRESHNESS_LAG, m, payload)
        self._emit_pg(market=m, metric_key=MetricKey.FRESHNESS_LAG, payload=payload)

    def record_universe_drift(
        self, market: Optional[str], *, current_size: int, prior_size: Optional[int]
    ) -> None:
        from .schema import universe_drift_payload

        m = normalize_market(market)
        payload = universe_drift_payload(current_size=current_size, prior_size=prior_size)
        self._set_gauge(MetricKey.UNIVERSE_DRIFT, m, payload)
        self._emit_pg(market=m, metric_key=MetricKey.UNIVERSE_DRIFT, payload=payload)

    def record_benchmark_age(
        self, market: Optional[str], *, benchmark_symbol: str
    ) -> None:
        from .schema import benchmark_age_payload

        m = normalize_market(market)
        payload = benchmark_age_payload(
            last_warmed_at_epoch=time.time(),
            benchmark_symbol=benchmark_symbol,
        )
        self._set_gauge(MetricKey.BENCHMARK_AGE, m, payload)
        self._emit_pg(market=m, metric_key=MetricKey.BENCHMARK_AGE, payload=payload)

    def record_extraction(
        self,
        market: Optional[str],
        *,
        language: str,
        success: bool,
        latency_ms: Optional[int] = None,
        provider: Optional[str] = None,
    ) -> None:
        """Record one theme-extraction attempt.

        Redis-only on the hot path: this fires per LLM call (~1/sec under
        load), and a PG row per call would dwarf the rest of the table for
        no extra signal — the Redis day-bucketed counters are exactly what the
        ``/v1/telemetry/markets`` endpoint returns. Daily PG rollup, if needed
        later, can be added by epic 10.4 (drift audit job) without changing
        the emission contract.
        """
        m = normalize_market(market)
        lang = _sanitize_language(language)
        self._incr_counter(MetricKey.EXTRACTION_SUCCESS, m, f"{lang}:total")
        if success:
            self._incr_counter(MetricKey.EXTRACTION_SUCCESS, m, f"{lang}:success")

    def record_completeness_from_db(self, market: Optional[str]) -> None:
        """Compute and emit the completeness distribution by querying the DB.

        Bucketizes ``stock_fundamentals.field_completeness_score`` server-side
        via a single ``GROUP BY`` query so we don't move every score over the
        wire. Best-effort: any failure is logged and swallowed so the calling
        refresh task is never broken by telemetry.
        """
        from sqlalchemy import case, func
        from .schema import COMPLETENESS_BUCKETS

        m = normalize_market(market)
        try:
            db = self._session()
        except Exception:
            return

        bucket_counts: Dict[str, int] = {b: 0 for b in COMPLETENESS_BUCKETS}
        symbols_total = 0
        try:
            from ...models.stock import StockFundamental
            from ...models.stock_universe import StockUniverse

            score = StockFundamental.field_completeness_score
            bucket_expr = case(
                (score < 25, "0-25"),
                (score < 50, "25-50"),
                (score < 75, "50-75"),
                (score < 90, "75-90"),
                else_="90-100",
            ).label("bucket")

            q = (
                db.query(bucket_expr, func.count().label("count"))
                .select_from(StockFundamental)
                .join(StockUniverse, StockUniverse.symbol == StockFundamental.symbol)
                .filter(StockUniverse.is_active.is_(True))
                .filter(score.isnot(None))
            )
            if m != SHARED_SENTINEL:
                q = q.filter(StockUniverse.market == m)
            for bucket, count in q.group_by(bucket_expr).all():
                bucket_counts[bucket] = int(count)
                symbols_total += int(count)
        except Exception as exc:
            logger.debug("telemetry: completeness DB query failed (%s)", exc)
            return
        finally:
            try:
                db.close()
            except Exception:
                pass

        self.record_completeness(
            m, bucket_counts=bucket_counts, symbols_total=symbols_total,
        )

    def record_completeness(
        self, market: Optional[str], *, bucket_counts: Dict[str, int], symbols_total: int
    ) -> None:
        from .schema import completeness_distribution_payload

        m = normalize_market(market)
        payload = completeness_distribution_payload(
            bucket_counts=bucket_counts, symbols_total=symbols_total,
        )
        # Snapshot gauge: latest bucket distribution per market.
        self._set_gauge(MetricKey.COMPLETENESS_DISTRIBUTION, m, payload)
        self._emit_pg(
            market=m, metric_key=MetricKey.COMPLETENESS_DISTRIBUTION, payload=payload,
        )

    def record_field_coverage_from_registry(self, market: Optional[str]) -> None:
        """Snapshot per-market field coverage (bead asia.10.5).

        Static part comes from ``field_capability_registry`` (policy-level
        supported/computed/unsupported counts per market). Dynamic part comes
        from a single GROUP BY on ``stock_fundamentals.growth_metric_basis``
        for the market's active universe — the comparable-period-YoY count
        gives the cadence-fallback rate.
        """
        from ..field_capability_registry import (
            SUPPORT_STATE_COMPUTED,
            SUPPORT_STATE_UNSUPPORTED,
            field_capability_registry,
        )
        from .schema import field_coverage_payload

        m = normalize_market(market)
        if m == SHARED_SENTINEL:
            # Coverage is inherently per-market (policy chains differ); SHARED
            # has no meaningful field-capability view.
            return

        try:
            entries = field_capability_registry.entries()
        except Exception as exc:
            logger.debug("telemetry: registry read failed (%s)", exc)
            return

        state_counts: Dict[str, int] = {}
        unsupported: list = []
        computed: list = []
        for entry in entries:
            cap = entry.markets.get(m)
            if cap is None:
                continue
            state_counts[cap.support_state] = state_counts.get(cap.support_state, 0) + 1
            if cap.support_state == SUPPORT_STATE_UNSUPPORTED:
                unsupported.append(entry.field)
            elif cap.support_state == SUPPORT_STATE_COMPUTED:
                computed.append(entry.field)

        cadence_counts, cadence_eligible = self._read_cadence_counts(m)

        payload = field_coverage_payload(
            total_fields=len(entries),
            support_state_counts=state_counts,
            unsupported_field_names=tuple(sorted(unsupported)),
            computed_field_names=tuple(sorted(computed)),
            cadence_counts=cadence_counts,
            cadence_eligible_universe=cadence_eligible,
        )
        self._set_gauge(MetricKey.FIELD_COVERAGE, m, payload)
        self._emit_pg(market=m, metric_key=MetricKey.FIELD_COVERAGE, payload=payload)

    def _read_cadence_counts(self, market: str) -> tuple:
        """Return ((basis→count) dict, eligible_universe) for one market.

        Eligible universe = active symbols with a non-NULL ``growth_metric_basis``.
        A NULL basis means fundamentals never landed or never produced a
        reportable basis — excluding from the denominator prevents false
        ``cadence_fallback_ratio`` signals driven by missing data.
        """
        from sqlalchemy import func as sa_func

        try:
            db = self._session()
        except Exception:
            return ({}, 0)

        counts: Dict[str, int] = {}
        total = 0
        try:
            from ...models.stock import StockFundamental
            from ...models.stock_universe import StockUniverse

            basis = StockFundamental.growth_metric_basis
            q = (
                db.query(basis, sa_func.count().label("n"))
                .select_from(StockFundamental)
                .join(StockUniverse, StockUniverse.symbol == StockFundamental.symbol)
                .filter(StockUniverse.is_active.is_(True))
                .filter(StockUniverse.market == market)
                .filter(basis.isnot(None))
                .group_by(basis)
            )
            for value, n in q.all():
                counts[str(value)] = int(n)
                total += int(n)
        except Exception as exc:
            logger.debug("telemetry: cadence DB query failed (%s)", exc)
            return ({}, 0)
        finally:
            try:
                db.close()
            except Exception:
                pass

        return (counts, total)

    # ------------------------------------------------------------------
    # Read API
    # ------------------------------------------------------------------
    def get_gauge(self, metric_key: str, market: Optional[str]) -> Optional[Dict[str, Any]]:
        """Return the latest gauge payload for ``(metric_key, market)`` from Redis."""
        client = self._redis()
        if client is None:
            return None
        try:
            raw = client.get(_gauge_key(metric_key, market))
            if raw is None:
                return None
            return json.loads(raw if isinstance(raw, str) else raw.decode())
        except Exception as exc:
            logger.debug("telemetry: Redis gauge read failed (%s)", exc)
            return None

    # Gauge metric keys read in market_summary, in fixed order so MGET
    # results can be zipped back deterministically.
    _SUMMARY_GAUGE_KEYS = (
        MetricKey.FRESHNESS_LAG,
        MetricKey.UNIVERSE_DRIFT,
        MetricKey.BENCHMARK_AGE,
        MetricKey.COMPLETENESS_DISTRIBUTION,
        MetricKey.FIELD_COVERAGE,
    )

    def market_summary(self, market: Optional[str]) -> Dict[str, Any]:
        """Return the latest gauges + derived lag for one market.

        Output shape (schema_version embedded in each sub-payload):
            {
                "market": "HK",
                "freshness_lag": {...payload..., "lag_seconds": 123},
                "universe_drift": {...payload...} | None,
                "benchmark_age": {..., "age_seconds": 456} | None,
                "completeness_distribution": {...} | None,
                "extraction_today": {"by_language": {"en": {"total": 10, "success": 9}}},
            }
        """
        m = normalize_market(market)
        out: Dict[str, Any] = {"market": m}

        gauges = self._mget_gauges(m, self._SUMMARY_GAUGE_KEYS)
        freshness = gauges[MetricKey.FRESHNESS_LAG]
        if freshness is not None:
            ts = float(freshness.get("last_refresh_at_epoch", 0))
            freshness["lag_seconds"] = max(0.0, time.time() - ts) if ts else None
        out[MetricKey.FRESHNESS_LAG] = freshness

        out[MetricKey.UNIVERSE_DRIFT] = gauges[MetricKey.UNIVERSE_DRIFT]

        bench = gauges[MetricKey.BENCHMARK_AGE]
        if bench is not None:
            ts = float(bench.get("last_warmed_at_epoch", 0))
            bench["age_seconds"] = max(0.0, time.time() - ts) if ts else None
        out[MetricKey.BENCHMARK_AGE] = bench

        out[MetricKey.COMPLETENESS_DISTRIBUTION] = gauges[MetricKey.COMPLETENESS_DISTRIBUTION]
        out[MetricKey.FIELD_COVERAGE] = gauges[MetricKey.FIELD_COVERAGE]
        # Extraction is recorded under SHARED scope (the meaningful dimension is
        # language, not market), so every per-market summary surfaces the same
        # global counters. Pass SHARED explicitly rather than ``m`` — otherwise
        # the per-market scan_iter pattern would never match the SHARED keys.
        out["extraction_today"] = self._read_extraction_counters(SHARED_SENTINEL)
        return out

    def _mget_gauges(
        self, market: str, metric_keys: tuple
    ) -> Dict[str, Optional[Dict[str, Any]]]:
        """Coalesce N gauge reads into a single Redis MGET round-trip."""
        client = self._redis()
        if client is None:
            return {k: None for k in metric_keys}
        try:
            keys = [_gauge_key(k, market) for k in metric_keys]
            raws = client.mget(keys)
        except Exception as exc:
            logger.debug("telemetry: Redis MGET gauges failed (%s)", exc)
            return {k: None for k in metric_keys}

        out: Dict[str, Optional[Dict[str, Any]]] = {}
        for metric, raw in zip(metric_keys, raws):
            if raw is None:
                out[metric] = None
                continue
            try:
                out[metric] = json.loads(raw if isinstance(raw, str) else raw.decode())
            except Exception:
                out[metric] = None
        return out

    def _read_extraction_counters(self, market: str) -> Dict[str, Any]:
        """Aggregate today's per-language extraction counters from Redis.

        Uses one SCAN + one MGET to avoid an N+1 round-trip pattern as the
        language cardinality grows.
        """
        client = self._redis()
        if client is None:
            return {"by_language": {}}
        day = _today_str()
        prefix = f"{_COUNTER_NS}:{MetricKey.EXTRACTION_SUCCESS}:{market_suffix(market)}:"
        suffix = f":{day}"
        by_language: Dict[str, Dict[str, int]] = {}
        try:
            keys = [
                (k.decode() if isinstance(k, bytes) else k)
                for k in client.scan_iter(match=f"{prefix}*{suffix}")
            ]
            if not keys:
                return {"by_language": {}}
            values = client.mget(keys)
            for key_str, raw in zip(keys, values):
                # dim has the shape "<lang>:<bucket>" (e.g. "en:total" or "en:success").
                dim = key_str[len(prefix):-len(suffix)]
                if ":" not in dim:
                    continue
                lang, bucket = dim.split(":", 1)
                count = int(raw) if raw is not None else 0
                by_language.setdefault(lang, {})[bucket] = count
        except Exception as exc:
            logger.debug("telemetry: extraction counter read failed (%s)", exc)
        return {"by_language": by_language}

    def list_markets(self) -> List[str]:
        return list(SUPPORTED_MARKETS)

    # ------------------------------------------------------------------
    # Retention
    # ------------------------------------------------------------------
    def cleanup_old_events(self, retention_days: int = _RETENTION_DAYS) -> int:
        """Delete PG event rows older than ``retention_days``. Returns row count.

        Called opportunistically from ``weekly_full_refresh`` so we don't add a
        dedicated beat entry. Best-effort: returns 0 on any failure.
        """
        try:
            db = self._session()
        except Exception:
            return 0
        try:
            from ...models.market_telemetry import MarketTelemetryEvent

            cutoff = datetime.now(timezone.utc) - timedelta(days=retention_days)
            deleted = (
                db.query(MarketTelemetryEvent)
                .filter(MarketTelemetryEvent.recorded_at < cutoff)
                .delete(synchronize_session=False)
            )
            db.commit()
            return int(deleted or 0)
        except Exception as exc:
            logger.debug("telemetry: PG cleanup failed (%s)", exc)
            try:
                db.rollback()
            except Exception:
                pass
            return 0
        finally:
            try:
                db.close()
            except Exception:
                pass


_default: Optional[PerMarketTelemetry] = None
_default_lock = threading.Lock()


def get_telemetry() -> PerMarketTelemetry:
    """Return the process-wide telemetry singleton."""
    global _default
    if _default is None:
        with _default_lock:
            if _default is None:
                _default = PerMarketTelemetry()
    return _default


__all__ = [
    "PerMarketTelemetry",
    "get_telemetry",
    "SHARED_SENTINEL",
]

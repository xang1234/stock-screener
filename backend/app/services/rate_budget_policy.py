"""
Per-market × per-provider rate budget policy (bead StockScreenClaude-asia.9.2).

Builds on the per-market queue topology from 9.1: each market gets its own
share of every external provider's capacity (yfinance / finviz / etc.) so one
market's refresh can't starve another market of provider tokens. Pairs with
``services.rate_limiter.RedisRateLimiter`` whose key-based design partitions
naturally — this policy module owns *which key* and *what interval* to use.

Defaults: per-market intervals are derived by **universe-weighted split** of
the global per-provider budget. Operators override per market via env
settings (``yfinance_rate_limit_hk`` etc.) when 9.3 measurements show the
auto-computed split is wrong. Universe weights are refreshed weekly by the
``refresh_universe_weights`` Celery beat task.
"""
from __future__ import annotations

import logging
import threading
from datetime import datetime
from typing import Dict, Optional

from ..config import settings
from ..tasks.market_queues import SHARED_SENTINEL, SUPPORTED_MARKETS, market_suffix, normalize_market

logger = logging.getLogger(__name__)


# Cache invalidation: weights TTL safety net. The beat task refreshes weekly,
# but if it fails for >2 weeks we recompute on next read so stale weights from
# a since-decommissioned universe shape can't persist forever.
_WEIGHTS_STALE_AFTER_S = 14 * 86400


class RateBudgetPolicy:
    """Centralized per-market × per-provider rate budget lookups.

    All providers (``yfinance``, ``finviz``, ``alpha_vantage``, ``sec_edgar``)
    share the same lookup pattern so adding a new provider only requires
    settings-side defaults, not policy-code changes.
    """

    # Built-in default global intervals when settings don't expose a key.
    # Used for providers without an explicit global setting.
    _DEFAULT_GLOBAL_INTERVALS_S: Dict[str, float] = {
        "yfinance": 1.0,
        # Tightened from 5.0s → 2.0s. yf.download is one HTTP round-trip
        # per batch; 2s spacing between batches is well under any plausible
        # per-batch rate cap, and the adaptive shrink + circuit breaker
        # handle the failure mode if Yahoo objects.
        "yfinance:batch": 2.0,
        "finviz": 0.5,
        "alpha_vantage": 3.0,  # 25 req/day → ~3s spacing if batched
        "sec_edgar": 0.15,     # 10 req/sec
    }

    # Per-market batch sizes. Keep the defaults uniform unless operators
    # explicitly override them via settings.
    #
    # CA: defaults copied from HK/JP/KR/TW (50 batch, 1 worker for yfinance;
    # 50 batch, 2 workers for finviz). Not measured — the 9.3 measurement
    # guidance applies: revise after the first sustained CA refresh run if
    # empirical throughput diverges from peer markets.
    _DEFAULT_BATCH_SIZE: Dict[str, Dict[str, int]] = {
        # US bumped to 150 — Yahoo accepts batches up to MAX_PRICE_BATCH_SIZE
        # (200) and the adaptive shrink in fetch_prices_in_batches halves the
        # batch on transient failure, so this is safe.
        "yfinance": {"US": 150, "HK": 50, "IN": 50, "JP": 50, "KR": 50, "TW": 50, "CN": 25, "CA": 50, "DE": 50},
        "finviz":   {"US": 100, "HK": 50, "IN": 50, "JP": 50, "KR": 50, "TW": 50, "CN": 50, "CA": 50, "DE": 50},
    }

    # Per-market default worker counts for providers that benefit from
    # concurrency. The Redis rate limiter still serializes egress, so this
    # only fills idle time during in-flight HTTP RTT — total req/sec to the
    # upstream IP stays at the configured cadence.
    _DEFAULT_PROVIDER_WORKERS: Dict[str, Dict[str, int]] = {
        # finvizfinance has no batch endpoint; concurrency is the only
        # speedup lever. US gets more workers because its rate-limit
        # interval is 0.5s and per-call RTT is typically ~1-2s.
        "finviz": {"US": 4, "HK": 2, "IN": 2, "JP": 2, "KR": 2, "TW": 2, "CN": 1, "CA": 2, "DE": 2},
        # yfinance batch is already a single bulk HTTP call; threading
        # there caused fork issues in Celery workers, so default to 1.
        "yfinance": {"US": 1, "HK": 1, "IN": 1, "JP": 1, "KR": 1, "TW": 1, "CN": 1, "CA": 1, "DE": 1},
    }

    # Per-market backoff caps. Non-US markets get longer caps because
    # observed 429 windows from non-US queries are sometimes longer than US.
    # 9.3 measurements will refine these.
    #
    # US keeps ``base_s=30`` to preserve the legacy 30/60/120 retry schedule
    # used by ``BulkDataFetcher._fetch_price_batch_with_retries``. Non-US
    # markets bump to ``base_s=60`` so an IN refresh that hits 429s waits
    # 60/120/240s before giving up — empirically Yahoo's IN throttle windows
    # are longer than US.
    _DEFAULT_BACKOFF: Dict[str, Dict[str, dict]] = {
        "yfinance": {
            "US": {"base_s": 30, "max_s": 480, "factor": 2.0},
            "HK": {"base_s": 60, "max_s": 600, "factor": 2.0},
            "IN": {"base_s": 60, "max_s": 600, "factor": 2.0},
            "JP": {"base_s": 60, "max_s": 600, "factor": 2.0},
            "KR": {"base_s": 60, "max_s": 600, "factor": 2.0},
            "TW": {"base_s": 60, "max_s": 600, "factor": 2.0},
            "CN": {"base_s": 60, "max_s": 900, "factor": 2.0},
            "CA": {"base_s": 60, "max_s": 600, "factor": 2.0},
            "DE": {"base_s": 60, "max_s": 600, "factor": 2.0},
        },
        "finviz": {
            "US": {"base_s": 30, "max_s": 240, "factor": 2.0},
            "HK": {"base_s": 60, "max_s": 480, "factor": 2.0},
            "IN": {"base_s": 60, "max_s": 480, "factor": 2.0},
            "JP": {"base_s": 60, "max_s": 480, "factor": 2.0},
            "KR": {"base_s": 60, "max_s": 480, "factor": 2.0},
            "TW": {"base_s": 60, "max_s": 480, "factor": 2.0},
            "CN": {"base_s": 60, "max_s": 480, "factor": 2.0},
            "CA": {"base_s": 60, "max_s": 480, "factor": 2.0},
            "DE": {"base_s": 60, "max_s": 480, "factor": 2.0},
        },
    }

    def __init__(self, redis_client_factory=None):
        self._lock = threading.Lock()
        self._weights_cache: Optional[Dict[str, float]] = None
        self._weights_computed_at: Optional[float] = None
        self._redis_client_factory = redis_client_factory

    # ------------------------------------------------------------------
    # Key naming
    # ------------------------------------------------------------------
    @staticmethod
    def provider_key(provider: str, market: Optional[str]) -> str:
        """Return the rate-limiter key for a provider in a given market.

        ``provider_key("yfinance", "HK")`` -> ``"yfinance:hk"``
        ``provider_key("yfinance", None)`` -> ``"yfinance:shared"``
        Suffix matches the existing ``yfinance:batch`` convention.
        """
        return f"{provider}:{market_suffix(market)}"

    # ------------------------------------------------------------------
    # Universe weights
    # ------------------------------------------------------------------
    def _universe_weights(self, force_refresh: bool = False) -> Dict[str, float]:
        """Compute per-market weights from the active universe sizes.

        Returns weights summing to 1.0 across SUPPORTED_MARKETS. Falls back to
        equal split when the universe is too small to be meaningful (fresh
        deploy before initial seed). Cached in-process; the beat task
        ``refresh_universe_weights`` invalidates the cache weekly.

        Double-checked locking: the lock is only acquired on cache miss or
        forced refresh, so the steady-state read path (called once per
        rate-limited call) is lock-free.
        """
        if not force_refresh:
            cached = self._weights_cache
            computed_at = self._weights_computed_at
            if cached is not None and computed_at is not None:
                if (datetime.now().timestamp() - computed_at) <= _WEIGHTS_STALE_AFTER_S:
                    return cached

        with self._lock:
            now = datetime.now().timestamp()
            stale = (
                self._weights_computed_at is None
                or (now - self._weights_computed_at) > _WEIGHTS_STALE_AFTER_S
            )
            if not force_refresh and self._weights_cache is not None and not stale:
                return self._weights_cache

            weights = self._compute_weights_from_db()
            self._weights_cache = weights
            self._weights_computed_at = now
            return weights

    @staticmethod
    def _compute_weights_from_db() -> Dict[str, float]:
        """Query StockUniverse for per-market counts and normalize to weights."""
        try:
            from ..database import SessionLocal
            from ..models.stock_universe import StockUniverse
            from sqlalchemy import func

            db = SessionLocal()
            try:
                rows = (
                    db.query(StockUniverse.market, func.count(StockUniverse.symbol))
                    .filter(StockUniverse.is_active == True)
                    .group_by(StockUniverse.market)
                    .all()
                )
                counts = {
                    market.upper(): count
                    for market, count in rows
                    if market and market.upper() in SUPPORTED_MARKETS
                }
            finally:
                db.close()
        except Exception as exc:
            logger.warning("RateBudgetPolicy: universe weight query failed (%s); using equal split", exc)
            counts = {}

        total = sum(counts.values())
        if total < 100:
            logger.info(
                "RateBudgetPolicy: universe size %d below threshold (100); using equal weights across %s",
                total, SUPPORTED_MARKETS,
            )
            return {m: 1.0 / len(SUPPORTED_MARKETS) for m in SUPPORTED_MARKETS}

        weights: Dict[str, float] = {}
        for market in SUPPORTED_MARKETS:
            weights[market] = counts.get(market, 0) / total
        # Guarantee no zero-weight market starves entirely; floor at 5%.
        floor = 0.05
        for market in SUPPORTED_MARKETS:
            if weights[market] < floor:
                weights[market] = floor
        # Renormalize to sum to 1.0 after flooring.
        total_w = sum(weights.values())
        return {m: w / total_w for m, w in weights.items()}

    def invalidate_weights_cache(self) -> None:
        """Force the next ``_universe_weights`` call to recompute.

        Called by the weekly ``refresh_universe_weights`` beat task. NOTE:
        the policy singleton is per-process, so invalidating from one worker
        does NOT reach other workers. The 14-day local stale fallback
        (``_WEIGHTS_STALE_AFTER_S``) ensures every worker eventually
        recomputes — adequate for the slow pace of universe-size drift, but
        a Redis-shared weights cache would be the cleaner long-term design.
        """
        with self._lock:
            self._weights_cache = None
            self._weights_computed_at = None

    # ------------------------------------------------------------------
    # Rate interval lookup
    # ------------------------------------------------------------------
    def get_rate_interval(self, provider: str, market: Optional[str]) -> float:
        """Return ``min_interval_s`` to pass to ``RedisRateLimiter.wait``.

        Resolution order:
        1. Per-market env override (``settings.yfinance_rate_limit_hk``)
        2. Universe-weighted division of the global setting
        3. Built-in default for the provider

        For ``market=None`` (shared scope), returns the global interval.
        """
        global_interval = self._global_interval_for(provider)
        normalized = normalize_market(market)
        if normalized == SHARED_SENTINEL:
            return global_interval

        override = self._per_market_override(provider, normalized)
        if override is not None and override > 0:
            return 1.0 / override  # override is given as req/sec

        # Universe-weighted: bigger universe → bigger share → smaller interval
        weights = self._universe_weights()
        weight = weights.get(normalized, 1.0 / len(SUPPORTED_MARKETS))
        if weight <= 0:
            return global_interval
        # weight is "fraction of total budget". interval scales as 1/weight
        # of the global interval (smaller weight = wait longer).
        return global_interval / weight

    @classmethod
    def _global_interval_for(cls, provider: str) -> float:
        """Return the global aggregate interval for a provider, in seconds."""
        # Map provider -> settings attribute lookups (for back-compat with
        # existing settings keys that pre-date 9.2).
        attr_map = {
            "yfinance": ("yfinance_rate_limit", lambda v: 1.0 / v if v > 0 else 1.0),
            "yfinance:batch": ("yfinance_batch_rate_limit_interval", lambda v: float(v)),
            "finviz": ("finviz_rate_limit_interval", lambda v: float(v)),
        }
        if provider in attr_map:
            attr, transform = attr_map[provider]
            value = getattr(settings, attr, None)
            if value is not None:
                return transform(value)
        return cls._DEFAULT_GLOBAL_INTERVALS_S.get(provider, 1.0)

    @staticmethod
    def _per_market_override(provider: str, market: str) -> Optional[float]:
        """Look up ``settings.<provider>_rate_limit_<market_lower>`` if defined."""
        attr_provider = provider.replace(":", "_")
        attr = f"{attr_provider}_rate_limit_{market.lower()}"
        value = getattr(settings, attr, None)
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            logger.warning("RateBudgetPolicy: invalid override %s=%r; ignoring", attr, value)
            return None

    # ------------------------------------------------------------------
    # Batch size lookup
    # ------------------------------------------------------------------
    def get_batch_size(self, provider: str, market: Optional[str]) -> int:
        """Return the batch size to use for a provider×market combination.

        Resolution: per-market env override → built-in default → 50.
        """
        normalized = normalize_market(market)
        if normalized == SHARED_SENTINEL:
            # Shared scope = use US default as a reasonable midpoint.
            normalized = "US"

        attr = f"{provider.replace(':', '_')}_batch_size_{normalized.lower()}"
        override = getattr(settings, attr, None)
        if override is not None and override > 0:
            return int(override)
        return self._DEFAULT_BATCH_SIZE.get(provider, {}).get(normalized, 50)

    # ------------------------------------------------------------------
    # Worker count per provider × market
    # ------------------------------------------------------------------
    def get_provider_workers(self, provider: str, market: Optional[str]) -> int:
        """Return the number of parallel workers to use for ``provider`` calls
        targeting ``market``.

        Resolution: ``settings.<provider>_workers_<market>`` env override →
        built-in default → 1.

        The Redis rate limiter still serializes egress globally for the
        provider×market key, so this number does not increase req/sec —
        it only fills idle time during HTTP RTT.
        """
        normalized = normalize_market(market)
        if normalized == SHARED_SENTINEL:
            normalized = "US"

        attr_provider = provider.replace(":", "_")
        attr = f"{attr_provider}_workers_{normalized.lower()}"
        override = getattr(settings, attr, None)
        if override is not None:
            try:
                value = int(override)
                if value > 0:
                    return value
            except (TypeError, ValueError):
                logger.warning("RateBudgetPolicy: invalid workers override %s=%r", attr, override)
        return self._DEFAULT_PROVIDER_WORKERS.get(provider, {}).get(normalized, 1)

    # ------------------------------------------------------------------
    # Backoff parameters
    # ------------------------------------------------------------------
    def get_backoff_params(self, provider: str, market: Optional[str]) -> dict:
        """Return ``{"base_s": int, "max_s": int, "factor": float}``.

        Used by ``BulkDataFetcher`` to compute consecutive-backoff delays:
        ``min(base_s * factor**(consecutive-1), max_s)``.
        """
        normalized = normalize_market(market)
        if normalized == SHARED_SENTINEL:
            normalized = "US"

        attr_max = f"{provider.replace(':', '_')}_backoff_max_s_{normalized.lower()}"
        override_max = getattr(settings, attr_max, None)

        defaults = self._DEFAULT_BACKOFF.get(provider, {}).get(
            normalized,
            {"base_s": 60, "max_s": 480, "factor": 2.0},
        )
        if override_max is not None and override_max > 0:
            return {**defaults, "max_s": int(override_max)}
        return dict(defaults)

    # ------------------------------------------------------------------
    # 429 / throttle counters (Redis-only; epic 10 surfaces them)
    # ------------------------------------------------------------------
    def _redis(self):
        if self._redis_client_factory is not None:
            try:
                return self._redis_client_factory()
            except Exception:
                return None
        try:
            from .redis_pool import get_redis_client
            return get_redis_client()
        except Exception:
            return None

    _COUNTER_TTL_S = 90 * 86400  # 90-day retention for daily buckets

    @staticmethod
    def counter_key_429(provider: str, market: Optional[str]) -> str:
        """Return the running 429 counter Redis key. Single source of truth
        used by both ``record_429`` writes and external readers (load harness,
        epic-10 observability) so the format stays in sync."""
        return f"ratelimit:429:{provider}:{market_suffix(market)}"

    @staticmethod
    def counter_keys_throttle(
        provider: str, market: Optional[str], day: Optional[str] = None
    ) -> tuple[str, str]:
        """Return ``(count_key, seconds_key)`` for the per-day throttle bucket."""
        day = day or datetime.now().strftime('%Y%m%d')
        label = market_suffix(market)
        return (
            f"ratelimit:throttle_count:{provider}:{label}:{day}",
            f"ratelimit:throttle_seconds:{provider}:{label}:{day}",
        )

    def _pipeline_counters(self, ops, *, expire_keys=None) -> None:
        """Execute a Redis pipeline of counter ops, swallowing errors.

        ``ops`` is a list of ``(method_name, key, *args)`` tuples. ``expire_keys``
        is the set of keys that should receive the 90-day TTL; keys not in the
        set are left as monotonic (no expiry).
        """
        client = self._redis()
        if client is None:
            return
        try:
            pipe = client.pipeline()
            for method, key, *extra in ops:
                getattr(pipe, method)(key, *extra)
            for key in expire_keys or ():
                pipe.expire(key, self._COUNTER_TTL_S)
            pipe.execute()
        except Exception as exc:
            logger.debug("RateBudgetPolicy: failed to record counter: %s", exc)

    def record_429(self, provider: str, market: Optional[str]) -> None:
        """Increment the per-market×provider 429 counter (running + daily).

        Running key is monotonic (no TTL); daily-bucketed key gets 90-day TTL
        so daily counters self-expire while the running total stays authoritative.
        """
        running_key = self.counter_key_429(provider, market)
        day = datetime.now().strftime('%Y%m%d')
        daily_key = f"{running_key}:{day}"
        self._pipeline_counters(
            [("incr", running_key), ("incr", daily_key)],
            expire_keys={daily_key},
        )

    def record_throttle_wait(self, provider: str, market: Optional[str], wait_s: float) -> None:
        """Record that the rate limiter actually slept for `wait_s` seconds.

        Both keys are daily-bucketed, so both get the 90-day TTL.
        """
        if wait_s <= 0:
            return
        count_key, seconds_key = self.counter_keys_throttle(provider, market)
        self._pipeline_counters(
            [("incr", count_key), ("incrbyfloat", seconds_key, wait_s)],
            expire_keys={count_key, seconds_key},
        )


# Module-level singleton for the common case. Tests can construct their own
# instance with a redis_client_factory override.
_default_policy: Optional[RateBudgetPolicy] = None
_default_policy_lock = threading.Lock()


def get_rate_budget_policy() -> RateBudgetPolicy:
    """Return the process-wide RateBudgetPolicy singleton."""
    global _default_policy
    if _default_policy is None:
        with _default_policy_lock:
            if _default_policy is None:
                _default_policy = RateBudgetPolicy()
    return _default_policy

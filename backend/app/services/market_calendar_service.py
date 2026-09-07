"""Market calendar abstraction for supported market session-aware decisions."""

from __future__ import annotations

import os
from collections.abc import Callable, Iterable, Mapping
from datetime import date, datetime, timedelta, timezone
from importlib import metadata
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from ..domain.markets.calendar_coverage import (
    AnnualCalendarManifest,
    CalendarCoverageRegistry,
    MarketCalendarCoverage,
)
from ..domain.markets.calendar_policy import (
    DEFAULT_CALENDAR_SESSION_OVERRIDES,
    REGULAR_MARKET_CLOSE_TIMES,
    CalendarProvider,
    CalendarSessionOverride,
)
from ..domain.markets.catalog import (
    MarketCatalog,
    MarketCatalogError,
    get_market_catalog,
)
from ..domain.markets.mic import MicFacts
from .market_calendar_adapters import (
    CalendarScheduleUnavailable,
    MarketCalendarAdapter,
    RawMarketCalendarAdapter,
)

try:
    import exchange_calendars as xcals  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - runtime guard
    xcals = None  # type: ignore

try:
    import pandas_market_calendars as pmc  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - runtime guard
    pmc = None  # type: ignore


class MarketCalendarService:
    """Unified market calendar contract backed by provider-specific calendars."""

    CALENDAR_DATA_ROOT = Path(__file__).resolve().parents[2] / "data/market_calendars"
    CALENDAR_MAINTENANCE_DOC = "docs/OPERATIONS.md#market-calendar-maintenance"

    def __init__(
        self,
        calendar_providers: Mapping[CalendarProvider, Callable[[str], object]]
        | None = None,
        market_catalog: MarketCatalog | None = None,
        session_overrides: Iterable[CalendarSessionOverride] | None = None,
        calendar_coverage_registry: CalendarCoverageRegistry | None = None,
    ):
        self._market_catalog = market_catalog or get_market_catalog()
        self._calendar_coverage_registry = (
            calendar_coverage_registry
            if calendar_coverage_registry is not None
            else CalendarCoverageRegistry.load(
                self._configured_calendar_data_root(),
                market_catalog=self._market_catalog,
            )
        )
        self._calendar_providers: dict[
            CalendarProvider,
            Callable[[str], object] | None,
        ] = {
            CalendarProvider.EXCHANGE_CALENDARS: (
                xcals.get_calendar if xcals is not None else None
            ),
            CalendarProvider.PANDAS_MARKET_CALENDARS: (
                pmc.get_calendar if pmc is not None else None
            ),
        }
        self._calendar_providers.update(calendar_providers or {})
        self._session_overrides = self._normalize_session_overrides(
            (
                *DEFAULT_CALENDAR_SESSION_OVERRIDES,
                *(session_overrides or ()),
            )
        )
        self._calendar_cache: dict[
            tuple[CalendarProvider, str, str],
            MarketCalendarAdapter,
        ] = {}

    def _configured_calendar_data_root(self) -> Path:
        configured = os.getenv("MARKET_CALENDAR_DATA_ROOT")
        return Path(configured).expanduser() if configured else self.CALENDAR_DATA_ROOT

    def normalize_market(self, market: str | None) -> str:
        try:
            normalized = self._market_catalog.get(market or "US").code
        except MarketCatalogError as exc:
            raise ValueError(
                f"Unsupported market for calendar service: {market}"
            ) from exc
        return normalized

    def _mic_facts(self, market: str | None, *, mic: str | None = None) -> MicFacts:
        normalized = self.normalize_market(market)
        return self._market_catalog.get(normalized).mic_facts_for(mic)

    def market_timezone(self, market: str, *, mic: str | None = None) -> ZoneInfo:
        return ZoneInfo(self._mic_facts(market, mic=mic).timezone)

    def market_now(
        self,
        market: str,
        now: datetime | None = None,
        *,
        mic: str | None = None,
    ) -> datetime:
        tz = self.market_timezone(market, mic=mic)
        if now is None:
            return datetime.now(tz)
        if now.tzinfo is None:
            return now.replace(tzinfo=tz)
        return now.astimezone(tz)

    def calendar_id(self, market: str, *, mic: str | None = None) -> str:
        return self._mic_facts(market, mic=mic).calendar_id

    def provider_calendar_id(
        self,
        market: str,
        *,
        mic: str | None = None,
    ) -> str | None:
        return self._mic_facts(market, mic=mic).provider_calendar_id

    def calendar_provider(
        self,
        market: str,
        *,
        mic: str | None = None,
    ) -> CalendarProvider:
        return self._calendar_provider_for_facts(self._mic_facts(market, mic=mic))

    def calendar_metadata(
        self,
        market: str,
        *,
        mic: str | None = None,
    ) -> dict[str, str | None]:
        facts = self._mic_facts(market, mic=mic)
        provider = self._calendar_provider_for_facts(facts)
        return {
            "market": self.normalize_market(market),
            "calendar_id": facts.calendar_id,
            "provider_calendar_id": facts.provider_calendar_id or facts.calendar_id,
            "calendar_provider": provider.value,
            "provider_package_version": self._provider_version(provider),
        }

    def default_currency(self, market: str, *, mic: str | None = None) -> str:
        return self._mic_facts(market, mic=mic).default_currency

    def _get_calendar(
        self, market: str, *, mic: str | None = None
    ) -> MarketCalendarAdapter:
        normalized = self.normalize_market(market)
        facts = self._mic_facts(normalized, mic=mic)
        calendar_id = facts.calendar_id
        provider_calendar_id = facts.provider_calendar_id or calendar_id
        provider_engine = self._calendar_provider_for_facts(facts)
        provider = self._calendar_providers.get(provider_engine)
        if provider is None:
            raise RuntimeError(
                f"{provider_engine.value} is required for MarketCalendarService"
            )
        cache_key = (provider_engine, calendar_id, provider_calendar_id)
        if cache_key not in self._calendar_cache:
            self._calendar_cache[cache_key] = RawMarketCalendarAdapter(
                provider(provider_calendar_id),
                cache_namespace=(
                    f"{provider_engine.value}:{calendar_id}:{provider_calendar_id}"
                ),
            )
        return self._calendar_cache[cache_key]

    @staticmethod
    def _normalize_session_overrides(
        overrides: Iterable[CalendarSessionOverride],
    ) -> dict[str, dict[date, bool]]:
        normalized: dict[str, dict[date, bool]] = {}
        for override in overrides:
            if override.is_trading_day:
                raise ValueError(
                    "positive trading-day overrides are not supported without "
                    "effective session bounds"
                )
            normalized.setdefault(override.normalized_market(), {})[override.day] = (
                override.is_trading_day
            )
        return normalized

    @staticmethod
    def _calendar_provider_for_facts(facts: MicFacts) -> CalendarProvider:
        return facts.calendar_provider

    @staticmethod
    def _provider_version(provider_engine: CalendarProvider) -> str | None:
        try:
            return metadata.version(provider_engine.package_name)
        except metadata.PackageNotFoundError:
            return None

    def _override_for_day(self, market: str, day: date) -> bool | None:
        normalized = self.normalize_market(market)
        return self._session_overrides.get(normalized, {}).get(day)

    def _apply_session_overrides(
        self,
        market: str,
        sessions: Iterable[date],
        *,
        start: date,
        end: date,
    ) -> list[date]:
        normalized = self.normalize_market(market)
        effective_sessions = set(sessions)
        for override_day, is_trading_day in self._session_overrides.get(
            normalized,
            {},
        ).items():
            if start <= override_day <= end and not is_trading_day:
                effective_sessions.discard(override_day)
        return sorted(effective_sessions)

    def _require_verified_calculation_date(
        self,
        market: str,
        requested_date: date,
        *,
        mic: str | None = None,
    ) -> MarketCalendarCoverage:
        # Expiry is Market-scoped. Official session overlays are applied only
        # to the primary MIC; explicit alternate MICs remain provider-backed.
        del mic
        coverage = self._calendar_coverage_registry.coverage_for(market)
        if requested_date > coverage.verified_through:
            raise CalendarCoverageExpired(
                market=coverage.market,
                requested_date=requested_date,
                verified_through=coverage.verified_through,
                source_url=coverage.source.url,
                operations_doc=self.CALENDAR_MAINTENANCE_DOC,
            )
        return coverage

    @staticmethod
    def _official_manifest_for_day(
        coverage: MarketCalendarCoverage,
        day: date,
        *,
        mic: str | None = None,
    ) -> AnnualCalendarManifest | None:
        if mic is not None and str(mic).strip().upper() != coverage.mic:
            return None
        annual = coverage.annual.get(day.year)
        if annual is None or annual.status != "official":
            return None
        return annual

    def _provider_trading_days(
        self,
        market: str,
        start: date,
        end: date,
        *,
        mic: str | None,
    ) -> list[date]:
        if start > end:
            return []
        return self._get_calendar(market, mic=mic).sessions_in_range(start, end)

    def _effective_trading_days(
        self,
        market: str,
        start: date,
        end: date,
        coverage: MarketCalendarCoverage,
        *,
        mic: str | None,
    ) -> list[date]:
        sessions: set[date] = set()
        cursor = start
        while cursor <= end:
            year_end = min(end, date(cursor.year, 12, 31))
            annual = self._official_manifest_for_day(coverage, cursor, mic=mic)
            if annual is not None:
                sessions.update(
                    session
                    for session in annual.sessions
                    if cursor <= session <= year_end
                )
            else:
                sessions.update(
                    self._provider_trading_days(
                        market,
                        cursor,
                        year_end,
                        mic=mic,
                    )
                )
            cursor = year_end + timedelta(days=1)
        return self._apply_session_overrides(
            market,
            sessions,
            start=start,
            end=end,
        )

    def _previous_effective_session_date(
        self,
        market: str,
        day: date,
        coverage: MarketCalendarCoverage,
        *,
        mic: str | None = None,
    ) -> date:
        start = day - timedelta(days=370)
        end = day - timedelta(days=1)
        sessions = self._effective_trading_days(
            market, start, end, coverage, mic=mic
        )
        if sessions:
            return sessions[-1]
        raise ValueError(
            f"No previous effective trading session available before {day.isoformat()}"
        )

    def _is_effective_trading_day(
        self,
        market: str,
        day: date,
        coverage: MarketCalendarCoverage,
        *,
        mic: str | None,
    ) -> bool:
        override = self._override_for_day(market, day)
        if override is not None:
            return override
        official = self._official_manifest_for_day(coverage, day, mic=mic)
        if official is not None:
            return day in official.sessions
        return self._get_calendar(market, mic=mic).is_session(pd.Timestamp(day))

    @staticmethod
    def _is_calendar_bounds_error(exc: Exception) -> bool:
        class_name = exc.__class__.__name__.lower()
        message = str(exc).lower()
        return (
            "outofbounds" in class_name
            or "out of bounds" in message
            or "last session" in message
            or "first session" in message
            or "latest date" in message
            or "earliest date" in message
            or "last date" in message
            or "first date" in message
        )

    def is_trading_day(
        self,
        market: str,
        day: date | None = None,
        *,
        mic: str | None = None,
    ) -> bool:
        normalized = self.normalize_market(market)
        candidate_day = day or self.market_now(normalized, mic=mic).date()
        coverage = self._require_verified_calculation_date(
            normalized, candidate_day, mic=mic
        )
        return self._is_effective_trading_day(
            normalized, candidate_day, coverage, mic=mic
        )

    def trading_days(
        self,
        market: str,
        start: date,
        end: date,
        *,
        mic: str | None = None,
    ) -> list[date]:
        """Trading days in ``[start, end]`` (inclusive), chronological order.

        Official annual manifests replace provider membership for represented
        years; older historical years remain provider-backed.
        """
        normalized = self.normalize_market(market)
        if end < start:
            return []
        coverage = self._require_verified_calculation_date(normalized, end, mic=mic)
        return self._effective_trading_days(
            normalized, start, end, coverage, mic=mic
        )

    def session_anchors(
        self,
        market: str,
        as_of_date: date,
        *,
        offsets: tuple[int, ...],
        mic: str | None = None,
    ) -> dict[int, date]:
        """Resolve exact prior Market sessions for fixed lookback offsets."""
        normalized = self.normalize_market(market)
        if not offsets or min(offsets) < 1:
            raise ValueError("session offsets must be positive")
        coverage = self._require_verified_calculation_date(
            normalized, as_of_date, mic=mic
        )
        if not self._is_effective_trading_day(
            normalized, as_of_date, coverage, mic=mic
        ):
            raise ValueError(
                f"{as_of_date.isoformat()} is not a {normalized} trading session"
            )
        maximum = max(offsets)
        start = as_of_date - timedelta(days=maximum * 2 + 30)
        sessions = self._effective_trading_days(
            normalized, start, as_of_date, coverage, mic=mic
        )
        if len(sessions) <= maximum:
            raise ValueError(
                f"{normalized} calendar has {len(sessions)} sessions; "
                f"{maximum + 1} required"
            )
        return {
            0: sessions[-1],
            **{offset: sessions[-1 - offset] for offset in offsets},
        }

    def is_market_open(
        self,
        market: str,
        now: datetime | None = None,
        *,
        mic: str | None = None,
    ) -> bool:
        normalized = self.normalize_market(market)
        market_now = self.market_now(normalized, now=now, mic=mic)
        current_day = market_now.date()
        coverage = self._require_verified_calculation_date(
            normalized, current_day, mic=mic
        )
        if not self._is_effective_trading_day(
            normalized, current_day, coverage, mic=mic
        ):
            return False
        official = self._official_manifest_for_day(
            coverage, current_day, mic=mic
        )
        if official is not None:
            close_time = official.close_exceptions.get(
                current_day,
                REGULAR_MARKET_CLOSE_TIMES[normalized],
            )
            official_close = datetime.combine(
                current_day,
                close_time,
                tzinfo=self.market_timezone(normalized, mic=mic),
            )
            if market_now >= official_close:
                return False

        calendar = self._get_calendar(normalized, mic=mic)
        current_session = pd.Timestamp(current_day)
        minute_utc = pd.Timestamp(market_now).tz_convert("UTC").floor("min")
        open_on_minute = calendar.is_open_on_minute(minute_utc)
        if open_on_minute is not None:
            return open_on_minute
        session_ranges = calendar.session_open_ranges(current_session.date())
        if session_ranges is None:
            return False
        return any(
            market_open.floor("min") <= minute_utc < market_close.floor("min")
            for market_open, market_close in session_ranges
        )

    def session_close(
        self, market: str, day: date, *, mic: str | None = None,
    ) -> datetime:
        """Verified effective session close in UTC, without an update buffer.

        Official primary-MIC close exceptions take precedence over the provider;
        an unavailable alternate-MIC schedule cannot borrow the primary close.
        """
        normalized = self.normalize_market(market)
        coverage = self._require_verified_calculation_date(normalized, day, mic=mic)
        # Validate the MIC even when an official manifest can answer membership.
        zone = self.market_timezone(normalized, mic=mic)
        if not self._is_effective_trading_day(normalized, day, coverage, mic=mic):
            raise ValueError("not_a_trading_session")
        official = self._official_manifest_for_day(coverage, day, mic=mic)
        if official is not None and day in official.close_exceptions:
            return datetime.combine(day, official.close_exceptions[day], tzinfo=zone).astimezone(timezone.utc)
        try:
            close = self._get_calendar(normalized, mic=mic).session_close(day)
        except Exception as exc:
            if official is None or not (self._is_calendar_bounds_error(exc) or isinstance(exc, CalendarScheduleUnavailable)):
                raise
            close = None
        if close is None:
            if official is None:
                raise CalendarScheduleUnavailable("Session close unavailable")
            return datetime.combine(day, REGULAR_MARKET_CLOSE_TIMES[normalized], tzinfo=zone).astimezone(timezone.utc)
        if pd.isna(close) or close.tzinfo is None:
            raise CalendarScheduleUnavailable("Session close is not an aware timestamp")
        return close.to_pydatetime().astimezone(timezone.utc)

    def last_completed_trading_day(
        self,
        market: str,
        now: datetime | None = None,
        *,
        mic: str | None = None,
    ) -> date:
        """Return the latest trading day that should already have end-of-day bars."""
        normalized = self.normalize_market(market)
        market_now = self.market_now(normalized, now=now, mic=mic)
        current_day = market_now.date()
        coverage = self._require_verified_calculation_date(
            normalized, current_day, mic=mic
        )

        if not self._is_effective_trading_day(
            normalized, current_day, coverage, mic=mic
        ):
            return self._previous_effective_session_date(
                normalized,
                current_day,
                coverage,
                mic=mic,
            )
        official = self._official_manifest_for_day(
            coverage, current_day, mic=mic
        )
        exceptional_close = (
            official.close_exceptions.get(current_day) if official is not None else None
        )
        if exceptional_close is not None:
            close_with_buffer = datetime.combine(
                current_day,
                exceptional_close,
                tzinfo=self.market_timezone(normalized, mic=mic),
            ) + timedelta(minutes=30)
        else:
            try:
                market_close = self._get_calendar(normalized, mic=mic).session_close(
                    current_day
                )
            except Exception as exc:
                if official is None or not (
                    self._is_calendar_bounds_error(exc)
                    or isinstance(exc, CalendarScheduleUnavailable)
                ):
                    raise
                market_close = None
            if market_close is None:
                regular_close = REGULAR_MARKET_CLOSE_TIMES[normalized]
                close_with_buffer = datetime.combine(
                    current_day,
                    regular_close,
                    tzinfo=self.market_timezone(normalized, mic=mic),
                ) + timedelta(minutes=30)
            else:
                close_with_buffer = market_close.tz_convert(
                    self.market_timezone(normalized, mic=mic)
                ).to_pydatetime() + timedelta(minutes=30)
        if market_now >= close_with_buffer:
            return current_day
        return self._previous_effective_session_date(
            normalized,
            current_day,
            coverage,
            mic=mic,
        )


class CalendarCoverageExpired(RuntimeError):
    """Raised when a calculation date is later than official calendar coverage."""

    def __init__(
        self,
        *,
        market: str,
        requested_date: date,
        verified_through: date,
        source_url: str,
        operations_doc: str,
    ) -> None:
        self.market = market
        self.requested_date = requested_date
        self.verified_through = verified_through
        self.source_url = source_url
        self.operations_doc = operations_doc
        super().__init__(
            f"{market} calendar coverage expired: requested "
            f"{requested_date.isoformat()}, verified through "
            f"{verified_through.isoformat()}; source={source_url}; "
            f"update guidance={operations_doc}"
        )

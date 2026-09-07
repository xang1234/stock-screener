"""Read-only daily snapshot facts; Task 9 owns Social run orchestration."""
from dataclasses import dataclass, replace
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
import math

from sqlalchemy import select

from app.domain.feature_store.run_metadata import feature_run_market
from app.domain.social_signals.records import validate_utc_timestamp
from app.infra.db.models.feature_store import FeatureRun, FeatureRunPointer, StockFeatureDaily
from app.models.market_exposure import MarketExposure
from app.services.benchmark_registry_service import BenchmarkRegistryService
from app.services.market_calendar_service import MarketCalendarService
from app.services.feature_run_rs_identity import resolve_feature_run_rs_identity, FeatureRunRsIdentityError


def score_value(value):
    """Only finite observed engine scores on the documented 0–100 scale."""
    if isinstance(value, bool) or not isinstance(value, (int, float, Decimal)):
        return None
    return Decimal(str(value)) if math.isfinite(value) and 0 <= value <= 100 else None


def _available_at(value, as_of):
    # SQLite loses timezone metadata on persisted timestamps; database values
    # follow the application's UTC convention, unlike untrusted caller clocks.
    if value is None:
        return False
    return value.replace(tzinfo=timezone.utc) <= as_of if value.tzinfo is None else value <= as_of


@dataclass(frozen=True, slots=True)
class DailyFreshness:
    required_session: date | None
    actual_session: date | None
    fresh: bool
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class PinnedFeatureRun:
    market: str
    run_id: int | None


@dataclass(frozen=True, slots=True)
class ConfirmationFacts:
    feature_run_id: int | None
    market_exposure_id: int | None
    feature_freshness: DailyFreshness
    market_freshness: DailyFreshness
    benchmark_symbol: str | None
    rs_rating: Decimal | None
    market_exposure: Decimal | None
    reasons: tuple[str, ...]


class SocialConfirmationReader:
    def __init__(self, db, *, calendar=None, benchmark_registry=None, grace_minutes=120):
        if type(grace_minutes) is not int or grace_minutes < 0:
            raise ValueError("invalid_grace_minutes")
        self.db = db
        self.calendar = calendar or MarketCalendarService()
        self.registry = benchmark_registry or BenchmarkRegistryService()
        self.grace = timedelta(minutes=grace_minutes)

    def freshness(self, market, as_of, actual_session, *, mic=None):
        validate_utc_timestamp(as_of, "as_of")
        try:
            day = self.calendar.market_now(market, as_of, mic=mic).date()
            sessions = self.calendar.trading_days(market, day - timedelta(days=370), day, mic=mic)
            required = next((session for session in reversed(sessions)
                             if self.calendar.session_close(market, session, mic=mic) + self.grace <= as_of), None)
            if required is None:
                return DailyFreshness(None, actual_session, False, "calendar_unavailable")
        except (ValueError, RuntimeError, KeyError):
            return DailyFreshness(None, actual_session, False, "calendar_unavailable")
        reason = ("missing_session" if actual_session is None else
                  "future_session" if actual_session > required else
                  "stale_session" if actual_session < required else None)
        return DailyFreshness(required, actual_session, reason is None, reason)

    def pin_feature_run(self, market: str) -> PinnedFeatureRun:
        """Resolve once and share across candidates and all Theme measurements."""
        with self.db.no_autoflush:
            pointer = self.db.get(FeatureRunPointer, f"latest_published_market:{market}")
            if pointer is None:
                pointer = self.db.get(FeatureRunPointer, "latest_published")
            return PinnedFeatureRun(market, pointer.run_id if pointer else None)

    def read(self, symbol, market, as_of, *, mic, pinned_run: PinnedFeatureRun | None = None):
        """Pin a published Market run and exposure; never mix symbol runs.

        Stale actual dates remain visible, but unavailable scores are null.
        Missing listing MIC fails closed rather than using a Market calendar.
        """
        validate_utc_timestamp(as_of, "as_of")
        if pinned_run is not None and pinned_run.market != market:
            raise ValueError("pinned_feature_market_mismatch")
        with self.db.no_autoflush:
            return self._read(symbol, market, as_of, mic=mic, pinned_run=pinned_run or self.pin_feature_run(market))

    def _read(self, symbol, market, as_of, *, mic, pinned_run):
        run = self.db.get(FeatureRun, pinned_run.run_id) if pinned_run.run_id is not None else None
        feature = self.db.get(StockFeatureDaily, (run.id, symbol)) if run else None
        ff = self.freshness(market, as_of, feature.as_of_date if feature else None, mic=mic)
        if mic is None:
            ff = replace(ff, fresh=False, reason="listing_mic_unknown")
        reasons = []
        feature_error = None
        if run is None or run.status != "published" or not _available_at(run.published_at, as_of) or not _available_at(run.completed_at, as_of):
            feature_error = "feature_not_available"
        elif feature_run_market(run) != market:
            feature_error = "feature_market_mismatch"
        elif feature is not None and feature.as_of_date != run.as_of_date:
            feature_error = "feature_run_date_mismatch"
        else:
            try:
                resolve_feature_run_rs_identity(run, ranking_date=run.as_of_date)
            except FeatureRunRsIdentityError:
                feature_error = "feature_rs_identity_mismatch"
        if feature_error:
            ff = replace(ff, fresh=False, reason=feature_error)
        market_fact = self.freshness(market, as_of, None)
        exposure = self.db.scalar(select(MarketExposure).where(
            MarketExposure.market == market,
            MarketExposure.date <= market_fact.required_session,
        ).order_by(MarketExposure.date.desc()).limit(1)) if market_fact.required_session else None
        mf = self.freshness(market, as_of, exposure.date if exposure else None)
        candidates = self.registry.get_candidate_symbols(market)
        benchmark = exposure.benchmark_symbol if exposure and exposure.benchmark_symbol in candidates else (candidates[0] if candidates else None)
        if exposure is not None:
            if not _available_at(exposure.created_at, as_of) or not _available_at(exposure.updated_at, as_of):
                mf = replace(mf, fresh=False, reason="market_not_available")
            elif exposure.benchmark_symbol not in candidates:
                mf = replace(mf, fresh=False, reason="market_benchmark_mismatch")
        for fact in (ff, mf):
            if fact.reason:
                reasons.append(fact.reason)
        details = (feature.details_json or {}) if feature else {}
        return ConfirmationFacts(run.id if run else None, exposure.id if exposure else None, ff, mf,
                                 benchmark, score_value(details.get("rs_rating")) if ff.fresh else None,
                                 score_value(exposure.exposure_score) if mf.fresh and exposure else None,
                                 tuple(reasons))

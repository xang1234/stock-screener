"""Read-only daily snapshot facts; Task 9 owns Social run orchestration."""
from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
import math

from sqlalchemy import select

from app.domain.feature_store.run_metadata import feature_run_market
from app.domain.social_signals.records import (ConfirmationInput, ThemeMarketEvidence, validate_utc_timestamp,
    DailyFreshness, PinnedFeatureRun, ConfirmationFacts, MarketConfirmationContext,
    GroupConfirmationContext, MarketConfirmationBatch)
from app.infra.db.models.feature_store import FeatureRun, FeatureRunPointer, StockFeatureDaily
from app.models.market_exposure import MarketExposure
from app.services.benchmark_registry_service import BenchmarkRegistryService
from app.services.market_calendar_service import MarketCalendarService
from app.services.feature_run_rs_identity import resolve_feature_run_rs_identity, FeatureRunRsIdentityError
from app.services.opportunity_state_service import read_liquidity_evidence
from app.services.group_rank_snapshot_reader import GroupRankSnapshotReader, GroupSnapshotIntegrityError
from app.models.industry import IBDGroupRank
from app.models.stock_universe import StockUniverse
from app.services.security_master_service import security_master_resolver


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

    def read_facts(self, symbol, market, as_of, *, mic, pinned_run: PinnedFeatureRun | None = None):
        """Pin a published Market run and exposure; never mix symbol runs.

        Stale actual dates remain visible, but unavailable scores are null.
        Missing listing MIC fails closed rather than using a Market calendar.
        """
        validate_utc_timestamp(as_of, "as_of")
        if pinned_run is not None and pinned_run.market != market:
            raise ValueError("pinned_feature_market_mismatch")
        with self.db.no_autoflush:
            return self._read(symbol, market, as_of, mic=mic, pinned_run=pinned_run or self.pin_feature_run(market))

    def read(self, market: str, symbols: tuple[str, ...], now: datetime) -> tuple[ConfirmationInput, ...]:
        """The provider-neutral batch ConfirmationReader port."""
        return self.read_market(market, symbols, now).inputs

    def read_market(self, market, symbols, now, *, pinned_run=None, theme_keys=None, membership_reader=None):
        """Return immutable local evidence; pin every shared dependency once."""
        validate_utc_timestamp(now, "now")
        pin = pinned_run or self.pin_feature_run(market)
        if pin.market != market:
            raise ValueError("pinned_feature_market_mismatch")
        with self.db.no_autoflush:
            context = self._market_context(market, now)
            group = self._group_context(pin, now)
            inputs, facts = [], []
            for symbol in sorted(set(symbols)):
                security = self.db.scalar(select(StockUniverse).where(
                    StockUniverse.symbol == symbol, StockUniverse.market == market,
                    StockUniverse.active_filter()))
                mic = security_master_resolver.resolve_identity(symbol=symbol, market=market,
                    exchange=security.exchange).mic if security else None
                fact = self._read(symbol, market, now, mic=mic, pinned_run=pin, context=context)
                feature = self.db.get(StockFeatureDaily, (pin.run_id, symbol)) if pin.run_id else None
                details = feature.details_json or {} if feature else {}
                rank = dict(group.cohort).get(details.get("ibd_industry_group"))
                if (not fact.feature_freshness.fresh or group.reason or rank is None
                        or details.get("ibd_group_rank_date") != str(group.session_date)
                        or type(details.get("ibd_group_rank")) is not int
                        or details.get("ibd_group_rank") != rank):
                    rank = None
                    fact = replace(fact, reasons=fact.reasons + (group.reason or "group_feature_mismatch",))
                inputs.append(ConfirmationInput(f"{market}:{symbol}", market, now,
                    fact.setup_score, fact.rs_rating_1m, fact.rs_rating_3m, rank,
                    len(group.cohort) if rank is not None else None, market_benchmark=context.benchmark_symbol))
                facts.append((symbol, fact))
            batch = MarketConfirmationBatch(pin, context, group, tuple(inputs), tuple(facts))
            return self.with_themes(batch, theme_keys=theme_keys, membership_reader=membership_reader)

    def with_themes(self, batch, *, theme_keys=None, membership_reader=None):
        """Measure trusted live/staged baskets using the already frozen Market pins.

        Pass PreparedThemeApplication plus all measured keys for staged additions.
        This method never repins features or exposure, and never accepts raw proposals.
        """
        from app.models.theme import ThemeCluster
        from app.services.social_theme_market_service import (
            LiveAcceptedBasketReader, SocialThemeMarketService, MeasurementUnavailable,
        )
        context = batch.market_context
        evidence, reasons = [], []
        members = membership_reader or LiveAcceptedBasketReader(self.db)
        with self.db.no_autoflush:
            keys = theme_keys if theme_keys is not None else tuple(self.db.scalars(select(ThemeCluster.canonical_key)
                .where(ThemeCluster.pipeline == "technical", ThemeCluster.is_active.is_(True))))
            service = SocialThemeMarketService(self.db, calendar=self.calendar, benchmark_registry=self.registry,
                membership_reader=members, grace_minutes=int(self.grace.total_seconds() // 60),
                pinned_feature_run=batch.pinned_run, benchmark_symbol=context.benchmark_symbol)
            symbols = {symbol for symbol, _ in batch.facts}
            for key in sorted(set(keys)):
                try:
                    basket = members.read(key, context.market)
                    if not any(m.canonical_symbol in symbols for m in basket.membership):
                        continue
                    if (not context.freshness.fresh or context.benchmark_symbol is None
                            or context.benchmark_symbol not in context.benchmark_candidates):
                        raise MeasurementUnavailable("market_benchmark_unavailable")
                    evidence.append(service.measure(key, context.market, context.observed_at,
                        tuple(m.canonical_symbol for m in basket.membership)))
                except MeasurementUnavailable as exc:
                    reasons.append((key, exc.reason))
        inputs = tuple(replace(row, theme_confirmations=tuple(e for e in evidence
            if any(f"{m.market}:{m.canonical_symbol}" == row.candidate_key for m in e.membership))) for row in batch.inputs)
        return replace(batch, inputs=inputs, theme_evidence=tuple(evidence), theme_reasons=tuple(reasons))

    def _group_context(self, pin, now):
        run = self.db.get(FeatureRun, pin.run_id) if pin.run_id else None
        empty = GroupConfirmationContext(run.as_of_date if run else None, None, None, (), reason="group_unavailable")
        if not run or feature_run_market(run) != pin.market:
            return empty
        try:
            identity = resolve_feature_run_rs_identity(run, ranking_date=run.as_of_date)
            rows = GroupRankSnapshotReader().load_publication(self.db, publication=identity.publication,
                include_top_symbol_names=False)
        except (FeatureRunRsIdentityError, GroupSnapshotIntegrityError):
            return replace(empty, reason="group_identity_mismatch")
        records = tuple(self.db.scalars(select(IBDGroupRank).where(IBDGroupRank.market == pin.market,
            IBDGroupRank.date == run.as_of_date, IBDGroupRank.rs_formula_version == identity.identity.formula_version)
            .order_by(IBDGroupRank.rank, IBDGroupRank.industry_group)))
        reason = "group_not_available" if any(not _available_at(r.created_at, now) for r in records) else None
        if len(rows) < 2:
            reason = "group_cohort_too_small"
        return GroupConfirmationContext(run.as_of_date, identity.identity.formula_version, identity.market_rs_run_id,
            tuple((r["industry_group"], r["rank"]) for r in rows), tuple(r.id for r in records), reason)

    def _market_context(self, market, as_of):
        required = self.freshness(market, as_of, None).required_session
        exposure = self.db.scalar(select(MarketExposure).where(MarketExposure.market == market,
            MarketExposure.date <= required).order_by(MarketExposure.date.desc()).limit(1)) if required else None
        freshness = self.freshness(market, as_of, exposure.date if exposure else None)
        candidates = tuple(self.registry.get_candidate_symbols(market))
        benchmark = exposure.benchmark_symbol if exposure else None
        if exposure is not None:
            if not _available_at(exposure.created_at, as_of) or not _available_at(exposure.updated_at, as_of):
                freshness = replace(freshness, fresh=False, reason="market_not_available")
            elif benchmark not in candidates:
                freshness = replace(freshness, fresh=False, reason="market_benchmark_mismatch")
        return MarketConfirmationContext(market, as_of, exposure.id if exposure else None, freshness,
            score_value(exposure.exposure_score) if exposure and freshness.fresh else None,
            benchmark, candidates, self.registry.TABLE_VERSION)

    def _read(self, symbol, market, as_of, *, mic, pinned_run, context=None):
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
        context = context or self._market_context(market, as_of)
        mf = context.freshness
        for fact in (ff, mf):
            if fact.reason:
                reasons.append(fact.reason)
        details = (feature.details_json or {}) if feature else {}
        setup = details.get("setup_engine")
        setup = setup if isinstance(setup, dict) else {}
        liquidity = read_liquidity_evidence(market, details.get("avg_dollar_volume"))
        volume = Decimal(str(details["avg_dollar_volume"])) if liquidity.available else None
        facts = ConfirmationFacts(run.id if run else None, context.exposure_id, ff, mf,
            context.benchmark_symbol, score_value(details.get("rs_rating")) if ff.fresh else None,
            context.exposure_score, tuple(reasons),
            score_value(setup.get("setup_score")) if ff.fresh else None,
            setup.get("setup_ready") if ff.fresh and type(setup.get("setup_ready")) is bool else None,
            score_value(details.get("rs_rating_1m")) if ff.fresh else None,
            score_value(details.get("rs_rating_3m")) if ff.fresh else None,
            liquidity.value if ff.fresh and liquidity.available else None, volume)
        missing = tuple(f"missing_{name}" for name in ("setup_score", "setup_ready", "rs_rating_1m", "rs_rating_3m")
            if getattr(facts, name) is None)
        if facts.liquidity_eligible is None:
            missing += ("missing_liquidity",)
        return replace(facts, reasons=facts.reasons + missing)

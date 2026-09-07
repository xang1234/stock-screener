"""Market price confirmation from a pinned, accepted company basket.

All I/O is local database reading. Staged publication may supply an alternative
trusted AcceptedBasketReader; ordinary callers cannot make proposals accepted
by passing symbols. Task 9 must recheck registry/basket versions at publication.
"""
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from hashlib import sha256
import json
from typing import Protocol

import pandas as pd
from sqlalchemy import select

from app.domain.relative_strength.price_validity import is_valid_adjusted_price
from app.domain.social_signals.records import EffectiveThemeMembership, ThemeMarketEvidence, validate_utc_timestamp
from app.models.stock import StockPrice
from app.models.stock_universe import StockUniverse
from app.models.theme import ThemeCluster
from app.services.benchmark_registry_service import BenchmarkRegistryService
from app.services.social_company_identity_service import SocialCompanyIdentityService
from app.services.social_confirmation_reader import PinnedFeatureRun, SocialConfirmationReader, _available_at
from app.services.social_theme_projection_service import SocialThemeProjectionService
from app.services.social_ticker_resolver import SocialTickerResolver
from app.services.security_master_service import security_master_resolver
from app.services.theme_discovery_service import compound_theme_returns, theme_relative_return_score

COMPONENTS = ("basket_rs_vs_benchmark", "avg_rs_rating", "pct_above_50ma")


class MeasurementUnavailable(RuntimeError):
    """An optional Theme has no trustworthy measurement identity, not run failure."""
    def __init__(self, reason):
        self.reason = reason
        super().__init__(reason)


@dataclass(frozen=True, slots=True)
class AcceptedBasketSnapshot:
    theme_key: str
    market: str
    membership: tuple[EffectiveThemeMembership, ...]
    company_stock_symbols: tuple[str, ...]
    identity_version: int
    identity_policy_version: str
    registry_version: int


class AcceptedBasketReader(Protocol):
    def read(self, theme_key: str, market: str) -> AcceptedBasketSnapshot: ...


class LiveAcceptedBasketReader:
    def __init__(self, db, *, pipeline="technical"):
        self.db, self.pipeline = db, pipeline

    def read(self, theme_key, market):
        identity = SocialCompanyIdentityService(self.db).read()
        theme = self.db.scalar(select(ThemeCluster).where(
            ThemeCluster.canonical_key == theme_key, ThemeCluster.pipeline == self.pipeline,
            ThemeCluster.is_active.is_(True)))
        if theme is None:
            raise MeasurementUnavailable("theme_unavailable")
        members = tuple(m for m in SocialThemeProjectionService(self.db, pipeline=self.pipeline).effective_live_membership(theme.id) if m.market == market)
        resolver = SocialTickerResolver(self.db, verified_company_ids=identity.verified_company_ids)
        stocks = tuple(m.canonical_symbol for m in members if resolver.resolve(m.canonical_symbol, market).security_kind == "stock")
        if SocialCompanyIdentityService(self.db).read() != identity:
            raise MeasurementUnavailable("identity_configuration_changed")
        return AcceptedBasketSnapshot(theme_key, market, members, stocks, identity.version, identity.policy_version, identity.registry_version)


class SocialThemeMarketService:
    def __init__(self, db, *, calendar=None, benchmark_registry=None, membership_reader: AcceptedBasketReader | None = None,
                 grace_minutes=120, pipeline="technical", pinned_feature_run: PinnedFeatureRun | None = None,
                 benchmark_symbol: str | None = None):
        self.db = db
        self.registry = benchmark_registry or BenchmarkRegistryService()
        self.reader = SocialConfirmationReader(db, calendar=calendar, benchmark_registry=self.registry, grace_minutes=grace_minutes)
        self.calendar = self.reader.calendar
        self.membership_reader = membership_reader or LiveAcceptedBasketReader(db, pipeline=pipeline)
        self.pinned_feature_run = pinned_feature_run
        self.benchmark_symbol = benchmark_symbol

    def measure(self, theme_key: str, market: str, as_of: datetime, accepted_symbols: tuple[str, ...]) -> ThemeMarketEvidence:
        validate_utc_timestamp(as_of, "as_of")
        with self.db.no_autoflush:
            return self._measure(theme_key, market, as_of, accepted_symbols)

    def _measure(self, theme_key, market, as_of, accepted_symbols):
        basket = self.membership_reader.read(theme_key, market)
        if basket.theme_key != theme_key or basket.market != market or any(m.market != market for m in basket.membership):
            raise MeasurementUnavailable("basket_identity_mismatch")
        if set(accepted_symbols) != {m.canonical_symbol for m in basket.membership}:
            raise MeasurementUnavailable("membership_changed")
        session = self.reader.freshness(market, as_of, None).required_session
        if session is None:
            raise MeasurementUnavailable("calendar_unavailable")
        candidates = self.registry.get_candidate_symbols(market)
        if not candidates:
            raise MeasurementUnavailable("benchmark_identity_unavailable")
        if self.benchmark_symbol is not None and self.benchmark_symbol not in candidates:
            raise MeasurementUnavailable("benchmark_not_registered")
        companies = {m.company_key for m in basket.membership if m.company_count_eligible}
        unknown = any(m.canonical_symbol in basket.company_stock_symbols and not m.company_count_eligible for m in basket.membership)
        market_days = self._session_window(market, session, 22)
        eligible_candidates = (self.benchmark_symbol,) if self.benchmark_symbol is not None else candidates
        benchmark = eligible_candidates[0]
        benchmark_prices = None
        for candidate in eligible_candidates:
            prices = self._prices(candidate, market_days, as_of)
            if prices is not None and len(market_days) == 22:
                benchmark, benchmark_prices = candidate, prices
                break
        benchmark_return = (compound_theme_returns(pd.Series(benchmark_prices).pct_change(fill_method=None).iloc[1:], 21)
                            if benchmark_prices is not None else None)
        values = {key: {} for key in COMPONENTS}
        selected, dates, runs, details, price_dates, observations = [], [], [], [], [], []
        pinned_run = self.pinned_feature_run or self.reader.pin_feature_run(market)
        for member in sorted(basket.membership, key=lambda m: m.canonical_symbol):
            symbol = member.canonical_symbol
            security = self.db.scalar(select(StockUniverse).where(StockUniverse.symbol == symbol, StockUniverse.market == market, StockUniverse.active_filter()))
            mic = security_master_resolver.resolve_identity(symbol=symbol, market=market, exchange=security.exchange).mic if security else None
            facts = self.reader.read(symbol, market, as_of, mic=mic, pinned_run=pinned_run)
            dates.append((symbol, facts.feature_freshness.actual_session))
            runs.append((symbol, facts.feature_run_id))
            required = facts.feature_freshness.required_session
            latest = self.db.scalar(select(StockPrice).where(StockPrice.symbol == symbol, StockPrice.date <= required).order_by(StockPrice.date.desc()).limit(1)) if required is not None else None
            price_dates.append((symbol, latest.date if latest is not None and _available_at(latest.created_at, as_of) else None))
            if not member.company_count_eligible:
                continue
            company = member.company_key
            if facts.rs_rating is not None:
                values["avg_rs_rating"].setdefault(company, (symbol, facts.rs_rating))
            if mic is None or required is None:
                details.append((symbol, "listing_calendar_unavailable"))
                continue
            days = self._session_window(market, required, 50, mic=mic)
            if not days:
                details.append((symbol, "listing_history_calendar_unavailable"))
            prices50 = self._prices(symbol, days, as_of)
            if prices50 is not None and len(days) == 50:
                # Identical engine validity: a 50-observation rolling mean with
                # min_periods=50. Invalid/missing sessions never become estimates.
                ma = pd.Series(prices50).rolling(window=50, min_periods=50).mean().iloc[-1]
                values["pct_above_50ma"].setdefault(company, (symbol, Decimal(100 if prices50[-1] > ma else 0)))
            if benchmark_prices is not None and days[-22:] == market_days:
                prices22 = self._prices(symbol, market_days, as_of)
                if prices22 is not None:
                    returns = pd.Series(prices22).pct_change(fill_method=None).iloc[1:]
                    values["basket_rs_vs_benchmark"].setdefault(company, (symbol, returns))
        components, counts, reasons = [], [], []
        for key in COMPONENTS:
            cohort = values[key]
            count = len(cohort)
            reason = ("company_identity_coverage_unknown" if unknown else
                      "benchmark_history_unavailable" if key == COMPONENTS[0] and benchmark_prices is None else
                      "insufficient_companies" if count < 3 else
                      "insufficient_company_coverage" if count * 10 < len(companies) * 7 else None)
            value = None
            if reason is None:
                if key == COMPONENTS[0]:
                    returns = pd.concat([item[1] for item in cohort.values()], axis=1).mean(axis=1)
                    value = Decimal(str(round(theme_relative_return_score(compound_theme_returns(returns, 21), benchmark_return), 1)))
                else:
                    value = sum(item[1] for item in cohort.values()) / count
            else:
                reasons.append((key, reason))
            components.append((key, value))
            counts.append((key, count))
            selected.extend((key, company, item[0]) for company, item in sorted(cohort.items()))
            observations.extend((key, company, item[0], tuple(Decimal(str(v)) for v in item[1]) if key == COMPONENTS[0] else (item[1],))
                                for company, item in sorted(cohort.items()))
        reasons.extend(details)
        version = sha256(json.dumps(asdict(basket), sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        return ThemeMarketEvidence(
            theme_key=theme_key, market=market, session_date=session,
            benchmark_symbol=benchmark, basket_version=version,
            accepted_company_count=len(companies), components=tuple(components),
            measured_company_counts=tuple(counts), reasons=tuple(reasons),
            membership=basket.membership, identity_version=basket.identity_version,
            identity_policy_version=basket.identity_policy_version,
            registry_version=basket.registry_version, input_sessions=tuple(dates),
            feature_run_ids=tuple(runs), selected_listings=tuple(selected),
            price_sessions=tuple(price_dates), company_observations=tuple(observations),
            benchmark_return_1m=Decimal(str(benchmark_return)) if benchmark_return is not None else None,
            benchmark_candidates=tuple(candidates), benchmark_registry_version=self.registry.TABLE_VERSION,
            benchmark_selection="explicit_pin" if self.benchmark_symbol is not None else "registry_order",
        )

    def _session_window(self, market, session, count, *, mic=None):
        try:
            return self.calendar.trading_days(market, session - timedelta(days=370), session, mic=mic)[-count:]
        except (ValueError, RuntimeError, KeyError):
            return []

    def _prices(self, symbol, days, as_of):
        if not days:
            return None
        rows = self.db.scalars(select(StockPrice).where(StockPrice.symbol == symbol, StockPrice.date.in_(days))).all()
        prices = {row.date: row.adj_close for row in rows if _available_at(row.created_at, as_of) and is_valid_adjusted_price(row.adj_close)}
        return [prices[day] for day in days] if all(day in prices for day in days) else None

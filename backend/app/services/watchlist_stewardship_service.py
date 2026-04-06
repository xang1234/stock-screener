"""Profile-aware watchlist stewardship built from published runs and recent context."""

from __future__ import annotations

from collections import Counter
from datetime import UTC, date, datetime, timedelta
from typing import Any

from sqlalchemy import func
from sqlalchemy.orm import Session

from app.infra.db.models.feature_store import FeatureRun, StockFeatureDaily
from app.models.market_breadth import MarketBreadth
from app.models.stock_universe import StockUniverse
from app.models.theme import ThemeAlert
from app.models.user_watchlist import UserWatchlist, WatchlistItem
from app.schemas.strategy_profile import StrategyProfileDetail
from app.schemas.user_watchlist import (
    WatchlistStewardshipFreshness,
    WatchlistStewardshipItem,
    WatchlistStewardshipResponse,
    WatchlistStewardshipSummaryCounts,
)
from app.services.stock_event_context_service import StockEventContextService
from app.services.strategy_profile_service import DEFAULT_PROFILE, StrategyProfileService
from app.utils.market_hours import eastern_day_bounds_utc, to_eastern_date

SUPPORTED_THEME_ALERT_TYPES = ("breakout", "velocity_spike")
THEME_SUPPORT_LOOKBACK_DAYS = 7


def _round_or_none(value: float | None, digits: int = 4) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _details_value(row: StockFeatureDaily | None, key: str) -> Any:
    if row is None:
        return None
    details = row.details_json or {}
    return details.get(key)


def _compute_regime_label(breadth: MarketBreadth | None, latest_run: FeatureRun | None, as_of_date: date) -> str:
    if breadth is None:
        return "unavailable"

    score = 0
    if breadth.stocks_up_4pct > breadth.stocks_down_4pct:
        score += 1
    elif breadth.stocks_up_4pct < breadth.stocks_down_4pct:
        score -= 1
    if breadth.ratio_5day is not None:
        score += 1 if breadth.ratio_5day >= 1 else -1
    if breadth.ratio_10day is not None:
        score += 1 if breadth.ratio_10day >= 1 else -1
    if latest_run is None or latest_run.as_of_date is None or (as_of_date - latest_run.as_of_date).days > 3:
        score -= 1
    if score >= 2:
        return "offense"
    if score <= -1:
        return "defense"
    return "balanced"


class WatchlistStewardshipService:
    """Build deterministic stewardship summaries for a watchlist."""

    def __init__(
        self,
        *,
        profile_service: StrategyProfileService | None = None,
        event_context_service: StockEventContextService | None = None,
    ) -> None:
        self._profile_service = profile_service or StrategyProfileService()
        self._event_context_service = event_context_service or StockEventContextService(
            profile_service=self._profile_service
        )

    def get_watchlist_stewardship(
        self,
        db: Session,
        *,
        watchlist_id: int,
        as_of_date: date | None = None,
        profile: str | None = None,
    ) -> WatchlistStewardshipResponse:
        watchlist = db.query(UserWatchlist).filter(UserWatchlist.id == watchlist_id).first()
        if watchlist is None:
            raise ValueError("watchlist_not_found")

        effective_as_of_date = self._resolve_as_of_date(db, as_of_date)
        profile_detail = self._profile_service.get_profile(profile or DEFAULT_PROFILE)
        degraded_reasons: list[str] = []

        latest_run = self._load_latest_feature_run(db, effective_as_of_date)
        previous_run = self._load_previous_feature_run(db, latest_run)
        breadth = self._load_latest_breadth(db, effective_as_of_date)
        regime_label = _compute_regime_label(breadth, latest_run, effective_as_of_date)

        if latest_run is None:
            degraded_reasons.append("missing_latest_feature_run")
        if previous_run is None:
            degraded_reasons.append("missing_previous_feature_run")
        if breadth is None:
            degraded_reasons.append("missing_breadth_snapshot")

        items = (
            db.query(WatchlistItem)
            .filter(WatchlistItem.watchlist_id == watchlist_id)
            .order_by(WatchlistItem.position.asc(), WatchlistItem.id.asc())
            .all()
        )
        symbols = [item.symbol.upper() for item in items if item.symbol]
        company_names = {
            row.symbol: row.name
            for row in db.query(StockUniverse.symbol, StockUniverse.name)
            .filter(StockUniverse.symbol.in_(symbols))
            .all()
        }

        latest_rows = self._load_feature_rows(db, latest_run, symbols)
        previous_rows = self._load_feature_rows(db, previous_run, symbols)
        recent_support = self._load_theme_support_symbols(
            db,
            effective_as_of_date - timedelta(days=THEME_SUPPORT_LOOKBACK_DAYS - 1),
            end_date=effective_as_of_date,
        )
        previous_window_end = effective_as_of_date - timedelta(days=THEME_SUPPORT_LOOKBACK_DAYS)
        previous_support = self._load_theme_support_symbols(
            db,
            previous_window_end - timedelta(days=THEME_SUPPORT_LOOKBACK_DAYS - 1),
            end_date=previous_window_end,
        )
        latest_theme_alert_at = self._load_latest_theme_alert_at(db, effective_as_of_date)

        stewardship_items: list[WatchlistStewardshipItem] = []
        for item in items:
            symbol = item.symbol.upper()
            current_row = latest_rows.get(symbol)
            previous_row = previous_rows.get(symbol)
            stewardship_items.append(
                self._build_item(
                    symbol=symbol,
                    company_name=item.display_name or company_names.get(symbol) or _details_value(current_row, "company_name"),
                    current_row=current_row,
                    previous_row=previous_row,
                    recent_support=recent_support,
                    previous_support=previous_support,
                    effective_as_of_date=effective_as_of_date,
                    regime_label=regime_label,
                    profile_detail=profile_detail,
                )
            )

        status_priority = {
            status: idx
            for idx, status in enumerate(profile_detail.stewardship.status_priority)
        }
        stewardship_items.sort(
            key=lambda item: (
                status_priority.get(item.status, len(status_priority)),
                -(item.current_composite_score if item.current_composite_score is not None else -1),
                item.symbol,
            )
        )

        counts = Counter(item.status for item in stewardship_items)
        summary_counts = WatchlistStewardshipSummaryCounts(
            all=len(stewardship_items),
            strengthening=counts.get("strengthening", 0),
            unchanged=counts.get("unchanged", 0),
            deteriorating=counts.get("deteriorating", 0),
            exit_risk=counts.get("exit_risk", 0),
            missing_from_run=counts.get("missing_from_run", 0),
        )

        return WatchlistStewardshipResponse(
            watchlist_id=watchlist.id,
            watchlist_name=watchlist.name,
            as_of_date=effective_as_of_date.isoformat(),
            freshness=WatchlistStewardshipFreshness(
                latest_feature_as_of_date=latest_run.as_of_date.isoformat() if latest_run else None,
                previous_feature_as_of_date=previous_run.as_of_date.isoformat() if previous_run else None,
                latest_theme_alert_at=latest_theme_alert_at.isoformat() if latest_theme_alert_at else None,
                breadth_date=breadth.date.isoformat() if breadth else None,
            ),
            summary_counts=summary_counts,
            items=stewardship_items,
            degraded_reasons=_dedupe(degraded_reasons),
        )

    def _resolve_as_of_date(self, db: Session, requested_date: date | None) -> date:
        if requested_date is not None:
            return requested_date
        candidates: list[date] = []
        latest_feature_date = (
            db.query(func.max(FeatureRun.as_of_date))
            .filter(FeatureRun.status == "published")
            .scalar()
        )
        if latest_feature_date is not None:
            candidates.append(latest_feature_date)
        latest_breadth_date = db.query(func.max(MarketBreadth.date)).scalar()
        if latest_breadth_date is not None:
            candidates.append(latest_breadth_date)
        latest_theme_alert_at = self._load_latest_theme_alert_at(db, None)
        if latest_theme_alert_at is not None:
            candidates.append(to_eastern_date(latest_theme_alert_at))
        return max(candidates) if candidates else to_eastern_date(datetime.now(UTC))

    def _load_latest_feature_run(self, db: Session, as_of_date: date) -> FeatureRun | None:
        return (
            db.query(FeatureRun)
            .filter(
                FeatureRun.status == "published",
                FeatureRun.as_of_date <= as_of_date,
            )
            .order_by(FeatureRun.as_of_date.desc(), FeatureRun.published_at.desc(), FeatureRun.id.desc())
            .first()
        )

    def _load_previous_feature_run(self, db: Session, latest_run: FeatureRun | None) -> FeatureRun | None:
        if latest_run is None or latest_run.as_of_date is None:
            return None
        return (
            db.query(FeatureRun)
            .filter(
                FeatureRun.status == "published",
                FeatureRun.as_of_date < latest_run.as_of_date,
            )
            .order_by(FeatureRun.as_of_date.desc(), FeatureRun.published_at.desc(), FeatureRun.id.desc())
            .first()
        )

    def _load_latest_breadth(self, db: Session, as_of_date: date) -> MarketBreadth | None:
        return (
            db.query(MarketBreadth)
            .filter(MarketBreadth.date <= as_of_date)
            .order_by(MarketBreadth.date.desc())
            .first()
        )

    def _load_feature_rows(
        self,
        db: Session,
        run: FeatureRun | None,
        symbols: list[str],
    ) -> dict[str, StockFeatureDaily]:
        if run is None or not symbols:
            return {}
        rows = (
            db.query(StockFeatureDaily)
            .filter(
                StockFeatureDaily.run_id == run.id,
                StockFeatureDaily.symbol.in_(symbols),
            )
            .all()
        )
        return {row.symbol.upper(): row for row in rows}

    def _load_theme_support_symbols(
        self,
        db: Session,
        start_date: date,
        end_date: date | None = None,
    ) -> set[str]:
        effective_end_date = end_date or start_date
        start_at, _ = eastern_day_bounds_utc(start_date)
        _, end_at = eastern_day_bounds_utc(effective_end_date)
        rows = (
            db.query(ThemeAlert.related_tickers)
            .filter(
                ThemeAlert.alert_type.in_(SUPPORTED_THEME_ALERT_TYPES),
                ThemeAlert.is_dismissed.is_(False),
                ThemeAlert.triggered_at >= start_at,
                ThemeAlert.triggered_at < end_at,
            )
            .all()
        )
        symbols: set[str] = set()
        for (tickers,) in rows:
            if not isinstance(tickers, list):
                continue
            for ticker in tickers:
                if isinstance(ticker, str) and ticker.strip():
                    symbols.add(ticker.strip().upper())
        return symbols

    def _load_latest_theme_alert_at(self, db: Session, as_of_date: date | None) -> datetime | None:
        query = db.query(func.max(ThemeAlert.triggered_at)).filter(
            ThemeAlert.alert_type.in_(SUPPORTED_THEME_ALERT_TYPES)
        )
        if as_of_date is not None:
            _, end_at = eastern_day_bounds_utc(as_of_date)
            query = query.filter(ThemeAlert.triggered_at < end_at)
        return query.scalar()

    def _build_item(
        self,
        *,
        symbol: str,
        company_name: str | None,
        current_row: StockFeatureDaily | None,
        previous_row: StockFeatureDaily | None,
        recent_support: set[str],
        previous_support: set[str],
        effective_as_of_date: date,
        regime_label: str,
        profile_detail: StrategyProfileDetail,
    ) -> WatchlistStewardshipItem:
        config = profile_detail.stewardship
        reasons: list[str] = []

        if current_row is None:
            return WatchlistStewardshipItem(
                symbol=symbol,
                company_name=company_name,
                status="missing_from_run",
                current_composite_score=None,
                previous_composite_score=_round_or_none(previous_row.composite_score if previous_row else None),
                score_delta=None,
                current_rs_rating=None,
                previous_rs_rating=_round_or_none(_details_value(previous_row, "rs_rating")),
                rs_delta=None,
                next_earnings_date=None,
                days_until_earnings=None,
                theme_support=self._theme_support_label(symbol, recent_support, previous_support),
                reasons=["Symbol is missing from the latest published feature run."],
            )

        current_score = current_row.composite_score
        previous_score = previous_row.composite_score if previous_row else None
        score_delta = (
            current_score - previous_score
            if current_score is not None and previous_score is not None
            else None
        )
        current_rating = current_row.overall_rating
        previous_rating = previous_row.overall_rating if previous_row else None
        current_stage = _details_value(current_row, "stage")
        previous_stage = _details_value(previous_row, "stage")
        stage_delta = (
            current_stage - previous_stage
            if current_stage is not None and previous_stage is not None
            else None
        )
        current_rs = _details_value(current_row, "rs_rating")
        previous_rs = _details_value(previous_row, "rs_rating")
        rs_delta = (
            float(current_rs) - float(previous_rs)
            if current_rs is not None and previous_rs is not None
            else None
        )
        theme_support = self._theme_support_label(symbol, recent_support, previous_support)
        next_earnings_date, days_until_earnings = self._event_context_service.get_next_earnings_summary(
            symbol,
            as_of_date=effective_as_of_date,
        )

        status = "unchanged"
        if current_score is not None and current_score < config.exit_score_max:
            reasons.append(f"Composite score {current_score:.1f} is below the exit threshold.")
        if current_rating is not None and current_rating <= config.exit_rating_max:
            reasons.append("Overall rating has slipped into exit-risk territory.")
        if score_delta is not None and score_delta <= config.exit_score_delta_max:
            reasons.append(f"Composite score dropped {score_delta:.1f} points.")
        if stage_delta is not None and stage_delta <= config.exit_stage_delta_max:
            reasons.append("Stage quality deteriorated materially.")
        if (
            days_until_earnings is not None
            and days_until_earnings <= config.defense_earnings_exit_window_days
            and regime_label == "defense"
        ):
            reasons.append("Earnings are too close for a defensive regime.")

        if reasons:
            status = "exit_risk"
        else:
            if score_delta is not None and score_delta <= config.deteriorating_score_delta_max:
                reasons.append(f"Composite score slipped {score_delta:.1f} points.")
            if (
                current_rating is not None
                and previous_rating is not None
                and current_rating <= previous_rating - 1
            ):
                reasons.append("Overall rating dropped at least one tier.")
            if rs_delta is not None and rs_delta <= config.deteriorating_rs_delta_max:
                reasons.append(f"RS rating faded {rs_delta:.1f} points.")
            if theme_support == "lost":
                reasons.append("Recent theme-alert support disappeared.")
            if reasons:
                status = "deteriorating"
            else:
                if score_delta is not None and score_delta >= config.strengthening_score_delta_min:
                    reasons.append(f"Composite score improved {score_delta:.1f} points.")
                if (
                    current_rating is not None
                    and previous_rating is not None
                    and current_rating >= previous_rating + 1
                ):
                    reasons.append("Overall rating improved by at least one tier.")
                if rs_delta is not None and rs_delta >= config.strengthening_rs_delta_min:
                    reasons.append(f"RS rating improved {rs_delta:.1f} points.")
                if theme_support == "new":
                    reasons.append("New theme-alert support appeared this week.")
                if reasons:
                    status = "strengthening"
                else:
                    reasons.append("No material change versus the previous published run.")

        return WatchlistStewardshipItem(
            symbol=symbol,
            company_name=company_name,
            status=status,
            current_composite_score=_round_or_none(current_score),
            previous_composite_score=_round_or_none(previous_score),
            score_delta=_round_or_none(score_delta),
            current_rs_rating=_round_or_none(current_rs),
            previous_rs_rating=_round_or_none(previous_rs),
            rs_delta=_round_or_none(rs_delta),
            next_earnings_date=next_earnings_date.isoformat() if next_earnings_date else None,
            days_until_earnings=days_until_earnings,
            theme_support=theme_support,
            reasons=_dedupe(reasons),
        )

    def _theme_support_label(
        self,
        symbol: str,
        recent_support: set[str],
        previous_support: set[str],
    ) -> str:
        in_recent = symbol in recent_support
        in_previous = symbol in previous_support
        if in_recent and not in_previous:
            return "new"
        if in_recent and in_previous:
            return "active"
        if not in_recent and in_previous:
            return "lost"
        return "none"

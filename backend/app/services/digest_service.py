"""Compose the daily digest from existing deterministic data sources."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from typing import Any

from sqlalchemy import case, func
from sqlalchemy.orm import Session

from app.domain.feature_store.models import FeatureRow
from app.infra.db.models.feature_store import FeatureRun, StockFeatureDaily
from app.models.market_breadth import MarketBreadth
from app.models.stock_universe import StockUniverse
from app.models.theme import ThemeAlert, ThemeCluster, ThemeMetrics
from app.models.user_watchlist import UserWatchlist, WatchlistItem
from app.schemas.digest import (
    DailyDigestResponse,
    DigestBreadthMetrics,
    DigestFreshness,
    DigestLeaderItem,
    DigestMarketSection,
    DigestRiskNote,
    DigestThemeAlertItem,
    DigestThemeItem,
    DigestThemeSection,
    DigestValidationSection,
    DigestValidationSourceSnapshot,
    DigestWatchlistHighlight,
)
from app.schemas.validation import ValidationHorizonSummary, ValidationSourceKind
from app.services.strategy_profile_service import DEFAULT_PROFILE, StrategyProfileService
from app.services.validation_service import ValidationService
from app.use_cases.scanning.explain_stock import ExplainStockUseCase
from app.utils.market_hours import eastern_day_bounds_utc, to_eastern_date

DIGEST_LEADERS_LIMIT = 5
WATCHLIST_MATCH_LIMIT = 5
WATCHLIST_LEADER_DEPTH = 20
THEME_SECTION_LIMIT = 5
THEME_ALERT_LIMIT = 5
THEME_ALERT_LOOKBACK_DAYS = 7
VALIDATION_LOOKBACK_DAYS = 90
SUPPORTED_THEME_ALERT_TYPES = ("breakout", "velocity_spike")


@dataclass(frozen=True)
class _ThemeSectionResult:
    section: DigestThemeSection
    recent_alert_symbols: set[str]
    degraded_reasons: list[str]
    latest_metrics_date: date | None


class DigestService:
    """Build JSON and markdown daily digests from cached application state."""

    def __init__(
        self,
        *,
        validation_service: ValidationService | None = None,
        profile_service: StrategyProfileService | None = None,
    ) -> None:
        self._validation_service = validation_service or ValidationService()
        self._profile_service = profile_service or StrategyProfileService()

    def get_daily_digest(
        self,
        db: Session,
        *,
        as_of_date: date | None = None,
        profile: str | None = None,
    ) -> DailyDigestResponse:
        effective_as_of_date = self._resolve_as_of_date(db, as_of_date)
        profile_detail = self._profile_service.get_profile(profile or DEFAULT_PROFILE)
        degraded_reasons: list[str] = []

        latest_run = self._load_latest_feature_run(db, effective_as_of_date)
        breadth = self._load_latest_breadth(db, effective_as_of_date)

        market = self._build_market_section(
            breadth=breadth,
            latest_run=latest_run,
            as_of_date=effective_as_of_date,
        )
        if breadth is None:
            degraded_reasons.append("missing_breadth_snapshot")
        if latest_run is None:
            degraded_reasons.append("missing_published_feature_run")

        leaders, leader_degraded = self._build_leaders(
            db,
            latest_run,
            profile_detail=profile_detail,
        )
        degraded_reasons.extend(leader_degraded)

        themes_result = self._build_theme_section(
            db,
            effective_as_of_date,
            profile_detail=profile_detail,
        )
        degraded_reasons.extend(themes_result.degraded_reasons)

        validation = self._build_validation_section(db, effective_as_of_date)
        degraded_reasons.extend(validation["degraded_reasons"])

        watchlists = self._build_watchlist_highlights(
            db,
            latest_run=latest_run,
            recent_alert_symbols=themes_result.recent_alert_symbols,
        )
        if latest_run is None and watchlists:
            degraded_reasons.append("watchlists_missing_feature_run_context")

        latest_theme_alert_at = self._load_latest_theme_alert_at(db, effective_as_of_date)
        freshness = self._build_freshness(
            latest_run=latest_run,
            breadth=breadth,
            latest_theme_metrics_date=themes_result.latest_metrics_date,
            latest_theme_alert_at=latest_theme_alert_at,
        )

        risks = self._build_risk_notes(
            market=market,
            leaders=leaders,
            validation=validation["section"],
            degraded_reasons=_dedupe(degraded_reasons),
            profile_detail=profile_detail,
        )

        return DailyDigestResponse(
            as_of_date=effective_as_of_date.isoformat(),
            freshness=freshness,
            market=market,
            leaders=leaders,
            themes=themes_result.section,
            validation=validation["section"],
            watchlists=watchlists,
            risks=risks,
            degraded_reasons=_dedupe(degraded_reasons),
        )

    def render_markdown(self, payload: DailyDigestResponse) -> str:
        """Render a markdown view from the normalized digest payload."""

        lines: list[str] = [
            f"# Daily Digest ({payload.as_of_date})",
            "",
            f"## Market Stance: {payload.market.stance.title()}",
            payload.market.summary,
            "",
            "## Leaders",
        ]

        if payload.leaders:
            for leader in payload.leaders:
                score = f"{leader.composite_score:.1f}" if leader.composite_score is not None else "-"
                rating = leader.rating or "-"
                industry = leader.industry_group or "Unknown group"
                lines.append(
                    f"- **{leader.symbol}** ({leader.name or 'Unknown'}) | Score {score} | {rating} | {industry} | {leader.reason_summary}"
                )
        else:
            lines.append("- No leader candidates are available.")

        lines.extend(["", "## Themes", "### Leaders"])
        if payload.themes.leaders:
            for theme in payload.themes.leaders:
                momentum = f"{theme.momentum_score:.1f}" if theme.momentum_score is not None else "-"
                lines.append(f"- **{theme.display_name}** | Momentum {momentum} | {theme.status or 'unclassified'}")
        else:
            lines.append("- No ranked themes are available.")

        lines.extend(["", "### Laggards"])
        if payload.themes.laggards:
            for theme in payload.themes.laggards:
                momentum = f"{theme.momentum_score:.1f}" if theme.momentum_score is not None else "-"
                lines.append(f"- **{theme.display_name}** | Momentum {momentum} | {theme.status or 'unclassified'}")
        else:
            lines.append("- No theme laggards are available.")

        lines.extend(["", "### Recent Theme Alerts"])
        if payload.themes.recent_alerts:
            for alert in payload.themes.recent_alerts:
                tickers = ", ".join(alert.related_tickers) if alert.related_tickers else "No tickers"
                lines.append(
                    f"- **{alert.title}** ({alert.alert_type}, {alert.severity or 'info'}) | {alert.theme or 'Unknown theme'} | {tickers}"
                )
        else:
            lines.append("- No recent theme alerts are available.")

        lines.extend(["", "## Validation Snapshot"])
        for snapshot in (payload.validation.scan_pick, payload.validation.theme_alert):
            lines.append(f"### {snapshot.source_kind.value.replace('_', ' ').title()}")
            for horizon in snapshot.horizons:
                avg_return = _format_percent(horizon.avg_return_pct)
                positive_rate = _format_ratio(horizon.positive_rate)
                lines.append(
                    f"- {horizon.horizon_sessions} session: sample {horizon.sample_size}, positive {positive_rate}, avg return {avg_return}"
                )
            if snapshot.degraded_reasons:
                lines.append(f"- Degraded: {', '.join(snapshot.degraded_reasons)}")

        lines.extend(["", "## Watchlist Highlights"])
        if payload.watchlists:
            for watchlist in payload.watchlists:
                matched = ", ".join(watchlist.matched_symbols) if watchlist.matched_symbols else "none"
                alerted = ", ".join(watchlist.alert_symbols) if watchlist.alert_symbols else "none"
                lines.append(
                    f"- **{watchlist.watchlist_name}** | leaders: {matched} | alerts: {alerted} | {watchlist.notes}"
                )
        else:
            lines.append("- No watchlist highlights are available.")

        lines.extend(["", "## Risks"])
        if payload.risks:
            for risk in payload.risks:
                lines.append(f"- **{risk.kind}** ({risk.severity}): {risk.message}")
        else:
            lines.append("- No material risk notes.")

        if payload.degraded_reasons:
            lines.extend(["", "## Degraded Signals", f"- {', '.join(payload.degraded_reasons)}"])

        return "\n".join(lines).strip() + "\n"

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

        latest_theme_metrics_date = db.query(func.max(ThemeMetrics.date)).scalar()
        if latest_theme_metrics_date is not None:
            candidates.append(latest_theme_metrics_date)

        latest_theme_alert_at = (
            db.query(func.max(ThemeAlert.triggered_at))
            .filter(ThemeAlert.alert_type.in_(SUPPORTED_THEME_ALERT_TYPES))
            .scalar()
        )
        if latest_theme_alert_at is not None:
            candidates.append(to_eastern_date(latest_theme_alert_at))

        return max(candidates) if candidates else datetime.now(UTC).date()

    def _load_latest_feature_run(self, db: Session, as_of_date: date) -> FeatureRun | None:
        return (
            db.query(FeatureRun)
            .filter(
                FeatureRun.status == "published",
                FeatureRun.as_of_date <= as_of_date,
            )
            .order_by(
                FeatureRun.as_of_date.desc(),
                case((FeatureRun.published_at.is_(None), 1), else_=0),
                FeatureRun.published_at.desc(),
                FeatureRun.id.desc(),
            )
            .first()
        )

    def _load_latest_breadth(self, db: Session, as_of_date: date) -> MarketBreadth | None:
        return (
            db.query(MarketBreadth)
            .filter(MarketBreadth.date <= as_of_date)
            .order_by(MarketBreadth.date.desc())
            .first()
        )

    def _load_latest_theme_alert_at(self, db: Session, as_of_date: date) -> datetime | None:
        _, alerts_until = eastern_day_bounds_utc(as_of_date)
        return (
            db.query(func.max(ThemeAlert.triggered_at))
            .filter(
                ThemeAlert.alert_type.in_(SUPPORTED_THEME_ALERT_TYPES),
                ThemeAlert.triggered_at < alerts_until,
            )
            .scalar()
        )

    def _build_market_section(
        self,
        *,
        breadth: MarketBreadth | None,
        latest_run: FeatureRun | None,
        as_of_date: date,
    ) -> DigestMarketSection:
        feature_run_stale = latest_run is None or latest_run.as_of_date is None or (
            (as_of_date - latest_run.as_of_date).days > 3
        )
        if breadth is None:
            return DigestMarketSection(
                stance="unavailable",
                summary="Market breadth snapshot is unavailable for the digest date.",
                breadth_metrics=DigestBreadthMetrics(),
            )

        score = 0
        if breadth.stocks_up_4pct > breadth.stocks_down_4pct:
            score += 1
        elif breadth.stocks_up_4pct < breadth.stocks_down_4pct:
            score -= 1

        if breadth.ratio_5day is not None:
            score += 1 if breadth.ratio_5day >= 1 else -1
        if breadth.ratio_10day is not None:
            score += 1 if breadth.ratio_10day >= 1 else -1
        if feature_run_stale:
            score -= 1

        if score >= 2:
            stance = "offense"
        elif score <= -1:
            stance = "defense"
        else:
            stance = "balanced"

        stale_note = " Feature-run freshness is stale." if feature_run_stale else ""
        summary = (
            f"Breadth on {breadth.date.isoformat()} shows {breadth.stocks_up_4pct} stocks up 4%+ versus "
            f"{breadth.stocks_down_4pct} down 4%+, with 5-day/10-day ratios at "
            f"{_format_float(breadth.ratio_5day)} / {_format_float(breadth.ratio_10day)}. "
            f"Current stance is {stance}.{stale_note}"
        )

        return DigestMarketSection(
            stance=stance,
            summary=summary,
            breadth_metrics=DigestBreadthMetrics(
                up_4pct=breadth.stocks_up_4pct,
                down_4pct=breadth.stocks_down_4pct,
                ratio_5day=breadth.ratio_5day,
                ratio_10day=breadth.ratio_10day,
                total_stocks_scanned=breadth.total_stocks_scanned,
            ),
        )

    def _build_leaders(
        self,
        db: Session,
        latest_run: FeatureRun | None,
        *,
        profile_detail,
    ) -> tuple[list[DigestLeaderItem], list[str]]:
        if latest_run is None:
            return [], ["missing_leader_candidates"]

        rows = (
            db.query(StockFeatureDaily)
            .filter(StockFeatureDaily.run_id == latest_run.id)
            .order_by(
                case((StockFeatureDaily.composite_score.is_(None), 1), else_=0),
                StockFeatureDaily.composite_score.desc(),
                StockFeatureDaily.symbol.asc(),
            )
            .limit(max(profile_detail.digest.leader_limit * 4, DIGEST_LEADERS_LIMIT))
            .all()
        )
        if not rows:
            return [], ["missing_leader_candidates"]

        name_map = {
            row.symbol: row.name
            for row in db.query(StockUniverse.symbol, StockUniverse.name)
            .filter(StockUniverse.symbol.in_([item.symbol for item in rows]))
            .all()
        }

        candidate_rows = [
            row
            for row in rows
            if row.composite_score is None
            or row.composite_score >= profile_detail.digest.leader_min_composite_score
        ]
        if not candidate_rows:
            candidate_rows = rows

        def _leader_sort_key(row: StockFeatureDaily) -> tuple[Any, ...]:
            details = row.details_json or {}
            score = float(row.composite_score) if row.composite_score is not None else float("-inf")
            rs_rating = float(details.get("rs_rating")) if details.get("rs_rating") is not None else float("-inf")
            eps_growth = float(details.get("eps_growth_qq")) if details.get("eps_growth_qq") is not None else float("-inf")
            sales_growth = float(details.get("sales_growth_qq")) if details.get("sales_growth_qq") is not None else float("-inf")
            leader_sort = profile_detail.digest.leader_sort
            if leader_sort == "growth_then_score":
                return (-(eps_growth + sales_growth), -score, row.symbol)
            if leader_sort == "rs_then_score":
                return (-rs_rating, -score, row.symbol)
            return (-score, -rs_rating, row.symbol)

        leaders: list[DigestLeaderItem] = []
        for row in sorted(candidate_rows, key=_leader_sort_key)[: profile_detail.digest.leader_limit]:
            feature_row = FeatureRow(
                run_id=row.run_id,
                symbol=row.symbol,
                as_of_date=row.as_of_date,
                composite_score=row.composite_score,
                overall_rating=row.overall_rating,
                passes_count=row.passes_count,
                details=row.details_json,
            )
            explanation_item = ExplainStockUseCase._build_item_from_feature_row(feature_row)
            explanation = ExplainStockUseCase.build_explanation_from_item(explanation_item)
            leaders.append(
                DigestLeaderItem(
                    symbol=row.symbol,
                    name=name_map.get(row.symbol) or (row.details_json or {}).get("company_name"),
                    composite_score=_round_or_none(row.composite_score),
                    rating=explanation.rating,
                    industry_group=(row.details_json or {}).get("ibd_industry_group"),
                    reason_summary=_build_reason_summary(explanation),
                )
            )
        return leaders, []

    def _build_theme_section(
        self,
        db: Session,
        as_of_date: date,
        *,
        profile_detail,
    ) -> _ThemeSectionResult:
        degraded_reasons: list[str] = []
        latest_metrics_date = (
            db.query(func.max(ThemeMetrics.date))
            .filter(ThemeMetrics.date <= as_of_date)
            .scalar()
        )

        theme_leaders: list[DigestThemeItem] = []
        theme_laggards: list[DigestThemeItem] = []
        if latest_metrics_date is None:
            degraded_reasons.append("missing_theme_metrics")
        else:
            theme_rows = (
                db.query(ThemeMetrics, ThemeCluster)
                .join(ThemeCluster, ThemeCluster.id == ThemeMetrics.theme_cluster_id)
                .filter(
                    ThemeMetrics.date == latest_metrics_date,
                    ThemeCluster.is_active.is_(True),
                )
                .all()
            )
            metric_name = profile_detail.digest.theme_sort

            def _metric_value(metrics: ThemeMetrics) -> float:
                value = getattr(metrics, metric_name, None)
                if value is None:
                    return float("-inf")
                return float(value)

            leader_rows = sorted(
                theme_rows,
                key=lambda item: (-_metric_value(item[0]), item[1].display_name.lower()),
            )[:THEME_SECTION_LIMIT]
            leader_theme_ids = {cluster.id for _, cluster in leader_rows}
            laggard_rows = [
                item
                for item in sorted(
                    theme_rows,
                    key=lambda item: (_metric_value(item[0]), item[1].display_name.lower()),
                )
                if item[1].id not in leader_theme_ids
            ][:THEME_SECTION_LIMIT]
            theme_leaders = [self._theme_item(metrics, cluster) for metrics, cluster in leader_rows]
            theme_laggards = [self._theme_item(metrics, cluster) for metrics, cluster in laggard_rows]

        recent_alert_symbols: set[str] = set()
        alerts_cutoff, _ = eastern_day_bounds_utc(
            as_of_date - timedelta(days=THEME_ALERT_LOOKBACK_DAYS - 1)
        )
        _, alerts_until = eastern_day_bounds_utc(as_of_date)
        alert_rows = (
            db.query(ThemeAlert, ThemeCluster.display_name)
            .outerjoin(ThemeCluster, ThemeCluster.id == ThemeAlert.theme_cluster_id)
            .filter(
                ThemeAlert.alert_type.in_(SUPPORTED_THEME_ALERT_TYPES),
                ThemeAlert.is_dismissed.is_(False),
                ThemeAlert.triggered_at >= alerts_cutoff,
                ThemeAlert.triggered_at < alerts_until,
            )
            .order_by(ThemeAlert.triggered_at.desc(), ThemeAlert.id.desc())
            .limit(THEME_ALERT_LIMIT)
            .all()
        )
        recent_alerts: list[DigestThemeAlertItem] = []
        for alert, theme_name in alert_rows:
            related_tickers = [
                ticker.strip().upper()
                for ticker in (alert.related_tickers or [])
                if isinstance(ticker, str) and ticker.strip()
            ]
            recent_alert_symbols.update(related_tickers)
            recent_alerts.append(
                DigestThemeAlertItem(
                    alert_id=alert.id,
                    alert_type=alert.alert_type,
                    severity=alert.severity,
                    triggered_at=_serialize_temporal(alert.triggered_at),
                    theme=theme_name,
                    title=alert.title,
                    related_tickers=related_tickers[:WATCHLIST_MATCH_LIMIT],
                )
            )

        if not recent_alerts:
            degraded_reasons.append("missing_recent_theme_alerts")

        return _ThemeSectionResult(
            section=DigestThemeSection(
                leaders=theme_leaders,
                laggards=theme_laggards,
                recent_alerts=recent_alerts,
            ),
            recent_alert_symbols=recent_alert_symbols,
            degraded_reasons=degraded_reasons,
            latest_metrics_date=latest_metrics_date,
        )

    def _theme_item(self, metrics: ThemeMetrics, cluster: ThemeCluster) -> DigestThemeItem:
        return DigestThemeItem(
            theme_id=cluster.id,
            display_name=cluster.display_name,
            category=cluster.category,
            momentum_score=_round_or_none(metrics.momentum_score),
            mention_velocity=_round_or_none(metrics.mention_velocity),
            basket_return_1m=_round_or_none(metrics.basket_return_1m),
            status=metrics.status,
        )

    def _build_validation_section(
        self,
        db: Session,
        as_of_date: date,
    ) -> dict[str, Any]:
        scan_pick = self._validation_service.get_overview(
            db,
            source_kind=ValidationSourceKind.SCAN_PICK,
            lookback_days=VALIDATION_LOOKBACK_DAYS,
            as_of_date=as_of_date,
        )
        theme_alert = self._validation_service.get_overview(
            db,
            source_kind=ValidationSourceKind.THEME_ALERT,
            lookback_days=VALIDATION_LOOKBACK_DAYS,
            as_of_date=as_of_date,
        )

        degraded_reasons = _dedupe(scan_pick.degraded_reasons + theme_alert.degraded_reasons)
        return {
            "section": DigestValidationSection(
                lookback_days=VALIDATION_LOOKBACK_DAYS,
                scan_pick=DigestValidationSourceSnapshot(
                    source_kind=ValidationSourceKind.SCAN_PICK,
                    horizons=scan_pick.horizons,
                    degraded_reasons=scan_pick.degraded_reasons,
                ),
                theme_alert=DigestValidationSourceSnapshot(
                    source_kind=ValidationSourceKind.THEME_ALERT,
                    horizons=theme_alert.horizons,
                    degraded_reasons=theme_alert.degraded_reasons,
                ),
            ),
            "degraded_reasons": degraded_reasons,
        }

    def _build_watchlist_highlights(
        self,
        db: Session,
        *,
        latest_run: FeatureRun | None,
        recent_alert_symbols: set[str],
    ) -> list[DigestWatchlistHighlight]:
        watchlists = db.query(UserWatchlist).order_by(UserWatchlist.position.asc(), UserWatchlist.id.asc()).all()
        if not watchlists:
            return []

        items = (
            db.query(WatchlistItem)
            .order_by(WatchlistItem.watchlist_id.asc(), WatchlistItem.position.asc(), WatchlistItem.id.asc())
            .all()
        )
        items_by_watchlist: dict[int, list[str]] = defaultdict(list)
        for item in items:
            if item.symbol:
                items_by_watchlist[item.watchlist_id].append(item.symbol.upper())

        leader_symbols: set[str] = set()
        if latest_run is not None:
            leader_rows = (
                db.query(StockFeatureDaily.symbol)
                .filter(StockFeatureDaily.run_id == latest_run.id)
                .order_by(
                    case((StockFeatureDaily.composite_score.is_(None), 1), else_=0),
                    StockFeatureDaily.composite_score.desc(),
                    StockFeatureDaily.symbol.asc(),
                )
                .limit(WATCHLIST_LEADER_DEPTH)
                .all()
            )
            leader_symbols = {row.symbol.upper() for row in leader_rows if row.symbol}

        highlights: list[DigestWatchlistHighlight] = []
        for watchlist in watchlists:
            symbols = items_by_watchlist.get(watchlist.id, [])
            matched_symbols = [symbol for symbol in symbols if symbol in leader_symbols][:WATCHLIST_MATCH_LIMIT]
            alert_symbols = [symbol for symbol in symbols if symbol in recent_alert_symbols][:WATCHLIST_MATCH_LIMIT]
            if latest_run is None:
                notes = "Published run context is unavailable; recent alert overlap is shown where possible."
            elif matched_symbols or alert_symbols:
                notes = (
                    f"{len(matched_symbols)} leader overlap and {len(alert_symbols)} alert overlap "
                    f"out of {len(symbols)} tracked symbols."
                )
            else:
                notes = f"No overlap with the latest leaders or recent theme alerts across {len(symbols)} symbols."
            highlights.append(
                DigestWatchlistHighlight(
                    watchlist_id=watchlist.id,
                    watchlist_name=watchlist.name,
                    matched_symbols=matched_symbols,
                    alert_symbols=alert_symbols,
                    notes=notes,
                )
            )

        highlights.sort(
            key=lambda item: (
                -(len(item.matched_symbols) + len(item.alert_symbols)),
                item.watchlist_name.lower(),
            )
        )
        return highlights

    def _build_freshness(
        self,
        *,
        latest_run: FeatureRun | None,
        breadth: MarketBreadth | None,
        latest_theme_metrics_date: date | None,
        latest_theme_alert_at: datetime | None,
    ) -> DigestFreshness:
        return DigestFreshness(
            latest_feature_as_of_date=latest_run.as_of_date.isoformat() if latest_run and latest_run.as_of_date else None,
            latest_feature_published_at=_serialize_temporal(latest_run.published_at) if latest_run and latest_run.published_at else None,
            latest_breadth_date=breadth.date.isoformat() if breadth else None,
            latest_theme_metrics_date=latest_theme_metrics_date.isoformat() if latest_theme_metrics_date else None,
            latest_theme_alert_at=_serialize_temporal(latest_theme_alert_at) if latest_theme_alert_at else None,
            validation_lookback_days=VALIDATION_LOOKBACK_DAYS,
        )

    def _build_risk_notes(
        self,
        *,
        market: DigestMarketSection,
        leaders: list[DigestLeaderItem],
        validation: DigestValidationSection,
        degraded_reasons: list[str],
        profile_detail,
    ) -> list[DigestRiskNote]:
        notes: list[DigestRiskNote] = []

        if market.stance == "defense":
            notes.append(
                DigestRiskNote(
                    kind="regime",
                    message="Breadth conditions currently favor defense over aggressive offense.",
                    severity="warning",
                )
            )
        elif market.stance == "unavailable":
            notes.append(
                DigestRiskNote(
                    kind="regime",
                    message="Breadth context is unavailable, so market stance confidence is reduced.",
                    severity="warning",
                )
            )

        scan_pick_five = _find_horizon(validation.scan_pick.horizons, 5)
        if scan_pick_five and (
            (
                scan_pick_five.avg_return_pct is not None
                and scan_pick_five.avg_return_pct <= profile_detail.digest.weak_validation_avg_return_floor
            )
            or (
                scan_pick_five.positive_rate is not None
                and scan_pick_five.positive_rate < profile_detail.digest.weak_validation_positive_rate_floor
            )
        ):
            notes.append(
                DigestRiskNote(
                    kind="validation",
                    message="Recent scan-pick validation is weak at the 5-session horizon.",
                    severity="warning",
                )
            )

        theme_alert_five = _find_horizon(validation.theme_alert.horizons, 5)
        if theme_alert_five and (
            (
                theme_alert_five.avg_return_pct is not None
                and theme_alert_five.avg_return_pct <= profile_detail.digest.weak_validation_avg_return_floor
            )
            or (
                theme_alert_five.positive_rate is not None
                and theme_alert_five.positive_rate < profile_detail.digest.weak_validation_positive_rate_floor
            )
        ):
            notes.append(
                DigestRiskNote(
                    kind="theme_alert_validation",
                    message="Recent theme-alert follow-through is weak at the 5-session horizon.",
                    severity="warning",
                )
            )

        industry_counts = Counter(
            leader.industry_group for leader in leaders if leader.industry_group
        )
        if industry_counts:
            top_industry, count = industry_counts.most_common(1)[0]
            if count >= 3:
                notes.append(
                    DigestRiskNote(
                        kind="concentration",
                        message=f"Leader concentration is elevated in {top_industry}.",
                        severity="info",
                    )
                )

        if degraded_reasons:
            notes.append(
                DigestRiskNote(
                    kind="degraded_data",
                    message=f"Digest sections are partially degraded: {', '.join(degraded_reasons)}.",
                    severity="info",
                )
            )

        return notes


def _build_reason_summary(explanation) -> str:
    passed: list[tuple[float, str]] = []
    for screener in explanation.screener_explanations:
        for criterion in screener.criteria:
            max_score = criterion.max_score or 0.0
            if not criterion.passed or max_score <= 0:
                continue
            passed.append((criterion.score / max_score, criterion.name.replace("_", " ")))

    if passed:
        top_names = [name for _, name in sorted(passed, key=lambda item: (-item[0], item[1]))[:2]]
        return f"Strengths led by {', '.join(top_names)}."

    if explanation.rating:
        return f"Rated {explanation.rating} with limited stored screener detail."
    return "Stored screener detail is limited."


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _find_horizon(horizons: list[ValidationHorizonSummary], target: int) -> ValidationHorizonSummary | None:
    for horizon in horizons:
        if horizon.horizon_sessions == target:
            return horizon
    return None


def _format_float(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.2f}"


def _format_percent(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.1f}%"


def _format_ratio(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value * 100:.0f}%"


def _round_or_none(value: float | None, digits: int = 4) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def _serialize_temporal(value: date | datetime) -> str:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=UTC)
        return value.isoformat()
    return value.isoformat()

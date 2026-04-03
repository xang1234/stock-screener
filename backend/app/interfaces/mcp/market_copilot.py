"""Hermes Market Copilot tool adapters exposed over MCP."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Any, Callable

from pydantic import ValidationError
from sqlalchemy import desc, func
from sqlalchemy.orm import Session, sessionmaker

from app.config import settings
from app.domain.common.query import FilterSpec, PageSpec, QuerySpec, SortOrder, SortSpec
from app.infra.db.models.feature_store import FeatureRun
from app.infra.db.uow import SqlUnitOfWork
from app.models.market_breadth import MarketBreadth
from app.models.stock_universe import StockUniverse
from app.models.theme import ThemeAlert, ThemeCluster, ThemeConstituent, ThemeMetrics
from app.models.user_watchlist import UserWatchlist, WatchlistItem
from app.schemas.scanning import ExplainResponse
from app.services.task_registry_service import SCHEDULED_TASKS, TaskRegistryService
from app.use_cases.feature_store.compare_runs import CompareFeatureRunsUseCase, CompareRunsQuery
from app.use_cases.scanning.explain_stock import ExplainStockUseCase

from .models import (
    CandidateFilters,
    CompareFeatureRunsArgs,
    ExplainSymbolArgs,
    FindCandidatesArgs,
    MarketOverviewArgs,
    TaskStatusArgs,
    ThemeStateArgs,
    ToolCitation,
    ToolEnvelope,
    ToolFact,
    ToolFreshness,
    WatchlistAddArgs,
    WatchlistSnapshotArgs,
)

logger = logging.getLogger(__name__)

_SUPPORTED_SORT_FIELDS = {
    "composite_score",
    "rs_rating",
    "price",
    "market_cap",
    "volume",
    "eps_growth_qq",
    "sales_growth_qq",
    "stage",
}


@dataclass(frozen=True)
class ToolSpec:
    """Static description of one MCP tool."""

    name: str
    description: str
    args_model: type
    handler: Callable[[Any], ToolEnvelope]


class MarketCopilotService:
    """Thin, deterministic adapters over the existing backend data model."""

    def __init__(
        self,
        session_factory: sessionmaker,
        app_settings: Any = settings,
    ) -> None:
        self._session_factory = session_factory
        self._settings = app_settings
        self._tool_specs: dict[str, ToolSpec] = {
            spec.name: spec
            for spec in (
                ToolSpec(
                    name="market_overview",
                    description="Summarize the latest published market state, including feature-run freshness, breadth, alerts, tasks, and a quick diff from the prior published run.",
                    args_model=MarketOverviewArgs,
                    handler=self._market_overview,
                ),
                ToolSpec(
                    name="compare_feature_runs",
                    description="Compare two published feature runs and return added symbols, removed symbols, movers, and rating changes. Defaults to the latest two published runs.",
                    args_model=CompareFeatureRunsArgs,
                    handler=self._compare_feature_runs,
                ),
                ToolSpec(
                    name="find_candidates",
                    description="Find current stock candidates from the latest published feature run using opinionated score, RS, price, growth, and classification filters.",
                    args_model=FindCandidatesArgs,
                    handler=self._find_candidates,
                ),
                ToolSpec(
                    name="explain_symbol",
                    description="Explain why a symbol is rated the way it is in the latest published feature run. Use depth='full' for setup details and peers.",
                    args_model=ExplainSymbolArgs,
                    handler=self._explain_symbol,
                ),
                ToolSpec(
                    name="watchlist_snapshot",
                    description="Summarize a watchlist against the latest published feature run, including which symbols are present, missing, strong, or weak.",
                    args_model=WatchlistSnapshotArgs,
                    handler=self._watchlist_snapshot,
                ),
                ToolSpec(
                    name="theme_state",
                    description="Inspect theme momentum, constituents, and alerts for one theme or return a ranked snapshot of current themes.",
                    args_model=ThemeStateArgs,
                    handler=self._theme_state,
                ),
                ToolSpec(
                    name="task_status",
                    description="Inspect scheduled task health and last-run outcomes for StockScreenClaude background jobs.",
                    args_model=TaskStatusArgs,
                    handler=self._task_status,
                ),
                ToolSpec(
                    name="watchlist_add",
                    description="Add symbols to an existing watchlist when MCP watchlist writes are explicitly enabled.",
                    args_model=WatchlistAddArgs,
                    handler=self._watchlist_add,
                ),
            )
        }

    def list_tools(self) -> list[dict[str, Any]]:
        """Return MCP tool definitions."""

        return [
            {
                "name": spec.name,
                "description": spec.description,
                "inputSchema": spec.args_model.model_json_schema(),
            }
            for spec in self._tool_specs.values()
        ]

    def call_tool(self, name: str, arguments: dict[str, Any] | None) -> dict[str, Any]:
        """Validate, execute, and format an MCP tool call."""

        spec = self._tool_specs.get(name)
        if spec is None:
            return self._tool_error(
                f"Unknown tool: {name}",
                code="unknown_tool",
                details={"available_tools": sorted(self._tool_specs)},
            )

        try:
            parsed_args = spec.args_model.model_validate(arguments or {})
        except ValidationError as exc:
            return self._tool_error(
                f"Invalid arguments for {name}",
                code="invalid_arguments",
                details={"errors": exc.errors()},
            )

        try:
            payload = spec.handler(parsed_args).model_dump(mode="json")
        except ValueError as exc:
            return self._tool_error(
                f"Invalid arguments for {name}",
                code="invalid_arguments",
                details={"message": str(exc)},
            )
        except Exception as exc:  # pragma: no cover - defensive guard
            logger.exception("Unhandled MCP tool failure for %s", name)
            return self._tool_error(
                f"{name} failed: {exc}",
                code="tool_execution_failed",
            )

        return {
            "content": [
                {
                    "type": "text",
                    "text": json.dumps(payload, indent=2, sort_keys=True),
                }
            ],
            "structuredContent": payload,
        }

    def _market_overview(self, args: MarketOverviewArgs) -> ToolEnvelope:
        selected_as_of = args.as_of_date
        with self._session_scope() as db:
            selected_run = self._select_run_for_overview(db, selected_as_of)
            previous_run = self._previous_published_run(db, selected_run)
            breadth = self._latest_breadth(db, selected_as_of or getattr(selected_run, "as_of_date", None))
            alerts = self._recent_alerts(db, limit=5)
            tasks = self._task_rows(db)

        comparison = None
        if selected_run is not None and previous_run is not None:
            comparison = self._compare_runs_payload(previous_run.id, selected_run.id, limit=5)

        top_candidates: list[dict[str, Any]] = []
        if selected_run is not None:
            with self._uow_scope() as uow:
                page = uow.feature_store.query_run_as_scan_results(
                    selected_run.id,
                    QuerySpec(page=PageSpec(page=1, per_page=5)),
                    include_sparklines=False,
                    include_setup_payload=False,
                )
                top_candidates = [self._candidate_record(item) for item in page.items]

        failed_tasks = [
            task
            for task in tasks
            if (task.get("last_run") or {}).get("status") == "failed"
        ]
        summary_parts: list[str] = []
        next_actions: list[str] = []
        facts: list[ToolFact] = []
        citations: list[ToolCitation] = []

        if selected_run is None:
            summary_parts.append("No published feature run is available.")
            next_actions.append("Publish a feature run before using market_overview.")
        else:
            summary_parts.append(
                f"Published feature run {selected_run.id} is available for {selected_run.as_of_date.isoformat()}."
            )
            facts.extend(
                [
                    self._fact("latest_run_id", selected_run.id, "feature_runs", selected_run.as_of_date),
                    self._fact("latest_run_status", selected_run.status, "feature_runs", selected_run.as_of_date),
                ]
            )
            citations.append(
                self._citation(
                    "feature_run",
                    f"Feature run {selected_run.id}",
                    f"feature_runs:{selected_run.id}",
                    selected_run.as_of_date,
                )
            )
            if comparison is not None:
                diff_summary = comparison["summary"]
                summary_parts.append(
                    f"Versus run {comparison['run_a']['id']}, the latest run added {len(comparison['added'])} symbols, removed {len(comparison['removed'])}, and recorded {len(comparison['movers'])} movers."
                )
                facts.append(
                    self._fact(
                        "avg_score_change",
                        diff_summary["avg_score_change"],
                        "feature_run_compare",
                        selected_run.as_of_date,
                    )
                )

        if breadth is None:
            summary_parts.append("No breadth snapshot is available for the requested date.")
            next_actions.append("Calculate market breadth to complete the market overview.")
        else:
            summary_parts.append(f"Breadth data is available for {breadth.date.isoformat()}.")
            facts.extend(
                [
                    self._fact("breadth_up_4pct", breadth.stocks_up_4pct, "market_breadth", breadth.date),
                    self._fact("breadth_down_4pct", breadth.stocks_down_4pct, "market_breadth", breadth.date),
                ]
            )
            citations.append(
                self._citation(
                    "market_breadth",
                    f"Breadth snapshot {breadth.date.isoformat()}",
                    f"market_breadth:{breadth.date.isoformat()}",
                    breadth.date,
                )
            )

        summary_parts.append(f"{len(alerts)} recent unread theme alerts are in scope.")
        facts.append(self._fact("unread_theme_alerts", len(alerts), "theme_alerts"))
        if alerts:
            next_actions.append("Review unread theme alerts for new or accelerating themes.")
            citations.extend(
                [
                    self._citation(
                        "theme_alert",
                        alert["title"],
                        f"theme_alerts:{alert['id']}",
                        alert["triggered_at"][:10] if alert.get("triggered_at") else None,
                    )
                    for alert in alerts[:3]
                ]
            )

        summary_parts.append(f"{len(failed_tasks)} scheduled tasks most recently failed.")
        facts.append(self._fact("failed_scheduled_tasks", len(failed_tasks), "task_execution_history"))
        if failed_tasks:
            next_actions.append("Investigate failed scheduled tasks before trusting stale market state.")
            citations.extend(
                [
                    self._citation(
                        "scheduled_task",
                        task["display_name"],
                        f"scheduled_tasks:{task['name']}",
                        (self._last_run(task).get("started_at") or "")[:10] or None,
                    )
                    for task in failed_tasks[:3]
                ]
            )

        return self._envelope(
            " ".join(summary_parts),
            facts=facts,
            citations=citations,
            next_actions=next_actions,
            freshness=self._freshness(
                as_of_date=(selected_run.as_of_date if selected_run is not None else selected_as_of),
                feature_run=selected_run.published_at.isoformat() if selected_run and selected_run.published_at else None,
                market_breadth=breadth.date.isoformat() if breadth is not None else None,
                theme_alerts=alerts[0]["triggered_at"] if alerts else None,
                task_execution_history=self._last_run(failed_tasks[0]).get("started_at") if failed_tasks else None,
            ),
            runs={
                "selected": self._run_record(selected_run),
                "previous": self._run_record(previous_run),
                "comparison": comparison,
            },
            breadth=self._breadth_record(breadth),
            alerts=alerts,
            tasks=failed_tasks or tasks[:5],
            top_candidates=top_candidates,
        )

    def _compare_feature_runs(self, args: CompareFeatureRunsArgs) -> ToolEnvelope:
        with self._session_scope() as db:
            run_a, run_b = self._resolve_compare_runs(db, args)
            if run_a is None or run_b is None:
                return self._envelope(
                    "Two published feature runs are required for comparison.",
                    facts=[],
                    citations=[],
                    next_actions=["Publish another feature run or provide explicit run_a and run_b values."],
                    freshness=self._freshness(),
                    runs={"requested": {"run_a": args.run_a, "run_b": args.run_b}},
                )

        comparison = self._compare_runs_payload(run_a.id, run_b.id, args.limit)
        return self._envelope(
            f"Compared published runs {run_a.id} ({run_a.as_of_date.isoformat()}) and {run_b.id} ({run_b.as_of_date.isoformat()}): {len(comparison['added'])} added, {len(comparison['removed'])} removed, {len(comparison['movers'])} movers.",
            facts=[
                self._fact("run_a_id", run_a.id, "feature_runs", run_a.as_of_date),
                self._fact("run_b_id", run_b.id, "feature_runs", run_b.as_of_date),
                self._fact("upgraded_count", comparison["summary"]["upgraded_count"], "feature_run_compare", run_b.as_of_date),
                self._fact("downgraded_count", comparison["summary"]["downgraded_count"], "feature_run_compare", run_b.as_of_date),
            ],
            citations=[
                self._citation("feature_run", f"Feature run {run_a.id}", f"feature_runs:{run_a.id}", run_a.as_of_date),
                self._citation("feature_run", f"Feature run {run_b.id}", f"feature_runs:{run_b.id}", run_b.as_of_date),
            ],
            next_actions=[],
            freshness=self._freshness(
                as_of_date=run_b.as_of_date,
                feature_run_compare=run_b.as_of_date.isoformat(),
            ),
            runs={
                "run_a": comparison["run_a"],
                "run_b": comparison["run_b"],
            },
            added=comparison["added"],
            removed=comparison["removed"],
            movers=comparison["movers"],
            summary_stats=comparison["summary"],
        )

    def _find_candidates(self, args: FindCandidatesArgs) -> ToolEnvelope:
        with self._session_scope() as db:
            latest_run = self._latest_published_run(db)
            if latest_run is None:
                return self._envelope(
                    "No published feature run is available for candidate search.",
                    facts=[],
                    citations=[],
                    next_actions=["Publish a feature run before using find_candidates."],
                    freshness=self._freshness(),
                    symbols=[],
                )

            watchlist_scope = None
            if args.universe and args.universe.lower() not in {"all", "latest_published"}:
                watchlist_scope = self._resolve_watchlist(db, args.universe)
                if watchlist_scope is None:
                    return self._envelope(
                        f"Universe scope {args.universe!r} was not found as a watchlist.",
                        facts=[],
                        citations=[],
                        next_actions=["Use 'all' or pass the name of an existing watchlist."],
                        freshness=self._freshness(as_of_date=latest_run.as_of_date),
                        symbols=[],
                    )

        if watchlist_scope is not None:
            symbols = self._find_candidates_for_watchlist(latest_run.id, watchlist_scope.id, args)
        else:
            symbols = self._find_candidates_for_run(latest_run.id, args)

        scope_label = watchlist_scope.name if watchlist_scope is not None else "latest published run"
        next_actions: list[str] = []
        if not symbols:
            next_actions.append("Broaden the filter thresholds or use a wider universe scope.")

        return self._envelope(
            f"Found {len(symbols)} candidates in {scope_label}.",
            facts=[
                self._fact("candidate_count", len(symbols), "feature_runs", latest_run.as_of_date),
                self._fact("run_id", latest_run.id, "feature_runs", latest_run.as_of_date),
            ],
            citations=[
                self._citation("feature_run", f"Feature run {latest_run.id}", f"feature_runs:{latest_run.id}", latest_run.as_of_date),
            ],
            next_actions=next_actions,
            freshness=self._freshness(as_of_date=latest_run.as_of_date, feature_run=latest_run.published_at.isoformat() if latest_run.published_at else None),
            run=self._run_record(latest_run),
            scope={"type": "watchlist" if watchlist_scope is not None else "published_run", "value": scope_label},
            symbols=symbols,
        )

    def _explain_symbol(self, args: ExplainSymbolArgs) -> ToolEnvelope:
        explain_use_case = ExplainStockUseCase()
        with self._uow_scope() as uow:
            latest_run = uow.feature_runs.get_latest_published()
            if latest_run is None:
                return self._envelope(
                    "No published feature run is available for symbol explanation.",
                    facts=[],
                    citations=[],
                    next_actions=["Publish a feature run before using explain_symbol."],
                    freshness=self._freshness(),
                    symbol=args.symbol,
                )

            row = uow.feature_store.get_row_by_symbol(latest_run.id, args.symbol)
            scan_item = uow.feature_store.get_by_symbol_for_run(
                latest_run.id,
                args.symbol,
                include_sparklines=False,
                include_setup_payload=True,
            )
            if row is None:
                return self._envelope(
                    f"{args.symbol} is not present in the latest published feature run.",
                    facts=[self._fact("symbol", args.symbol, "feature_runs", latest_run.as_of_date)],
                    citations=[self._citation("feature_run", f"Feature run {latest_run.id}", f"feature_runs:{latest_run.id}", latest_run.as_of_date)],
                    next_actions=["Try a symbol that exists in the latest published run."],
                    freshness=self._freshness(as_of_date=latest_run.as_of_date),
                    symbol=args.symbol,
                )
            if scan_item is None:
                return self._envelope(
                    f"{args.symbol} could not be mapped from the latest published feature run.",
                    facts=[self._fact("symbol", args.symbol, "feature_runs", latest_run.as_of_date)],
                    citations=[self._citation("feature_run", f"Feature run {latest_run.id}", f"feature_runs:{latest_run.id}", latest_run.as_of_date)],
                    next_actions=["Inspect the stored feature row for missing detail fields."],
                    freshness=self._freshness(as_of_date=latest_run.as_of_date),
                    symbol=args.symbol,
                )

            explanation_item = explain_use_case._build_item_from_feature_row(row)
            explanation = self._build_stock_explanation(explain_use_case, explanation_item)
            explanation_payload = ExplainResponse.from_domain(explanation).model_dump(mode="json")
            peers = []
            setup_payload = None
            if args.depth == "full":
                group_name = scan_item.extended_fields.get("ibd_industry_group")
                if group_name:
                    peers = [
                        self._candidate_record(peer)
                        for peer in uow.feature_store.get_peers_by_industry_for_run(latest_run.id, group_name)
                        if peer.symbol != args.symbol
                    ][:5]
                setup_payload = uow.feature_store.get_setup_payload_for_run(latest_run.id, args.symbol)

        summary = (
            f"{args.symbol} is rated {explanation.rating} in published run {latest_run.id} "
            f"with composite score {round(explanation.composite_score, 2)}."
        )
        if scan_item.extended_fields.get("stage") is not None:
            summary += f" It is currently classified as stage {scan_item.extended_fields['stage']}."

        return self._envelope(
            summary,
            facts=[
                self._fact("symbol", args.symbol, "feature_runs", latest_run.as_of_date),
                self._fact("composite_score", round(explanation.composite_score, 2), "feature_runs", latest_run.as_of_date),
                self._fact("rating", explanation.rating, "feature_runs", latest_run.as_of_date),
            ],
            citations=[
                self._citation("feature_run", f"Feature run {latest_run.id}", f"feature_runs:{latest_run.id}", latest_run.as_of_date),
            ],
            next_actions=[] if explanation_payload["screener_explanations"] else ["Inspect the stored screener payload for missing breakdowns."],
            freshness=self._freshness(as_of_date=latest_run.as_of_date, feature_run=latest_run.published_at.isoformat() if latest_run.published_at else None),
            symbol=args.symbol,
            depth=args.depth,
            run=self._run_record(latest_run),
            result=self._candidate_record(scan_item),
            explanation=explanation_payload,
            peers=peers,
            setup_payload=setup_payload if args.depth == "full" else None,
        )

    def _watchlist_snapshot(self, args: WatchlistSnapshotArgs) -> ToolEnvelope:
        with self._session_scope() as db:
            watchlist = self._resolve_watchlist(db, args.watchlist)
            if watchlist is None:
                return self._envelope(
                    f"Watchlist {args.watchlist!r} was not found.",
                    facts=[],
                    citations=[],
                    next_actions=["Create the watchlist in StockScreenClaude before requesting a snapshot."],
                    freshness=self._freshness(),
                    watchlist=None,
                )

            watchlist_items = (
                db.query(WatchlistItem)
                .filter(WatchlistItem.watchlist_id == watchlist.id)
                .order_by(WatchlistItem.position.asc(), WatchlistItem.symbol.asc())
                .all()
            )
            latest_run = self._latest_published_run(db)

        if latest_run is None:
            return self._envelope(
                f"Watchlist {watchlist.name} has {len(watchlist_items)} symbols, but there is no published feature run to snapshot against.",
                facts=[self._fact("watchlist_size", len(watchlist_items), "user_watchlists")],
                citations=[self._citation("watchlist", watchlist.name, f"user_watchlists:{watchlist.id}")],
                next_actions=["Publish a feature run before requesting a watchlist snapshot."],
                freshness=self._freshness(),
                watchlist={
                    "id": watchlist.id,
                    "name": watchlist.name,
                    "symbols": [item.symbol for item in watchlist_items],
                },
                items=[],
                missing_symbols=[],
            )

        with self._uow_scope() as uow:
            present_items = []
            missing_symbols = []
            for item in watchlist_items:
                candidate = uow.feature_store.get_by_symbol_for_run(
                    latest_run.id,
                    item.symbol,
                    include_sparklines=False,
                    include_setup_payload=False,
                )
                if candidate is None:
                    missing_symbols.append(item.symbol)
                else:
                    present_items.append(self._candidate_record(candidate))

        strongest = sorted(
            present_items,
            key=lambda row: row.get("composite_score") or -1,
            reverse=True,
        )[:3]
        next_actions: list[str] = []
        if missing_symbols:
            next_actions.append("Review symbols missing from the latest published feature run.")

        return self._envelope(
            f"Watchlist {watchlist.name} has {len(watchlist_items)} symbols; {len(present_items)} are present in published run {latest_run.id} and {len(missing_symbols)} are missing.",
            facts=[
                self._fact("watchlist_id", watchlist.id, "user_watchlists"),
                self._fact("watchlist_size", len(watchlist_items), "user_watchlists"),
                self._fact("present_in_latest_run", len(present_items), "feature_runs", latest_run.as_of_date),
            ],
            citations=[
                self._citation("watchlist", watchlist.name, f"user_watchlists:{watchlist.id}"),
                self._citation("feature_run", f"Feature run {latest_run.id}", f"feature_runs:{latest_run.id}", latest_run.as_of_date),
            ],
            next_actions=next_actions,
            freshness=self._freshness(as_of_date=latest_run.as_of_date, feature_run=latest_run.published_at.isoformat() if latest_run.published_at else None),
            watchlist={"id": watchlist.id, "name": watchlist.name},
            items=present_items,
            strongest=strongest,
            missing_symbols=missing_symbols,
            run=self._run_record(latest_run),
        )

    def _theme_state(self, args: ThemeStateArgs) -> ToolEnvelope:
        with self._session_scope() as db:
            if args.theme_name:
                theme = self._resolve_theme(db, args.theme_name)
                if theme is None:
                    return self._envelope(
                        f"No active theme matched {args.theme_name!r}.",
                        facts=[],
                        citations=[],
                        next_actions=["Use a broader theme name or omit theme_name to inspect the ranked theme snapshot."],
                        freshness=self._freshness(),
                        themes=[],
                    )

                metrics = (
                    db.query(ThemeMetrics)
                    .filter(ThemeMetrics.theme_cluster_id == theme.id)
                    .order_by(ThemeMetrics.date.desc())
                    .first()
                )
                constituents = (
                    db.query(ThemeConstituent)
                    .filter(ThemeConstituent.theme_cluster_id == theme.id, ThemeConstituent.is_active == True)
                    .order_by(desc(ThemeConstituent.confidence), desc(ThemeConstituent.mention_count))
                    .limit(args.limit)
                    .all()
                )
                alerts = (
                    db.query(ThemeAlert)
                    .filter(ThemeAlert.theme_cluster_id == theme.id, ThemeAlert.is_dismissed == False)
                    .order_by(ThemeAlert.triggered_at.desc())
                    .limit(args.limit)
                    .all()
                )

                theme_payload = self._theme_record(theme, metrics)
                return self._envelope(
                    f"Theme {theme.display_name} has {len(constituents)} active constituents and {len(alerts)} active alerts.",
                    facts=[
                        self._fact("theme_cluster_id", theme.id, "theme_clusters"),
                        self._fact("active_constituents", len(constituents), "theme_constituents"),
                        self._fact("active_alerts", len(alerts), "theme_alerts"),
                    ],
                    citations=[
                        self._citation("theme_cluster", theme.display_name, f"theme_clusters:{theme.id}", theme.last_seen_at.date() if theme.last_seen_at else None),
                    ],
                    next_actions=[] if alerts else ["Review the theme's latest metrics and constituent quality for new alert opportunities."],
                    freshness=self._freshness(
                        as_of_date=metrics.date if metrics is not None else None,
                        theme_metrics=metrics.date.isoformat() if metrics is not None else None,
                        theme_alerts=alerts[0].triggered_at.isoformat() if alerts else None,
                    ),
                    theme=theme_payload,
                    constituents=[self._theme_constituent_record(row) for row in constituents],
                    alerts=[self._alert_record(row) for row in alerts],
                )

            latest_metrics_date = db.query(func.max(ThemeMetrics.date)).scalar()
            ranked_themes = []
            if latest_metrics_date is not None:
                ranked_rows = (
                    db.query(ThemeCluster, ThemeMetrics)
                    .join(ThemeMetrics, ThemeMetrics.theme_cluster_id == ThemeCluster.id)
                    .filter(ThemeMetrics.date == latest_metrics_date, ThemeCluster.is_active == True)
                    .order_by(ThemeMetrics.rank.asc().nullslast(), desc(ThemeMetrics.momentum_score))
                    .limit(args.limit)
                    .all()
                )
                ranked_themes = [self._theme_record(theme, metrics) for theme, metrics in ranked_rows]

            alerts = self._recent_alerts(db, limit=min(args.limit, 10))

        return self._envelope(
            f"Theme snapshot includes {len(ranked_themes)} ranked themes and {len(alerts)} unread alerts.",
            facts=[
                self._fact("ranked_themes", len(ranked_themes), "theme_metrics", latest_metrics_date),
                self._fact("unread_alerts", len(alerts), "theme_alerts", latest_metrics_date),
            ],
            citations=[],
            next_actions=[] if ranked_themes else ["Run the theme pipeline to populate current theme metrics."],
            freshness=self._freshness(
                as_of_date=latest_metrics_date,
                theme_metrics=latest_metrics_date.isoformat() if latest_metrics_date is not None else None,
                theme_alerts=alerts[0]["triggered_at"] if alerts else None,
            ),
            themes=ranked_themes,
            alerts=alerts,
        )

    def _task_status(self, args: TaskStatusArgs) -> ToolEnvelope:
        with self._session_scope() as db:
            task_rows = self._task_rows(db)

        if args.task_name:
            task_rows = [row for row in task_rows if row["name"] == args.task_name]
            if not task_rows:
                return self._envelope(
                    f"Task {args.task_name!r} is not registered.",
                    facts=[],
                    citations=[],
                    next_actions=[f"Use one of: {', '.join(sorted(SCHEDULED_TASKS))}"],
                    freshness=self._freshness(),
                    tasks=[],
                )

        failed = [task for task in task_rows if self._last_run(task).get("status") == "failed"]
        running = [task for task in task_rows if self._last_run(task).get("status") == "running"]
        return self._envelope(
            f"{len(task_rows)} scheduled tasks tracked; {len(failed)} recently failed and {len(running)} are currently running.",
            facts=[
                self._fact("tracked_tasks", len(task_rows), "scheduled_tasks"),
                self._fact("failed_tasks", len(failed), "task_execution_history"),
                self._fact("running_tasks", len(running), "task_execution_history"),
            ],
            citations=[
                self._citation("scheduled_task", task["display_name"], f"scheduled_tasks:{task['name']}")
                for task in task_rows[:3]
            ],
            next_actions=(["Inspect the failed task records before relying on stale outputs."] if failed else []),
            freshness=self._freshness(
                task_execution_history=self._last_run(task_rows[0]).get("started_at") if task_rows else None,
            ),
            tasks=task_rows,
        )

    def _watchlist_add(self, args: WatchlistAddArgs) -> ToolEnvelope:
        writes_enabled = bool(getattr(self._settings, "mcp_watchlist_writes_enabled", False))
        with self._session_scope() as db:
            watchlist = self._resolve_watchlist(db, args.watchlist)
            if watchlist is None:
                return self._envelope(
                    f"Watchlist {args.watchlist!r} was not found.",
                    facts=[],
                    citations=[],
                    next_actions=["Create the watchlist in StockScreenClaude before adding symbols."],
                    freshness=self._freshness(),
                    watchlist=None,
                    added=[],
                    skipped=args.symbols,
                    writes_enabled=writes_enabled,
                )

            if not writes_enabled:
                return self._envelope(
                    "Watchlist writes are disabled for this MCP server.",
                    facts=[self._fact("writes_enabled", False, "settings")],
                    citations=[self._citation("watchlist", watchlist.name, f"user_watchlists:{watchlist.id}")],
                    next_actions=["Set MCP_WATCHLIST_WRITES_ENABLED=true to allow watchlist_add."],
                    freshness=self._freshness(),
                    watchlist={"id": watchlist.id, "name": watchlist.name},
                    added=[],
                    skipped=args.symbols,
                    writes_enabled=False,
                )

            existing_items = (
                db.query(WatchlistItem)
                .filter(WatchlistItem.watchlist_id == watchlist.id)
                .order_by(WatchlistItem.position.asc())
                .all()
            )
            watchlist_id = watchlist.id
            watchlist_name = watchlist.name
            existing_symbols = {item.symbol for item in existing_items}
            max_position = existing_items[-1].position if existing_items else -1
            name_map = {
                row.symbol: row.name
                for row in db.query(StockUniverse)
                .filter(StockUniverse.symbol.in_(args.symbols))
                .all()
            }

            added: list[dict[str, Any]] = []
            skipped: list[str] = []
            for symbol in args.symbols:
                if symbol in existing_symbols:
                    skipped.append(symbol)
                    continue
                max_position += 1
                item = WatchlistItem(
                    watchlist_id=watchlist_id,
                    symbol=symbol,
                    display_name=name_map.get(symbol),
                    notes=args.reason,
                    position=max_position,
                )
                db.add(item)
                added.append({"symbol": symbol, "display_name": name_map.get(symbol), "reason": args.reason})
            db.commit()

        next_actions = []
        if skipped:
            next_actions.append("Review skipped symbols that were already present in the watchlist.")

        return self._envelope(
            f"Added {len(added)} symbols to watchlist {watchlist_name}; {len(skipped)} symbols were skipped.",
            facts=[
                self._fact("writes_enabled", True, "settings"),
                self._fact("added_count", len(added), "user_watchlists"),
                self._fact("skipped_count", len(skipped), "user_watchlists"),
            ],
            citations=[self._citation("watchlist", watchlist_name, f"user_watchlists:{watchlist_id}")],
            next_actions=next_actions,
            freshness=self._freshness(),
            watchlist={"id": watchlist_id, "name": watchlist_name},
            added=added,
            skipped=skipped,
            writes_enabled=True,
        )

    def _find_candidates_for_run(self, run_id: int, args: FindCandidatesArgs) -> list[dict[str, Any]]:
        with self._uow_scope() as uow:
            page = uow.feature_store.query_run_as_scan_results(
                run_id,
                QuerySpec(
                    filters=self._build_filter_spec(args.filters),
                    sort=self._build_sort_spec(args.filters),
                    page=PageSpec(page=1, per_page=args.limit),
                ),
                include_sparklines=False,
                include_setup_payload=False,
            )
            return [self._candidate_record(item) for item in page.items]

    def _find_candidates_for_watchlist(self, run_id: int, watchlist_id: int, args: FindCandidatesArgs) -> list[dict[str, Any]]:
        sort_spec = self._build_sort_spec(args.filters)
        with self._session_scope() as db:
            symbols = [
                row.symbol
                for row in db.query(WatchlistItem)
                .filter(WatchlistItem.watchlist_id == watchlist_id)
                .order_by(WatchlistItem.position.asc())
                .all()
            ]

        with self._uow_scope() as uow:
            items = []
            for symbol in symbols:
                item = uow.feature_store.get_by_symbol_for_run(
                    run_id,
                    symbol,
                    include_sparklines=False,
                    include_setup_payload=False,
                )
                if item is not None and self._matches_candidate_filters(item, args.filters):
                    items.append(item)

        items.sort(
            key=lambda row: self._candidate_sort_value(self._candidate_record(row), sort_spec.field),
            reverse=sort_spec.order == SortOrder.DESC,
        )
        return [self._candidate_record(item) for item in items[: args.limit]]

    def _build_filter_spec(self, filters: CandidateFilters) -> FilterSpec:
        spec = FilterSpec()
        spec.add_range("composite_score", min_value=filters.min_score)
        spec.add_range("rs_rating", min_value=filters.min_rs_rating)
        spec.add_range("eps_growth_qq", min_value=filters.min_eps_growth_qq)
        spec.add_range("sales_growth_qq", min_value=filters.min_sales_growth_qq)
        spec.add_range("market_cap", min_value=filters.min_market_cap, max_value=filters.max_market_cap)
        spec.add_range("price", min_value=filters.min_price, max_value=filters.max_price)
        spec.add_range("volume", min_value=filters.min_volume)
        if filters.stage is not None:
            spec.add_range("stage", min_value=filters.stage, max_value=filters.stage)
        ratings = self._rating_values(filters.rating)
        if ratings:
            spec.add_categorical("rating", ratings)
        if filters.sector:
            spec.add_text_search("gics_sector", filters.sector)
        if filters.industry_group:
            spec.add_text_search("ibd_industry_group", filters.industry_group)
        if filters.text_query:
            spec.add_text_search("symbol", filters.text_query)
        return spec

    def _build_sort_spec(self, filters: CandidateFilters) -> SortSpec:
        if filters.sort_field not in _SUPPORTED_SORT_FIELDS:
            raise ValueError(
                f"Unsupported sort_field {filters.sort_field!r}; use one of {sorted(_SUPPORTED_SORT_FIELDS)}"
            )
        return SortSpec(field=filters.sort_field, order=SortOrder(filters.sort_order))

    def _matches_candidate_filters(self, item: Any, filters: CandidateFilters) -> bool:
        record = self._candidate_record(item)
        return all(
            (
                filters.min_score is None or (record.get("composite_score") or 0) >= filters.min_score,
                filters.min_rs_rating is None or (record.get("rs_rating") or 0) >= filters.min_rs_rating,
                filters.min_eps_growth_qq is None or (record.get("eps_growth_qq") or 0) >= filters.min_eps_growth_qq,
                filters.min_sales_growth_qq is None or (record.get("sales_growth_qq") or 0) >= filters.min_sales_growth_qq,
                filters.min_market_cap is None or (record.get("market_cap") or 0) >= filters.min_market_cap,
                filters.max_market_cap is None or (record.get("market_cap") or 0) <= filters.max_market_cap,
                filters.min_price is None or (record.get("current_price") or 0) >= filters.min_price,
                filters.max_price is None or (record.get("current_price") or 0) <= filters.max_price,
                filters.min_volume is None or (record.get("volume") or 0) >= filters.min_volume,
                filters.stage is None or record.get("stage") == filters.stage,
                not self._rating_values(filters.rating) or record.get("rating") in self._rating_values(filters.rating),
                not filters.sector or filters.sector.lower() in (record.get("gics_sector") or "").lower(),
                not filters.industry_group or filters.industry_group.lower() in (record.get("ibd_industry_group") or "").lower(),
                not filters.text_query
                or filters.text_query.lower() in record["symbol"].lower()
                or filters.text_query.lower() in (record.get("company_name") or "").lower(),
            )
        )

    def _candidate_sort_value(self, record: dict[str, Any], sort_field: str) -> Any:
        field_map = {
            "price": "current_price",
            "volume": "volume",
        }
        key = field_map.get(sort_field, sort_field)
        value = record.get(key)
        return value if value is not None else float("-inf")

    def _build_stock_explanation(self, use_case: ExplainStockUseCase, item: Any) -> Any:
        return use_case.build_explanation_from_item(item)

    def _compare_runs_payload(self, run_a_id: int, run_b_id: int, limit: int) -> dict[str, Any]:
        result = CompareFeatureRunsUseCase().execute(
            SqlUnitOfWork(self._session_factory),
            CompareRunsQuery(run_a=run_a_id, run_b=run_b_id, limit=limit),
        )
        return {
            "run_a": {"id": result.run_a_id, "as_of_date": result.run_a_date.isoformat()},
            "run_b": {"id": result.run_b_id, "as_of_date": result.run_b_date.isoformat()},
            "summary": {
                "total_common": result.summary.total_common,
                "upgraded_count": result.summary.upgraded_count,
                "downgraded_count": result.summary.downgraded_count,
                "avg_score_change": result.summary.avg_score_change,
            },
            "added": [self._symbol_entry_record(entry) for entry in result.added],
            "removed": [self._symbol_entry_record(entry) for entry in result.removed],
            "movers": [self._symbol_delta_record(entry) for entry in result.movers],
        }

    def _resolve_compare_runs(self, db: Session, args: CompareFeatureRunsArgs) -> tuple[FeatureRun | None, FeatureRun | None]:
        if args.run_a is not None and args.run_b is not None:
            return db.get(FeatureRun, args.run_a), db.get(FeatureRun, args.run_b)

        published = (
            db.query(FeatureRun)
            .filter(FeatureRun.status == "published")
            .order_by(FeatureRun.as_of_date.desc(), FeatureRun.id.desc())
            .limit(3)
            .all()
        )
        if not published:
            return None, None

        latest = published[0]
        if args.run_a is not None:
            run_a = db.get(FeatureRun, args.run_a)
            if run_a is None:
                return None, None
            run_b = latest if latest.id != run_a.id else (published[1] if len(published) > 1 else None)
            return run_a, run_b
        if args.run_b is not None:
            run_b = db.get(FeatureRun, args.run_b)
            if run_b is None:
                return None, None
            run_a = latest if latest.id != run_b.id else (published[1] if len(published) > 1 else None)
            return run_a, run_b
        if len(published) < 2:
            return None, None
        return published[1], published[0]

    def _select_run_for_overview(self, db: Session, as_of_date: date | None) -> FeatureRun | None:
        query = db.query(FeatureRun).filter(FeatureRun.status == "published")
        if as_of_date is not None:
            query = query.filter(FeatureRun.as_of_date == as_of_date)
        return query.order_by(FeatureRun.as_of_date.desc(), FeatureRun.id.desc()).first()

    def _previous_published_run(self, db: Session, selected_run: FeatureRun | None) -> FeatureRun | None:
        if selected_run is None:
            return None
        return (
            db.query(FeatureRun)
            .filter(
                FeatureRun.status == "published",
                FeatureRun.id != selected_run.id,
                FeatureRun.as_of_date <= selected_run.as_of_date,
            )
            .order_by(FeatureRun.as_of_date.desc(), FeatureRun.id.desc())
            .first()
        )

    def _latest_published_run(self, db: Session) -> FeatureRun | None:
        return (
            db.query(FeatureRun)
            .filter(FeatureRun.status == "published")
            .order_by(FeatureRun.as_of_date.desc(), FeatureRun.id.desc())
            .first()
        )

    def _latest_breadth(self, db: Session, as_of_date: date | None) -> MarketBreadth | None:
        query = db.query(MarketBreadth)
        if as_of_date is not None:
            query = query.filter(MarketBreadth.date <= as_of_date)
        return query.order_by(MarketBreadth.date.desc()).first()

    def _recent_alerts(self, db: Session, limit: int) -> list[dict[str, Any]]:
        rows = (
            db.query(ThemeAlert)
            .filter(ThemeAlert.is_dismissed == False, ThemeAlert.is_read == False)
            .order_by(ThemeAlert.triggered_at.desc())
            .limit(limit)
            .all()
        )
        return [self._alert_record(row) for row in rows]

    def _resolve_watchlist(self, db: Session, watchlist: str) -> UserWatchlist | None:
        needle = watchlist.strip()
        query = db.query(UserWatchlist)
        if needle.isdigit():
            row = query.filter(UserWatchlist.id == int(needle)).first()
            if row is not None:
                return row
        exact = query.filter(func.lower(UserWatchlist.name) == needle.lower()).first()
        if exact is not None:
            return exact
        escaped = self._escape_like_pattern(needle)
        return (
            query.filter(UserWatchlist.name.ilike(f"%{escaped}%", escape="\\"))
            .order_by(UserWatchlist.position.asc())
            .first()
        )

    def _resolve_theme(self, db: Session, theme_name: str) -> ThemeCluster | None:
        exact = (
            db.query(ThemeCluster)
            .filter(ThemeCluster.is_active == True, func.lower(ThemeCluster.display_name) == theme_name.lower())
            .first()
        )
        if exact is not None:
            return exact
        escaped = self._escape_like_pattern(theme_name)
        return (
            db.query(ThemeCluster)
            .filter(ThemeCluster.is_active == True, ThemeCluster.display_name.ilike(f"%{escaped}%", escape="\\"))
            .order_by(ThemeCluster.last_seen_at.desc().nullslast(), ThemeCluster.id.desc())
            .first()
        )

    def _task_rows(self, db: Session) -> list[dict[str, Any]]:
        rows = TaskRegistryService.get_instance().get_all_scheduled_tasks(db)
        rows.sort(key=lambda row: row["name"])
        return rows

    def _last_run(self, task: dict[str, Any]) -> dict[str, Any]:
        return task.get("last_run") or {}

    def _candidate_record(self, item: Any) -> dict[str, Any]:
        extended = getattr(item, "extended_fields", {}) or {}
        return {
            "symbol": item.symbol,
            "company_name": extended.get("company_name"),
            "composite_score": round(item.composite_score, 2) if item.composite_score is not None else None,
            "rating": item.rating,
            "current_price": item.current_price,
            "rs_rating": extended.get("rs_rating"),
            "stage": extended.get("stage"),
            "gics_sector": extended.get("gics_sector"),
            "ibd_industry_group": extended.get("ibd_industry_group"),
            "market_cap": extended.get("market_cap"),
            "volume": extended.get("volume"),
            "eps_growth_qq": extended.get("eps_growth_qq"),
            "sales_growth_qq": extended.get("sales_growth_qq"),
            "se_setup_score": extended.get("se_setup_score"),
            "se_setup_ready": extended.get("se_setup_ready"),
        }

    def _theme_record(self, theme: ThemeCluster, metrics: ThemeMetrics | None) -> dict[str, Any]:
        return {
            "id": theme.id,
            "display_name": theme.display_name,
            "canonical_key": theme.canonical_key,
            "category": theme.category,
            "lifecycle_state": theme.lifecycle_state,
            "is_emerging": theme.is_emerging,
            "is_validated": theme.is_validated,
            "last_seen_at": theme.last_seen_at.isoformat() if theme.last_seen_at else None,
            "metrics": None if metrics is None else {
                "date": metrics.date.isoformat(),
                "rank": metrics.rank,
                "momentum_score": metrics.momentum_score,
                "mentions_7d": metrics.mentions_7d,
                "basket_rs_vs_spy": metrics.basket_rs_vs_spy,
                "avg_rs_rating": metrics.avg_rs_rating,
                "status": metrics.status,
            },
        }

    def _theme_constituent_record(self, row: ThemeConstituent) -> dict[str, Any]:
        return {
            "symbol": row.symbol,
            "confidence": row.confidence,
            "mention_count": row.mention_count,
            "source": row.source,
            "is_active": row.is_active,
        }

    def _alert_record(self, row: ThemeAlert) -> dict[str, Any]:
        return {
            "id": row.id,
            "theme_cluster_id": row.theme_cluster_id,
            "alert_type": row.alert_type,
            "title": row.title,
            "description": row.description,
            "severity": row.severity,
            "related_tickers": row.related_tickers or [],
            "metrics": row.metrics or {},
            "triggered_at": row.triggered_at.isoformat() if row.triggered_at else None,
            "is_read": row.is_read,
        }

    def _breadth_record(self, row: MarketBreadth | None) -> dict[str, Any] | None:
        if row is None:
            return None
        return {
            "date": row.date.isoformat(),
            "stocks_up_4pct": row.stocks_up_4pct,
            "stocks_down_4pct": row.stocks_down_4pct,
            "ratio_5day": row.ratio_5day,
            "ratio_10day": row.ratio_10day,
            "total_stocks_scanned": row.total_stocks_scanned,
        }

    def _run_record(self, run: FeatureRun | None) -> dict[str, Any] | None:
        if run is None:
            return None
        return {
            "id": run.id,
            "as_of_date": run.as_of_date.isoformat(),
            "status": run.status,
            "published_at": run.published_at.isoformat() if run.published_at else None,
            "completed_at": run.completed_at.isoformat() if run.completed_at else None,
        }

    def _symbol_entry_record(self, entry: Any) -> dict[str, Any]:
        return {
            "symbol": entry.symbol,
            "score": entry.score,
            "rating": entry.rating,
        }

    def _symbol_delta_record(self, delta: Any) -> dict[str, Any]:
        return {
            "symbol": delta.symbol,
            "score_a": delta.score_a,
            "score_b": delta.score_b,
            "score_delta": delta.score_delta,
            "rating_a": delta.rating_a,
            "rating_b": delta.rating_b,
        }

    def _session_scope(self):
        return self._session_factory()

    def _uow_scope(self) -> SqlUnitOfWork:
        return SqlUnitOfWork(self._session_factory)

    def _tool_error(
        self,
        message: str,
        *,
        code: str,
        details: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = {
            "error": {
                "code": code,
                "message": message,
                "details": details or {},
            }
        }
        return {
            "content": [
                {
                    "type": "text",
                    "text": json.dumps(payload, indent=2, sort_keys=True),
                }
            ],
            "structuredContent": payload,
            "isError": True,
        }

    def _envelope(
        self,
        summary: str,
        *,
        facts: list[ToolFact],
        citations: list[ToolCitation],
        next_actions: list[str],
        freshness: ToolFreshness,
        **extra: Any,
    ) -> ToolEnvelope:
        return ToolEnvelope(
            summary=summary,
            facts=facts,
            citations=citations,
            next_actions=next_actions,
            freshness=freshness,
            **extra,
        )

    def _fact(self, key: str, value: Any, source: str, as_of: date | None = None) -> ToolFact:
        return ToolFact(
            key=key,
            value=value,
            source=source,
            as_of=as_of.isoformat() if isinstance(as_of, date) else None,
        )

    def _citation(
        self,
        source: str,
        label: str,
        reference: str,
        as_of: date | datetime | str | None = None,
    ) -> ToolCitation:
        return ToolCitation(
            source=source,
            label=label,
            reference=reference,
            as_of=self._as_of_string(as_of),
        )

    def _freshness(self, as_of_date: date | None = None, **sources: str | None) -> ToolFreshness:
        return ToolFreshness(
            generated_at=datetime.now(UTC).isoformat(),
            as_of_date=as_of_date.isoformat() if isinstance(as_of_date, date) else None,
            sources=sources,
        )

    def _as_of_string(self, value: date | datetime | str | None) -> str | None:
        if isinstance(value, datetime):
            return value.date().isoformat()
        if isinstance(value, date):
            return value.isoformat()
        if isinstance(value, str):
            candidate = value[:10]
            try:
                return date.fromisoformat(candidate).isoformat()
            except ValueError:
                return None
        return None

    def _rating_values(self, rating: str | list[str] | None) -> tuple[str, ...]:
        if rating is None:
            return ()
        if isinstance(rating, str):
            return (rating,)
        return tuple(rating)

    def _escape_like_pattern(self, value: str) -> str:
        return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")

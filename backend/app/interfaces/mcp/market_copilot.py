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
from app.domain.common.query import FilterSpec, PageSpec, SortOrder, SortSpec
from app.domain.feature_store.models import FeatureRunDomain
from app.domain.markets.symbol_suffixes import market_symbol_suffix_registry
from app.domain.scanning.filter_expression_model import QuerySpec
from app.infra.db.models.feature_store import FeatureRun
from app.infra.db.uow import SqlUnitOfWork
from app.models.market_breadth import MarketBreadth
from app.models.stock import StockFundamental
from app.models.stock_universe import StockUniverse
from app.models.theme import ThemeAlert, ThemeCluster, ThemeConstituent, ThemeMetrics
from app.models.user_watchlist import UserWatchlist, WatchlistItem
from app.schemas.scanning import ExplainResponse, ScanResultItem
from app.services import yahoo_earnings_calendar
from app.services.breadth.query import breadth_query, latest_breadth
from app.services.economic_theme_read_service import EconomicThemeReader
from app.services.market_calendar_service import MarketCalendarService
from app.services.task_registry_service import SCHEDULED_TASKS
from app.services.watchlist_capacity import check_watchlist_capacity, lock_watchlist
from app.tasks.market_queues import SUPPORTED_MARKETS
from app.use_cases.feature_store.compare_runs import (
    CompareFeatureRunsUseCase,
    CompareRunsQuery,
)
from app.use_cases.scanning.explain_stock import ExplainStockUseCase
from app.utils.market_hours import to_eastern_date
from app.wiring.bootstrap import (
    get_group_rank_service,
    get_market_calendar_service,
    get_task_registry_service,
)

from .models import (
    BreadthSnapshotArgs,
    CandidateFilters,
    CompareFeatureRunsArgs,
    DailyDigestArgs,
    EarningsLookupArgs,
    ExplainSymbolArgs,
    FindCandidatesArgs,
    GroupRankingsArgs,
    MarketCalendarStatusArgs,
    MarketOverviewArgs,
    StockLookupArgs,
    StockSnapshotArgs,
    TaskStatusArgs,
    ThemeStateArgs,
    ToolCitation,
    ToolEnvelope,
    ToolFact,
    ToolFreshness,
    WatchlistAddArgs,
    WatchlistSnapshotArgs,
)

# Fields a candidate record projects from ``ScanResultItem``.
# The record is a projection of the shared schema, not a second field list: a new
# fact is added in ``ScanResultItem`` and appears here by being named once.
CANDIDATE_FIELDS = frozenset(
    {
        "symbol",
        "company_name",
        "composite_score",
        "rating",
        "current_price",
        "rs_rating",
        "stage",
        "gics_sector",
        "ibd_industry_group",
        "market_cap",
        "volume",
        "eps_growth_qq",
        "sales_growth_qq",
        "minervini_rating",
        "canslim_rating",
        "volume_breakthrough_rating",
        "data_status",
        "is_scannable",
        "action_state",
        "opportunity_state",
        "vcp_detected",
        "vcp_num_bases",
        "volume_surge",
        "se_setup_score",
        "se_setup_ready",
        "se_readiness_score",
        "se_quality_score",
        "se_pattern_primary",
        "se_pivot_price",
        "se_pivot_type",
        "se_pivot_date",
        "se_distance_to_pivot_pct",
        "se_in_early_zone",
        "se_extended_from_pivot",
        "se_atr14_pct",
        "se_volume_vs_50d",
    }
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
_MARKET_OVERVIEW_TASK_LIMIT = max(5, len(SUPPORTED_MARKETS))

# Names reported next to the stored observation. They identify the canonical
# modules to a consumer; neither module exports a source constant, so the values
# are literals here and have to be kept in step with the module names.
_EARNINGS_SOURCE = "xang_yahoo_calendar"
_CALENDAR_SOURCE = "xang_market_calendar_service"


class ToolSourceUnavailable(RuntimeError):
    """A canonical source could not answer; the tool must fail closed.

    Raised instead of returning a half-filled payload: a caller must not read
    an ``expected_trading_date`` or an ``available`` flag off a source that did
    not respond. ``call_tool`` turns this into an ``isError`` response with the
    given code and a message that carries no internal detail.
    """

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


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
        calendar_service: Any = None,
        now_provider: Callable[[], datetime] | None = None,
    ) -> None:
        self._session_factory = session_factory
        self._settings = app_settings
        # The canonical calendar is resolved lazily: importing the service must
        # not require a calendar data tree, and tests inject their own.
        self._calendar_service = calendar_service
        self._now_provider = now_provider or (lambda: datetime.now(UTC))
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
                ToolSpec(
                    name="group_rankings",
                    description="Get current IBD industry group rankings and biggest rank movers over a configurable period (1w, 1m, 3m, 6m).",
                    args_model=GroupRankingsArgs,
                    handler=self._group_rankings,
                ),
                ToolSpec(
                    name="stock_lookup",
                    description="Look up a stock by symbol and return universe info, fundamentals, and optionally RS rating and stage from the latest feature run.",
                    args_model=StockLookupArgs,
                    handler=self._stock_lookup,
                ),
                ToolSpec(
                    name="stock_snapshot",
                    description="Return a combined snapshot for one symbol including universe metadata, latest feature-run posture, theme membership, breadth context, and watchlist presence.",
                    args_model=StockSnapshotArgs,
                    handler=self._stock_snapshot,
                ),
                ToolSpec(
                    name="breadth_snapshot",
                    description="Return recent market breadth snapshots (up/down 4%+ counts and ratios) for the last N trading days.",
                    args_model=BreadthSnapshotArgs,
                    handler=self._breadth_snapshot,
                ),
                ToolSpec(
                    name="market_calendar_status",
                    description=(
                        "Return the canonical market-calendar facts for one market: market_now, market_date, "
                        "is_trading_day, last_completed_trading_day and expected_trading_date. Read-only; no "
                        "trading decision."
                    ),
                    args_model=MarketCalendarStatusArgs,
                    handler=self._market_calendar_status,
                ),
                ToolSpec(
                    name="earnings_lookup",
                    description=(
                        "Read the stored earnings-calendar observation for one symbol: next_earnings_date, "
                        "event_calendar_as_of_date, freshness_status, date_confirmation and machine-readable "
                        "reasons. Missing or stale facts are named, never filled in."
                    ),
                    args_model=EarningsLookupArgs,
                    handler=self._earnings_lookup,
                ),
                ToolSpec(
                    name="daily_digest",
                    description="Return the full daily digest including market stance, leaders, themes, watchlist highlights, and risk notes.",
                    args_model=DailyDigestArgs,
                    handler=self._daily_digest,
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
                # loc/msg/type only: ctx carries the original exception object,
                # which the JSON envelope cannot serialise.
                details={
                    "errors": [
                        {
                            "loc": list(error.get("loc", ())),
                            "msg": error.get("msg"),
                            "type": error.get("type"),
                        }
                        for error in exc.errors()
                    ]
                },
            )

        try:
            payload = spec.handler(parsed_args).model_dump(mode="json")
        except ValidationError as exc:
            # ValidationError subclasses ValueError, so without this clause a
            # stored row the schema rejects reads as invalid_arguments - the
            # caller's request was fine, the persisted data is not.
            logger.warning("Stored data rejected by the schema in %s: %s", name, exc)
            return self._tool_error(
                f"{name} could not read the stored row",
                code="invalid_stored_data",
                details={
                    "errors": [
                        {"loc": list(error.get("loc", ())), "msg": error.get("msg"), "type": error.get("type")}
                        for error in exc.errors()
                    ]
                },
            )
        except ValueError as exc:
            return self._tool_error(
                f"Invalid arguments for {name}",
                code="invalid_arguments",
                details={"message": str(exc)},
            )
        except ToolSourceUnavailable as exc:
            # The message is written for the caller; the cause belongs in the
            # log, otherwise an expired calendar and a missing data root look
            # identical from the outside.
            logger.warning(
                "%s unavailable (%s)", name, exc.code, exc_info=exc.__cause__ or exc
            )
            return self._tool_error(exc.message, code=exc.code)
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

    # ------------------------------------------------------------------
    # Canonical calendar and stored earnings facts (read-only)
    # ------------------------------------------------------------------
    @staticmethod
    def _calendar_unavailable(
        message: str = "The canonical market calendar could not be loaded.",
    ) -> ToolSourceUnavailable:
        """Build the one error every calendar failure reports."""
        return ToolSourceUnavailable("calendar_unavailable", message)

    @staticmethod
    def _runtime_services_missing(exc: RuntimeError) -> bool:
        # resolve_runtime_services raises a plain RuntimeError with this text.
        # Checked by exact type and message on purpose: CalendarCoverageExpired
        # and provider drift also subclass RuntimeError, and falling back for
        # those would hide a broken calendar behind a private instance.
        return type(exc) is RuntimeError and "RuntimeServices are not initialized" in str(exc)

    def _calendar(self) -> Any:
        """Return the canonical calendar service, resolved once per process."""
        if self._calendar_service is None:
            try:
                self._calendar_service = get_market_calendar_service()
            except RuntimeError as exc:
                if not self._runtime_services_missing(exc):
                    raise self._calendar_unavailable() from exc
                # The stdio MCP process never runs the FastAPI startup, so no
                # process-scoped instance exists there. Same canonical class,
                # built on first use - not a second implementation.
                try:
                    self._calendar_service = MarketCalendarService()
                except Exception as build_exc:  # noqa: BLE001 - fail closed
                    raise self._calendar_unavailable() from build_exc
            except Exception as exc:  # noqa: BLE001 - fail closed, never emulate
                raise self._calendar_unavailable() from exc
        return self._calendar_service

    def _earnings_reference_date(self, symbol: str, now: datetime) -> tuple[date, str]:
        """Calendar date the earnings freshness is measured against.

        Returns the date and the basis it came from: ``market`` when the
        symbol's own market answered, ``eastern`` for a plain US ticker, and
        ``eastern_fallback`` when a suffixed symbol's calendar was unavailable
        and the Eastern date had to stand in. The basis travels to the caller so
        a successful answer cannot be mistaken for market-local evidence.
        """
        market = market_symbol_suffix_registry.market_for_symbol(symbol)
        if market is None:
            return to_eastern_date(now), "eastern"
        try:
            market_date = self._calendar().market_now(market, now=now).date()
        except Exception as exc:  # noqa: BLE001 - the read must stay available
            logger.warning(
                "market date for %s (%s) unavailable, using the Eastern date: %s",
                symbol,
                market,
                type(exc).__name__,
            )
            return to_eastern_date(now), "eastern_fallback"
        return market_date, "market"

    def _now(self) -> datetime:
        current = self._now_provider()
        return current if current.tzinfo is not None else current.replace(tzinfo=UTC)

    def _market_calendar_status(self, args: MarketCalendarStatusArgs) -> ToolEnvelope:
        """Return the canonical calendar facts the run-freshness check needs.

        ``expected_trading_date`` keeps the meaning the temporary gateway gave
        it: the market-calendar day while the market trades (mid-session
        included), otherwise the last completed session. It is therefore *not*
        ``last_completed_trading_day`` during a running session, which is the
        value that only exists after the close.
        """

        calendar = self._calendar()
        try:
            market = calendar.normalize_market(args.market)
        except ValueError as exc:
            raise ValueError(f"Unsupported market: {args.market}") from exc
        except Exception as exc:  # noqa: BLE001 - provider/catalog drift
            raise self._calendar_unavailable(
                "The canonical market calendar could not answer for this market."
            ) from exc

        now = self._now()
        try:
            market_now = calendar.market_now(market, now=now)
            market_date = market_now.date()
            is_trading_day = bool(calendar.is_trading_day(market, market_date))
            last_completed = calendar.last_completed_trading_day(market, now=now)
            market_timezone = str(calendar.market_timezone(market))
            is_open = bool(calendar.is_market_open(market, now=now))
        except Exception as exc:  # noqa: BLE001 - missing coverage, provider drift
            logger.warning(
                "calendar lookup failed for %s: %s",
                market,
                type(exc).__name__,
                exc_info=True,
            )
            raise self._calendar_unavailable(
                "The canonical market calendar could not answer for this market."
            ) from exc

        # Today's session only counts once it is running. Before the open the
        # freshest bars any cache can hold belong to the last completed session;
        # pointing at a session that has not traded would make an up-to-date
        # cache look stale for the rest of the day.
        session_open = is_trading_day and is_open
        session_complete = is_trading_day and last_completed == market_date
        expected = market_date if (session_open or session_complete) else last_completed

        if session_open:
            summary = (
                f"{market} trades on {market_date.isoformat()} and the session is open; the "
                f"expected trading date is {expected.isoformat()}."
            )
        elif session_complete:
            summary = (
                f"{market} traded on {market_date.isoformat()} and the session is complete; the "
                f"expected trading date is that session, {expected.isoformat()}."
            )
        elif is_trading_day:
            summary = (
                f"{market} trades on {market_date.isoformat()}, but the session has not opened "
                f"yet; the expected trading date is the last completed session, "
                f"{expected.isoformat()}."
            )
        else:
            summary = (
                f"{market} has no session on {market_date.isoformat()}; the expected trading date "
                f"is the last completed session, {expected.isoformat()}."
            )

        payload = {
            "market": market,
            "market_timezone": market_timezone,
            "market_now": market_now.isoformat(),
            "market_date": market_date.isoformat(),
            "is_trading_day": is_trading_day,
            "last_completed_trading_day": last_completed.isoformat(),
            "expected_trading_date": expected.isoformat(),
            "calendar_source": _CALENDAR_SOURCE,
        }

        facts = self._facts(
            {key: value for key, value in payload.items() if key != "calendar_source"},
            _CALENDAR_SOURCE,
            as_of=market_date,
            key_dates={
                "last_completed_trading_day": last_completed,
                "expected_trading_date": expected,
            },
        )

        return self._envelope(
            summary,
            facts=facts,
            citations=[
                self._citation(
                    _CALENDAR_SOURCE,
                    "canonical market calendar",
                    f"market={market}, day={market_date.isoformat()}",
                    market_date,
                )
            ],
            next_actions=[],
            freshness=self._freshness(
                market_date,
                generated_at=now,
                market_calendar=_CALENDAR_SOURCE,
            ),
            **payload,
        )

    def _earnings_lookup(self, args: EarningsLookupArgs) -> ToolEnvelope:
        """Read one stored earnings observation; never fetch, never guess.

        The freshness verdict comes from ``app.services.yahoo_earnings_calendar``
        - the same module the scanners use. This handler only names *which*
        side of the window was missed; it re-derives no threshold.

        Contract notes (kept from the temporary gateway, deviations named):

        * ``available`` answers "is there a usable observation", not "is a date
          known": a fresh observation that knows of no upcoming date is
          ``available=True`` with
          ``reasons=["no_upcoming_earnings_date", "date_confirmation_unrecorded"]``.
        * ``date_confirmation`` stays ``unknown``: ``stock_fundamentals`` stores
          no confirmation proof, so neither ``confirmed`` nor ``estimated`` can
          be claimed.
        * Deviations from the gateway: a malformed symbol and an unreadable
          database are ``isError`` responses here (``invalid_arguments`` /
          ``database_unavailable``) instead of a success payload, because a
          consumer must not read a filled-in envelope off a failed lookup.
        """

        rules = yahoo_earnings_calendar
        now = self._now()
        # Freshness is judged in the symbol's own market date: 0700.HK at 08:00
        # HKT must not be measured against the previous Eastern date. A plain
        # US ticker has no suffix and keeps the Eastern date the scanners use.
        reference, reference_basis = self._earnings_reference_date(args.symbol, now)

        payload: dict[str, Any] = {
            "symbol": args.symbol,
            "next_earnings_date": None,
            "event_calendar_as_of_date": None,
            "source": _EARNINGS_SOURCE,
            "freshness_status": "unknown",
            "date_confirmation": "unknown",
            "available": False,
            "reasons": [],
            "generated_at": now.isoformat(),
            "freshness_reference_date": reference.isoformat(),
            "freshness_reference_basis": reference_basis,
            "event_calendar_max_age_days": rules.EVENT_CALENDAR_MAX_AGE_DAYS,
            "event_calendar_producer_refresh_after_days": (
                rules.EVENT_CALENDAR_PRODUCER_REFRESH_AFTER_DAYS
            ),
            "event_calendar_future_tolerance_days": rules.EVENT_CALENDAR_FUTURE_TOLERANCE_DAYS,
        }

        try:
            with self._session_scope() as db:
                row = (
                    db.query(StockFundamental)
                    .filter(StockFundamental.symbol == args.symbol)
                    .first()
                )
        except Exception as exc:  # noqa: BLE001 - fail closed on every driver error
            logger.warning("earnings lookup failed for %s: %s", args.symbol, type(exc).__name__)
            raise ToolSourceUnavailable(
                "database_unavailable",
                "The stored earnings observation could not be read.",
            ) from exc

        if row is None:
            payload["reasons"].append("symbol_not_found")
            summary = f"No stored earnings observation exists for {args.symbol}."
        else:
            observed = row.event_calendar_as_of_date
            if observed is None:
                payload["reasons"].append("observation_date_missing")
                summary = (
                    f"{args.symbol} has no event-calendar observation date; the freshness is unknown."
                )
            else:
                payload["event_calendar_as_of_date"] = observed.isoformat()
                age_days = (reference - observed).days
                payload["observation_age_days"] = age_days

                if not rules.is_event_calendar_observation_fresh(observed, reference_date=reference):
                    # The canonical rule decides; the branch only names the side.
                    if age_days > rules.EVENT_CALENDAR_MAX_AGE_DAYS:
                        payload["freshness_status"] = "stale"
                        payload["reasons"].append("observation_stale")
                        summary = (
                            f"The earnings observation for {args.symbol} is {age_days} days old "
                            f"and no longer counts as evidence."
                        )
                    else:
                        payload["freshness_status"] = "invalid"
                        payload["reasons"].append("observation_date_in_future")
                        summary = (
                            f"The earnings observation for {args.symbol} is dated in the future, "
                            "beyond the accepted tolerance."
                        )
                else:
                    payload["freshness_status"] = "ok"
                    stored_next, is_evidence = rules.persisted_next_earnings_date(
                        {
                            "event_calendar_as_of_date": observed,
                            "next_earnings_date": row.next_earnings_date,
                        },
                        as_of_date=reference,
                    )
                    if not is_evidence:
                        # The column is a SQL date and the canonical helper
                        # treats a missing date as usable evidence, so the only
                        # remaining cause is a date before the reference date.
                        # With a weekly producer cadence that is a normal state,
                        # not damaged storage: report it as stale.
                        payload["freshness_status"] = "stale"
                        payload["reasons"].append("earnings_date_in_past")
                        summary = (
                            f"The earnings observation for {args.symbol} is current, but its stored "
                            f"date {row.next_earnings_date.isoformat()} has already passed."
                        )
                    elif stored_next is None:
                        payload["available"] = True
                        payload["reasons"].extend(
                            ["no_upcoming_earnings_date", "date_confirmation_unrecorded"]
                        )
                        summary = (
                            f"The earnings observation for {args.symbol} is current and reports no "
                            "upcoming date."
                        )
                    else:
                        payload["next_earnings_date"] = stored_next.isoformat()
                        payload["available"] = True
                        payload["reasons"].append("date_confirmation_unrecorded")
                        summary = (
                            f"The earnings observation for {args.symbol} is current; the next stored "
                            f"date is {stored_next.isoformat()}, unconfirmed."
                        )

        fact_keys = (
            "symbol",
            "freshness_status",
            "date_confirmation",
            "available",
            "event_calendar_as_of_date",
            "next_earnings_date",
            "observation_age_days",
        )
        facts = self._facts(
            {key: payload[key] for key in fact_keys if payload.get(key) is not None},
            _EARNINGS_SOURCE,
        )

        next_actions: list[str] = []
        if payload["freshness_status"] == "stale":
            next_actions.append(
                "Refresh the fundamentals snapshot before treating this date as evidence."
            )
        elif payload["freshness_status"] == "invalid":
            next_actions.append("Inspect the stored event-calendar observation for this symbol.")
        elif not payload["available"]:
            next_actions.append("No usable observation is stored for this symbol.")

        return self._envelope(
            summary,
            facts=facts,
            citations=[
                self._citation(
                    "stock_fundamentals",
                    "stored event-calendar observation",
                    f"symbol={args.symbol}",
                    payload["event_calendar_as_of_date"],
                )
            ],
            next_actions=next_actions,
            freshness=self._freshness(
                None,
                generated_at=now,
                earnings_calendar=_EARNINGS_SOURCE,
                fundamentals="stock_fundamentals",
            ),
            **payload,
        )

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

        unavailable = {}
        if alerts is None:  # economic authority: not a measured zero
            summary_parts.append("Theme alerts are unavailable under economic Theme authority.")
            unavailable["theme_alerts"] = "not_defined_for_economic_taxonomy"
            alerts = []
        else:
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
            tasks=failed_tasks or tasks[:_MARKET_OVERVIEW_TASK_LIMIT],
            top_candidates=top_candidates,
            **({"unavailable": unavailable} if unavailable else {}),
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
        """Build the explanation for one symbol from the run that applies to it.

        The run is resolved market-first (see ``_latest_published_run_for_symbol``):
        publication writes per-market pointers, so looking at the global pointer alone
        reported "no published feature run" while a valid market run was published.
        """
        explain_use_case = ExplainStockUseCase()
        with self._uow_scope() as uow:
            latest_run = self._latest_published_run_for_symbol(uow, args.symbol)
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

        composite_score = explanation.composite_score
        if composite_score is None:
            summary = (
                f"{args.symbol} is rated {explanation.rating} in published run {latest_run.id} "
                "with no composite score yet."
            )
        else:
            summary = (
                f"{args.symbol} is rated {explanation.rating} in published run {latest_run.id} "
                f"with composite score {round(composite_score, 2)}."
            )
        if scan_item.extended_fields.get("stage") is not None:
            summary += f" It is currently classified as stage {scan_item.extended_fields['stage']}."

        return self._envelope(
            summary,
            facts=[
                self._fact("symbol", args.symbol, "feature_runs", latest_run.as_of_date),
                self._fact(
                    "composite_score",
                    round(composite_score, 2) if composite_score is not None else None,
                    "feature_runs",
                    latest_run.as_of_date,
                ),
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
            economic = EconomicThemeReader(db)
            if economic.source_name == "economic":
                return self._economic_theme_state(economic, args)
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

    def _economic_theme_state(
        self, reader: EconomicThemeReader, args: ThemeStateArgs
    ) -> ToolEnvelope:
        catalog = reader.read_current_catalog()
        if args.theme_name:
            theme = reader.find_theme(args.theme_name)
            if theme is None:
                return self._envelope(
                    f"No published Economic Theme matched {args.theme_name!r}.",
                    facts=[],
                    citations=[],
                    next_actions=[
                        "Use a broader theme name or omit theme_name to inspect the published generation."
                    ],
                    freshness=self._freshness(),
                    themes=[],
                )
            theme_id = str(theme["economic_theme_id"])
            constituents = list(theme.get("constituents", []))[: args.limit]
            return self._envelope(
                f"Economic Theme {theme['display_name']} has {len(constituents)} published constituents.",
                facts=[
                    self._fact("economic_theme_id", theme_id, "economic_themes"),
                    self._fact(
                        "published_constituents",
                        len(constituents),
                        "reader_snapshot_bundles",
                    ),
                ],
                citations=[
                    self._citation(
                        "economic_theme",
                        theme["display_name"],
                        f"economic_themes:{theme_id}",
                    )
                ],
                next_actions=[],
                freshness=self._freshness(
                    theme_generation=catalog["generation_id"],
                    theme_published_at=catalog["generation"].get("published_at"),
                ),
                theme=theme,
                constituents=constituents,
                alerts=[],
                unavailable={"theme_alerts": "not_defined_for_economic_taxonomy"},
            )

        ranked_themes = reader.ranked_themes(limit=args.limit)
        return self._envelope(
            f"Published Economic Theme generation includes {len(ranked_themes)} ranked themes; legacy alerts are unavailable.",
            facts=[
                self._fact(
                    "ranked_themes", len(ranked_themes), "reader_snapshot_bundles"
                ),
                self._fact(
                    "generation_id", catalog["generation_id"], "serving_generations"
                ),
            ],
            citations=[],
            next_actions=(
                []
                if ranked_themes
                else ["Publish an Economic Theme generation with calculated metrics."]
            ),
            freshness=self._freshness(
                theme_generation=catalog["generation_id"],
                theme_published_at=catalog["generation"].get("published_at"),
            ),
            themes=ranked_themes,
            alerts=[],
            unavailable={"theme_alerts": "not_defined_for_economic_taxonomy"},
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

            watchlist = lock_watchlist(db, watchlist.id) or watchlist
            existing_items = (
                db.query(WatchlistItem)
                .filter(WatchlistItem.watchlist_id == watchlist.id)
                .order_by(WatchlistItem.position.asc())
                .all()
            )
            watchlist_id = watchlist.id
            watchlist_name = watchlist.name
            existing_symbols = {item.symbol for item in existing_items}
            max_position = max((item.position for item in existing_items), default=-1)
            # Same rule as the REST endpoints: only active-universe symbols.
            name_map = {
                row.symbol: row.name
                for row in db.query(StockUniverse)
                .filter(
                    StockUniverse.active_filter(),
                    StockUniverse.symbol.in_(args.symbols),
                )
                .all()
            }
            skipped = [symbol for symbol in args.symbols if symbol in existing_symbols]
            invalid = [
                symbol
                for symbol in args.symbols
                if symbol not in existing_symbols and symbol not in name_map
            ]
            addable = [
                symbol
                for symbol in args.symbols
                if symbol not in existing_symbols and symbol in name_map
            ]
            # A WatchlistCapacityError is a ValueError, which call_tool reports.
            check_watchlist_capacity(len(existing_items), len(addable))

            added: list[dict[str, Any]] = []
            for symbol in addable:
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
        if invalid:
            next_actions.append("Check the invalid symbols; they are not in the active stock universe.")

        return self._envelope(
            f"Added {len(added)} symbols to watchlist {watchlist_name}; {len(skipped)} symbols were skipped"
            f" and {len(invalid)} were invalid.",
            facts=[
                self._fact("writes_enabled", True, "settings"),
                self._fact("added_count", len(added), "user_watchlists"),
                self._fact("skipped_count", len(skipped), "user_watchlists"),
                self._fact("invalid_count", len(invalid), "stock_universe"),
            ],
            citations=[self._citation("watchlist", watchlist_name, f"user_watchlists:{watchlist_id}")],
            next_actions=next_actions,
            freshness=self._freshness(),
            watchlist={"id": watchlist_id, "name": watchlist_name},
            added=added,
            skipped=skipped,
            invalid=invalid,
            writes_enabled=True,
        )

    def _group_rankings(self, args: GroupRankingsArgs) -> ToolEnvelope:
        with self._session_scope() as db:
            service = get_group_rank_service()
            rankings = service.get_current_rankings(db, limit=args.limit)
            movers = service.get_rank_movers(db, period=args.period, limit=min(args.limit, 10))

        if not rankings:
            return self._envelope(
                "No IBD group ranking data is available.",
                facts=[],
                citations=[],
                next_actions=["Run a group ranking calculation before using group_rankings."],
                freshness=self._freshness(),
                rankings=[],
                movers={"gainers": [], "losers": []},
            )

        ranking_date_str = rankings[0].get("date")
        ranking_date = date.fromisoformat(ranking_date_str) if ranking_date_str else None
        rank_change_key = f"rank_change_{args.period}"
        top_rankings = [
            {
                "industry_group": r["industry_group"],
                "rank": r["rank"],
                "avg_rs_rating": r.get("avg_rs_rating"),
                "num_stocks": r.get("num_stocks"),
                "top_symbol": r.get("top_symbol"),
                "rank_change": r.get(rank_change_key),
            }
            for r in rankings
        ]

        return self._envelope(
            f"Top {len(top_rankings)} IBD group rankings as of {ranking_date}. "
            f"Showing {args.period} movers ({len(movers.get('gainers', []))} gainers, {len(movers.get('losers', []))} losers).",
            facts=[
                self._fact("ranking_date", str(ranking_date), "ibd_group_ranks"),
                self._fact("returned_groups", len(rankings), "ibd_group_ranks"),
                self._fact("top_group", rankings[0]["industry_group"], "ibd_group_ranks", ranking_date),
            ],
            citations=[self._citation("ibd_group_ranks", "Group Rankings", f"ibd_group_ranks:{ranking_date}", ranking_date)],
            next_actions=[
                "Use find_candidates with industry_group filter to drill into a top-ranked group.",
                "Use explain_symbol for the top_symbol in a leading group.",
            ],
            freshness=self._freshness(as_of_date=ranking_date, ibd_group_ranks=ranking_date_str or ""),
            rankings=top_rankings,
            movers={
                "period": movers.get("period", args.period),
                "gainers": movers.get("gainers", []),
                "losers": movers.get("losers", []),
            },
        )

    def _stock_lookup(self, args: StockLookupArgs) -> ToolEnvelope:
        with self._session_scope() as db:
            stock = (
                db.query(StockUniverse)
                .filter(StockUniverse.symbol == args.symbol, StockUniverse.is_active.is_(True))
                .first()
            )
            run = self._latest_published_run(db) if args.include_technicals else None

        if stock is None:
            return self._envelope(
                f"Symbol {args.symbol!r} was not found in the active stock universe.",
                facts=[],
                citations=[],
                next_actions=["Verify the ticker symbol or check if it was delisted."],
                freshness=self._freshness(),
                stock=None,
                technicals=None,
            )

        stock_info = {
            "symbol": stock.symbol,
            "name": stock.name,
            "sector": stock.sector,
            "industry": stock.industry,
            "market_cap": stock.market_cap,
            "exchange": stock.exchange,
        }

        technicals = None
        if run is not None:
            with self._uow_scope() as uow:
                row = uow.feature_store.get_by_symbol_for_run(
                    run.id,
                    args.symbol,
                    include_sparklines=False,
                    include_setup_payload=False,
                )
            if row is not None:
                extended = getattr(row, "extended_fields", {}) or {}
                technicals = {
                    "run_id": run.id,
                    "as_of_date": run.as_of_date.isoformat(),
                    "composite_score": row.composite_score,
                    "overall_rating": row.rating,
                    "rs_rating": extended.get("rs_rating"),
                    "stage": extended.get("stage"),
                    "current_price": row.current_price,
                }

        run_date = run.as_of_date if run is not None else None
        facts = [
            self._fact("symbol", stock.symbol, "stock_universe"),
            self._fact("name", stock.name, "stock_universe"),
            self._fact("sector", stock.sector, "stock_universe"),
        ]
        citations = [self._citation("stock_universe", stock.symbol, f"stock_universe:{stock.symbol}")]
        if technicals:
            facts.append(self._fact("composite_score", technicals["composite_score"], "feature_runs", run_date))

        return self._envelope(
            f"{stock.symbol} ({stock.name}) — {stock.sector} / {stock.industry}."
            + (f" Score {technicals['composite_score']}, RS {technicals['rs_rating']}, Stage {technicals['stage']}." if technicals else ""),
            facts=facts,
            citations=citations,
            next_actions=["Use explain_symbol for a detailed breakdown." if technicals else "Run a feature snapshot to get technicals."],
            freshness=self._freshness(
                as_of_date=run_date,
                stock_universe="current",
                **({"feature_run": run_date.isoformat()} if run_date else {}),
            ),
            stock=stock_info,
            technicals=technicals,
        )

    def _stock_snapshot(self, args: StockSnapshotArgs) -> ToolEnvelope:
        with self._session_scope() as db:
            stock = (
                db.query(StockUniverse)
                .filter(StockUniverse.symbol == args.symbol, StockUniverse.is_active.is_(True))
                .first()
            )
            run = self._latest_published_run(db)
            breadth = self._latest_breadth(db, getattr(run, "as_of_date", None))
            watchlists = (
                db.query(UserWatchlist)
                .join(WatchlistItem, WatchlistItem.watchlist_id == UserWatchlist.id)
                .filter(WatchlistItem.symbol == args.symbol)
                .order_by(UserWatchlist.position.asc())
                .all()
            )
            economic = EconomicThemeReader(db)
            if economic.source_name == "economic":
                themes = economic.theme_summaries_for_symbol(args.symbol)
                has_theme_metrics = any(
                    row.get("momentum_score") is not None for row in themes
                )
            else:
                constituent_base_query = (
                    db.query(ThemeConstituent, ThemeCluster)
                    .join(
                        ThemeCluster,
                        ThemeCluster.id == ThemeConstituent.theme_cluster_id,
                    )
                    .filter(
                        ThemeConstituent.symbol == args.symbol,
                        ThemeConstituent.is_active.is_(True),
                        ThemeCluster.is_active.is_(True),
                    )
                    .order_by(ThemeCluster.display_name.asc())
                )
                if run is not None:
                    constituent_rows = (
                        constituent_base_query.add_entity(ThemeMetrics)
                        .outerjoin(
                            ThemeMetrics,
                            (ThemeMetrics.theme_cluster_id == ThemeCluster.id)
                            & (ThemeMetrics.date == run.as_of_date),
                        )
                        .all()
                    )
                else:
                    constituent_rows = [
                        (constituent, cluster, None)
                        for constituent, cluster in constituent_base_query.all()
                    ]
                themes = [
                    {
                        "display_name": cluster.display_name,
                        "canonical_key": cluster.canonical_key,
                        "category": cluster.category,
                        "lifecycle_state": cluster.lifecycle_state,
                        "confidence": constituent.confidence,
                        "mention_count": constituent.mention_count,
                        "momentum_score": getattr(metrics, "momentum_score", None),
                        "status": getattr(metrics, "status", None),
                    }
                    for constituent, cluster, metrics in constituent_rows
                ]
                has_theme_metrics = any(
                    metrics is not None for _, _, metrics in constituent_rows
                )

        if stock is None:
            return self._envelope(
                f"Symbol {args.symbol!r} was not found in the active stock universe.",
                facts=[],
                citations=[],
                next_actions=["Verify the ticker symbol or check if it was delisted."],
                freshness=self._freshness(),
                stock=None,
                technicals=None,
                themes=[],
                watchlists=[],
                breadth=None,
            )

        technicals = None
        if run is not None:
            with self._uow_scope() as uow:
                row = uow.feature_store.get_by_symbol_for_run(
                    run.id,
                    args.symbol,
                    include_sparklines=False,
                    include_setup_payload=False,
                )
            if row is not None:
                technicals = self._candidate_record(row)
                technicals["run_id"] = run.id
                technicals["as_of_date"] = run.as_of_date.isoformat()

        breadth_record = self._breadth_record(breadth)
        watchlist_rows = [{"id": row.id, "name": row.name} for row in watchlists]
        run_date = run.as_of_date if run is not None else None

        facts = [
            self._fact("symbol", stock.symbol, "stock_universe"),
            self._fact("sector", stock.sector, "stock_universe"),
            self._fact("theme_count", len(themes), "theme_clusters", run_date),
            self._fact("watchlist_count", len(watchlist_rows), "user_watchlists"),
        ]
        if technicals is not None:
            facts.append(self._fact("composite_score", technicals.get("composite_score"), "feature_runs", run_date))
        citations = [self._citation("stock_universe", stock.symbol, f"stock_universe:{stock.symbol}")]
        citations.extend(
            self._citation(
                "economic_themes" if theme.get("theme_id") else "theme_clusters",
                theme["display_name"],
                (
                    f"economic_themes:{theme['theme_id']}"
                    if theme.get("theme_id")
                    else f"theme_clusters:{theme['canonical_key']}"
                ),
                run_date,
            )
            for theme in themes[:3]
        )
        if breadth is not None:
            citations.append(self._citation("market_breadth", "Breadth", f"market_breadth:{breadth.date.isoformat()}", breadth.date))

        summary = (
            f"{stock.symbol} ({stock.name}) is tracked in {len(watchlist_rows)} watchlist(s) and {len(themes)} active theme(s)."
        )
        if technicals is not None:
            summary += (
                f" Latest published run {run.id} scored it {technicals.get('composite_score')} "
                f"with RS {technicals.get('rs_rating')} and stage {technicals.get('stage')}."
            )
        if breadth is not None:
            summary += f" Latest breadth 5-day ratio is {breadth.ratio_5day} on {breadth.date.isoformat()}."

        return self._envelope(
            summary,
            facts=facts,
            citations=citations,
            next_actions=[
                "Use explain_symbol for setup detail and peers.",
                "Use theme_state to inspect any linked themes in more detail." if themes else "Use find_candidates to compare against other leadership names.",
            ],
            freshness=self._freshness(
                as_of_date=run_date,
                stock_universe="current",
                **{
                    key: value
                    for key, value in {
                        "feature_run": run_date.isoformat() if run_date else None,
                        "market_breadth": breadth.date.isoformat() if breadth is not None else None,
                        "theme_metrics": run_date.isoformat() if has_theme_metrics and run_date else None,
                    }.items()
                    if value is not None
                },
            ),
            stock={
                "symbol": stock.symbol,
                "name": stock.name,
                "sector": stock.sector,
                "industry": stock.industry,
                "market_cap": stock.market_cap,
                "exchange": stock.exchange,
            },
            technicals=technicals,
            themes=themes,
            watchlists=watchlist_rows,
            breadth=breadth_record,
        )

    def _breadth_snapshot(self, args: BreadthSnapshotArgs) -> ToolEnvelope:
        with self._session_scope() as db:
            rows = (
                breadth_query(db, market="US")
                .order_by(desc(MarketBreadth.date))
                .limit(args.days)
                .all()
            )

        if not rows:
            return self._envelope(
                "No market breadth data is available.",
                facts=[],
                citations=[],
                next_actions=["Run a breadth calculation to populate breadth data."],
                freshness=self._freshness(),
                snapshots=[],
            )

        snapshots = [self._breadth_record(row) for row in rows]
        latest = rows[0]

        return self._envelope(
            f"Market breadth for the last {len(snapshots)} trading days. "
            f"Latest ({latest.date.isoformat()}): {latest.stocks_up_4pct} up 4%+, {latest.stocks_down_4pct} down 4%+, "
            f"5-day ratio {latest.ratio_5day}.",
            facts=[
                self._fact("latest_date", latest.date.isoformat(), "market_breadth", latest.date),
                self._fact("days_returned", len(snapshots), "market_breadth"),
                self._fact("latest_ratio_5day", latest.ratio_5day, "market_breadth", latest.date),
            ],
            citations=[self._citation("market_breadth", "Breadth Data", f"market_breadth:{latest.date.isoformat()}", latest.date)],
            next_actions=[
                "Use market_overview for a fuller picture including feature runs and alerts.",
            ],
            freshness=self._freshness(as_of_date=latest.date, market_breadth=str(latest.date)),
            snapshots=snapshots,
        )

    def _daily_digest(self, args: DailyDigestArgs) -> ToolEnvelope:
        from app.services.digest_service import DigestService

        with self._session_scope() as db:
            service = DigestService()
            digest = service.get_daily_digest(db, as_of_date=args.as_of_date)

        digest_date = date.fromisoformat(digest.as_of_date) if digest.as_of_date else None

        return self._envelope(
            f"Daily digest as of {digest.as_of_date}. "
            f"Market stance: {digest.market.stance}. "
            f"{len(digest.leaders)} leaders, {len(digest.risks)} risk notes.",
            facts=[
                self._fact("as_of_date", digest.as_of_date, "digest"),
                self._fact("market_stance", digest.market.stance, "digest", digest_date),
                self._fact("leader_count", len(digest.leaders), "digest", digest_date),
            ],
            citations=[self._citation("digest", "Daily Digest", f"digest:{digest.as_of_date}", digest_date)],
            next_actions=[
                "Use find_candidates to drill into the leaders list.",
                "Use theme_state for deeper theme analysis.",
            ],
            freshness=self._freshness(
                as_of_date=digest_date,
                digest=digest.as_of_date or "",
                **{k: v for k, v in {
                    "feature_run": digest.freshness.latest_feature_as_of_date,
                    "market_breadth": digest.freshness.latest_breadth_date,
                    "theme_metrics": digest.freshness.latest_theme_metrics_date,
                }.items() if v is not None},
            ),
            digest=digest.model_dump(mode="json"),
        )

    def _find_candidates_for_run(self, run_id: int, args: FindCandidatesArgs) -> list[dict[str, Any]]:
        with self._uow_scope() as uow:
            page = uow.feature_store.query_run_as_scan_results(
                run_id,
                QuerySpec.from_filter_spec(
                    self._build_filter_spec(args.filters),
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

    def _latest_published_run_for_symbol(
        self,
        uow: SqlUnitOfWork,
        symbol: str,
    ) -> FeatureRunDomain | None:
        """Return the published run that applies to *symbol*, market first.

        Publication now writes per-market pointers (``latest_published_market:US``)
        and the global ``latest_published`` pointer is no longer maintained. Reading
        only the global pointer therefore reported "no published feature run" while a
        perfectly valid market run existed — exactly what ``find_candidates`` was
        still showing. Mirrors ``api.v1.stocks._get_latest_feature_run_for_symbol``:
        resolve the symbol's market, try its pointer, then fall back to the global
        pointer so a legacy single-market install keeps working.

        The market lookup is deliberately local rather than importing
        ``api.v1._price_history.resolve_symbol_market``: that module exists to
        share price-history helpers between two route modules and pulls in pandas,
        so it is not a dependency the MCP layer should take. This module already
        queries ``StockUniverse`` directly and imports it.

        Deliberately does **not** create a global ``latest_published`` pointer as a
        side effect: publication owns that key, and writing it here would mask a
        genuinely missing publication.
        """
        market = (
            uow.session.query(StockUniverse.market)
            .filter(
                StockUniverse.active_filter(),
                StockUniverse.symbol == symbol.upper(),
            )
            .scalar()
        )
        normalized_market = str(market or "").strip().upper()

        if normalized_market:
            latest_run = uow.feature_runs.get_latest_published(
                pointer_key=f"latest_published_market:{normalized_market}"
            )
            if latest_run is not None:
                return latest_run

        return uow.feature_runs.get_latest_published()

    def _latest_breadth(self, db: Session, as_of_date: date | None) -> MarketBreadth | None:
        # Market copilot is US-scoped today.
        return latest_breadth(db, market="US", as_of_date=as_of_date)

    def _recent_alerts(self, db: Session, limit: int) -> list[dict[str, Any]] | None:
        """Unread legacy alerts, or None under economic authority, where they are unavailable."""
        if EconomicThemeReader(db).source_name == "economic":
            return None  # legacy-only, as /themes/alerts in economic mode (#475)
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
        rows = get_task_registry_service().get_all_scheduled_tasks(db)
        rows.sort(key=lambda row: row["name"])
        return rows

    def _last_run(self, task: dict[str, Any]) -> dict[str, Any]:
        return task.get("last_run") or {}

    def _candidate_record(self, item: Any) -> dict[str, Any]:
        """Project one scan result into the flat MCP candidate record.

        The record is a projection of ``ScanResultItem`` - the same model the
        HTTP API, the UI and the static export use - so a new fact reaches every
        surface from one field definition. Nothing is recomputed here: the
        schema only reads the persisted ``extended_fields``.
        """
        record = ScanResultItem.from_domain(
            item,
            include_setup_payload=False,
        ).model_dump(mode="json", include=CANDIDATE_FIELDS)
        if record["composite_score"] is not None:
            record["composite_score"] = round(record["composite_score"], 2)
        return record

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

    def _facts(
        self,
        values: dict[str, Any],
        source: str,
        *,
        as_of: date | None = None,
        key_dates: dict[str, date] | None = None,
    ) -> list[ToolFact]:
        """Build one fact per value, keeping the dictionary's order."""
        return [
            self._fact(key, value, source, (key_dates or {}).get(key, as_of))
            for key, value in values.items()
        ]

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

    def _freshness(
        self,
        as_of_date: date | None = None,
        *,
        generated_at: datetime | None = None,
        **sources: str | None,
    ) -> ToolFreshness:
        return ToolFreshness(
            generated_at=(generated_at or datetime.now(UTC)).isoformat(),
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

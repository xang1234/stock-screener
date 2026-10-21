"""Per-market daily refresh pipeline orchestration."""

from __future__ import annotations

import logging
import time
from datetime import date, datetime
from typing import Any
from uuid import uuid4

from celery import chain

from app.celery_app import celery_app
from app.database import SessionLocal
from app.domain.markets.catalog import get_market_catalog
from app.domain.markets.registry import critical_price_symbols
from app.domain.relative_strength import BALANCED_RS_FORMULA_VERSION
from app.domain.relative_strength.price_validity import is_valid_adjusted_price
from app.models.stock import StockPrice
from app.services.market_calendar_service import MarketCalendarService
from app.tasks.market_queues import (
    SHARED_DATA_FETCH_QUEUE,
    data_fetch_queue_for_market,
    market_jobs_queue_for_market,
    normalize_market,
)

logger = logging.getLogger(__name__)

_MIN_DAILY_PRICE_REFRESH_SUCCESS_RATE = 0.90
# A missing primary benchmark bar: one refetch of the candidates, then this
# many re-runs of the whole pipeline, each this long after the failed guard.
_MAX_BENCHMARK_RERUNS = 2
_BENCHMARK_RERUN_COUNTDOWN_SECONDS = 45 * 60
_BENCHMARK_REFETCH_PERIOD = "1y"
PRIMARY_BENCHMARK_MISSING = "primary_benchmark_missing"


class PrimaryBenchmarkMissing(RuntimeError):
    """The primary benchmark has no bar for the pipeline's as-of date after a
    refetch. Market RS needs it (the as-of date is its first anchor) and US
    exposure is scored on the primary only, so the chain stops here, where a
    re-run can still recover the evening."""


def _normalize_pipeline_market(market: str | None) -> str:
    normalized = normalize_market(market)
    if normalized == "SHARED":
        raise ValueError("Daily market pipelines require an explicit market")
    return normalized


def _daily_pipeline_universe_name(market: str) -> str:
    return f"market:{market.upper()}"


def _pipeline_chain_claims():
    from app.wiring.bootstrap import get_workload_coordination

    return get_workload_coordination()


def _nonfatal_group_taxonomy_skip(result: dict) -> bool:
    return result.get("reason") == "no_taxonomy_for_market" and (
        result.get("skipped") is True
        or str(result.get("status", "")).lower() == "skipped"
    )


def _result_failed(result: Any) -> bool:
    if not isinstance(result, dict):
        return True
    if result.get("error") or result.get("reason") == "not_trading_day":
        return True
    if _nonfatal_group_taxonomy_skip(result):
        return False
    if result.get("skipped") is True:
        return True
    return str(result.get("status", "")).lower() in {
        "already_running",
        "failed",
        "partial",
        "skipped",
    }


def _partial_price_refresh_meets_minimum(result: dict) -> bool:
    try:
        coverage_refreshed = result.get("coverage_refreshed")
        coverage_total = result.get("coverage_total")
        refreshed = float(
            coverage_refreshed
            if coverage_refreshed is not None
            else result.get("refreshed", 0)
        )
        total = float(
            coverage_total
            if coverage_total is not None
            else result.get("total", 0)
        )
    except (TypeError, ValueError):
        return False
    return total > 0 and refreshed / total >= _MIN_DAILY_PRICE_REFRESH_SUCCESS_RATE


def _price_refresh_failed(result: Any) -> bool:
    if not isinstance(result, dict):
        return True
    if _nonfatal_group_taxonomy_skip(result):
        return False
    status = str(result.get("status", "")).lower()
    if status == "partial":
        if result.get("error") or result.get("reason") == "not_trading_day":
            return True
        if result.get("skipped") is True:
            return True
        return not _partial_price_refresh_meets_minimum(result)
    return _result_failed(result)


def _benchmark_bar_present(symbol: str, as_of: date) -> bool:
    """A StockPrice row with the adjusted close Market RS reads."""
    db = SessionLocal()
    try:
        adj_close = (
            db.query(StockPrice.adj_close)
            .filter(StockPrice.symbol == symbol, StockPrice.date == as_of)
            .scalar()
        )
    finally:
        db.close()
    return is_valid_adjusted_price(adj_close)


def _refetch_prices(market: str, symbols: tuple[str, ...]) -> None:
    """One pass of the bulk price path for ``symbols``: the market's provider
    plan (Yahoo quote repair where the plan has it), then the refresh
    persistence."""
    from app.services.bulk_data_fetcher import BulkDataFetcher
    from app.wiring.bootstrap import get_price_cache

    results = BulkDataFetcher().fetch_prices_in_batches(
        list(symbols), period=_BENCHMARK_REFETCH_PERIOD, market=market
    )
    frames = {
        symbol: payload["price_data"]
        for symbol, payload in results.items()
        if not payload.get("has_error") and payload.get("price_data") is not None
    }
    get_price_cache().store_refreshed_batch(
        frames,
        period=_BENCHMARK_REFETCH_PERIOD,
        market_by_symbol={symbol: market for symbol in frames},
    )


@celery_app.task(
    name="app.tasks.daily_market_pipeline_tasks.ensure_benchmark_prices",
    queue=SHARED_DATA_FETCH_QUEUE,
)
def ensure_benchmark_prices(
    result: dict | None = None, *, market: str, as_of_date: str
) -> dict:
    """Refetch the market's benchmark candidates once when the primary has no
    bar for ``as_of_date``, and add what the guard decides on to the price
    refresh result.

    Runs on the market's data_fetch queue so the refetch takes the provider
    plan the daily refresh takes. A refresh that already failed passes
    through untouched; its guard reports it.
    """
    if _price_refresh_failed(result):
        return result
    market_code = _normalize_pipeline_market(market)
    candidates = critical_price_symbols(market_code)
    primary = candidates[0]
    as_of = date.fromisoformat(as_of_date)
    report: dict[str, Any] = {
        "status": "present",
        "symbol": primary,
        "as_of_date": as_of_date,
        "refetched": False,
    }
    if not _benchmark_bar_present(primary, as_of):
        report["refetched"] = True
        try:
            _refetch_prices(market_code, candidates)
        except Exception as exc:
            # The guard turns a missing bar into the re-run; a refetch that
            # raised here would end the chain before it.
            logger.warning("Benchmark refetch for %s failed", market_code, exc_info=True)
            report["error"] = str(exc)
        if not _benchmark_bar_present(primary, as_of):
            report["status"] = "missing"
    return {**result, "benchmark": report}


def _schedule_benchmark_rerun(
    market: str, *, as_of_date: str | None, rerun_attempt: int
) -> dict:
    """Queue one re-run of the market's pipeline for ``as_of_date``, at most
    _MAX_BENCHMARK_RERUNS times.

    The attempt number rides in the re-run's kwargs: the count needs no
    store, cannot be left behind, and the next Beat dispatch starts at zero.
    The re-run is an ordinary dispatch, so it takes the chain claim (and
    skips while a pipeline still holds it) and checks the session again when
    it fires. Never queued while the market is open.
    """
    if as_of_date is None:
        return {"status": "skipped", "reason": "no_as_of_date"}
    attempt = rerun_attempt + 1
    if attempt > _MAX_BENCHMARK_RERUNS:
        return {"status": "exhausted", "attempts": rerun_attempt}
    try:
        market_open = _preflight_calendar().is_market_open(market)
    except Exception as exc:
        logger.warning("Cannot tell whether %s is open; no re-run queued", market, exc_info=True)
        return {"status": "skipped", "reason": f"calendar_unavailable: {exc}"}
    if market_open:
        return {"status": "skipped", "reason": "market_open"}
    try:
        task = queue_daily_market_pipeline.apply_async(
            kwargs={"market": market, "rerun_attempt": attempt, "rerun_as_of_date": as_of_date},
            countdown=_BENCHMARK_RERUN_COUNTDOWN_SECONDS,
            queue=market_jobs_queue_for_market(market),
        )
    except Exception as exc:
        logger.warning("Could not queue the %s pipeline re-run", market, exc_info=True)
        return {"status": "failed_to_queue", "reason": str(exc)}
    return {
        "status": "scheduled",
        "attempt": attempt,
        "countdown_seconds": _BENCHMARK_RERUN_COUNTDOWN_SECONDS,
        "task_id": task.id,
    }


@celery_app.task(
    name="app.tasks.daily_market_pipeline_tasks.guard_price_refresh",
    queue="celery",
)
def guard_price_refresh(
    result: dict | None = None,
    *,
    market: str,
    as_of_date: str | None = None,
    rerun_attempt: int = 0,
) -> dict:
    if _price_refresh_failed(result):
        raise RuntimeError(f"Daily price refresh failed for {market}: {result}")
    # Absent from a chain published before ensure_benchmark_prices existed.
    benchmark = result.get("benchmark")
    if benchmark is not None and benchmark.get("status") != "present":
        rerun = _schedule_benchmark_rerun(
            market, as_of_date=as_of_date, rerun_attempt=rerun_attempt
        )
        logger.error(
            "%s: %s has no %s bar for %s after a refetch (rerun: %s)",
            PRIMARY_BENCHMARK_MISSING, market, benchmark.get("symbol"), as_of_date, rerun,
            extra={"market": market, "reason": PRIMARY_BENCHMARK_MISSING, "rerun": rerun},
        )
        raise PrimaryBenchmarkMissing(
            f"{PRIMARY_BENCHMARK_MISSING}: {market} has no {benchmark.get('symbol')} bar for "
            f"{as_of_date} after a refetch; rerun={rerun}; benchmark={benchmark}"
        )
    return {"status": "ok", "market": market, "stage": "prices"}


@celery_app.task(
    name="app.tasks.daily_market_pipeline_tasks.guard_market_rs_result",
    queue="celery",
)
def guard_market_rs_result(
    result: dict | None = None,
    *,
    market: str,
    calculation_date: str,
) -> dict:
    market_code = _normalize_pipeline_market(market)
    exact_completed = (
        isinstance(result, dict)
        and result.get("status") == "completed"
        and result.get("market") == market_code
        and result.get("as_of_date") == calculation_date
        and result.get("formula_version") == BALANCED_RS_FORMULA_VERSION
        and result.get("market_rs_run_id") is not None
    )

    if not exact_completed:
        raise RuntimeError(
            f"Canonical Market RS failed for {market_code} on "
            f"{calculation_date}: {result}"
        )
    return {
        "status": "ok",
        "market": market_code,
        "stage": "market_rs",
        "as_of_date": calculation_date,
        "formula_version": BALANCED_RS_FORMULA_VERSION,
        "market_rs_run_id": result["market_rs_run_id"],
    }


@celery_app.task(
    name="app.tasks.daily_market_pipeline_tasks.guard_breadth_result",
    queue="celery",
)
def guard_breadth_result(result: dict | None = None, *, market: str) -> dict:
    if _result_failed(result):
        raise RuntimeError(f"Daily breadth calculation failed for {market}: {result}")
    return {"status": "ok", "market": market, "stage": "breadth"}


@celery_app.task(
    name="app.tasks.daily_market_pipeline_tasks.guard_exposure_result",
    queue="celery",
)
def guard_exposure_result(result: dict | None = None, *, market: str) -> dict:
    # Exposure is a non-critical leaf — groups/snapshot do NOT depend on it — so
    # a missing/lagging benchmark must not abort the pipeline. Log and continue.
    if _result_failed(result):
        logger.warning("Daily market exposure not stored for %s (continuing): %s", market, result)
        return {"status": "skipped", "market": market, "stage": "exposure"}
    return {"status": "ok", "market": market, "stage": "exposure"}


@celery_app.task(
    name="app.tasks.daily_market_pipeline_tasks.guard_group_result",
    queue="celery",
)
def guard_group_result(result: dict | None = None, *, market: str) -> dict:
    if _result_failed(result):
        raise RuntimeError(f"Daily group ranking failed for {market}: {result}")
    return {"status": "ok", "market": market, "stage": "groups"}


@celery_app.task(
    name="app.tasks.daily_market_pipeline_tasks.guard_snapshot_result",
    queue="celery",
)
def guard_snapshot_result(result: dict | None = None, *, market: str) -> dict:
    if isinstance(result, dict) and result.get("auto_scan_id"):
        guarded = {
            "status": "ok",
            "market": market,
            "stage": "scan",
            "auto_scan_id": result["auto_scan_id"],
        }
        run_id = result.get("run_id") or result.get("existing_run_id")
        if run_id is not None:
            guarded["run_id"] = run_id
        if result.get("as_of_date") is not None:
            guarded["as_of_date"] = result["as_of_date"]
        return guarded
    raise RuntimeError(f"Daily market scan did not publish for {market}: {result}")


@celery_app.task(
    name="app.tasks.daily_market_pipeline_tasks.dispatch_options_after_snapshot",
    queue="celery",
)
def dispatch_options_after_snapshot(
    result: dict | None = None, *, market: str
) -> dict:
    payload = dict(result or {})
    if _normalize_pipeline_market(market) != "US":
        payload["options_dispatch"] = {
            "status": "skipped",
            "reason": "market_unsupported",
        }
        return payload
    try:
        from app.interfaces.tasks.options_analytics_tasks import (
            refresh_options_analytics,
        )

        task = refresh_options_analytics.apply_async(
            kwargs={"source_run_id": payload.get("run_id"), "market": "US"},
            queue=data_fetch_queue_for_market("US"),
        )
        payload["options_dispatch"] = {"status": "queued", "task_id": task.id}
    except Exception as exc:
        logger.warning("Could not dispatch Options Analytics follow-on", exc_info=True)
        payload["options_dispatch"] = {
            "status": "failed_to_queue",
            "reason": str(exc),
        }
    return payload


def _build_daily_market_pipeline_signatures(
    market: str, trading_date: date, *, rerun_attempt: int = 0
) -> list:
    from app.interfaces.tasks.feature_store_tasks import build_daily_snapshot
    from app.tasks.breadth_tasks import (
        calculate_daily_breadth_with_gapfill,
        calculate_market_exposure,
    )
    from app.tasks.cache_tasks import smart_refresh_cache
    from app.tasks.group_rank_tasks import calculate_daily_group_rankings_with_gapfill
    from app.tasks.market_rs_tasks import calculate_market_rs_snapshot

    market_code = _normalize_pipeline_market(market)
    as_of_date = trading_date.isoformat()
    signatures = [
        smart_refresh_cache.si(mode="delta", market=market_code).set(
            queue=data_fetch_queue_for_market(market_code)
        ),
        ensure_benchmark_prices.s(market=market_code, as_of_date=as_of_date).set(
            queue=data_fetch_queue_for_market(market_code)
        ),
        guard_price_refresh.s(
            market=market_code, as_of_date=as_of_date, rerun_attempt=rerun_attempt
        ).set(queue=market_jobs_queue_for_market(market_code)),
        calculate_market_rs_snapshot.si(
            market=market_code,
            calculation_date=as_of_date,
            formula_version=BALANCED_RS_FORMULA_VERSION,
        ).set(queue=market_jobs_queue_for_market(market_code)),
        guard_market_rs_result.s(
            market=market_code,
            calculation_date=as_of_date,
        ).set(queue=market_jobs_queue_for_market(market_code)),
    ]
    signatures.extend([
        calculate_daily_group_rankings_with_gapfill.si(
            market=market_code,
            calculation_date=as_of_date,
            execution_policy="refresh_guarded",
        ).set(
            queue=market_jobs_queue_for_market(market_code)
        ),
        guard_group_result.s(market=market_code).set(
            queue=market_jobs_queue_for_market(market_code)
        ),
        build_daily_snapshot.si(
            market=market_code,
            as_of_date_str=as_of_date,
            universe_name=_daily_pipeline_universe_name(market_code),
            publish_pointer_key=f"latest_published_market:{market_code}",
            static_daily_mode=True,
        ).set(queue=market_jobs_queue_for_market(market_code)),
        guard_snapshot_result.s(market=market_code).set(
            queue=market_jobs_queue_for_market(market_code)
        ),
    ])
    if market_code == "US":
        signatures.append(
            dispatch_options_after_snapshot.s(market=market_code).set(
                queue=market_jobs_queue_for_market(market_code)
            )
        )
    # Breadth and exposure run last: groups and the snapshot never read them,
    # and a breadth failure (its guard raises, or an OOM kill the task cannot
    # catch) ends the chain. Here it can no longer cost the day's scan.
    if get_market_catalog().get(market_code).capabilities.breadth:
        signatures.extend([
            calculate_daily_breadth_with_gapfill.si(
            market=market_code,
            calculation_date=as_of_date,
            execution_policy="refresh_guarded",
            ).set(queue=market_jobs_queue_for_market(market_code)),
            guard_breadth_result.s(market=market_code).set(
                queue=market_jobs_queue_for_market(market_code)
            ),
            # Exposure blends breadth + index OHLCV and runs after its guard.
            calculate_market_exposure.si(
                market=market_code,
                calculation_date=as_of_date,
            ).set(queue=market_jobs_queue_for_market(market_code)),
            guard_exposure_result.s(market=market_code).set(
                queue=market_jobs_queue_for_market(market_code)
            ),
        ])
    return signatures


@celery_app.task(
    name="app.tasks.daily_market_pipeline_tasks.release_daily_market_pipeline_claim",
    queue="celery",
)
def release_daily_market_pipeline_claim(*, market: str, token: str) -> dict:
    """Free the market's pipeline-chain claim: the chain's last stage, and the
    error callback of every stage.

    Keyword-only on purpose: Celery calls an errback task inline with
    (request, exc, traceback) when it takes more than one positional argument;
    this way it is published as a task and its immutable signature ignores the
    failed task's id.
    """
    try:
        released = _pipeline_chain_claims().release_pipeline_chain(token, market=market)
    except Exception:
        # The claim's TTL (PIPELINE_CHAIN_TTL_SECONDS) frees it regardless.
        logger.warning("Could not release the %s pipeline chain claim", market, exc_info=True)
        released = False
    return {"status": "ok", "market": market, "stage": "release", "released": released}


# Backend states that end a chain. Every state other than these, PENDING and
# SUCCESS means a worker holds the stage: STARTED, RETRY, and custom ones such
# as the PROGRESS a running price refresh reports through update_state.
_CHAIN_DEAD_STATES = frozenset({"FAILURE", "REVOKED"})
_CHAIN_IDLE_STATES = frozenset({"PENDING", "SUCCESS"})


def _chain_state_verdict(task_ids: list[str], states: dict[str, str]) -> str | None:
    """``dead``, ``alive`` or ``done`` from the stages' backend states; None
    when only the broker can tell (every stage PENDING or SUCCESS).

    A chain publishes one stage at a time, so later stages read PENDING until
    then. Results expire (24h), so an old stage's SUCCESS can read PENDING
    too, which is why the broker is searched for every stage id, not only the
    first unfinished one.
    """
    chain_states = [states.get(task_id, "PENDING") for task_id in task_ids]
    if any(state in _CHAIN_DEAD_STATES for state in chain_states):
        return "dead"
    if any(state not in _CHAIN_IDLE_STATES for state in chain_states):
        return "alive"
    if chain_states[-1] == "SUCCESS":
        return "done"
    return None


def _broker_verdict(task_ids: list[str], in_broker: dict[str, bool | None]) -> str | None:
    """``alive`` if a stage's message is queued or delivered, ``missing`` if
    none is, None if the broker could not be read in full."""
    answers = [in_broker.get(task_id) for task_id in task_ids]
    if any(answer is True for answer in answers):
        return "alive"
    if any(answer is None for answer in answers):
        return None
    return "missing"


def _apply_chain_verdict(
    claims: Any, market: str, token: str, verdict: str, heartbeat: str
) -> str:
    from app.tasks.workload_coordination import PIPELINE_CHAIN_MISSING_RELEASE_SECONDS

    if verdict in ("alive", "unknown"):
        # An inconclusive search proved nothing gone. A stage buried past the
        # search budget is most likely still queued, and only renewing keeps a
        # backlog longer than the TTL from letting a second chain start; once
        # the backlog drains, a conclusive search frees a dead chain's claim.
        claims.clear_pipeline_chain_missing(token)
        renewed = claims.renew_pipeline_chain(token, market=market)
        if not renewed:
            return "lost"
        return "renewed" if verdict == "alive" else "renewed_unknown"
    if verdict == "missing":
        seconds, since = claims.pipeline_chain_missing(token)
        if since is None or seconds < PIPELINE_CHAIN_MISSING_RELEASE_SECONDS:
            # Unconfirmed: renew so a claim near its TTL (after a stretch of
            # failed checks) lasts until the confirming check releases it.
            return "missing" if claims.renew_pipeline_chain(token, market=market) else "lost"
        # Only while the marker read above is still there (a heartbeat that
        # saw the chain alive since then cleared it) and this heartbeat still
        # holds the heartbeat lock (#627).
        outcome = claims.release_missing_pipeline_chain(
            token, market=market, missing_since=since, heartbeat=heartbeat
        )
        return "released_missing" if outcome == "released" else outcome
    # Owner-checked: a newer chain that took the claim meanwhile keeps it.
    released = claims.release_pipeline_chain(token, market=market)
    return f"released_{verdict}" if released else "lost"


@celery_app.task(
    name="app.tasks.daily_market_pipeline_tasks.renew_daily_market_pipeline_claims",
    queue="celery",
)
def renew_daily_market_pipeline_claims() -> dict:
    """Keep each market's pipeline-chain claim while its chain is alive (#605).

    Renews the claim while a stage is queued, delivered or running, and
    releases it once a stage failed or was revoked (a revoked stage fires no
    errback), the chain finished, or heartbeats found no trace of it for
    PIPELINE_CHAIN_MISSING_RELEASE_SECONDS. Chains whose states do not decide
    share one broker search, bounded like the Schedules page's
    (``still_in_broker``); a search that runs out of budget or cannot read
    the broker renews the claim, since it proved nothing gone. A market whose
    check raises keeps its claim, unrenewed, until the next beat or
    PIPELINE_CHAIN_TTL_SECONDS: an error that never clears must not hold a
    market forever.

    A stage killed with its worker (OOM, SIGKILL), or whose retry message was
    lost, keeps its STARTED/PROGRESS/RETRY state until its result expires
    (24h), so such a chain keeps its claim that long; the Operations console
    can release it (#612).

    Heartbeats never overlap: one reads a chain's states seconds before it
    acts on them, so another acting on an older read could free a claim the
    first just saw alive (#627). A beat that finds another running skips.
    """
    claims = _pipeline_chain_claims()
    heartbeat = uuid4().hex
    if not claims.acquire_pipeline_chain_heartbeat(heartbeat):
        return {"status": "skipped", "reason": "heartbeat_running", "claims": {}}
    try:
        return {"status": "ok", "claims": _renew_pipeline_chain_claims(claims, heartbeat)}
    finally:
        claims.release_pipeline_chain_heartbeat(heartbeat)


def _renew_pipeline_chain_claims(claims: Any, heartbeat: str) -> dict[str, str]:
    from app.services.task_registry_service import still_in_broker

    outcomes: dict[str, str] = {}
    checks: dict[str, tuple[str, list[str], str | None]] = {}
    for market, holder in claims.get_pipeline_chain_holders().items():
        token = (holder or {}).get("task_id")
        if not token:
            continue
        try:
            task_ids = claims.get_pipeline_chain_task_ids(token)
            if not task_ids:
                # Claimed before #605 recorded stage ids: its fixed expiry frees it.
                outcomes[market] = "untracked"
                continue
            states = {task_id: celery_app.AsyncResult(task_id).state for task_id in task_ids}
            checks[market] = (token, task_ids, _chain_state_verdict(task_ids, states))
        except Exception:
            logger.warning(
                "Could not check the %s pipeline chain claim; it is kept", market, exc_info=True
            )
            outcomes[market] = "error"

    undecided = [
        task_id for _token, task_ids, verdict in checks.values() if verdict is None
        for task_id in task_ids
    ]
    # Fresh: a cached "not found" from before a stage was published would
    # read a live chain as missing.
    in_broker = still_in_broker(undecided, cache=False) if undecided else {}
    for market, (token, task_ids, verdict) in checks.items():
        # None: the search ran out of budget or the broker was unreadable.
        verdict = verdict or _broker_verdict(task_ids, in_broker) or "unknown"
        try:
            outcomes[market] = _apply_chain_verdict(claims, market, token, verdict, heartbeat)
        except Exception:
            logger.warning(
                "Could not update the %s pipeline chain claim; it is kept", market, exc_info=True
            )
            outcomes[market] = "error"
    return outcomes


def _chain_task_ids(workflow: Any) -> list[str]:
    """Freeze ``workflow`` and return its stages' task ids, first to last.

    Publishing the same chain object keeps these ids.
    """
    node = workflow.freeze()
    task_ids: list[str] = []
    while node is not None:
        task_ids.append(node.id)
        node = node.parent
    return task_ids[::-1]


def _claim_pipeline_chain(
    market: str, token: str, task_ids: list[str]
) -> tuple[Any, bool | None]:
    """(claims, True) when claimed, (claims, False) when another chain holds
    the claim, (claims or None, None) when the claim store is unavailable.

    None fails open, like the workload-holder check: the daily run matters
    more than duplicate protection.
    """
    claims = None
    try:
        claims = _pipeline_chain_claims()
        return claims, claims.claim_pipeline_chain(token, market=market, task_ids=task_ids)
    except Exception:
        logger.warning(
            "Could not claim the %s pipeline chain; dispatching without duplicate protection",
            market,
            exc_info=True,
        )
        return claims, None


def _pipeline_chain_holder(claims: Any, market: str) -> dict | None:
    """The current holder, None if the claim is free, {} if it cannot be read.

    {} keeps a lost SET NX a skip: a chain does hold the claim.
    """
    try:
        return claims.get_pipeline_chain_holder(market)
    except Exception:
        logger.warning("Could not read the %s pipeline chain holder", market, exc_info=True)
        return {}


_PREFLIGHT_CALENDAR_TTL_SECONDS = 300
_preflight_calendar_cache: list = []  # [(built_at_monotonic, calendar)]


def _preflight_calendar():
    """A shared calendar for pre-flight checks: the Scheduled jobs list polls
    every 5-30s and asks once per market, and each MarketCalendarService()
    re-reads every market's coverage manifests."""
    now = time.monotonic()
    if not _preflight_calendar_cache or now - _preflight_calendar_cache[0][0] > _PREFLIGHT_CALENDAR_TTL_SECONDS:
        _preflight_calendar_cache[:] = [(now, MarketCalendarService())]
    return _preflight_calendar_cache[0][1]


def pipeline_run_blockers(markets) -> dict[str, dict | None]:
    """{MARKET: why a manual Run would only skip right now, or None}.

    The two cheap dispatcher checks an operator can act on, run before
    anything is queued. All markets share one calendar and one MGET for their
    chain claims, so a hung Redis costs the polled list one socket timeout,
    not one per market. A check that cannot be read never blocks: the task
    still runs every check itself (Beat calls it directly anyway).
    """
    codes = [_normalize_pipeline_market(market) for market in markets]
    try:
        starts = _pipeline_chain_claims().get_pipeline_chain_starts(codes)
    except Exception:
        logger.debug("Pre-flight claim check failed for %s", codes, exc_info=True)
        starts = {}
    blockers: dict[str, dict | None] = {}
    for code in codes:
        blockers[code] = None
        try:
            calendar = _preflight_calendar()
            today = calendar.market_now(code).date()
            if not calendar.is_trading_day(code, today):
                blockers[code] = {
                    "reason": "not_trading_day",
                    "message": f"Not a trading day in {code} ({today.isoformat()})",
                    "as_of_date": today.isoformat(),
                }
                continue
        except Exception:
            logger.debug("Pre-flight calendar check failed for %s", code, exc_info=True)
        if starts.get(code) is not None:
            blockers[code] = {
                "reason": "market_pipeline_already_active",
                "message": f"The {code} pipeline is already in progress",
                "active_since": starts[code] or None,
            }
    return blockers


def pipeline_run_blocker(market: str) -> dict | None:
    """The pre-flight answer for one market (see pipeline_run_blockers)."""
    return pipeline_run_blockers([market])[_normalize_pipeline_market(market)]

def _market_pipeline_active(market: str) -> dict | None:
    try:
        from app.wiring.bootstrap import get_workload_coordination

        return get_workload_coordination().get_market_workload_holder(market)
    except Exception:
        logger.debug("Could not inspect market workload holder", exc_info=True)
        return None


def _local_runtime_bootstrap_blocker(market: str) -> dict | None:
    from app.services.runtime_preferences_service import get_runtime_bootstrap_status

    db = SessionLocal()
    try:
        bootstrap_status = get_runtime_bootstrap_status(db)
    finally:
        db.close()

    if not bootstrap_status.bootstrap_required:
        return None

    return {
        "status": "skipped",
        "reason": "local_runtime_bootstrap_not_ready",
        "market": market,
        "bootstrap_state": bootstrap_status.bootstrap_state,
        "bootstrap_required": bootstrap_status.bootstrap_required,
        "timestamp": datetime.now().isoformat(),
    }


@celery_app.task(
    bind=True,
    name="app.tasks.daily_market_pipeline_tasks.queue_daily_market_pipeline",
    queue="celery",
)
def queue_daily_market_pipeline(
    self, market: str, rerun_attempt: int = 0, rerun_as_of_date: str | None = None
) -> dict:
    """Dispatch the market's daily chain for its last completed session.

    A re-run (``rerun_attempt`` > 0, queued by guard_price_refresh for
    ``rerun_as_of_date``) runs only while that is still the last completed
    session and the market is closed; otherwise it skips, and the next Beat
    dispatch covers the newer session.
    """
    from app.services.runtime_preferences_service import is_market_enabled_now

    market_code = _normalize_pipeline_market(market)
    bootstrap_blocker = _local_runtime_bootstrap_blocker(market_code)
    if bootstrap_blocker is not None:
        return bootstrap_blocker

    if not is_market_enabled_now(market_code):
        return {
            "status": "skipped",
            "reason": f"market {market_code} is disabled in local runtime preferences",
            "market": market_code,
            "timestamp": datetime.now().isoformat(),
        }

    holder = _market_pipeline_active(market_code)
    if holder:
        return {
            "status": "skipped",
            "reason": "market_pipeline_already_active",
            "market": market_code,
            "active_task_name": holder.get("task_name"),
            "active_task_id": holder.get("task_id"),
            "timestamp": datetime.now().isoformat(),
        }

    calendar = MarketCalendarService()
    today = calendar.market_now(market_code).date()
    if not rerun_attempt and not calendar.is_trading_day(market_code, today):
        return {
            "status": "skipped",
            "reason": "not_trading_day",
            "market": market_code,
            "as_of_date": today.isoformat(),
            "timestamp": datetime.now().isoformat(),
        }

    trading_day = calendar.last_completed_trading_day(market_code)
    if rerun_attempt:
        if trading_day.isoformat() != rerun_as_of_date:
            return {
                "status": "skipped",
                "reason": "rerun_session_advanced",
                "market": market_code,
                "as_of_date": trading_day.isoformat(),
                "rerun_as_of_date": rerun_as_of_date,
                "timestamp": datetime.now().isoformat(),
            }
        if calendar.is_market_open(market_code):
            return {
                "status": "skipped",
                "reason": "market_open",
                "market": market_code,
                "as_of_date": trading_day.isoformat(),
                "timestamp": datetime.now().isoformat(),
            }
    signatures = _build_daily_market_pipeline_signatures(
        market_code, trading_day, rerun_attempt=rerun_attempt
    )
    token = uuid4().hex
    # Built before claiming, so nothing between the claim and the publish's
    # try can raise and leak the claim.
    release = release_daily_market_pipeline_claim.si(market=market_code, token=token).set(
        queue=market_jobs_queue_for_market(market_code)
    )
    # Frozen before claiming, so the claim records the ids it will publish
    # with; the heartbeat reads them to tell whether the chain is alive.
    tracked = chain(*signatures, release)
    task_ids = _chain_task_ids(tracked)
    claims, claimed = _claim_pipeline_chain(market_code, token, task_ids)
    chain_holder = None
    if claimed is False:
        chain_holder = _pipeline_chain_holder(claims, market_code)
        if chain_holder is None:
            # Freed between SET NX and GET: nothing is running, so try once more.
            claims, claimed = _claim_pipeline_chain(market_code, token, task_ids)
            if claimed is False:
                chain_holder = _pipeline_chain_holder(claims, market_code) or {}
    if claimed is False:
        return {
            "status": "skipped",
            "reason": "market_pipeline_already_active",
            "market": market_code,
            # The claim token, not a Celery task id.
            "active_chain_token": chain_holder.get("task_id"),
            "active_since": chain_holder.get("started_at"),
            "timestamp": datetime.now().isoformat(),
        }
    publish_options: dict = {}
    workflow = tracked if claimed else chain(*signatures)
    if claimed:
        publish_options["link_error"] = release

    try:
        task = workflow.apply_async(**publish_options)
    except Exception:
        if claimed:
            try:
                claims.release_pipeline_chain(token, market=market_code)
            except Exception:
                # Keep the publish error; the claim's TTL frees it regardless.
                logger.warning(
                    "Could not release the %s pipeline chain claim", market_code, exc_info=True
                )
        raise
    logger.info(
        "Queued daily market pipeline",
        extra={"market": market_code, "task_id": task.id, "as_of_date": trading_day.isoformat()},
    )
    return {
        "status": "queued",
        "market": market_code,
        "as_of_date": trading_day.isoformat(),
        "task_id": task.id,
        "timestamp": datetime.now().isoformat(),
    }

"""Celery tasks for GEX (Gamma Exposure) batch processing and persistence."""

from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
from datetime import date, datetime
from pathlib import Path
from uuid import uuid4
from zoneinfo import ZoneInfo

from ..celery_app import celery_app as app
from ..database import SessionLocal
from ..models.gex import GexBatch, GexSnapshot
from ..models.stock_universe import StockUniverse
from ..services.options_snapshot_upsert import upsert_snapshot

logger = logging.getLogger(__name__)

_ET = ZoneInfo("America/New_York")


def _trading_date_for(fetched_at_utc: datetime) -> date:
    """US/Eastern trading day for a naive-UTC fetched_at timestamp.

    See migration 20260805_0028 -- avoids bucketing history queries on a raw
    UTC date_trunc, which drifts across DST/midnight boundaries.
    """
    return fetched_at_utc.replace(tzinfo=ZoneInfo("UTC")).astimezone(_ET).date()


def _chunked(items: list[dict], chunk_size: int):
    """Yield fixed-size chunks from a list."""
    if chunk_size <= 0:
        chunk_size = 200
    for i in range(0, len(items), chunk_size):
        yield items[i:i + chunk_size]


def _as_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError:
        return None


def _safe_float(value):
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value):
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _resolve_all_active_tickers(db) -> list[dict]:
    rows = (
        db.query(StockUniverse.symbol, StockUniverse.name)
        .filter(StockUniverse.active_filter())
        .order_by(StockUniverse.symbol.asc())
        .all()
    )
    return [
        {
            "symbol": symbol,
            "yahoo_ticker": symbol,
            "company_name": company_name,
        }
        for symbol, company_name in rows
        if symbol
    ]


def parse_gex_output(json_path: str, batch_id: str, db) -> dict:
    """Parse gex_batch JSON output and persist rows in database."""
    payload = json.loads(Path(json_path).read_text(encoding="utf-8"))
    rows = payload.get("rows") or []

    tickers_ok = 0
    tickers_failed = 0
    gex_values: list[float] = []
    closest_to_flip = None
    closest_distance = None

    for row in rows:
        status = row.get("status", "FAILED")
        distance = _safe_float(row.get("distance_to_flip_pct"))
        total_gex = _safe_float(row.get("total_gex"))

        if status == "OK":
            tickers_ok += 1
            if total_gex is not None:
                gex_values.append(total_gex)
            if distance is not None and (closest_distance is None or abs(distance) < abs(closest_distance)):
                closest_distance = distance
                closest_to_flip = row.get("symbol")
        else:
            tickers_failed += 1

        call_gex = _safe_float(row.get("call_gex"))
        put_gex = _safe_float(row.get("put_gex"))

        fetched_at = datetime.utcnow()
        trading_date_val = _trading_date_for(fetched_at)
        upsert_snapshot(
            db,
            GexSnapshot,
            ticker=row.get("symbol"),
            trading_date=trading_date_val,
            expiration=_as_date(row.get("expiration")),
            values={
                "company_name": row.get("company_name"),
                "status": status,
                "error": row.get("error") if status != "OK" else None,
                "spot_price": _safe_float(row.get("spot_price")),
                "call_oi": _safe_int(row.get("call_oi")) or 0,
                "put_oi": _safe_int(row.get("put_oi")) or 0,
                "call_gex": call_gex,
                "put_gex": put_gex,
                "total_gex": total_gex,
                "flip_level": _safe_float(row.get("flip_level")),
                "distance_to_flip_pct": distance,
                "fetched_at": fetched_at,
                "batch_id": batch_id,
            },
        )

    db.commit()

    avg_total_gex = sum(gex_values) / len(gex_values) if gex_values else None
    return {
        "tickers_total": len(rows),
        "tickers_ok": tickers_ok,
        "tickers_failed": tickers_failed,
        "avg_total_gex": avg_total_gex,
        "closest_to_flip": closest_to_flip,
    }


@app.task(name="app.tasks.gex_tasks.update_gex", bind=True, max_retries=1)
def update_gex(
    self,
    strike_range_pct: float = 20.0,
    max_strikes: int | None = None,
    chain_next: bool = True,
) -> dict:
    """Run GEX batch job and store full-universe results in database.

    chain_next: If True (the default, used when chained from the Max Pain
        run), triggers the daily options update when this run finishes.
        Manual/API-triggered runs pass False so an operator re-running GEX
        alone doesn't unexpectedly cascade into Options too.
    """
    batch_id = str(uuid4())[:8]
    db = SessionLocal()
    config_path = None
    output_path = None

    try:
        tickers = _resolve_all_active_tickers(db)
        if not tickers:
            raise RuntimeError("No active symbols found in stock_universe")

        logger.info("[%s] Starting GEX batch for %s active symbols", batch_id, len(tickers))

        batch = GexBatch(
            id=batch_id,
            status="running",
            started_at=datetime.utcnow(),
            celery_task_id=self.request.id,
            tickers_total=len(tickers),
            strike_range_pct=float(strike_range_pct),
            max_strikes=max_strikes,
        )
        db.add(batch)
        db.commit()

        chunk_size = int(os.getenv("GEX_UNIVERSE_CHUNK_SIZE", "100"))
        retry_minutes = float(os.getenv("GEX_FULL_RETRY_MINUTES", "0.75"))
        retry_interval = float(os.getenv("GEX_FULL_RETRY_INTERVAL_SEC", "3.0"))
        fetch_concurrency = int(os.getenv("GEX_FETCH_CONCURRENCY", "6"))
        yfinance_rate_limit = float(os.getenv("GEX_YFINANCE_RATE_LIMIT", "3.5"))

        total_processed = 0
        total_ok = 0
        total_failed = 0
        gex_sum = 0.0
        gex_count = 0
        closest_to_flip = None
        closest_abs_distance = None

        logger.info(
            "[%s] Running chunked GEX for %s symbols: chunk_size=%s retry_window=%sm retry_interval=%ss "
            "fetch_concurrency=%s rate_limit=%s/s",
            batch_id,
            len(tickers),
            chunk_size,
            retry_minutes,
            retry_interval,
            fetch_concurrency,
            yfinance_rate_limit,
        )

        for chunk_index, ticker_chunk in enumerate(_chunked(tickers, chunk_size), start=1):
            chunk_config_path = None
            chunk_output_path = None
            try:
                config_payload = {
                    "tickers": ticker_chunk,
                    "strike_range_pct": float(strike_range_pct),
                    "max_strikes": max_strikes,
                }

                with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as cfg_file:
                    cfg_file.write(json.dumps(config_payload))
                    chunk_config_path = cfg_file.name

                with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as out_file:
                    chunk_output_path = out_file.name

                args = [
                    "python",
                    "scripts/gex_batch.py",
                    chunk_config_path,
                    "--output",
                    chunk_output_path,
                    "--max-retry-minutes",
                    str(retry_minutes),
                    "--retry-interval-sec",
                    str(retry_interval),
                    "--fetch-concurrency",
                    str(fetch_concurrency),
                    "--rate-limit-per-sec",
                    str(yfinance_rate_limit),
                ]

                logger.info(
                    "[%s] Running GEX chunk %s (%s symbols)",
                    batch_id,
                    chunk_index,
                    len(ticker_chunk),
                )
                result = subprocess.run(
                    args,
                    capture_output=True,
                    text=True,
                    timeout=3 * 3600,
                    cwd=str(Path(__file__).resolve().parents[2]),
                )

                if result.returncode != 0:
                    logger.error(
                        "[%s] GEX chunk %s failed (exit=%s): %s",
                        batch_id,
                        chunk_index,
                        result.returncode,
                        (result.stderr or result.stdout or "Unknown chunk error")[:1000],
                    )
                    total_failed += len(ticker_chunk)
                else:
                    summary = parse_gex_output(chunk_output_path, batch_id, db)
                    processed = summary.get("tickers_total") or len(ticker_chunk)
                    ok = summary.get("tickers_ok") or 0
                    failed = summary.get("tickers_failed") or 0

                    total_processed += processed
                    total_ok += ok
                    total_failed += failed

                    for row in (
                        db.query(GexSnapshot)
                        .filter(GexSnapshot.batch_id == batch_id)
                        .order_by(GexSnapshot.id.desc())
                        .limit(processed)
                        .all()
                    ):
                        if row.status != "OK":
                            continue
                        if row.total_gex is not None:
                            gex_sum += float(row.total_gex)
                            gex_count += 1
                        distance = row.distance_to_flip_pct
                        if distance is None:
                            continue
                        abs_distance = abs(float(distance))
                        if closest_abs_distance is None or abs_distance < closest_abs_distance:
                            closest_abs_distance = abs_distance
                            closest_to_flip = row.ticker

                    batch.tickers_total = len(tickers)
                    batch.tickers_ok = total_ok
                    batch.tickers_failed = total_failed
                    batch.avg_total_gex = (gex_sum / gex_count) if gex_count else None
                    batch.closest_to_flip = closest_to_flip
                    db.commit()

                self.update_state(
                    state='PROGRESS',
                    meta={
                        'current': min(chunk_index * chunk_size, len(tickers)),
                        'total': len(tickers),
                        'percent': round(min(chunk_index * chunk_size, len(tickers)) / len(tickers) * 100, 1),
                        'message': f"Processed chunk {chunk_index}: {total_ok} ok, {total_failed} failed",
                    },
                )
            finally:
                try:
                    if chunk_config_path:
                        Path(chunk_config_path).unlink(missing_ok=True)
                except Exception:
                    pass
                try:
                    if chunk_output_path:
                        Path(chunk_output_path).unlink(missing_ok=True)
                except Exception:
                    pass

        # A run where every chunk's subprocess failed used to still be marked
        # "completed" (per-chunk failures only incremented total_failed, never
        # the overall batch.status), so operators saw a green "Completed"
        # status even though zero real data was produced.
        if total_ok == 0 and total_failed > 0:
            batch.status = "failed"
            batch.error_message = (
                f"All {total_failed} symbols failed across chunked GEX run; "
                "see worker logs for per-chunk errors."
            )
        else:
            batch.status = "completed"
        batch.completed_at = datetime.utcnow()
        batch.tickers_total = len(tickers)
        batch.tickers_ok = total_ok
        batch.tickers_failed = total_failed
        batch.avg_total_gex = (gex_sum / gex_count) if gex_count else None
        batch.closest_to_flip = closest_to_flip
        db.commit()

        return {
            "batch_id": batch_id,
            "status": "success" if batch.status == "completed" else "failed",
            "summary": {
                "tickers_total": len(tickers),
                "tickers_ok": total_ok,
                "tickers_failed": total_failed,
                "avg_total_gex": batch.avg_total_gex,
                "closest_to_flip": closest_to_flip,
            },
        }

    except subprocess.TimeoutExpired:
        logger.error("[%s] GEX batch timed out", batch_id)
        batch = db.query(GexBatch).filter(GexBatch.id == batch_id).first()
        if batch:
            batch.status = "failed"
            batch.error_message = "Timeout after 12 hours"
            db.commit()
        return {"batch_id": batch_id, "status": "timeout"}

    except Exception as exc:
        logger.exception("[%s] Unexpected error in GEX batch", batch_id)
        try:
            batch = db.query(GexBatch).filter(GexBatch.id == batch_id).first()
            if batch:
                batch.status = "failed"
                batch.error_message = str(exc)[:1000]
                db.commit()
        except Exception:
            pass
        return {"batch_id": batch_id, "status": "error", "error": str(exc)}

    finally:
        try:
            if config_path:
                Path(config_path).unlink(missing_ok=True)
        except Exception:
            pass
        try:
            if output_path:
                Path(output_path).unlink(missing_ok=True)
        except Exception:
            pass
        db.close()

        # Chain into the daily options update regardless of this run's
        # outcome, so a bad GEX day doesn't stall the rest of the evening
        # pipeline. Explicit queue routing is required here -- these tasks
        # aren't in celery_app.py's default data_fetch task_routes map, so a
        # plain .delay() would silently land on the general 'celery' queue.
        if chain_next:
            try:
                from .market_queues import data_fetch_queue_for_market
                from .options_tasks import schedule_daily_update as schedule_daily_options_update
                schedule_daily_options_update.apply_async(queue=data_fetch_queue_for_market("US"))
                logger.info("[%s] Chained daily options update", batch_id)
            except Exception:
                logger.exception("[%s] Failed to chain daily options update", batch_id)


@app.task(name="app.tasks.gex_tasks.schedule_daily_update")
def schedule_daily_update() -> dict:
    """Manual-trigger wrapper that refreshes full-universe GEX snapshots.

    No longer reachable from Celery Beat -- the daily run is chained
    straight from update_max_pain's finally block instead (see
    max_pain_tasks.py). This wrapper remains as the "run GEX independently"
    entry point for the Scheduled Tasks dashboard, so it deliberately does
    NOT cascade into the options update (chain_next=False).
    """
    logger.info("Triggering GEX update for full active universe (manual, standalone)")
    # Explicit queue routing, kept even though update_gex now has a
    # celery_app.py task_routes backstop -- routing the wrapper task (this
    # function) would NOT propagate to update_gex, since a plain .delay()
    # from inside a task dispatches on that task's own route, not the
    # caller's queue. Same bug class as max_pain_tasks.schedule_daily_update
    # (see its comment).
    from .market_queues import data_fetch_queue_for_market
    task = update_gex.apply_async(kwargs={"chain_next": False}, queue=data_fetch_queue_for_market("US"))
    return {
        "status": "queued",
        "task_id": task.id,
    }

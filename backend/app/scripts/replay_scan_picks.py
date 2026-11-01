"""Operator CLI: replay Minervini scan picks point-in-time into backtest_replay_picks.

Replays only sessions with a completed Market RS run, which are completed
sessions, and never fetches from a provider, so it is safe while a market is open.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool

from app.database import SessionLocal, engine
from app.services.backtest_replay_service import SessionReplay, replay_market
from app.services.benchmark_registry_service import benchmark_registry


def _parse_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid ISO date: {value}") from exc


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Replay Minervini picks for the Backtest page")
    parser.add_argument(
        "--market", required=True, type=str.upper, choices=benchmark_registry.supported_markets()
    )
    parser.add_argument("--start", type=_parse_date)
    parser.add_argument("--end", type=_parse_date)
    parser.add_argument("--dry-run", action="store_true", help="print picks, write nothing")
    parser.add_argument(
        "--skip-existing", action="store_true", help="skip sessions that already have picks"
    )
    return parser


def _read_only_session_factory():
    """Sessions whose every transaction is READ ONLY. SQLite, the test database, has no such mode."""
    if engine.dialect.name != "postgresql":
        return SessionLocal
    factory = sessionmaker(
        bind=create_engine(
            engine.url,
            poolclass=NullPool,
            connect_args={
                "options": "-c default_transaction_read_only=on -c statement_timeout=300000"
            },
        )
    )
    with factory() as db:
        if db.execute(text("SHOW transaction_read_only")).scalar() != "on":
            raise RuntimeError("the dry run could not open a read-only transaction")
    return factory


def _session_line(replay: SessionReplay) -> str:
    picks = " ".join(f"{pick.symbol}:{pick.minervini_score:.1f}" for pick in replay.picks)
    return (
        f"{replay.as_of_date} run={replay.market_rs_run_id} scored={replay.scored} "
        f"passed={replay.passed} errors={replay.errors} {replay.seconds:.2f}s {picks}"
    ).rstrip()


def main(argv: list[str] | None = None) -> int:
    options = _build_parser().parse_args(argv)
    session_factory = _read_only_session_factory() if options.dry_run else SessionLocal
    mode = "dry run, nothing written" if options.dry_run else "writing picks"
    print(f"Replaying {options.market}: {mode}", flush=True)
    sessions = picks = errors = 0
    for replay in replay_market(
        session_factory,
        options.market,
        start=options.start,
        end=options.end,
        write=not options.dry_run,
        skip_existing=options.skip_existing,
    ):
        print(_session_line(replay), flush=True)
        sessions += 1
        picks += len(replay.picks)
        errors += replay.errors
    print(f"{options.market}: {sessions} sessions, {picks} picks, {errors} error rows")
    return 1 if errors else 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    sys.exit(main())

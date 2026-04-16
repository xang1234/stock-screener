#!/usr/bin/env python3
"""ASIA v2 market canary dress-rehearsal harness.

Seeds the DB-backed launch-gate telemetry into a PostgreSQL rehearsal
database, then runs the normal launch-gate runner with a market-specific
external evidence bundle.

Usage:
    DATABASE_URL=postgresql://user:pass@host:port/dbname \
    python backend/scripts/run_market_canary_rehearsal.py \
        --market JP \
        --evidence-dir data/governance/canary_evidence/jp-2026-04-16

The target database must be PostgreSQL. By default the script runs
``alembic upgrade head`` first and refuses to seed over a non-empty
``market_telemetry_events`` table so the rehearsal stays deterministic.

Exit codes:
    0 — verdict PASS
    1 — environment / setup problem, or verdict NO_GO
    2 — verdict FAIL
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Optional

from sqlalchemy import create_engine, text
from sqlalchemy.engine.url import make_url
from sqlalchemy.orm import sessionmaker

_BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from app.services.governance.gate_artifact import resolve_output_dir, write_artifacts  # noqa: E402
from app.services.governance.launch_gates import GateVerdict, run_all_gates  # noqa: E402
from app.services.telemetry.schema import (  # noqa: E402
    SCHEMA_VERSION,
    benchmark_age_payload,
    completeness_distribution_payload,
    freshness_lag_payload,
    universe_drift_payload,
)
from app.tasks.market_queues import SUPPORTED_MARKETS  # noqa: E402
from app.utils.db_url import redacted_database_url as _redact_url  # noqa: E402


@dataclass(frozen=True)
class MarketCanaryProfile:
    benchmark_symbol: str
    symbols_refreshed: int
    drift_ratios: Dict[str, float]
    low_bucket_ratios: Dict[str, float]


_PROFILE_BY_MARKET: Dict[str, MarketCanaryProfile] = {
    "HK": MarketCanaryProfile(
        benchmark_symbol="^HSI",
        symbols_refreshed=1240,
        drift_ratios={"US": 0.006, "HK": 0.008, "JP": 0.005, "TW": 0.007},
        low_bucket_ratios={"US": 0.01, "HK": 0.02, "JP": 0.03, "TW": 0.04},
    ),
    "JP": MarketCanaryProfile(
        benchmark_symbol="^N225",
        symbols_refreshed=1860,
        drift_ratios={"US": 0.007, "HK": 0.009, "JP": 0.012, "TW": 0.008},
        low_bucket_ratios={"US": 0.02, "HK": 0.02, "JP": 0.03, "TW": 0.04},
    ),
    "TW": MarketCanaryProfile(
        benchmark_symbol="^TWII",
        symbols_refreshed=1480,
        drift_ratios={"US": 0.006, "HK": 0.008, "JP": 0.010, "TW": 0.013},
        low_bucket_ratios={"US": 0.02, "HK": 0.03, "JP": 0.03, "TW": 0.04},
    ),
}

_UNIVERSE_PRIOR_SIZE = {"US": 5000, "HK": 2400, "JP": 3800, "TW": 1800}
_EVIDENCE_FILENAMES = {
    "G5": "multilingual_qa.json",
    "G6": "parity_regression.json",
    "G7": "load_soak.json",
}
_DEFAULT_EXECUTION_MODE = "ephemeral_postgres_dress_rehearsal"


def _parse_now(raw: Optional[str]) -> datetime:
    if not raw:
        return datetime.now(timezone.utc)
    normalized = raw.replace("Z", "+00:00")
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_path(raw: str, root: Path) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    return root / path


def _maybe_relative(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def _profile_for_market(market: str) -> MarketCanaryProfile:
    try:
        return _PROFILE_BY_MARKET[market]
    except KeyError as exc:
        raise ValueError(f"unsupported market {market!r}") from exc


def _default_provenance_note(market: str, evidence_dir: Path, root: Path) -> str:
    return (
        f"Seeded PostgreSQL rehearsal telemetry for {market} and attached external "
        f"evidence from {_maybe_relative(evidence_dir, root)}."
    )


def _load_database_url() -> str:
    database_url = (os.environ.get("DATABASE_URL") or "").strip()
    if not database_url:
        raise ValueError("DATABASE_URL is required")
    os.environ["DATABASE_URL"] = database_url
    return database_url


def _verify_postgres(database_url: str) -> None:
    try:
        backend = make_url(database_url).get_backend_name()
    except Exception as exc:
        raise ValueError(f"invalid DATABASE_URL: {exc}") from exc
    if not backend.startswith("postgresql"):
        raise ValueError(f"only PostgreSQL is supported; got backend {backend!r}")


def _alembic(database_url: str, args: list[str]) -> tuple[int, str]:
    env = {**os.environ, "DATABASE_URL": database_url}
    try:
        proc = subprocess.run(
            ["alembic", *args],
            cwd=str(_BACKEND_DIR),
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return (127, "alembic CLI not found on PATH")
    output = (proc.stdout or "") + (proc.stderr or "")
    return (proc.returncode, output)


def _ensure_migrated(database_url: str) -> None:
    code, output = _alembic(database_url, ["upgrade", "head"])
    if code != 0:
        tail = output.strip().splitlines()[-1] if output.strip() else "upgrade failed"
        raise RuntimeError(f"alembic upgrade head failed: {tail}")


def _engine(database_url: str):
    return create_engine(database_url, pool_pre_ping=True, future=True)


def _telemetry_row_count(database_url: str) -> int:
    engine = _engine(database_url)
    try:
        with engine.connect() as conn:
            return int(
                conn.execute(text("SELECT COUNT(*) FROM market_telemetry_events")).scalar() or 0
            )
    except Exception as exc:
        raise RuntimeError(
            "unable to inspect market_telemetry_events; run migrations or remove --skip-migrate"
        ) from exc
    finally:
        engine.dispose()


def _open_session(database_url: str):
    engine = _engine(database_url)
    factory = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    session = factory()
    session._canary_engine = engine  # keep engine alive for session lifetime
    return session


def _bucket_counts_for_low_ratio(low_ratio: float, *, total: int = 1000) -> Dict[str, int]:
    low = int(round(total * low_ratio))
    remainder = total - low
    mid_25_50 = min(120, remainder)
    remainder -= mid_25_50
    mid_50_75 = min(220, remainder)
    remainder -= mid_50_75
    mid_75_90 = min(260, remainder)
    remainder -= mid_75_90
    return {
        "0-25": low,
        "25-50": mid_25_50,
        "50-75": mid_50_75,
        "75-90": mid_75_90,
        "90-100": remainder,
    }


def _seed_rehearsal_telemetry(database_url: str, market: str, now: datetime) -> Dict[str, object]:
    profile = _profile_for_market(market)
    existing_rows = _telemetry_row_count(database_url)
    if existing_rows:
        raise RuntimeError(
            "market_telemetry_events is not empty; use a fresh rehearsal database"
        )

    rows = []
    for idx, code in enumerate(SUPPORTED_MARKETS):
        prior_size = _UNIVERSE_PRIOR_SIZE[code]
        delta = max(1, int(round(prior_size * profile.drift_ratios[code])))
        rows.append(
            {
                "market": code,
                "metric_key": "universe_drift",
                "schema_version": SCHEMA_VERSION,
                "payload": json.dumps(
                    universe_drift_payload(
                        current_size=prior_size + delta,
                        prior_size=prior_size,
                    )
                ),
                "recorded_at": now - timedelta(hours=6 - idx),
            }
        )
    for idx, code in enumerate(SUPPORTED_MARKETS):
        rows.append(
            {
                "market": code,
                "metric_key": "completeness_distribution",
                "schema_version": SCHEMA_VERSION,
                "payload": json.dumps(
                    completeness_distribution_payload(
                        bucket_counts=_bucket_counts_for_low_ratio(profile.low_bucket_ratios[code]),
                        symbols_total=1000,
                    )
                ),
                "recorded_at": now - timedelta(hours=2, minutes=idx),
            }
        )

    rows.append(
        {
            "market": market,
            "metric_key": "freshness_lag",
            "schema_version": SCHEMA_VERSION,
            "payload": json.dumps(
                freshness_lag_payload(
                    last_refresh_at_epoch=(now - timedelta(minutes=20)).timestamp(),
                    source="prices",
                    symbols_refreshed=profile.symbols_refreshed,
                )
            ),
            "recorded_at": now - timedelta(minutes=20),
        }
    )
    rows.append(
        {
            "market": market,
            "metric_key": "benchmark_age",
            "schema_version": SCHEMA_VERSION,
            "payload": json.dumps(
                benchmark_age_payload(
                    last_warmed_at_epoch=(now - timedelta(minutes=10)).timestamp(),
                    benchmark_symbol=profile.benchmark_symbol,
                )
            ),
            "recorded_at": now - timedelta(minutes=10),
        }
    )

    engine = _engine(database_url)
    try:
        with engine.begin() as conn:
            conn.execute(
                text(
                    "INSERT INTO market_telemetry_events "
                    "(market, metric_key, schema_version, payload, recorded_at) "
                    "VALUES (:market, :metric_key, :schema_version, :payload, :recorded_at)"
                ),
                rows,
            )
    except Exception as exc:
        raise RuntimeError(f"failed to seed canary telemetry: {exc}") from exc
    finally:
        engine.dispose()

    worst_market = max(profile.low_bucket_ratios, key=profile.low_bucket_ratios.__getitem__)
    return {
        "universe_drift_rows": len(SUPPORTED_MARKETS),
        "completeness_rows": len(SUPPORTED_MARKETS),
        "freshness_rows": 1,
        "benchmark_rows": 1,
        "worst_drift_ratio": max(profile.drift_ratios.values()),
        "worst_low_bucket_ratio": profile.low_bucket_ratios[worst_market],
        "worst_low_bucket_market": worst_market,
        "benchmark_symbol": profile.benchmark_symbol,
    }


def _resolve_evidence_bundle(raw_dir: str, root: Path) -> Dict[str, str]:
    evidence_dir = _resolve_path(raw_dir, root)
    bundle: Dict[str, str] = {}
    missing = []
    for gate_id, filename in _EVIDENCE_FILENAMES.items():
        path = evidence_dir / filename
        if path.is_file():
            bundle[gate_id] = str(path)
        else:
            missing.append(filename)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise FileNotFoundError(
            f"evidence bundle incomplete at {evidence_dir}: missing {missing_text}"
        )
    return bundle


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Seed Postgres canary telemetry and run the ASIA launch-gate runner."
    )
    parser.add_argument("--market", required=True, choices=sorted(_PROFILE_BY_MARKET))
    parser.add_argument(
        "--evidence-dir",
        required=True,
        help="Repo-relative or absolute directory containing multilingual_qa.json, parity_regression.json, and load_soak.json.",
    )
    parser.add_argument("--output-dir", default=None, help="Override artifact output directory.")
    parser.add_argument("--skip-migrate", action="store_true", help="Skip alembic upgrade head.")
    parser.add_argument(
        "--now",
        default=None,
        help="Optional ISO-8601 UTC timestamp for deterministic rehearsal output.",
    )
    parser.add_argument(
        "--execution-mode",
        default=_DEFAULT_EXECUTION_MODE,
        help="Provenance label embedded into the launch-gate artifact.",
    )
    parser.add_argument(
        "--provenance-note",
        default=None,
        help="Optional human-readable provenance note embedded into the artifact.",
    )
    args = parser.parse_args()

    root = _project_root()
    now = _parse_now(args.now)

    try:
        database_url = _load_database_url()
        _verify_postgres(database_url)
        evidence = _resolve_evidence_bundle(args.evidence_dir, root)
        if not args.skip_migrate:
            _ensure_migrated(database_url)
        seeded = _seed_rehearsal_telemetry(database_url, args.market, now)
        provenance_note = args.provenance_note or _default_provenance_note(
            args.market, _resolve_path(args.evidence_dir, root), root
        )

        db = _open_session(database_url)
        try:
            report = run_all_gates(
                project_root=root,
                db=db,
                external_evidence=evidence,
                now=now,
                execution_mode=args.execution_mode,
                provenance_note=provenance_note,
            )
        finally:
            db.close()
            engine = getattr(db, "_canary_engine", None)
            if engine is not None:
                engine.dispose()

        out_dir = resolve_output_dir(args.output_dir)
        paths = write_artifacts(report, out_dir)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1

    print(f"Market: {args.market}")
    print(f"Database: {_redact_url(database_url)}")
    print(f"Evidence dir: {_maybe_relative(_resolve_path(args.evidence_dir, root), root)}")
    print(
        "Seeded telemetry: "
        f"universe_drift={seeded['universe_drift_rows']} rows, "
        f"completeness={seeded['completeness_rows']} rows, "
        f"freshness={seeded['freshness_rows']} row, "
        f"benchmark_age={seeded['benchmark_rows']} row"
    )
    print(
        "Telemetry envelope: "
        f"worst_drift_ratio={seeded['worst_drift_ratio']:.3f}, "
        f"worst_low_bucket_ratio={seeded['worst_low_bucket_ratio']:.2f} "
        f"on {seeded['worst_low_bucket_market']}, "
        f"benchmark={seeded['benchmark_symbol']}"
    )
    print(f"Verdict: {report.verdict.upper()}")
    print(f"Execution mode: {report.execution_mode}")
    print(f"Provenance: {report.provenance_note}")
    print(f"Content hash: {report.content_hash}")
    print("Artifacts:")
    for key, value in paths.items():
        print(f"  {key}: {value}")

    if report.verdict == GateVerdict.PASS:
        return 0
    if report.verdict == GateVerdict.NO_GO:
        return 1
    return 2


if __name__ == "__main__":
    sys.exit(main())

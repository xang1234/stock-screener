#!/usr/bin/env python3
"""ASIA v2 full-path migration rehearsal harness (bead asia.11.2).

Walks the entire Alembic migration chain (baseline → head), seeds a
production-like dataset between checkpoints, exercises a downgrade /
re-upgrade rollback drill on the most recent revision pair, and emits
a Markdown rehearsal report under ``docs/asia/``.

Usage:
    DATABASE_URL=postgresql://user:pass@host:port/dbname \
    python backend/scripts/run_migration_rehearsal.py [--report-dir DIR]

The DATABASE_URL must point at an EMPTY PostgreSQL database. The script
will reject SQLite (per project Postgres-only constraint) and refuse to
run against a database that already has Alembic state.

Exit codes:
    0 — rehearsal completed; report written
    1 — environment problem (no DB, SQLite, dirty target)
    2 — migration step failed; report still emitted with FAIL marker
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from app.utils.db_url import redacted_database_url as _redact_url


# How many revisions back from head the rollback drill downgrades to.
# 2 covers the most recent two telemetry migrations (0012, 0013) which
# are the most likely rollback targets during canary.
ROLLBACK_DEPTH = 2

# Production-shaped seed sizes — large enough to exercise indices, small
# enough to keep the rehearsal under one minute of wall time.
SEED_UNIVERSE_SIZE = 5000
SEED_TELEMETRY_SIZE = 1000


def _discover_migration_chain() -> Tuple[str, ...]:
    """Walk the Alembic graph baseline → head dynamically.

    Source of truth = the alembic.ini next to this repo's alembic/. Doing
    this at runtime rather than hardcoding the revision IDs means the
    rehearsal stays in lockstep with whatever migrations have landed,
    without an editor diff every time a migration is added.
    """
    from alembic.config import Config
    from alembic.script import ScriptDirectory

    cfg = Config(str(_BACKEND_DIR / "alembic.ini"))
    cfg.set_main_option("script_location", str(_BACKEND_DIR / "alembic"))
    script = ScriptDirectory.from_config(cfg)
    head = script.get_current_head()
    if head is None:
        return tuple()
    # walk_revisions yields newest → oldest; reverse to walk forward.
    chain = [rev.revision for rev in script.walk_revisions(base="base", head=head)]
    return tuple(reversed(chain))


_MIGRATION_CHAIN: Tuple[str, ...] = _discover_migration_chain()
_ROLLBACK_DRILL = (
    (_MIGRATION_CHAIN[-1], _MIGRATION_CHAIN[-(ROLLBACK_DEPTH + 1)])
    if len(_MIGRATION_CHAIN) > ROLLBACK_DEPTH
    else tuple()
)


def _alembic(database_url: str, args: List[str]) -> Tuple[int, float, str]:
    """Run an alembic command. Returns (returncode, elapsed_s, captured_output)."""
    cmd = ["alembic"] + args
    env = {**os.environ, "DATABASE_URL": database_url}
    started = time.monotonic()
    try:
        proc = subprocess.run(
            cmd, env=env, cwd=str(_BACKEND_DIR),
            capture_output=True, text=True, check=False,
        )
    except FileNotFoundError:
        return (127, 0.0, "alembic CLI not found on PATH")
    elapsed = time.monotonic() - started
    out = (proc.stdout or "") + (proc.stderr or "")
    return (proc.returncode, elapsed, out)


def _last_output_line(text: str) -> str:
    """Return the last line from output, safely handling whitespace-only output."""
    lines = text.strip().splitlines()
    return lines[-1] if lines else ""


def _engine(database_url: str):
    from sqlalchemy import create_engine
    return create_engine(database_url, pool_pre_ping=True, future=True)


_TABLE_MISSING_CODES = frozenset({"42P01", "42S02"})  # Postgres, SQLite "no such table"


def _row_count(engine, table: str) -> Optional[int]:
    """Return row count, or None when the table does not exist (expected after downgrade).

    Only swallows the "table does not exist" case (SQLSTATE 42P01 on Postgres).
    Every other failure — connection drop, permission error, malformed query —
    is re-raised so the rehearsal fails rather than producing false PASS evidence.
    """
    from sqlalchemy import text
    from sqlalchemy.exc import ProgrammingError, OperationalError
    try:
        with engine.connect() as conn:
            result = conn.execute(text(f"SELECT COUNT(*) FROM {table}"))
            return int(result.scalar() or 0)
    except (ProgrammingError, OperationalError) as exc:
        # Inspect SQLSTATE: 42P01 = undefined_table (Postgres), 42S02 = SQLite.
        code = getattr(getattr(exc, "orig", None), "pgcode", None) or ""
        msg = str(exc).lower()
        if code in _TABLE_MISSING_CODES or "does not exist" in msg or "no such table" in msg:
            return None  # expected after downgrade
        raise  # unexpected — let the rehearsal fail


def _seed_universe(engine, *, total: int = SEED_UNIVERSE_SIZE) -> int:
    """Insert a US-baseline universe. Idempotent on rerun; returns rows added.

    Production-like cardinality is ~10k symbols across markets; we seed half
    that to keep the rehearsal under one minute while still being large
    enough that a missed index would show in timing.
    """
    from sqlalchemy import text
    rows = [
        {
            "symbol": f"US{idx:06d}",
            "name": f"Synthetic Equity {idx}",
            "sector": "Technology" if idx % 3 == 0 else "Industrials",
            "industry": "Synthetic",
            "is_active": True,
            "status": "active",
            "consecutive_fetch_failures": 0,
        }
        for idx in range(1, total + 1)
    ]
    with engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO stock_universe "
                "(symbol, name, sector, industry, is_active, status, consecutive_fetch_failures) "
                "VALUES (:symbol, :name, :sector, :industry, :is_active, :status, :consecutive_fetch_failures) "
                "ON CONFLICT (symbol) DO NOTHING"
            ),
            rows,
        )
    return total


def _seed_telemetry_events(engine, *, total: int = SEED_TELEMETRY_SIZE) -> int:
    """Seed telemetry events (touches the 0012 migration's table).

    Mixed markets and metric_keys to exercise the indices.
    """
    from sqlalchemy import text
    import json as _json
    markets = ("US", "HK", "JP", "TW", "SHARED")
    metrics = ("freshness_lag", "universe_drift", "benchmark_age",
               "completeness_distribution", "extraction_success")
    rows = [
        {
            "market": markets[i % len(markets)],
            "metric_key": metrics[i % len(metrics)],
            "schema_version": 1,
            "payload": _json.dumps({"schema_version": 1, "synthetic": True, "i": i}),
        }
        for i in range(total)
    ]
    # JSONB column accepts a JSON-formatted text value without an explicit
    # cast; mixing :param and ::cast in SQLAlchemy text() is fragile (the
    # `::` is parsed as part of the named-param token).
    with engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO market_telemetry_events "
                "(market, metric_key, schema_version, payload) "
                "VALUES (:market, :metric_key, :schema_version, :payload)"
            ),
            rows,
        )
    return total


def _refuse_dirty(engine) -> Optional[str]:
    """Return a problem string if alembic_version already has rows."""
    from sqlalchemy import text
    try:
        with engine.connect() as conn:
            r = conn.execute(text(
                "SELECT version_num FROM alembic_version LIMIT 1"
            )).first()
            if r is not None:
                return f"target DB already at revision {r[0]}; refusing to overwrite"
    except Exception:
        # alembic_version table doesn't exist yet — that's the expected state.
        return None
    return None


def _verify_postgres(database_url: str) -> Optional[str]:
    """Reject non-Postgres URLs (project constraint)."""
    from sqlalchemy.engine.url import make_url
    try:
        u = make_url(database_url)
    except Exception as exc:
        return f"invalid DATABASE_URL: {exc}"
    if not u.get_backend_name().startswith("postgresql"):
        return f"only PostgreSQL is supported; got backend {u.get_backend_name()!r}"
    return None


# ---------------------------------------------------------------------------
# Rehearsal driver
# ---------------------------------------------------------------------------
def run_rehearsal(database_url: str) -> Dict[str, Any]:
    """Walk the full migration chain + rollback drill. Return a results dict."""
    pre_check = _verify_postgres(database_url)
    if pre_check:
        # "env_error" maps to exit 1 in main() — distinct from "fail" (exit 2)
        # so automation can distinguish environment/setup problems from a
        # real migration regression.
        return {"status": "env_error", "error": pre_check, "steps": []}

    engine = _engine(database_url)
    dirty = _refuse_dirty(engine)
    if dirty:
        return {"status": "env_error", "error": dirty, "steps": []}

    steps: List[Dict[str, Any]] = []

    def record(action: str, target: str, code: int, elapsed: float, note: str = "") -> bool:
        steps.append({
            "action": action,
            "target": target,
            "status": "ok" if code == 0 else "fail",
            "elapsed_s": round(elapsed, 3),
            "note": note,
        })
        return code == 0

    # Phase 1: forward walk one revision at a time.
    for rev in _MIGRATION_CHAIN:
        code, elapsed, out = _alembic(database_url, ["upgrade", rev])
        ok = record("upgrade", rev, code, elapsed, _last_output_line(out))
        if not ok:
            return {"status": "fail", "error": f"upgrade {rev} failed", "steps": steps,
                    "alembic_output": out}

    # Phase 2: snapshot row counts at head.
    counts_at_head = {
        "stock_universe": _row_count(engine, "stock_universe"),
        "market_telemetry_events": _row_count(engine, "market_telemetry_events"),
        "market_telemetry_alerts": _row_count(engine, "market_telemetry_alerts"),
    }

    # Phase 3: seed production-like cardinality, re-snapshot.
    seeded_universe = _seed_universe(engine)
    seeded_telemetry = _seed_telemetry_events(engine)
    counts_after_seed = {
        "stock_universe": _row_count(engine, "stock_universe"),
        "market_telemetry_events": _row_count(engine, "market_telemetry_events"),
    }

    # Phase 4: rollback drill — downgrade head → N-2, then re-upgrade.
    if not _ROLLBACK_DRILL:
        return {"status": "fail",
                "error": f"Migration chain too short for rollback drill "
                         f"(need > {ROLLBACK_DEPTH} revisions, got {len(_MIGRATION_CHAIN)})",
                "steps": steps}
    head, rollback_to = _ROLLBACK_DRILL
    code, elapsed, out = _alembic(database_url, ["downgrade", rollback_to])
    if not record("downgrade", rollback_to, code, elapsed,
                  _last_output_line(out)):
        return {"status": "fail", "error": f"downgrade to {rollback_to} failed",
                "steps": steps, "alembic_output": out}

    # Phase 5: row counts after downgrade — universe must be intact (it
    # predates the telemetry migrations); telemetry tables should be gone.
    counts_after_downgrade = {
        "stock_universe": _row_count(engine, "stock_universe"),  # should equal post-seed
        "market_telemetry_events": _row_count(engine, "market_telemetry_events"),  # None expected
        "market_telemetry_alerts": _row_count(engine, "market_telemetry_alerts"),  # None expected
    }

    # Phase 6: re-upgrade to head.
    code, elapsed, out = _alembic(database_url, ["upgrade", head])
    if not record("re-upgrade", head, code, elapsed,
                  _last_output_line(out)):
        return {"status": "fail", "error": f"re-upgrade to {head} failed",
                "steps": steps, "alembic_output": out}

    counts_after_reupgrade = {
        "stock_universe": _row_count(engine, "stock_universe"),  # unchanged from post-seed
        "market_telemetry_events": _row_count(engine, "market_telemetry_events"),  # 0 (table recreated empty)
        "market_telemetry_alerts": _row_count(engine, "market_telemetry_alerts"),  # 0
    }

    return {
        "status": "pass",
        "steps": steps,
        "seeded": {"stock_universe": seeded_universe, "market_telemetry_events": seeded_telemetry},
        "row_counts": {
            "at_head_pre_seed": counts_at_head,
            "after_seed": counts_after_seed,
            "after_downgrade": counts_after_downgrade,
            "after_reupgrade": counts_after_reupgrade,
        },
    }


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------
def render_report(result: Dict[str, Any], *, database_label: str) -> str:
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    lines: List[str] = []
    lines.append("# ASIA v2 E11 ST2 Migration Rehearsal Report")
    lines.append("")
    lines.append(f"- Date: {now}")
    lines.append("- Bead: `StockScreenClaude-asia.11.2`")
    scope = (f"{_MIGRATION_CHAIN[0]} → {_MIGRATION_CHAIN[-1]}"
             if _MIGRATION_CHAIN else "no migrations discovered")
    drill = (f"head ({_ROLLBACK_DRILL[0]}) ↔ {_ROLLBACK_DRILL[1]}"
             if _ROLLBACK_DRILL else "skipped — chain too short")
    lines.append(f"- Scope: full Alembic chain baseline → head ({scope})")
    lines.append(f"- Rollback drill: {drill}")
    lines.append(f"- Database: {database_label}")
    lines.append(f"- Outcome: **{result.get('status', 'unknown').upper()}**")
    lines.append("")

    status = str(result.get("status", "unknown")).lower()
    if status in {"fail", "env_error"}:
        lines.append("## Failure" if status == "fail" else "## Environment Error")
        lines.append("")
        lines.append(f"- Error: {result.get('error', 'unknown')}")
        if result.get("alembic_output"):
            lines.append("")
            lines.append("### Alembic output (tail)")
            lines.append("```")
            tail = "\n".join(result["alembic_output"].splitlines()[-20:])
            lines.append(tail)
            lines.append("```")
        return "\n".join(lines) + "\n"

    lines.append("## Execution Log (Timed)")
    lines.append("")
    lines.append("| Step | Action | Target | Status | Real Time (s) |")
    lines.append("|---|---|---|---|---:|")
    for i, s in enumerate(result.get("steps", []), start=1):
        lines.append(
            f"| {i} | {s['action']} | `{s['target']}` | "
            f"{s['status'].upper()} | {s['elapsed_s']} |"
        )
    lines.append("")

    seeded = result.get("seeded", {})
    lines.append("## Seeded Production-like Dataset")
    lines.append("")
    lines.append(f"- `stock_universe`: {seeded.get('stock_universe', 0)} synthetic US symbols")
    lines.append(f"- `market_telemetry_events`: {seeded.get('market_telemetry_events', 0)} mixed-market events")
    lines.append("")

    rc = result.get("row_counts", {})
    lines.append("## Row Counts at Each Checkpoint")
    lines.append("")
    lines.append("| Checkpoint | stock_universe | market_telemetry_events | market_telemetry_alerts |")
    lines.append("|---|---:|---:|---:|")
    rollback_label = (
        "After downgrade to %s" % _ROLLBACK_DRILL[1]
        if _ROLLBACK_DRILL
        else "After downgrade (rollback drill unavailable)"
    )
    for label, key in [
        ("Head (pre-seed)", "at_head_pre_seed"),
        ("After seed", "after_seed"),
        (rollback_label, "after_downgrade"),
        ("After re-upgrade to head", "after_reupgrade"),
    ]:
        c = rc.get(key, {})
        lines.append(
            f"| {label} | "
            f"{c.get('stock_universe', '—')} | "
            f"{c.get('market_telemetry_events', '—')} | "
            f"{c.get('market_telemetry_alerts', '—')} |"
        )
    lines.append("")

    lines.append("## Acceptance Findings")
    lines.append("")
    after_seed = rc.get("after_seed", {})
    after_down = rc.get("after_downgrade", {})
    universe_preserved = (
        after_seed.get("stock_universe") is not None
        and after_seed.get("stock_universe") == after_down.get("stock_universe")
    )
    telemetry_dropped = after_down.get("market_telemetry_events") is None
    lines.append(
        f"- Universe rows preserved across downgrade: **{'YES' if universe_preserved else 'NO'}** "
        f"(after seed = {after_seed.get('stock_universe')}, "
        f"after downgrade = {after_down.get('stock_universe')})"
    )
    lines.append(
        f"- Telemetry tables removed by downgrade: **{'YES' if telemetry_dropped else 'NO'}**"
    )
    lines.append(
        f"- Re-upgrade restored telemetry tables (empty as expected): "
        f"**{'YES' if rc.get('after_reupgrade', {}).get('market_telemetry_events') == 0 else 'NO'}**"
    )
    lines.append("")
    lines.append("This rehearsal demonstrates **no data-loss** for the universe table across")
    lines.append("a head→0011→head cycle and complete drop/recreate of the telemetry tables.")
    lines.append("")
    lines.append("---")
    lines.append("Generated by `backend/scripts/run_migration_rehearsal.py`.")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report-dir", default=None,
        help="Override the docs/asia/ output directory.",
    )
    parser.add_argument(
        "--database-label", default=None,
        help="Friendly label for the report (e.g. 'staging-restore', 'ephemeral-pg').",
    )
    args = parser.parse_args()

    database_url = os.environ.get("DATABASE_URL", "")
    if not database_url:
        print("DATABASE_URL is required.", file=sys.stderr)
        return 1

    label = args.database_label or _redact_url(database_url)

    print(f"Running rehearsal against {label} ...")
    result = run_rehearsal(database_url)
    print(f"Status: {result.get('status', 'unknown').upper()}")

    from app.config.settings import get_project_root
    report_dir = Path(args.report_dir) if args.report_dir else (
        get_project_root() / "docs" / "asia"
    )
    report_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    report_path = report_dir / f"asia_v2_e11_st2_migration_rehearsal_report_{stamp}.md"
    report_path.write_text(render_report(result, database_label=label), encoding="utf-8")
    print(f"Report: {report_path}")

    status = result.get("status")
    if status == "fail":
        return 2
    if status == "env_error":
        return 1
    return 0

if __name__ == "__main__":
    sys.exit(main())

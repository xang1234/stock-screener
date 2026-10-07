"""Report how far each static-site market lags its last completed session (#484).

Runs in the static-site publisher after the bundle is combined, so it sees what
the site will actually serve (the newest valid artifact per market, or nothing). Writes a table
to the run summary and emits a GitHub annotation per lagging market. It never
fails: the site still deploys, and the annotations make staleness visible.

Freshness is judged against ``last_completed_trading_day`` at report time, so a
market whose session is still in progress is not counted behind for today. No
prices are read or fetched.
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from app.scripts.validate_static_market_artifacts import parse_selected_markets
from app.services.daily_price_bundle_contract import latest_daily_price_manifest_name
from app.services.market_calendar_service import MarketCalendarService
from app.services.static_market_artifact_contract import STATIC_MARKET_METADATA_FILENAME
from app.services.static_site_export_service import STATIC_SUPPORTED_MARKETS

DEFAULT_MAX_SESSIONS_BEHIND = 3


@dataclass(frozen=True)
class MarketFreshness:
    market: str
    served_as_of: date | None
    expected_session: date | None
    sessions_behind: int | None
    price_bundle_as_of: date | None
    bundle_sessions_behind: int | None
    source: str
    reason: str | None
    level: str  # ok | warning | error


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def _as_date(value: Any) -> date | None:
    try:
        return date.fromisoformat(str(value)[:10]) if value else None
    except ValueError:
        return None


def _current_artifact_as_of(artifacts_dir: Path, market: str) -> date | None:
    metadata_path = next(
        (artifacts_dir / f"static-market-{market}").rglob(STATIC_MARKET_METADATA_FILENAME),
        None,
    )
    entry = (_read_json(metadata_path) or {}).get("entry") if metadata_path else None
    return _as_date(entry.get("as_of_date")) if isinstance(entry, dict) else None


def _sessions_behind(calendar, market: str, as_of: date | None, expected: date | None) -> int | None:
    if as_of is None or expected is None:
        return None
    if as_of >= expected:
        return 0
    return len(calendar.trading_days(market, as_of + timedelta(days=1), expected))


def build_freshness_rows(
    *,
    manifest: Mapping[str, Any],
    markets: Sequence[str],
    artifacts_dir: Path,
    price_manifest_dir: Path,
    calendar,
    max_sessions_behind: int,
    selected_markets: Collection[str] = (),
) -> list[MarketFreshness]:
    served = manifest.get("markets") or {}
    rows = []
    for market in markets:
        entry = served.get(market)
        served_as_of = _as_date(entry.get("as_of_date")) if isinstance(entry, dict) else None
        try:
            expected = calendar.last_completed_trading_day(market)
        except Exception:  # noqa: BLE001 - e.g. calendar coverage expired; report, don't fail
            expected = None
        bundle = _read_json(price_manifest_dir / latest_daily_price_manifest_name(market)) or {}
        bundle_as_of = _as_date(bundle.get("as_of_date"))
        behind = _sessions_behind(calendar, market, served_as_of, expected)
        bundle_behind = _sessions_behind(calendar, market, bundle_as_of, expected)

        # Status/diagnostics exist only for markets built in this run, and not
        # even then if the build job died before uploading them.
        built_this_run = market in selected_markets
        status = _read_json(artifacts_dir / f"static-market-status-{market}" / "status.json")
        diagnostics = _read_json(
            artifacts_dir / f"static-market-diagnostics-{market}" / "snapshot-failure.json"
        ) or {}
        if served_as_of is None:
            source = "not served"
        elif status is None:
            source = "fallback" if built_this_run else "previous run"
        elif status.get("has_current_artifact") and served_as_of == _current_artifact_as_of(
            artifacts_dir, market
        ):
            source = "current"
        else:
            # No current artifact, or the combiner preferred a newer fallback
            # over a current export that rewound to an older as-of date.
            source = "fallback"
        reason = diagnostics.get("reason") or (status or {}).get("reason")
        if reason is None and status is None and built_this_run:
            reason = "no status from this run's build"

        lag = max((n for n in (behind, bundle_behind) if n is not None), default=0)
        if served_as_of is None or lag > max_sessions_behind:
            level = "error"
        elif lag >= 1 or expected is None or bundle_as_of is None:
            level = "warning"
        else:
            level = "ok"
        rows.append(
            MarketFreshness(
                market=market,
                served_as_of=served_as_of,
                expected_session=expected,
                sessions_behind=behind,
                price_bundle_as_of=bundle_as_of,
                bundle_sessions_behind=bundle_behind,
                source=source,
                reason=reason,
                level=level,
            )
        )
    return rows


def _cell(value: Any) -> str:
    return "—" if value is None else str(value).replace("|", "\\|").replace("\n", " ")


def summary_markdown(rows: Sequence[MarketFreshness]) -> str:
    icon = {"ok": "✅", "warning": "⚠️", "error": "❌"}
    lines = [
        "## Static site market freshness",
        "",
        "| Market | Served as of | Expected session | Sessions behind | Price bundle | Bundle behind | Source | Reason | |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row.market} | {_cell(row.served_as_of)} | {_cell(row.expected_session)} | "
            f"{_cell(row.sessions_behind)} | {_cell(row.price_bundle_as_of)} | "
            f"{_cell(row.bundle_sessions_behind)} | {row.source} | {_cell(row.reason)} | {icon[row.level]} |"
        )
    return "\n".join(lines) + "\n"


def annotation(row: MarketFreshness) -> str:
    if row.served_as_of is None:
        message = f"{row.market} is not on the site: no current or fallback artifact"
    elif row.sessions_behind is None:
        message = f"{row.market} freshness unknown: no expected session from the market calendar"
    else:
        message = (
            f"{row.market} is {row.sessions_behind} session(s) behind "
            f"(served {row.served_as_of}, expected {row.expected_session})"
        )
    if row.price_bundle_as_of is None:
        message += "; price bundle manifest unavailable"
    elif row.bundle_sessions_behind:
        message += f"; price bundle {row.bundle_sessions_behind} session(s) behind ({row.price_bundle_as_of})"
    if row.reason:
        message += f"; reason: {row.reason}"
    # Workflow-command escaping: artifact text must not split or inject commands.
    escaped = message.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")
    return f"::{row.level} title=Static site freshness::{escaped}"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--artifacts-dir", type=Path, required=True)
    parser.add_argument("--price-manifest-dir", type=Path, required=True)
    parser.add_argument("--max-sessions-behind", type=int, default=DEFAULT_MAX_SESSIONS_BEHIND)
    parser.add_argument("--selected-markets", default="[]", help="JSON list of markets built this run")
    args = parser.parse_args(argv)

    rows = build_freshness_rows(
        manifest=_read_json(args.manifest) or {},
        markets=STATIC_SUPPORTED_MARKETS,
        artifacts_dir=args.artifacts_dir,
        price_manifest_dir=args.price_manifest_dir,
        calendar=MarketCalendarService(),
        max_sessions_behind=args.max_sessions_behind,
        selected_markets=parse_selected_markets(args.selected_markets),
    )
    summary = summary_markdown(rows)
    print(summary)
    for row in rows:
        if row.level != "ok":
            print(annotation(row))
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a", encoding="utf-8") as fh:
            fh.write(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

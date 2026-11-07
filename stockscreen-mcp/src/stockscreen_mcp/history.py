"""The market history archive: one immutable file per published session, compared two at a time.

backend/app/scripts/market_history.py writes it. The tests build their fixtures with that
writer, so a format change there fails here.
"""

from __future__ import annotations

import gzip
import http.client
import json
import os
import threading
import time
import urllib.error
import urllib.request
import zlib
from collections import Counter, OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from typing import Any
from urllib.parse import urljoin

from mcp.server.mcpserver.exceptions import ToolError

from . import __version__
from .data import MANIFEST_TTL_SECONDS
from .screens import expression_fields, row_value

DEFAULT_HISTORY_URL = "https://github.com/xang1234/stock-screener/releases/download/"
SESSION_SCHEMA = "market-history-v1"
INDEX_SCHEMA = "market-history-index-v1"
INDEX_ASSET = "index.json"
# A projected US session held 5.5 MB on 2026-11-06, so four stay small.
MAX_CACHED_SESSIONS = 4
PRE_STAGE_FIX = "artifact_pre_stage_fix"
# The fields what_changed reports from and to for a named symbol, in this order.
TRACKED_FIELDS = (
    "current_price", "rs_rating", "rs_rating_1m", "rs_rating_3m", "rs_rating_12m", "stage", "composite_score",
    "minervini_score", "canslim_score", "ibd_industry_group", "ibd_group_rank", "eps_growth_qq",
    "sales_growth_qq", "growth_metric_basis", "week_52_high_distance", "market_cap_usd",
)
# HK and JP rows can switch between a year-over-year and a quarter-over-quarter comparison.
GROWTH_FIELDS = ("eps_growth_qq", "sales_growth_qq")
BASIS = TRACKED_FIELDS.index("growth_metric_basis")
# How a symbol with a row in both sessions can move through a screen. The _with_ moves are those where a field
# the screen's rules read went from null to a value or back, so data coverage may have moved it, not the stock.
MOVES = ("entered", "entered_with_new_data", "left", "left_with_lost_data")
UPGRADE = (
    "Update the client's command to `uvx stockscreen-mcp@latest` and restart it, or install the latest bundle "
    "if you use one."
)


def release_tag(market: str) -> str:
    return f"market-history-{market.lower()}-v1"


def session_asset(market: str, as_of_date: str) -> str:
    return f"{market.lower()}-{as_of_date}.json.gz"


def fetch_bytes(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": f"stockscreen-mcp/{__version__}"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


@dataclass(frozen=True)
class Screen:
    name: str | None
    definition: str | None
    symbols: frozenset[str]
    fields: frozenset[str]  # the fields its rules read


ABSENT = Screen(name=None, definition=None, symbols=frozenset(), fields=frozenset())


@dataclass(frozen=True)
class Session:
    as_of_date: str
    published_at: str
    origin: str
    rs_universe_size: int | None
    rs_formula_version: str | None
    screens: dict[str, Screen]
    # symbol -> its TRACKED_FIELDS values in that order; every published row, so presence is known.
    rows: dict[str, tuple[Any, ...]]
    # symbol -> the fields some screen's rules read that are null on its row.
    nulls: dict[str, frozenset[str]]


def _most_common(values: Any) -> Any:
    counts = Counter(value for value in values if value is not None)
    return counts.most_common(1)[0][0] if counts else None


def parse_index(document: Any, market: str) -> list[str]:
    if document.get("schema_version") != INDEX_SCHEMA:
        raise ToolError(
            f"The {market} history index uses schema {document.get('schema_version')!r}, which this version of "
            f"stockscreen-mcp cannot read. {UPGRADE}"
        )
    return sorted(document["sessions"])


def parse_session(document: Any, market: str, as_of_date: str) -> Session:
    """Keep only what a comparison reads, since a US file holds about 10k rows of every published field."""
    if document.get("schema_version") != SESSION_SCHEMA:
        raise ToolError(
            f"The {market} {as_of_date} session uses schema {document.get('schema_version')!r}, which this version "
            f"of stockscreen-mcp cannot read. {UPGRADE}"
        )
    if (document["market"], document["as_of_date"]) != (market, as_of_date):
        raise ToolError(f"The {market} {as_of_date} file holds {document['market']} {document['as_of_date']}.")
    columns = {name: i for i, name in enumerate(document["columns"])}
    rows = document["rows"]
    symbol = columns["symbol"]
    tracked = [columns.get(field) for field in TRACKED_FIELDS]
    screens = {
        preset: Screen(
            entry["name"], entry["definition"], frozenset(entry["symbols"]),
            frozenset(expression_fields(entry["filter_expression"])),
        )
        for preset, entry in document["presets"].items()
    }
    screened = sorted(set().union(*(screen.fields for screen in screens.values())))
    shared: dict[frozenset[str], frozenset[str]] = {}

    def nulls(row: list[Any]) -> frozenset[str]:
        record = dict(zip(document["columns"], row))
        found = frozenset(field for field in screened if row_value(record, field) is None)
        return shared.setdefault(found, found)

    def column(name: str) -> list[Any]:
        return [] if name not in columns else [row[columns[name]] for row in rows]

    return Session(
        as_of_date=as_of_date,
        published_at=document["published_at"],
        origin=document["origin"],
        rs_universe_size=_most_common(column("rs_universe_size")),
        rs_formula_version=_most_common(column("rs_formula_version")),
        screens=screens,
        rows={row[symbol]: tuple(None if i is None else row[i] for i in tracked) for row in rows},
        nulls={row[symbol]: nulls(row) for row in rows},
    )


class History:
    """Reads the archive Releases, caching each market's index for the manifest's TTL and the last few sessions."""

    def __init__(
        self,
        base_url: str | None = None,
        fetch: Callable[[str], bytes] = fetch_bytes,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        base = base_url or os.environ.get("STOCKSCREEN_HISTORY_URL") or DEFAULT_HISTORY_URL
        self.base_url = base.rstrip("/") + "/"
        self._fetch = fetch
        self._clock = clock
        # ponytail: one lock for every load, as in StaticData; per-market locks if concurrent calls ever matter.
        self._lock = threading.Lock()
        self._indexes: dict[str, tuple[float, list[str]]] = {}
        self._sessions: OrderedDict[tuple[str, str], Session] = OrderedDict()

    def _get(self, url: str) -> bytes:
        try:
            return self._fetch(url)
        except urllib.error.HTTPError:
            raise
        # OSError covers URLError and timeouts, ValueError a malformed STOCKSCREEN_HISTORY_URL.
        except (OSError, ValueError, http.client.HTTPException) as exc:
            raise ToolError(f"Could not reach the market history at {url}: {exc!r}") from exc

    def _read(self, url: str, parse: Callable[[Any], Any]) -> Any:
        body = self._get(url)
        # Release assets redirect to a blob store that serves .json.gz as plain bytes, with no Content-Encoding.
        try:
            return parse(json.loads(gzip.decompress(body) if body[:2] == b"\x1f\x8b" else body))
        except (OSError, EOFError, zlib.error, ValueError, KeyError, TypeError, IndexError, AttributeError) as exc:
            raise ToolError(f"Could not read the market history file {url}: {exc!r}") from exc

    def _missing_index(self, market: str, exc: urllib.error.HTTPError) -> ToolError:
        if exc.code != 404:
            return ToolError(f"The {market} history index is unavailable ({exc.code}); try again in a minute.")
        # Release downloads 404 alike for a missing Release and a missing asset, so ask the Release page.
        # ponytail: a GET of about 150 KB, only while an existing Release has no index; HEAD if this gets hot.
        try:
            self._get(urljoin(self.base_url, f"../tag/{release_tag(market)}"))
        except urllib.error.HTTPError as probe:
            if probe.code != 404:
                return ToolError(
                    f"The {market} history index and its Release page are unavailable ({probe.code}); try again in "
                    "a minute."
                )
            return ToolError(
                f"The market history archive has not started for {market}, so there are no sessions to compare yet. "
                f"It records every publish from its first run on; try again after {market}'s next publish."
            )
        return ToolError(
            f"The {market} history index is missing from its Release. A publish rewriting it is back within a minute, "
            "but if a publish failed after creating the Release, the index stays missing until the next publish. Try "
            "again in a few minutes, not in a tight loop."
        )

    def sessions(self, market: str) -> list[str]:
        """The archived session dates, oldest first."""
        with self._lock:
            cached = self._indexes.get(market)
            if cached is not None and self._clock() - cached[0] <= MANIFEST_TTL_SECONDS:
                return cached[1]
            url = f"{self.base_url}{release_tag(market)}/{INDEX_ASSET}"
            try:
                sessions = self._read(url, lambda document: parse_index(document, market))
            except urllib.error.HTTPError as exc:
                raise self._missing_index(market, exc) from exc
            self._indexes[market] = (self._clock(), sessions)
            return sessions

    def session(self, market: str, as_of_date: str) -> Session:
        key = (market, as_of_date)
        with self._lock:
            if key in self._sessions:
                self._sessions.move_to_end(key)
                return self._sessions[key]
            url = f"{self.base_url}{release_tag(market)}/{session_asset(market, as_of_date)}"
            try:
                session = self._read(url, lambda document: parse_session(document, market, as_of_date))
            except urllib.error.HTTPError as exc:
                raise ToolError(f"Archived session {url} is unavailable ({exc.code}); try again in a minute.") from exc
            self._sessions[key] = session
            while len(self._sessions) > MAX_CACHED_SESSIONS:
                self._sessions.popitem(last=False)
            return session


def pick_sessions(market: str, archived: list[str], since: date | None, until: date | None) -> tuple[str, str]:
    """The newest session on or before `until`, and the newest before it on or before `since`."""
    if since is not None and until is not None and since > until:
        raise ToolError(f"since {since} is after until {until}. Pass the older date as since.")
    if not archived:
        raise ToolError(f"The {market} archive holds no session yet, so there is nothing to compare.")
    first = archived[0]
    if len(archived) == 1:
        raise ToolError(
            f"The {market} archive started at {first} and holds one session, so there is nothing to compare until "
            "the next publish is recorded."
        )
    newer = [s for s in archived if until is None or s <= until.isoformat()]
    if not newer:
        raise ToolError(f"No archived {market} session on or before {until}. The archive starts at {first}.")
    new = newer[-1]
    older = [s for s in archived if s < new and (since is None or s <= since.isoformat())]
    if not older:
        bound = "" if since is None else f" and on or before {since}"
        raise ToolError(f"No archived {market} session before {new}{bound}. The archive starts at {first}.")
    return older[-1], new


def _listed(symbols: set[str], limit: int) -> dict[str, Any]:
    ordered = sorted(symbols)
    return {"count": len(ordered), "symbols": ordered[:limit], "truncated": len(ordered) > limit}


def _side(session: Session) -> dict[str, Any]:
    return {
        "as_of_date": session.as_of_date,
        "published_at": session.published_at,
        "origin": session.origin,
        "rs_universe_size": session.rs_universe_size,
        "rs_formula_version": session.rs_formula_version,
    }


def _move(old: Session, new: Session, preset: str, symbol: str) -> str | None:
    """One of MOVES, or None when the symbol stayed in or out; it needs a row in both sessions."""
    before, after = old.screens.get(preset, ABSENT), new.screens.get(preset, ABSENT)
    was, now = symbol in before.symbols, symbol in after.symbols
    if was == now:
        return None
    if now:
        return "entered_with_new_data" if (old.nulls[symbol] - new.nulls[symbol]) & after.fields else "entered"
    return "left_with_lost_data" if (new.nulls[symbol] - old.nulls[symbol]) & before.fields else "left"


def _screen_change(old: Session, new: Session, preset: str, limit: int) -> dict[str, Any]:
    before, after = old.screens.get(preset, ABSENT), new.screens.get(preset, ABSENT)
    was, now = before.symbols, after.symbols
    moved: dict[str, set[str]] = {move: set() for move in MOVES}
    # A symbol with no row on one side changed the publish, not its screens.
    for symbol in now ^ was:
        if symbol in old.rows and symbol in new.rows:
            moved[_move(old, new, preset, symbol)].add(symbol)
    return {
        "name": after.name or before.name,
        "definition_changed": before.definition != after.definition,
        **{move: _listed(symbols, limit) for move, symbols in moved.items()},
        "new_to_publish": _listed({s for s in now if s not in old.rows}, limit),
        "dropped_from_publish": _listed({s for s in was if s not in new.rows}, limit),
    }


def _symbol_change(old: Session, new: Session, symbol: str, presets: list[str]) -> dict[str, Any]:
    before, after = old.rows.get(symbol), new.rows.get(symbol)
    change: dict[str, Any] = {
        "in_since": before is not None, "in_until": after is not None, **{move: [] for move in MOVES},
        "changed": {}, "missing": [], "warnings": [],
    }
    if before is None or after is None:
        return change
    for preset in presets:
        move = _move(old, new, preset, symbol)
        if move is not None:
            change[move].append(preset)
    basis = {"from_basis": before[BASIS], "to_basis": after[BASIS]}
    change["changed"] = {
        field: {"from": a, "to": b, **(basis if field in GROWTH_FIELDS else {})}
        for field, a, b in zip(TRACKED_FIELDS, before, after) if a != b
    }
    change["missing"] = [field for field, a, b in zip(TRACKED_FIELDS, before, after) if a is None and b is None]
    if before[BASIS] != after[BASIS]:
        change["warnings"].append(
            f"growth_metric_basis went from {before[BASIS]} to {after[BASIS]}, so {' and '.join(GROWTH_FIELDS)} "
            "measure different comparisons in the two sessions, so their change can come from the comparison rather "
            "than the stock."
        )
    return change


def _warnings(old: Session, new: Session, redefined: list[str], newest_publish: str | None, newest: str) -> list[str]:
    warnings = []
    a, b = old.rs_universe_size, new.rs_universe_size
    if a and b and abs(b - a) > a / 10:
        warnings.append(
            f"The RS universe went from {a:,} to {b:,} stocks, a change of more than 10 percent, so RS ratings and "
            "the screens that use them moved partly because the universe changed."
        )
    a, b = old.rs_formula_version, new.rs_formula_version
    if a and b and a != b:
        warnings.append(
            f"The RS formula changed from {a} to {b}, so RS ratings do not compare across these sessions."
        )
    elif bool(a) != bool(b):
        side, session, other = ("since", old, "until") if b else ("until", new, "since")
        warnings.append(
            f"The RS formula is unknown in the {side} session, {session.as_of_date}, so its RS ratings may not "
            f"compare with the {other} session's."
        )
    for side, session in (("since", old), ("until", new)):
        if session.origin != "live":
            warnings.append(
                f"The {side} session, {session.as_of_date}, came from a retained build (origin {session.origin}), so "
                "its published_at only approximates publication and the site may never have shown that build."
            )
    if (old.origin == PRE_STAGE_FIX) != (new.origin == PRE_STAGE_FIX):
        warnings.append(
            "One session was built before the 2026-10-21 stage fix and the other after it, so stage changes between "
            "them can come from the fix rather than the stocks."
        )
    if redefined:
        warnings.append(
            f"The rules of {', '.join(redefined)} differ between the sessions, so their entered and left lists can "
            "come from the rules rather than the stocks."
        )
    if newest_publish is not None and newest_publish > newest:
        warnings.append(
            f"The site now shows {newest_publish}, which the archive has not recorded yet. Its newest session is "
            f"{newest}."
        )
    return warnings


def diff(
    old: Session,
    new: Session,
    archived: list[str],
    newest_publish: str | None,
    *,
    screen: str | None = None,
    symbols: list[str] | None = None,
    limit: int = 25,
) -> dict[str, Any]:
    """Everything what_changed returns about two sessions; `archived` is the index, oldest first."""
    known = sorted(old.screens.keys() | new.screens.keys())
    presets = known
    if screen is not None:
        presets = [p for p in known if p.lower() == screen.strip().lower()]
        if not presets:
            raise ToolError(f"Unknown screen {screen!r}. Screens in these sessions: {', '.join(known)}.")
    screens = {preset: _screen_change(old, new, preset, limit) for preset in presets}
    redefined = [preset for preset, change in screens.items() if change["definition_changed"]]
    result = {
        "since": _side(old),
        "until": _side(new),
        "sessions_between": sum(old.as_of_date < s < new.as_of_date for s in archived),
        "warnings": _warnings(old, new, redefined, newest_publish, archived[-1]),
        "screens": screens,
    }
    if symbols:
        requested = dict.fromkeys(s.strip().upper() for s in symbols)
        result["symbols"] = {symbol: _symbol_change(old, new, symbol, presets) for symbol in requested}
    return result

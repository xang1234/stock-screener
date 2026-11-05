"""Terminal front end: a day's screen as a table, as JSON, or as a TradingView watchlist."""

from __future__ import annotations

import argparse
import errno
import json
import os
import sys
import unicodedata
from pathlib import Path
from typing import Any, NamedTuple

from mcp.server.mcpserver.exceptions import ToolError

from . import server


class Venue(NamedTuple):
    exchange: str
    dash: str = "-"
    strip: str = ""


# Keyed by exchange and gics_industry, where None means any other industry. XNYS covers funds TradingView
# lists under AMEX: 2,693 of the 2,803 such rows resolve there and 78 under NYSE (scanner census, 2026-11-05).
US_VENUES = {
    ("XNYS", None): Venue("NYSE", dash="."),
    ("XNYS", "Exchange Traded Fund"): Venue("AMEX", dash="."),
    ("XNAS", None): Venue("NASDAQ", dash="."),
    ("XASE", None): Venue("AMEX", dash="."),
}
SUFFIX_VENUES = {
    ".HK": Venue("HKEX", strip="0"),
    ".T": Venue("TSE"),
    ".DE": Venue("XETR"),
    ".TO": Venue("TSX", dash="."),
    ".V": Venue("TSXV", dash="."),
    ".NS": Venue("NSE", dash="_"),
    ".TW": Venue("TWSE"),
    ".TWO": Venue("TPEX"),
    ".KS": Venue("KRX"),
    ".KQ": Venue("KRX"),
    ".SS": Venue("SSE"),
    ".SZ": Venue("SZSE"),
    ".SI": Venue("SGX"),
    ".AX": Venue("ASX"),
}
MALAYSIA_NOTE = "Malaysia is unsupported: TradingView lists its stocks by name instead of numeric code."

COLUMNS = (
    ("SYMBOL", "symbol", "{}", "<"),
    ("NAME", "company_name", "{}", "<"),
    ("PRICE", "current_price", "{:,.2f}", ">"),
    ("RS", "rs_rating", "{:.0f}", ">"),
    ("STAGE", "stage", "{:.0f}", ">"),
    ("1D", "price_change_1d", "{:+.2f}%", ">"),
    ("GROUP", "ibd_industry_group", "{}", "<"),
)
MAX_CELL_WIDTH = 32
# https://in.tradingview.com/support/solutions/43000611193: "Watchlist can contain no more than 1000 symbols."
TRADINGVIEW_LIST_CAP = 1000


def _stderr(message: str) -> None:
    print(message, file=sys.stderr)


def tradingview_symbol(row: dict[str, Any], market: str) -> str | None:
    if market == "US":
        exchange = row.get("exchange")
        code = row["symbol"]
        venue = US_VENUES.get((exchange, row.get("gics_industry"))) or US_VENUES.get((exchange, None))
    else:
        code, dot, suffix = row["symbol"].rpartition(".")
        venue = SUFFIX_VENUES.get(dot + suffix)
    if venue is None:
        return None
    ticker = code.replace("-", venue.dash).lstrip(venue.strip)
    return f"{venue.exchange}:{ticker}" if ticker else None


def _display_width(text: str) -> int:
    # JP, CN and KR company names are double-width in a terminal; combining marks take no column.
    return sum(
        0 if unicodedata.combining(char) else 2 if unicodedata.east_asian_width(char) in "WF" else 1
        for char in text
    )


def _cell(row: dict[str, Any], field: str, template: str) -> str:
    value = row.get(field)
    try:
        text = "-" if value is None else template.format(value)
    except (TypeError, ValueError):
        text = str(value)
    # Category C, not str.isprintable: names can hold an ideographic space, which isprintable rejects.
    text = "".join(" " if unicodedata.category(char).startswith("C") else char for char in text)
    while _display_width(text) > MAX_CELL_WIDTH:
        text = text[:-1]
    return text


def _table(rows: list[dict[str, Any]]) -> str:
    grid = [
        [header for header, *_ in COLUMNS],
        *[[_cell(row, field, template) for _, field, template, _ in COLUMNS] for row in rows],
    ]
    widths = [max(_display_width(line[i]) for line in grid) for i in range(len(COLUMNS))]
    lines = []
    for line in grid:
        cells = []
        for text, width, (*_, align) in zip(line, widths, COLUMNS):
            pad = " " * (width - _display_width(text))
            cells.append(pad + text if align == ">" else text + pad)
        lines.append("  ".join(cells).rstrip())
    return "\n".join(lines)


def _report_freshness(market: str, as_of_date: str | None, status: str) -> None:
    _stderr(f"{market} data as of {as_of_date}")
    if status == "late":
        _stderr(f"Warning: the {market} data is late, so {as_of_date} may not be the latest session.")


def _list_screens(market: str) -> int:
    listing = server.list_screens(market)
    _report_freshness(listing["market"], listing["as_of_date"], server.market_status(market))
    width = max((len(screen["id"]) for screen in listing["screens"]), default=0)
    for screen in listing["screens"]:
        print(f"{screen['id']:<{width}}  {screen['name']}")
    return 0


def _write_watchlist(result: server.ScreenResult, target: str) -> int:
    screen_id = result.preset["id"]
    if not result.rows:
        _stderr(f"No {result.market} stocks match {screen_id}, so no file was written.")
        return 0

    mapped = [(row["symbol"], tradingview_symbol(row, result.market)) for row in result.rows]
    tokens = [token for _, token in mapped if token]
    skipped = [symbol for symbol, token in mapped if not token]
    if skipped:
        _stderr(f"Skipped {len(skipped)} with no TradingView symbol: {', '.join(skipped)}")
    if any(symbol.endswith(".KL") for symbol in skipped):
        _stderr(MALAYSIA_NOTE)
    if not tokens:
        _stderr("Nothing to write: no match has a TradingView symbol.")
        return 1
    if len(tokens) > TRADINGVIEW_LIST_CAP:
        _stderr(
            f"Warning: {len(tokens)} symbols is over TradingView's limit of {TRADINGVIEW_LIST_CAP} per watchlist, "
            "so they will not all fit in one list."
        )

    text = ",".join(tokens)
    if target == "-":
        print(text, flush=True)
        destination = "stdout"
    else:
        destination = target or f"{screen_id}-{result.market}-{result.as_of_date}.txt"
        try:
            Path(destination).write_text(text, encoding="utf-8")
        except OSError as exc:
            _stderr(f"Could not write {destination}: {exc.strerror}")
            return 1
    _stderr(f"Wrote {len(tokens)} of {result.total} matches to {destination}")
    return 0


def _run_screen(args: argparse.Namespace) -> int:
    result = server.match_screen(args.market, args.screen)
    status = server.market_status(result.market)
    _report_freshness(result.market, result.as_of_date, status)
    if args.tradingview is not None:
        return _write_watchlist(result, args.tradingview)
    shown = result.rows[: args.limit]
    if args.json:
        print(json.dumps(
            {
                "market": result.market,
                "screen": result.preset["id"],
                "as_of_date": result.as_of_date,
                "status": status,
                "total": result.total,
                "rows": [server._pick(row, [*result.columns, "exchange"]) for row in shown],
            },
            indent=2,
        ))
        return 0
    print(_table(shown), flush=True)
    _stderr(f"{len(shown)} of {result.total}")
    return 0


def _positive_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError("must be a whole number") from None
    if value < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="stockscreen-mcp",
        description="Daily stock screens in the terminal. With no arguments, runs the MCP server over stdio.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    screen = commands.add_parser("screen", help="List a market's screens, or run one.")
    screen.add_argument("screen", nargs="?", help="Screen id, such as minervini. Omit it to list the screens.")
    screen.add_argument("--market", default="US", type=str.strip, help="Market code, such as US, HK or JP (default: US).")
    screen.add_argument("--limit", type=_positive_int, default=25, help="Rows to show (default: 25).")
    output = screen.add_mutually_exclusive_group()
    output.add_argument("--json", action="store_true", help="Print JSON instead of a table.")
    output.add_argument(
        "--tradingview", nargs="?", const="", metavar="FILE",
        help="Write every match as a TradingView watchlist, ignoring --limit. Use - for stdout.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        server.main()
        return 0

    parser = _parser()
    args = parser.parse_args(argv)
    if args.screen is None and (args.json or args.tradingview is not None):
        parser.error("--json and --tradingview need a screen id, such as: screen minervini --json")

    # Company names can fall outside a redirected Windows stdout's cp1252.
    sys.stdout.reconfigure(errors="replace")
    try:
        code = _list_screens(args.market) if args.screen is None else _run_screen(args)
        sys.stdout.flush()
        return code
    except ToolError as exc:
        _stderr(f"Error: {exc}")
        return 1
    except KeyboardInterrupt:
        return 130
    except OSError as exc:
        if exc.errno not in (errno.EPIPE, errno.EINVAL):
            raise
        # The reader left (`| head`), which is not a failure. Windows reports a closed pipe as EINVAL.
        # Shutdown flushes stdout again, so aim it at devnull.
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        return 0

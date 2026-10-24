"""Helpers for parsing pasted watchlist imports."""

from __future__ import annotations

import csv
import io
import re
from collections.abc import Iterable
from dataclasses import dataclass

from .symbol_format import SYMBOL_SHAPE_RE

# TradingView ``EXCHANGE:TICKER`` prefix -> universe symbol suffixes to try,
# best first. KRX covers both KOSPI (.KS) and KOSDAQ (.KQ). A prefix missing
# here gives no candidate, so ``EURONEXT:AIR`` is reported, never read as AIR.
_TRADINGVIEW_PREFIX_SUFFIXES = {
    "NASDAQ": ("",),
    "NYSE": ("",),
    "AMEX": ("",),
    "NYSEARCA": ("",),
    "BATS": ("",),
    "CBOE": ("",),
    "OTC": ("",),
    "HKEX": (".HK",),
    "TSE": (".T",),
    "TWSE": (".TW",),
    "TPEX": (".TWO",),
    "KRX": (".KS", ".KQ"),
    "SSE": (".SS",),
    "SZSE": (".SZ",),
    "SGX": (".SI",),
    "ASX": (".AX",),
    "MYX": (".KL",),
    "TSX": (".TO",),
    "TSXV": (".V",),
    "XETR": (".DE",),
    "FWB": (".F",),
    "NSE": (".NS",),
    "BSE": (".BO",),
}
# A ``###Section name`` cell runs to the next comma or newline. Dropping it
# whole keeps a multi-word name such as ``###Top AI Picks`` from leaking AI.
_TRADINGVIEW_SECTION_RE = re.compile(r"(?:^|(?<=[,\n]))[ \t]*###[^,\n]*")
# A slash marks a URL such as https://x, never a ticker.
_TRADINGVIEW_SYMBOL_RE = re.compile(r"^[A-Z][A-Z0-9_]*:[^\s/]+$")


@dataclass(frozen=True)
class ImportToken:
    """One imported token and the universe symbols it may name, best first."""

    text: str
    candidates: tuple[str, ...]


_HEADER_TOKENS = {
    "symbol",
    "symbols",
    "ticker",
    "tickers",
    "code",
}
# Heuristic requires at least one A-Z letter so numeric-only cells
# (price/volume columns) don't masquerade as symbols.
_SYMBOL_HEURISTIC_PATTERN = re.compile(r"^(?=.*[A-Z])[A-Z0-9][A-Z0-9.\-]{0,19}$")


def parse_watchlist_import_tokens(
    content: str,
    format_hint: str | None = "auto",
) -> list[ImportToken]:
    """Parse pasted watchlist content into de-duplicated uppercase tokens."""

    normalized_hint = (format_hint or "auto").lower()
    stripped = content.strip()
    if not stripped:
        return []

    if normalized_hint not in {"auto", "text", "csv"}:
        raise ValueError(f"Unsupported import format: {format_hint}")

    if normalized_hint == "csv" or (
        normalized_hint == "auto"
        and "\n" in stripped
        and any(delimiter in stripped for delimiter in (",", "\t", ";"))
    ):
        raw_tokens = _parse_tabular_symbols(stripped)
    else:
        raw_tokens = _parse_text_symbols(_TRADINGVIEW_SECTION_RE.sub("", stripped))
    tokens = [_import_token(token.strip().upper()) for token in raw_tokens]

    deduped: list[ImportToken] = []
    seen: set[str] = set()
    for token in tokens:
        if not token.text or token.text.lower() in _HEADER_TOKENS or token.text in seen:
            continue
        seen.add(token.text)
        deduped.append(token)
    return deduped


def _import_token(cell: str) -> ImportToken:
    prefix, separator, ticker = cell.partition(":")
    if not separator:
        ticker, suffixes = cell, ("",)
    else:
        suffixes = _TRADINGVIEW_PREFIX_SUFFIXES.get(prefix, ())
        if prefix == "HKEX" and ticker.isdigit():
            ticker = f"{int(ticker):04d}"
    # Class shares are often written BRK.B; the universe may hold BRK-B.
    tickers = dict.fromkeys((ticker, ticker.replace(".", "-")))
    return ImportToken(cell, tuple(t + suffix for t in tickers for suffix in suffixes))


def _looks_like_symbol_token(token: str) -> bool:
    # Heuristic parser should reject numeric-only cells (price/volume columns).
    return bool(_SYMBOL_HEURISTIC_PATTERN.match(token.strip().upper()))


def _parse_tabular_symbols(content: str) -> list[str]:
    sample = "\n".join(content.splitlines()[:5])
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t;")
        delimiter = dialect.delimiter
    except csv.Error:
        delimiter = "\t" if "\t" in sample else ","

    reader = csv.reader(io.StringIO(content), delimiter=delimiter)
    rows = [row for row in reader if any(cell.strip() for cell in row)]
    if not rows:
        return []

    first_row = rows[0]
    header_cells = [cell.strip().lower() for cell in first_row if cell.strip()]
    has_header = any(cell in _HEADER_TOKENS for cell in header_cells)
    symbol_column = next(
        (index for index, cell in enumerate(first_row) if cell.strip().lower() in _HEADER_TOKENS),
        0,
    )

    symbols: list[str] = []
    if has_header:
        data_rows = rows[1:]
        for row in data_rows:
            if symbol_column >= len(row):
                continue
            candidate = row[symbol_column].strip()
            if candidate:
                symbols.append(candidate)
        return symbols

    cells = [
        _unquote(cell).split()
        for cell in re.split(r"[,;\t\n]", _TRADINGVIEW_SECTION_RE.sub("", content))
    ]
    if all(_is_symbol_cell(words) for words in cells if words):
        return [word for words in cells for word in words]

    for row in rows:
        cells_after_sections = [cell.strip() for cell in row if not cell.strip().startswith("###")]
        if cells_after_sections and cells_after_sections[0]:
            symbols.append(cells_after_sections[0])
    return symbols


def _unquote(cell: str) -> str:
    # Only a fully quoted cell is unwrapped, so a quoted name holding a comma
    # such as "Apple, Inc." splits into pieces that are not symbols.
    cell = cell.strip()
    return cell[1:-1] if len(cell) >= 2 and cell[0] == cell[-1] == '"' else cell


def _is_symbol_cell(words: list[str]) -> bool:
    # Several words make a symbol cell only when each is EXCHANGE:TICKER, so a
    # name such as "NVIDIA Corp" keeps a header-less paste a table.
    if all(_TRADINGVIEW_SYMBOL_RE.match(word.upper()) for word in words):
        return True
    return len(words) == 1 and _looks_like_symbol_token(words[0])


def _parse_text_symbols(content: str) -> list[str]:
    return [token for token in re.split(r"[\s,;\t]+", content) if token.strip()]


def split_import_results(
    tokens: Iterable[ImportToken],
    known_symbols: set[str],
    existing_symbols: set[str],
) -> tuple[list[str], list[str], list[str]]:
    """Classify tokens into addable, existing, and invalid groups.

    Each token resolves to its first candidate in ``known_symbols`` (else its
    first one already in the watchlist). Unresolved tokens are reported under
    their original text.
    """

    added_candidates: list[str] = []
    skipped_existing: list[str] = []
    invalid_symbols: list[str] = []
    resolved_symbols: set[str] = set()

    for token in tokens:
        symbol = next(
            (c for c in token.candidates if c in known_symbols), None
        ) or next((c for c in token.candidates if c in existing_symbols), None)
        if symbol in resolved_symbols:
            continue
        if symbol is not None:
            resolved_symbols.add(symbol)
        if symbol in existing_symbols:
            skipped_existing.append(symbol)
        elif symbol is None or not SYMBOL_SHAPE_RE.match(symbol):
            invalid_symbols.append(token.text)
        else:
            added_candidates.append(symbol)

    return added_candidates, skipped_existing, invalid_symbols

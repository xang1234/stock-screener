"""Helpers for parsing pasted watchlist imports."""

from __future__ import annotations

import csv
import io
import re
from collections.abc import Iterable

from .symbol_format import SYMBOL_SHAPE_RE

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


def parse_watchlist_import_symbols(
    content: str,
    format_hint: str | None = "auto",
) -> list[str]:
    """Parse pasted watchlist content into de-duplicated uppercase symbols."""

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
        tokens = _parse_tabular_symbols(stripped)
    else:
        tokens = _parse_text_symbols(stripped)

    deduped: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        symbol = token.strip().upper()
        if not symbol or symbol.lower() in _HEADER_TOKENS or symbol in seen:
            continue
        seen.add(symbol)
        deduped.append(symbol)
    return deduped


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

    flattened_cells = [cell.strip() for row in rows for cell in row if cell.strip()]
    if flattened_cells and all(_looks_like_symbol_token(cell) for cell in flattened_cells):
        return flattened_cells

    for row in rows:
        if not row:
            continue
        candidate = row[0].strip()
        if candidate:
            symbols.append(candidate)
    return symbols


def _parse_text_symbols(content: str) -> list[str]:
    return [token for token in re.split(r"[\s,;\t]+", content) if token.strip()]


def split_import_results(
    symbols: Iterable[str],
    known_symbols: set[str],
    existing_symbols: set[str],
) -> tuple[list[str], list[str], list[str]]:
    """Classify parsed symbols into addable, existing, and invalid groups."""

    added_candidates: list[str] = []
    skipped_existing: list[str] = []
    invalid_symbols: list[str] = []

    for symbol in symbols:
        upper_symbol = symbol.upper()
        if upper_symbol in existing_symbols:
            skipped_existing.append(upper_symbol)
        elif upper_symbol not in known_symbols or not SYMBOL_SHAPE_RE.match(upper_symbol):
            invalid_symbols.append(upper_symbol)
        else:
            added_candidates.append(upper_symbol)

    return added_candidates, skipped_existing, invalid_symbols

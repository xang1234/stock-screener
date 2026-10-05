"""Read a whole Finviz screener filter despite its 1,000-row cap.

Since early October 2026 Finviz answers anonymous screener requests past row
1,000 (``r > 1000``) with HTTP 403, whatever the sort order, so NYSE (~5,000
rows) and NASDAQ (~4,700) can no longer be paged in one query. A slice of up to
1,980 rows is read from both ends (ticker ascending, then descending, with a
page of overlap); a bigger one is split by the next filter dimension,
recursively. On 2026-10-05 sectors
partitioned each exchange exactly, fund type partitioned Financial, price
partitioned ETFs and average volume partitioned NYSE ETFs under $50. Any read
that cannot account for every row raises FinvizReadError rather than return a
partial list: the universe refresh would deactivate the missing symbols.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable, Sequence
from typing import Any

FINVIZ_ROW_CAP = 1000
FINVIZ_PAGE_SIZE = 20
# Largest slice read from both ends with a page of overlap between the passes;
# the overlap absorbs a listing inserted between them.
TWO_ENDED_MAX_ROWS = 2 * FINVIZ_ROW_CAP - FINVIZ_PAGE_SIZE

SPLIT_DIMENSIONS: tuple[tuple[str, ...], ...] = (
    tuple(
        f"sec_{sector}"
        for sector in (
            "basicmaterials",
            "communicationservices",
            "consumercyclical",
            "consumerdefensive",
            "energy",
            "financial",
            "healthcare",
            "industrials",
            "realestate",
            "technology",
            "utilities",
        )
    ),
    # Financial is mostly funds: NYSE 3,465 = 372 stocks + 2,803 ETFs + 290 CEFs.
    (
        "ind_stocksonly",
        "ind_exchangetradedfund",
        "ind_closedendfunddebt",
        "ind_closedendfundequity",
        "ind_closedendfundforeign",
    ),
    ("sh_price_u50", "sh_price_o50"),
    ("sh_avgvol_u100", "sh_avgvol_o100"),
)

# A result page shows "#1 / N Total"; an empty one shows "0 Total". A page with
# neither (a challenge or redesigned page) is not a screener answer.
_TOTAL_PATTERN = re.compile(r"#\s*1\s*/\s*([\d,]+)\s*Total")
_EMPTY_TOTAL_PATTERN = re.compile(r"(?<![\w/,.])0\s+Total\b")

FetchPage = Callable[[str, str, int], Any]


class FinvizReadError(RuntimeError):
    """Finviz could not be read completely; callers treat it like a provider outage."""


class FinvizSliceTooLarge(FinvizReadError):
    """A slice still exceeds what two-ended reading covers after every split."""


class FinvizIncompleteRead(FinvizReadError):
    """A read cannot account for every row its filter matches."""


def screener_total(soup: Any) -> int | None:
    """Rows matching the page's filters, or None if the page shows no row total."""
    text = soup.get_text(" ")
    match = _TOTAL_PATTERN.search(text)
    if match:
        return int(match.group(1).replace(",", ""))
    return 0 if _EMPTY_TOTAL_PATTERN.search(text) else None


def read_screener(
    base_filters: str,
    fetch_page: FetchPage,
    *,
    parse_rows: Callable[[Any], list[dict[str, Any]]],
    row_key: Callable[[dict[str, Any]], str],
    total_of: Callable[[Any], int | None] = screener_total,
    pause: Callable[[], Any] = lambda: time.sleep(1),
    dimensions: Sequence[Sequence[str]] = SPLIT_DIMENSIONS,
) -> list[dict[str, Any]]:
    """All rows matching ``base_filters``, one per ``row_key``.

    ``fetch_page(filters, order, first_row)`` returns one screener page, where
    ``order`` is ``"ticker"`` or ``"-ticker"`` and ``first_row`` is Finviz's
    1-based ``r``. ``pause`` runs before every request (the Finviz rate budget).
    """
    rows_by_key: dict[str, dict[str, Any]] = {}

    def fetch(filters: str, order: str, first_row: int) -> Any:
        pause()
        return fetch_page(filters, order, first_row)

    def read_pages(filters: str, order: str, rows: list[dict[str, Any]], wanted: int) -> list[dict[str, Any]]:
        next_row = len(rows) + 1
        while len(rows) < wanted and next_row <= FINVIZ_ROW_CAP:
            page_rows = parse_rows(fetch(filters, order, next_row))
            if not page_rows:
                break
            rows = rows + page_rows
            next_row += FINVIZ_PAGE_SIZE
        return rows

    def visit(filters: str, remaining: Sequence[Sequence[str]]) -> int:
        first_page = fetch(filters, "ticker", 1)
        first_rows = parse_rows(first_page)
        total = total_of(first_page)
        if total is None:
            # Not a screener answer (e.g. an HTTP-200 challenge page): reading it as
            # empty would drop the slice from an otherwise complete read.
            raise FinvizIncompleteRead(f"Finviz page for {filters!r} shows no row total")
        if total > TWO_ENDED_MAX_ROWS:
            if not remaining:
                raise FinvizSliceTooLarge(
                    f"Finviz slice {filters!r} has {total} rows; no filter left to split it "
                    f"below {TWO_ENDED_MAX_ROWS}"
                )
            children = sum(visit(f"{filters},{value}", remaining[1:]) for value in remaining[0])
            if children < total:
                raise FinvizIncompleteRead(
                    f"Finviz split of {filters!r} covers {children} of {total} rows"
                )
            return total

        rows = read_pages(filters, "ticker", first_rows, min(total, FINVIZ_ROW_CAP))
        if total > FINVIZ_ROW_CAP:
            # One extra page of overlap absorbs a listing change during the read.
            rows += read_pages(filters, "-ticker", [], total - FINVIZ_ROW_CAP + FINVIZ_PAGE_SIZE)
        slice_rows = {key: row for row in rows if (key := row_key(row))}
        if len(slice_rows) < total:
            # Short pages, keyless rows or ordering drift between page requests.
            raise FinvizIncompleteRead(
                f"Finviz read of {filters!r} found {len(slice_rows)} of {total} rows"
            )
        rows_by_key.update(slice_rows)
        return total

    total = visit(base_filters, dimensions)
    if len(rows_by_key) < total:
        # A ticker reclassified mid-read can fill two split slices while another is missed.
        raise FinvizIncompleteRead(
            f"Finviz read of {base_filters!r} found {len(rows_by_key)} of {total} rows"
        )
    return list(rows_by_key.values())

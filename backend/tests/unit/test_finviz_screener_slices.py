"""Reading whole Finviz screener filters through its 1,000-row cap."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import requests

from app.services import finviz_screener_slices as slices


def _row(ticker, *, sector="technology", fund="stocksonly", price=10.0, avgvol=50.0):
    return {"ticker": ticker, "sector": sector, "fund": fund, "price": price, "avgvol": avgvol}


def _matches(row, token):
    kind, _, value = token.partition("_")
    if kind == "exch":
        return True
    if kind == "sec":
        return row["sector"] == value
    if kind == "ind":
        return row["fund"] == value
    if token.startswith("sh_price_"):
        bound = float(token[len("sh_price_") + 1:])
        return row["price"] < bound if token[len("sh_price_")] == "u" else row["price"] > bound
    if token.startswith("sh_avgvol_"):
        bound = float(token[len("sh_avgvol_") + 1:])
        return row["avgvol"] < bound if token[len("sh_avgvol_")] == "u" else row["avgvol"] > bound
    raise AssertionError(f"unexpected filter {token}")


class _FakeFinviz:
    """Applies filters and sort order; refuses rows past 1,000 like Finviz since Oct 2026."""

    def __init__(self, rows):
        self.rows = rows
        self.requests = []

    def fetch_page(self, filters, order, first_row):
        self.requests.append((filters, order, first_row))
        if first_row > slices.FINVIZ_ROW_CAP:
            raise requests.HTTPError("403 Client Error: Forbidden")
        tokens = filters.split(",")
        matched = sorted(
            (row for row in self.rows if all(_matches(row, t) for t in tokens)),
            key=lambda row: row["ticker"],
            reverse=order == "-ticker",
        )
        start = first_row - 1
        return SimpleNamespace(
            total=len(matched), rows=matched[start:start + slices.FINVIZ_PAGE_SIZE]
        )


def _read(fake):
    return slices.read_screener(
        "exch_nyse",
        fake.fetch_page,
        parse_rows=lambda page: page.rows,
        row_key=lambda row: row["ticker"],
        total_of=lambda page: page.total,
        pause=lambda: None,
    )


def test_reads_every_row_of_an_exchange_larger_than_the_cap():
    rows = (
        [_row(f"T{i:04d}") for i in range(400)]
        + [_row(f"S{i:04d}", sector="financial") for i in range(300)]
        + [_row(f"E{i:04d}", sector="financial", fund="exchangetradedfund", price=20.0) for i in range(1500)]
        + [_row(f"F{i:04d}", sector="financial", fund="exchangetradedfund", price=80.0) for i in range(800)]
    )
    fake = _FakeFinviz(rows)

    collected = _read(fake)

    assert sorted(row["ticker"] for row in collected) == sorted(row["ticker"] for row in rows)
    assert all(first_row <= slices.FINVIZ_ROW_CAP for _, _, first_row in fake.requests)
    # The 1,500 cheap ETFs exceed the cap, so that slice is also read from the far end.
    assert any(order == "-ticker" for _, order, _ in fake.requests)


def test_a_slice_that_no_filter_can_split_below_twice_the_cap_fails_loudly():
    fake = _FakeFinviz([_row(f"E{i:04d}", sector="financial", fund="exchangetradedfund") for i in range(2500)])

    with pytest.raises(slices.FinvizSliceTooLarge, match="2500 rows"):
        _read(fake)


def test_a_small_exchange_is_read_in_one_ascending_pass():
    rows = [_row(f"A{i:03d}") for i in range(284)]
    fake = _FakeFinviz(rows)

    assert len(_read(fake)) == 284
    assert {order for _, order, _ in fake.requests} == {"ticker"}
    assert [filters for filters, _, _ in fake.requests] == ["exch_nyse"] * 15


def test_a_split_that_loses_rows_fails_instead_of_returning_a_partial_read():
    """A row no child filter matches (e.g. no sector yet) would otherwise vanish and
    the universe refresh would deactivate it."""
    rows = (
        [_row(f"T{i:04d}") for i in range(1000)]
        + [_row(f"F{i:04d}", sector="financial") for i in range(1100)]
        + [_row("NOSECTOR", sector="")]
    )
    fake = _FakeFinviz(rows)

    with pytest.raises(slices.FinvizIncompleteRead, match="2100 of 2101"):
        _read(fake)


def test_an_unreadable_total_on_a_full_page_fails_instead_of_truncating():
    fake = _FakeFinviz([_row(f"T{i:04d}") for i in range(50)])

    with pytest.raises(slices.FinvizIncompleteRead, match="no row total"):
        slices.read_screener(
            "exch_nyse",
            fake.fetch_page,
            parse_rows=lambda page: page.rows,
            row_key=lambda row: row["ticker"],
            total_of=lambda page: None,
            pause=lambda: None,
        )


def test_a_slice_with_rows_that_have_no_key_fails():
    """A row we cannot identify is a row we cannot account for."""
    fake = _FakeFinviz([_row("A001"), _row("")])

    with pytest.raises(slices.FinvizIncompleteRead, match="1 of 2"):
        _read(fake)


def test_a_slice_whose_pages_repeat_a_ticker_fails():
    """Ordering drift between page requests can repeat one ticker and skip another."""
    rows = [_row(f"A{i:03d}") for i in range(40)]
    fake = _FakeFinviz(rows)
    fetch = fake.fetch_page

    def drifting(filters, order, first_row):
        page = fetch(filters, order, first_row)
        if first_row == 21:  # second page repeats the first page's last ticker
            page.rows = [rows[19], *page.rows[1:]]
        return page

    fake.fetch_page = drifting

    with pytest.raises(slices.FinvizIncompleteRead, match="39 of 40"):
        _read(fake)


def test_reader_errors_are_finviz_read_errors():
    assert issubclass(slices.FinvizSliceTooLarge, slices.FinvizReadError)
    assert issubclass(slices.FinvizIncompleteRead, slices.FinvizReadError)


def test_a_page_without_a_row_total_fails_even_when_empty():
    """An HTTP-200 challenge or changed page has no table and no total; reading it as an
    empty exchange would let the universe refresh drop that exchange."""
    fake = _FakeFinviz([])

    with pytest.raises(slices.FinvizIncompleteRead, match="no row total"):
        slices.read_screener(
            "exch_amex",
            fake.fetch_page,
            parse_rows=lambda page: page.rows,
            row_key=lambda row: row["ticker"],
            total_of=lambda page: None,
            pause=lambda: None,
        )


def test_a_ticker_counted_in_two_split_slices_fails():
    """A ticker reclassified mid-read can satisfy two child slices while another goes missing."""
    rows = [_row(f"T{i:04d}") for i in range(1000)] + [
        _row(f"F{i:04d}", sector="financial") for i in range(1100)
    ]
    fake = _FakeFinviz(rows)
    fetch = fake.fetch_page

    def reclassifying(filters, order, first_row):
        page = fetch(filters, order, first_row)
        if filters.endswith("sec_financial") and order == "ticker" and first_row == 1:
            page.rows = [rows[0], *page.rows[1:]]  # T0000 shows up, F0000 does not
        return page

    fake.fetch_page = reclassifying

    with pytest.raises(slices.FinvizIncompleteRead, match="2099 of 2100"):
        _read(fake)


@pytest.mark.parametrize(
    ("text", "expected"),
    [("Stats #1 / 5,029 Total Off", 5029), ("Stats 0 Total Off Set Alert", 0), ("Total Debt/Equity", None)],
)
def test_screener_total_reads_both_markers(text, expected):
    from bs4 import BeautifulSoup

    assert slices.screener_total(BeautifulSoup(f"<div>{text}</div>", "lxml")) == expected


def test_a_slice_too_big_for_an_overlapping_two_ended_read_is_split():
    """At 2,000 rows each direction returns exactly 1,000, leaving no overlap page to
    absorb a listing inserted between the passes; split instead."""
    rows = [_row(f"T{i:04d}") for i in range(1000)] + [
        _row(f"F{i:04d}", sector="financial") for i in range(1000)
    ]
    fake = _FakeFinviz(rows)

    assert len(_read(fake)) == 2000
    assert any(",sec_" in filters for filters, _, _ in fake.requests)
    assert all(order == "ticker" for _, order, _ in fake.requests)  # each half fits one pass

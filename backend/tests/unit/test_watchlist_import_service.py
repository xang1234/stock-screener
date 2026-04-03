from app.services.watchlist_import_service import (
    parse_watchlist_import_symbols,
    split_import_results,
)


def test_parse_watchlist_import_symbols_handles_text_and_dedupes():
    content = "NVDA, msft\nAAPL\tNVDA\n\n"

    assert parse_watchlist_import_symbols(content, format_hint="text") == [
        "NVDA",
        "MSFT",
        "AAPL",
    ]


def test_parse_watchlist_import_symbols_skips_csv_headers():
    content = "symbol,name\nNVDA,NVIDIA\nMSFT,Microsoft\n"

    assert parse_watchlist_import_symbols(content, format_hint="csv") == [
        "NVDA",
        "MSFT",
    ]


def test_split_import_results_classifies_existing_and_invalid_symbols():
    added, existing, invalid = split_import_results(
        ["NVDA", "MSFT", "BAD$", "AAPL"],
        known_symbols={"NVDA", "MSFT", "AAPL"},
        existing_symbols={"AAPL"},
    )

    assert added == ["NVDA", "MSFT"]
    assert existing == ["AAPL"]
    assert invalid == ["BAD$"]

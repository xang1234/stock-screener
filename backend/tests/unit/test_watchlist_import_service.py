import pytest

from app.services.watchlist_import_service import (
    ImportToken,
    parse_watchlist_import_tokens,
    split_import_results,
)


def _plain_import_symbols(content, format_hint="auto"):
    tokens = parse_watchlist_import_tokens(content, format_hint)
    assert all(token.candidates == (token.text,) for token in tokens)
    return [token.text for token in tokens]


def test_parse_watchlist_import_symbols_handles_text_and_dedupes():
    content = "NVDA, msft\nAAPL\tNVDA\n\n"

    assert _plain_import_symbols(content, format_hint="text") == [
        "NVDA",
        "MSFT",
        "AAPL",
    ]


def test_parse_watchlist_import_symbols_skips_text_headers():
    content = "symbol NVDA msft"

    assert _plain_import_symbols(content, format_hint="text") == [
        "NVDA",
        "MSFT",
    ]


def test_parse_watchlist_import_symbols_skips_csv_headers():
    content = "symbol,name\nNVDA,NVIDIA\nMSFT,Microsoft\n"

    assert _plain_import_symbols(content, format_hint="csv") == [
        "NVDA",
        "MSFT",
    ]


def test_parse_watchlist_import_symbols_handles_tab_delimited_spreadsheet_paste():
    content = "symbol\tname\nAAPL\tApple Inc.\nMSFT\tMicrosoft\n"

    assert _plain_import_symbols(content) == [
        "AAPL",
        "MSFT",
    ]


def test_parse_watchlist_import_symbols_preserves_headerless_symbol_matrix():
    content = "AAPL,MSFT\nNVDA"

    assert _plain_import_symbols(content) == [
        "AAPL",
        "MSFT",
        "NVDA",
    ]


def _texts_and_candidates(content):
    return [(token.text, token.candidates) for token in parse_watchlist_import_tokens(content)]


def test_tradingview_section_names_never_become_symbols():
    content = "###Top AI Picks,NASDAQ:NVDA,###ON Watch,NYSE:ANET"

    assert _texts_and_candidates(content) == [
        ("NASDAQ:NVDA", ("NVDA",)),
        ("NYSE:ANET", ("ANET",)),
    ]


def test_tradingview_export_spanning_lines_keeps_every_row():
    assert _texts_and_candidates("NASDAQ:NVDA,NYSE:ANET\nHKEX:700,TSE:6758") == [
        ("NASDAQ:NVDA", ("NVDA",)),
        ("NYSE:ANET", ("ANET",)),
        ("HKEX:700", ("0700.HK",)),
        ("TSE:6758", ("6758.T",)),
    ]
    assert _texts_and_candidates("###Leaders,NASDAQ:NVDA\r\n###Watch,HKEX:700\n") == [
        ("NASDAQ:NVDA", ("NVDA",)),
        ("HKEX:700", ("0700.HK",)),
    ]


def test_prefixed_tokens_pasted_with_spaces_still_map():
    assert _texts_and_candidates("NASDAQ:NVDA NYSE:BRK.B\tTVC:DXY") == [
        ("NASDAQ:NVDA", ("NVDA",)),
        ("NYSE:BRK.B", ("BRK.B", "BRK-B")),
        ("TVC:DXY", ()),
    ]


def test_tradingview_unmapped_prefixes_have_no_candidate():
    assert _texts_and_candidates("EURONEXT:AIR,LSE:BP,EURONEXT:MC,TVC:DXY,BINANCE:BTCUSDT") == [
        ("EURONEXT:AIR", ()),
        ("LSE:BP", ()),
        ("EURONEXT:MC", ()),
        ("TVC:DXY", ()),
        ("BINANCE:BTCUSDT", ()),
    ]


def test_tradingview_prefixes_map_to_universe_candidates():
    content = (
        "NYSE:BRK.B,AMEX:SPY,HKEX:5,HKEX:9988,TSE:6758,TWSE:2330,TPEX:3008,"
        "KRX:005930,SSE:600519,SZSE:000001,SGX:D05,ASX:BHP,TSX:SHOP,TSXV:ABC,"
        "XETR:SAP,FWB:SAP,NSE:RELIANCE,BSE:500325,MYX:1155"
    )

    assert _texts_and_candidates(content) == [
        ("NYSE:BRK.B", ("BRK.B", "BRK-B")),
        ("AMEX:SPY", ("SPY",)),
        ("HKEX:5", ("0005.HK",)),
        ("HKEX:9988", ("9988.HK",)),
        ("TSE:6758", ("6758.T",)),
        ("TWSE:2330", ("2330.TW",)),
        ("TPEX:3008", ("3008.TWO",)),
        ("KRX:005930", ("005930.KS", "005930.KQ")),
        ("SSE:600519", ("600519.SS",)),
        ("SZSE:000001", ("000001.SZ",)),
        ("SGX:D05", ("D05.SI",)),
        ("ASX:BHP", ("BHP.AX",)),
        ("TSX:SHOP", ("SHOP.TO",)),
        ("TSXV:ABC", ("ABC.V",)),
        ("XETR:SAP", ("SAP.DE",)),
        ("FWB:SAP", ("SAP.F",)),
        ("NSE:RELIANCE", ("RELIANCE.NS",)),
        ("BSE:500325", ("500325.BO",)),
        ("MYX:1155", ("1155.KL",)),
    ]


def test_split_picks_the_candidate_the_universe_holds():
    tokens = parse_watchlist_import_tokens(
        "KRX:247540,KRX:005930,NYSE:BRK.B,NYSE:BF.B,EURONEXT:AIR,NASDAQ:NVDA,NVDA"
    )

    added, existing, invalid = split_import_results(
        tokens,
        known_symbols={"247540.KQ", "005930.KS", "BRK-B", "BF.B", "AIR", "NVDA"},
        existing_symbols=set(),
    )

    assert added == ["247540.KQ", "005930.KS", "BRK-B", "BF.B", "NVDA"]
    assert existing == []
    assert invalid == ["EURONEXT:AIR"]


def test_split_import_results_classifies_existing_and_invalid_symbols():
    added, existing, invalid = split_import_results(
        [ImportToken(symbol, (symbol,)) for symbol in ["NVDA", "MSFT", "BAD$", "AAPL"]],
        known_symbols={"NVDA", "MSFT", "AAPL"},
        existing_symbols={"AAPL"},
    )

    assert added == ["NVDA", "MSFT"]
    assert existing == ["AAPL"]
    assert invalid == ["BAD$"]


def _added(content, known_symbols, format_hint="auto"):
    added, _, _ = split_import_results(
        parse_watchlist_import_tokens(content, format_hint), known_symbols, set()
    )
    return added


_TRAP_TICKERS = {"ON", "NOW", "USA", "AI", "TOP", "PICKS", "WATCH", "LINK", "PT"}


def test_csv_with_prefix_shaped_notes_reads_only_the_symbol_column():
    known = {"NVDA", "MSFT", "AAPL", *_TRAP_TICKERS}

    assert _added("symbol,notes\nNVDA,BUY:NOW\nMSFT,ON", known) == ["NVDA", "MSFT"]
    assert _added("Symbol,Notes\nNVDA,PT:200\nMSFT,NOW\nAAPL,", known) == ["NVDA", "MSFT", "AAPL"]
    assert _added("symbol,notes\nNVDA,NYSE:ON", known, format_hint="csv") == ["NVDA"]


def test_csv_with_a_link_column_reads_only_the_symbol_column():
    known = {"NVDA", "AAPL", *_TRAP_TICKERS}

    assert _added(
        "Ticker,Country,Link\nNVDA,USA,https://x\nAAPL,USA,https://y", known
    ) == ["NVDA", "AAPL"]


def test_mixed_plain_and_prefixed_pastes_keep_every_symbol():
    known = {"NVDA", "AAPL", "MSFT", "TSLA"}

    assert _added("NVDA AAPL MSFT\nNASDAQ:TSLA", known) == ["NVDA", "AAPL", "MSFT", "TSLA"]
    assert _added("NVDA MSFT,NASDAQ:AAPL", known, format_hint="text") == ["NVDA", "MSFT", "AAPL"]


def test_tradingview_cells_holding_several_symbols_are_split():
    known = {"NVDA", "ANET", "AAPL", "BRK-B", *_TRAP_TICKERS}

    assert _added("NASDAQ:NVDA NYSE:BRK.B\nNASDAQ:AAPL", known) == ["NVDA", "BRK-B", "AAPL"]
    assert _added("###Top AI Picks\nNASDAQ:NVDA NYSE:ANET", known) == ["NVDA", "ANET"]
    assert _added("NASDAQ:NVDA;NYSE:ANET", known) == ["NVDA", "ANET"]


def test_section_names_never_leak_from_a_mixed_paste():
    known = {"NVDA", "ANET", *_TRAP_TICKERS}

    assert _added("###Top AI Picks,NVDA,NASDAQ:ANET", known) == ["NVDA", "ANET"]
    assert _added("###Top AI Picks,NVDA\n###ON Watch,NASDAQ:ANET", known) == ["NVDA", "ANET"]


def test_plain_dotted_class_share_also_tries_the_dashed_form():
    assert _texts_and_candidates("BRK.B") == [("BRK.B", ("BRK.B", "BRK-B"))]
    assert _added("BRK.B,BF.B", {"BRK-B", "BF.B"}) == ["BRK-B", "BF.B"]


def test_multi_line_paste_mixing_prefixed_and_plain_cells_keeps_every_cell():
    known = {"NVDA", "AMD", "ANET", "MSFT", "AAPL", "TSLA", *_TRAP_TICKERS}

    assert _added("NASDAQ:NVDA,NASDAQ:AMD\nMSFT,AAPL", known) == ["NVDA", "AMD", "MSFT", "AAPL"]
    assert _added("NASDAQ:NVDA,NASDAQ:AMD,NYSE:ANET\nMSFT", known) == ["NVDA", "AMD", "ANET", "MSFT"]
    assert _added("NVDA,NASDAQ:AAPL\nMSFT,NASDAQ:TSLA", known) == ["NVDA", "AAPL", "MSFT", "TSLA"]


def test_headerless_table_with_a_link_column_reads_only_the_first_column():
    known = {"NVDA", "AAPL", *_TRAP_TICKERS}

    assert _added("NVDA,USA,https://x\nAAPL,USA,https://y", known) == ["NVDA", "AAPL"]


def test_headerless_table_with_names_drops_section_cells_before_the_first_column():
    known = {"NVDA", "AMD", "CORP", "MICRO", *_TRAP_TICKERS}

    assert _added(
        "###Top AI Picks,NVDA,NVIDIA Corp\n###ON Watch,AMD,Advanced Micro", known
    ) == ["NVDA", "AMD"]


_UNIVERSE = {
    "NVDA", "AMD", "ANET", "MSFT", "AAPL", "TSLA", "BRK-B", "BF.B", "247540.KQ",
    "005930.KS", "0700.HK", "0005.HK", "9988.HK", "6758.T", "SPY", "AIR", "BP",
    "MC", "DXY", "BTCUSDT", *_TRAP_TICKERS,
}
_ALREADY_IN_WATCHLIST = {"MSFT"}


def _outcome_by_cell(content):
    tokens = parse_watchlist_import_tokens(content, "auto")
    added, existing, invalid = split_import_results(tokens, _UNIVERSE, _ALREADY_IN_WATCHLIST)
    outcome = {}
    for token in tokens:
        if token.text in invalid:
            outcome[token.text] = "invalid"
        elif any(candidate in added for candidate in token.candidates):
            outcome[token.text] = "added"
        elif any(candidate in existing for candidate in token.candidates):
            outcome[token.text] = "existing"
        else:
            outcome[token.text] = "lost"
    return outcome


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        (
            "NASDAQ:NVDA,NASDAQ:AMD\nMSFT,AAPL",
            {"NASDAQ:NVDA": "added", "NASDAQ:AMD": "added", "MSFT": "existing", "AAPL": "added"},
        ),
        (
            "NASDAQ:NVDA,NASDAQ:AMD,NYSE:ANET\nMSFT",
            {"NASDAQ:NVDA": "added", "NASDAQ:AMD": "added", "NYSE:ANET": "added", "MSFT": "existing"},
        ),
        (
            "NVDA,NASDAQ:AAPL\nMSFT,NASDAQ:TSLA",
            {"NVDA": "added", "NASDAQ:AAPL": "added", "MSFT": "existing", "NASDAQ:TSLA": "added"},
        ),
        ("###Top AI Picks,NVDA\n###ON Watch,NASDAQ:ANET", {"NVDA": "added", "NASDAQ:ANET": "added"}),
        ("###Top AI Picks,NVDA,NASDAQ:ANET", {"NVDA": "added", "NASDAQ:ANET": "added"}),
        ("###Top AI Picks,NASDAQ:NVDA,###ON Watch,NYSE:ANET", {"NASDAQ:NVDA": "added", "NYSE:ANET": "added"}),
        ("###Top AI Picks\nNASDAQ:NVDA NYSE:ANET", {"NASDAQ:NVDA": "added", "NYSE:ANET": "added"}),
        (
            "###Leaders,NASDAQ:NVDA\r\n###Watch,HKEX:700\n",
            {"NASDAQ:NVDA": "added", "HKEX:700": "added"},
        ),
        (
            "NASDAQ:NVDA,NYSE:ANET\nHKEX:700,TSE:6758",
            {"NASDAQ:NVDA": "added", "NYSE:ANET": "added", "HKEX:700": "added", "TSE:6758": "added"},
        ),
        ("symbol,notes\nNVDA,BUY:NOW\nMSFT,ON", {"NVDA": "added", "MSFT": "existing"}),
        ("Symbol,Notes\nNVDA,PT:200\nMSFT,NOW\nAAPL,", {"NVDA": "added", "MSFT": "existing", "AAPL": "added"}),
        ("symbol,notes\nNVDA,NYSE:ON", {"NVDA": "added"}),
        ("Ticker,Country,Link\nNVDA,USA,https://x\nAAPL,USA,https://y", {"NVDA": "added", "AAPL": "added"}),
        ("symbol,name\nNVDA,NVIDIA\nMSFT,Microsoft\n", {"NVDA": "added", "MSFT": "existing"}),
        ("symbol\tname\nAAPL\tApple Inc.\nMSFT\tMicrosoft\n", {"AAPL": "added", "MSFT": "existing"}),
        (
            "NVDA AAPL MSFT\nNASDAQ:TSLA",
            {"NVDA": "added", "AAPL": "added", "MSFT": "existing", "NASDAQ:TSLA": "added"},
        ),
        ("NVDA MSFT,NASDAQ:AAPL", {"NVDA": "added", "MSFT": "existing", "NASDAQ:AAPL": "added"}),
        (
            "NASDAQ:NVDA NYSE:BRK.B\nNASDAQ:AAPL",
            {"NASDAQ:NVDA": "added", "NYSE:BRK.B": "added", "NASDAQ:AAPL": "added"},
        ),
        ("NASDAQ:NVDA;NYSE:ANET", {"NASDAQ:NVDA": "added", "NYSE:ANET": "added"}),
        (
            "NASDAQ:NVDA NYSE:BRK.B\tTVC:DXY",
            {"NASDAQ:NVDA": "added", "NYSE:BRK.B": "added", "TVC:DXY": "invalid"},
        ),
        (
            "EURONEXT:AIR,LSE:BP,EURONEXT:MC,TVC:DXY,BINANCE:BTCUSDT",
            {
                "EURONEXT:AIR": "invalid",
                "LSE:BP": "invalid",
                "EURONEXT:MC": "invalid",
                "TVC:DXY": "invalid",
                "BINANCE:BTCUSDT": "invalid",
            },
        ),
        (
            "KRX:247540,KRX:005930,NYSE:BRK.B,NYSE:BF.B,EURONEXT:AIR,NASDAQ:NVDA,NVDA",
            {
                "KRX:247540": "added",
                "KRX:005930": "added",
                "NYSE:BRK.B": "added",
                "NYSE:BF.B": "added",
                "EURONEXT:AIR": "invalid",
                "NASDAQ:NVDA": "added",
                "NVDA": "added",
            },
        ),
        ("NVDA, msft\nAAPL\tNVDA\n\n", {"NVDA": "added", "MSFT": "existing", "AAPL": "added"}),
        ("AAPL,MSFT\nNVDA", {"AAPL": "added", "MSFT": "existing", "NVDA": "added"}),
        ("BRK.B,BF.B", {"BRK.B": "added", "BF.B": "added"}),
        ('"NVDA","MSFT"\n"AAPL"', {"NVDA": "added", "MSFT": "existing", "AAPL": "added"}),
        ('AAPL,"Apple, Inc."\nMSFT,"Microsoft, Corp"', {"AAPL": "added", "MSFT": "existing"}),
    ],
)
def test_every_symbol_cell_is_imported_skipped_or_reported(content, expected):
    assert _outcome_by_cell(content) == expected

"""The terminal commands, called the way the console script calls them."""

import io
import json
import os
import shutil
import subprocess
import sys
import threading
import urllib.error
from pathlib import Path

import pytest
from mcp.types import LATEST_PROTOCOL_VERSION

from stockscreen_mcp import cli, server

from conftest import publish, stock

pytestmark = pytest.mark.usefixtures("use_fake_data")


def passing(symbol, **fields):
    return stock(symbol, **{"rs_rating": 95, "stage": 2, "minervini_score": 90, "ma_alignment": True, **fields})


def set_rows(static_data, market, rows):
    static_data.scan(market)[1][:] = rows


def test_bare_invocation_runs_the_server_without_touching_stdout(monkeypatch, capsys):
    started = []
    monkeypatch.setattr(server, "main", lambda: started.append("server"))
    monkeypatch.setattr(sys, "argv", ["stockscreen-mcp"])

    assert cli.main() == 0

    assert started == ["server"]
    assert capsys.readouterr().out == ""


def test_screen_alone_lists_ids_and_names(capsys):
    assert cli.main(["screen"]) == 0

    captured = capsys.readouterr()
    assert captured.out == (
        "minervini             Minervini Trend Template\n"
        "correction_survivors  Correction Survivors\n"
        "se_cup_handle         Cup with Handle\n"
    )
    assert captured.err == "US data as of 2026-10-09\n"


def test_table_is_aligned_and_counts_what_it_shows(use_fake_data, capsys):
    set_rows(use_fake_data, "US", [
        passing("AAA", company_name="Alpha Holdings", current_price=1234.5, price_change_1d=1.5,
                ibd_industry_group="Semiconductors"),
        passing("BRK-B", company_name=None, current_price=9.0, price_change_1d=-0.25, minervini_score=80),
        passing("LOW", minervini_score=70),
    ])

    assert cli.main(["screen", "minervini", "--limit", "2"]) == 0

    captured = capsys.readouterr()
    assert captured.out == (
        "SYMBOL  NAME               PRICE  RS  STAGE      1D  GROUP\n"
        "AAA     Alpha Holdings  1,234.50  95      2  +1.50%  Semiconductors\n"
        "BRK-B   -                   9.00  95      2  -0.25%  -\n"
        "2 of 3\n"
    )
    assert captured.err == "US data as of 2026-10-09\n"


def test_table_aligns_double_width_names_and_clips_long_ones(use_fake_data, capsys):
    set_rows(use_fake_data, "HK", [
        passing("0001.HK", company_name="華智數媒", current_price=1.0, price_change_1d=1.0),
        passing("0002.HK", company_name="ABCD", current_price=1.0, price_change_1d=1.0),
        passing("0003.HK", company_name="X" * 40, current_price=1.0, price_change_1d=1.0),
    ])

    assert cli.main(["screen", "minervini", "--market", "hk"]) == 0

    assert capsys.readouterr().out == (
        "SYMBOL   NAME                              PRICE  RS  STAGE      1D  GROUP\n"
        "0001.HK  華智數媒                           1.00  95      2  +1.00%  -\n"
        "0002.HK  ABCD                               1.00  95      2  +1.00%  -\n"
        f"0003.HK  {'X' * 32}   1.00  95      2  +1.00%  -\n"
        "3 of 3\n"
    )


def test_names_a_legacy_windows_stdout_cannot_encode_do_not_crash(use_fake_data, monkeypatch):
    set_rows(use_fake_data, "HK", [passing("0001.HK", company_name="華", current_price=1.0, price_change_1d=1.0)])
    stdout = io.TextIOWrapper(io.BytesIO(), encoding="cp1252")
    monkeypatch.setattr(sys, "stdout", stdout)

    assert cli.main(["screen", "minervini", "--market", "HK"]) == 0

    stdout.flush()
    assert b"0001.HK  ?" in stdout.buffer.getvalue()


def test_json_prints_limited_full_rows_with_date_and_total(capsys):
    assert cli.main(["screen", "minervini", "--json", "--limit", "1"]) == 0

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert {k: payload[k] for k in ("market", "screen", "as_of_date", "status", "total")} == {
        "market": "US", "screen": "minervini", "as_of_date": "2026-10-09", "status": "current", "total": 2,
    }
    assert [(row["symbol"], row["exchange"]) for row in payload["rows"]] == [("AAA", "XNAS")]
    assert captured.err == "US data as of 2026-10-09\n"


def test_a_late_market_warns_and_still_succeeds(fake_publish, capsys):
    fake_publish.manifest["markets"]["US"]["publication"]["state"] = "stale"

    assert cli.main(["screen", "minervini", "--limit", "1"]) == 0

    captured = capsys.readouterr()
    assert captured.err == (
        "US data as of 2026-10-09\n"
        "Warning: the US data is late, so 2026-10-09 may not be the latest session.\n"
    )
    assert captured.out.endswith("1 of 2\n")


def test_watchlist_defaults_to_a_dated_file_with_exact_contents(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)

    assert cli.main(["screen", "minervini", "--tradingview"]) == 0

    assert (tmp_path / "minervini-US-2026-10-09.txt").read_bytes() == b"NASDAQ:AAA,NYSE:BBB"
    assert capsys.readouterr().err == (
        "US data as of 2026-10-09\n"
        "Wrote 2 of 2 matches to minervini-US-2026-10-09.txt\n"
    )


def test_watchlist_writes_every_match_past_the_tool_cap_and_ignores_limit(use_fake_data, tmp_path, capsys):
    set_rows(use_fake_data, "US", [passing(f"S{i:03d}") for i in range(150)])
    target = tmp_path / "all.txt"

    assert cli.main(["screen", "minervini", "--limit", "5", "--tradingview", str(target)]) == 0

    tokens = target.read_text().split(",")
    assert len(tokens) == 150
    assert (tokens[0], tokens[-1]) == ("NASDAQ:S000", "NASDAQ:S149")
    assert capsys.readouterr().err.endswith(f"Wrote 150 of 150 matches to {target}\n")


def test_watchlist_dash_writes_stdout(capsys):
    assert cli.main(["screen", "minervini", "--tradingview", "-"]) == 0

    captured = capsys.readouterr()
    assert captured.out == "NASDAQ:AAA,NYSE:BBB\n"
    assert captured.err == "US data as of 2026-10-09\nWrote 2 of 2 matches to stdout\n"


@pytest.mark.parametrize(
    ("market", "symbol", "exchange", "token"),
    [
        ("US", "NVDA", "XNAS", "NASDAQ:NVDA"),
        ("US", "BRK-B", "XNYS", "NYSE:BRK.B"),
        ("US", "IMO", "XASE", "AMEX:IMO"),
        ("HK", "0700.HK", None, "HKEX:700"),
        ("HK", "9988.HK", None, "HKEX:9988"),
        ("HK", "7203.T", None, "TSE:7203"),
        ("HK", "130A.T", None, "TSE:130A"),
        ("HK", "SAP.DE", None, "XETR:SAP"),
        ("HK", "BIP-UN.TO", None, "TSX:BIP.UN"),
        ("HK", "NET-UN.V", None, "TSXV:NET.UN"),
        ("HK", "BAJAJ-AUTO.NS", None, "NSE:BAJAJ_AUTO"),
        ("HK", "1301.TW", None, "TWSE:1301"),
        ("HK", "6716.TWO", None, "TPEX:6716"),
        ("HK", "034730.KS", None, "KRX:034730"),
        ("HK", "263860.KQ", None, "KRX:263860"),
        ("HK", "600189.SS", None, "SSE:600189"),
        ("HK", "002172.SZ", None, "SZSE:002172"),
        ("HK", "D05.SI", None, "SGX:D05"),
        ("HK", "BHP.AX", None, "ASX:BHP"),
    ],
)
def test_every_mapping_row_produces_its_tradingview_token(use_fake_data, capsys, market, symbol, exchange, token):
    set_rows(use_fake_data, market, [passing(symbol, exchange=exchange)])

    assert cli.main(["screen", "minervini", "--market", market, "--tradingview", "-"]) == 0

    assert capsys.readouterr().out == f"{token}\n"


def test_unmappable_symbols_are_listed_and_the_rest_still_written(use_fake_data, capsys):
    set_rows(use_fake_data, "HK", [
        passing("0700.HK"), passing("430047.BJ"), passing("1155.KL"), passing("ODD.XX"), passing("NOSUFFIX"),
    ])

    assert cli.main(["screen", "minervini", "--market", "HK", "--tradingview", "-"]) == 0

    captured = capsys.readouterr()
    assert captured.out == "HKEX:700\n"
    assert captured.err == (
        "HK data as of 2026-10-09\n"
        "Skipped 4 with no TradingView symbol: 1155.KL, 430047.BJ, NOSUFFIX, ODD.XX\n"
        "Malaysia is unsupported: TradingView lists its stocks by name instead of numeric code.\n"
        "Wrote 1 of 5 matches to stdout\n"
    )


def test_a_us_row_with_an_unknown_exchange_is_skipped(use_fake_data, capsys):
    set_rows(use_fake_data, "US", [passing("AAA"), passing("OTC", exchange="OTCM")])

    assert cli.main(["screen", "minervini", "--tradingview", "-"]) == 0

    captured = capsys.readouterr()
    assert captured.out == "NASDAQ:AAA\n"
    assert "Skipped 1 with no TradingView symbol: OTC\n" in captured.err
    assert "Malaysia" not in captured.err


def test_nothing_mappable_exits_1_and_writes_no_file(use_fake_data, tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    set_rows(use_fake_data, "HK", [passing("1155.KL")])

    assert cli.main(["screen", "minervini", "--market", "HK", "--tradingview"]) == 1

    assert list(tmp_path.iterdir()) == []
    assert capsys.readouterr().err == (
        "HK data as of 2026-10-09\n"
        "Skipped 1 with no TradingView symbol: 1155.KL\n"
        "Malaysia is unsupported: TradingView lists its stocks by name instead of numeric code.\n"
        "Nothing to write: no match has a TradingView symbol.\n"
    )


def test_zero_matches_exit_0_and_write_no_file(use_fake_data, tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    set_rows(use_fake_data, "US", [])

    assert cli.main(["screen", "minervini", "--tradingview"]) == 0
    assert list(tmp_path.iterdir()) == []
    assert capsys.readouterr().err == (
        "US data as of 2026-10-09\n"
        "No US stocks match minervini, so no file was written.\n"
    )

    assert cli.main(["screen", "minervini"]) == 0
    assert capsys.readouterr().out == "SYMBOL  NAME  PRICE  RS  STAGE  1D  GROUP\n0 of 0\n"


def test_an_unwritable_target_is_one_line_and_exits_1(tmp_path, capsys):
    target = tmp_path / "missing" / "list.txt"

    assert cli.main(["screen", "minervini", "--tradingview", str(target)]) == 1

    assert capsys.readouterr().err == (
        "US data as of 2026-10-09\n"
        f"Could not write {target}: No such file or directory\n"
    )


def test_unknown_screen_suggests_close_matches_and_exits_1(capsys):
    assert cli.main(["screen", "minerva"]) == 1

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == (
        "Error: Unknown screen 'minerva'. Did you mean: minervini? "
        "Available: minervini, correction_survivors, se_cup_handle.\n"
    )


def test_unknown_market_exits_1(capsys):
    assert cli.main(["screen", "--market", "ZZ"]) == 1

    assert capsys.readouterr().err == "Error: Unknown market 'ZZ'. Available markets: AU, HK, US.\n"


def test_unreachable_data_is_one_line_and_exits_1(fake_publish, capsys):
    def offline(url):
        raise urllib.error.URLError("offline")

    fake_publish.on_fetch = offline

    assert cli.main(["screen", "minervini"]) == 1

    assert capsys.readouterr().err == (
        "Error: Could not reach the published data at https://example.test/static-data/manifest.json: "
        "URLError('offline')\n"
    )


@pytest.mark.parametrize(
    "argv",
    [
        ["bogus"],
        ["screen", "minervini", "--limit", "0"],
        ["screen", "minervini", "--limit", "many"],
        ["screen", "minervini", "--json", "--tradingview"],
        ["screen", "--tradingview", "minervini"],
        ["screen", "--json"],
    ],
)
def test_usage_errors_exit_2(argv, capsys):
    with pytest.raises(SystemExit) as exit_info:
        cli.main(argv)

    assert exit_info.value.code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.startswith("usage: stockscreen-mcp")


def _entry_point() -> str:
    path = shutil.which("stockscreen-mcp", path=str(Path(sys.executable).parent))
    assert path, "stockscreen-mcp is not installed next to this interpreter"
    return path


def _env(data_url: str) -> dict[str, str]:
    return {**os.environ, "STOCKSCREEN_DATA_URL": data_url}


def test_installed_entry_point_answers_an_mcp_handshake_over_stdio():
    # Port 9 refuses connections, so this stays offline: a handshake must not read the publish.
    process = subprocess.Popen(
        [_entry_point()], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        text=True, encoding="utf-8", env=_env("http://127.0.0.1:9/"),
    )
    watchdog = threading.Timer(60, process.kill)
    watchdog.start()

    def send(message):
        process.stdin.write(json.dumps({"jsonrpc": "2.0", **message}) + "\n")
        process.stdin.flush()

    try:
        send({"id": 1, "method": "initialize", "params": {
            "protocolVersion": LATEST_PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": {"name": "test", "version": "0"},
        }})
        initialized = json.loads(process.stdout.readline())
        send({"method": "notifications/initialized"})
        send({"id": 2, "method": "tools/list"})
        tools = json.loads(process.stdout.readline())
        process.stdin.close()
        process.stdout.close()
        process.wait()
    finally:
        watchdog.cancel()
        process.kill()

    assert initialized["id"] == 1
    assert initialized["result"]["serverInfo"]["name"] == "stockscreen"
    assert {tool["name"] for tool in tools["result"]["tools"]} == {
        "list_markets", "list_screens", "run_screen", "get_stock", "market_overview", "group_rankings",
    }


def test_a_reader_that_leaves_early_is_not_a_traceback(tmp_path):
    manifest, files = publish("g/1/")
    for name, content in {"manifest.json": manifest, **files}.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(content), encoding="utf-8")

    process = subprocess.Popen(
        [_entry_point(), "screen", "minervini"], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, encoding="utf-8", env=_env(tmp_path.as_uri()),
    )
    process.stdout.close()
    stderr = process.stderr.read()
    process.stderr.close()

    assert process.wait(timeout=60) == 0
    assert stderr == "US data as of 2026-10-09\n"

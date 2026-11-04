from __future__ import annotations

import ast
import gzip
import json
import subprocess
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import pytest
from app.domain.markets import SUPPORTED_MARKET_CODES
from app.scripts import scoreboard_ledger as ledger
from app.scripts.scoreboard_ledger import (
    LedgerCorrupt,
    LedgerDay,
    Origin,
    PresetMatch,
    Slot,
    asset_name,
)

BACKEND = Path(ledger.__file__).resolve().parents[2]
DEFINITION = "241a0f65f493c25e"
LINE_1006 = (
    '{"as_of_date":"2026-10-06","published_at":"2026-10-07T01:00:00Z","origin":"artifact_pre_stage_fix",'
    '"presets":{"canslim":{"definition":"241a0f65f493c25e","symbols":[]},'
    '"minervini":{"definition":"241a0f65f493c25e","symbols":["AAPL","MSFT"]}}}'
)


def write_gz(path: Path, text: str) -> Path:
    path.write_bytes(gzip.compress(text.encode("utf-8")))
    return path


def us_path(tmp_path: Path) -> Path:
    return tmp_path / "scoreboard-ledger-us-v1.jsonl.gz"


def day_line(as_of: str, *, published_at: str = "2026-11-05T01:00:00Z", origin: str = "live") -> str:
    return (
        f'{{"as_of_date":"{as_of}","published_at":"{published_at}","origin":"{origin}",'
        f'"presets":{{"minervini":{{"definition":"{DEFINITION}","symbols":["AAPL"]}}}}}}'
    )


def test_asset_names_follow_the_options_history_pattern() -> None:
    assert asset_name("US") == "scoreboard-ledger-us-v1.jsonl.gz"
    assert asset_name("HK", Slot.PREVIOUS) == "scoreboard-ledger-hk-v1.previous.jsonl.gz"


def test_ledger_markets_match_the_market_catalog() -> None:
    assert set(ledger.MARKETS) == set(SUPPORTED_MARKET_CODES)


def test_a_written_line_loads_back_into_the_named_shape(tmp_path) -> None:
    path = write_gz(us_path(tmp_path), LINE_1006 + "\n")
    assert ledger.load_ledger(path, "US") == {
        date(2026, 10, 6): LedgerDay(
            as_of_date=date(2026, 10, 6),
            published_at=datetime(2026, 10, 7, 1, tzinfo=timezone.utc),
            origin=Origin.ARTIFACT_PRE_STAGE_FIX,
            presets={
                "canslim": PresetMatch(DEFINITION, ()),
                "minervini": PresetMatch(DEFINITION, ("AAPL", "MSFT")),
            },
        )
    }


def test_canonical_line_reproduces_the_stored_bytes() -> None:
    assert ledger.canonical_line(ledger.parse_line(LINE_1006)) == LINE_1006


def test_an_empty_gzip_is_a_valid_empty_ledger(tmp_path) -> None:
    assert ledger.load_ledger(write_gz(us_path(tmp_path), ""), "US") == {}


def test_truncated_gzip_is_corrupt(tmp_path) -> None:
    whole = gzip.compress((LINE_1006 + "\n").encode())
    path = us_path(tmp_path)
    path.write_bytes(whole[: len(whole) // 2])
    with pytest.raises(LedgerCorrupt, match="does not decompress"):
        ledger.load_ledger(path, "US")


def test_zero_byte_file_is_corrupt_not_an_empty_ledger(tmp_path) -> None:
    path = us_path(tmp_path)
    path.write_bytes(b"")
    with pytest.raises(LedgerCorrupt, match="zero bytes"):
        ledger.load_ledger(path, "US")


def test_plain_text_is_corrupt(tmp_path) -> None:
    path = us_path(tmp_path)
    path.write_text(LINE_1006)
    with pytest.raises(LedgerCorrupt, match="does not decompress"):
        ledger.load_ledger(path, "US")


@pytest.mark.parametrize(
    "bad_line",
    [
        "not json",
        '["2026-10-06"]',
        day_line("2026-13-40"),
        day_line("2026-10-06", published_at="2026-10-07"),
        day_line("2026-10-06", origin="recovered"),
        day_line("2026-10-06").replace(DEFINITION, "SHORT"),
        day_line("2026-10-06").replace('["AAPL"]', '["MSFT","AAPL"]'),
        day_line("2026-10-06").replace('["AAPL"]', '["AAPL","AAPL"]'),
        day_line("2026-10-06").replace('["AAPL"]', '"AAPL"'),
        day_line("2026-10-06").replace(',"origin"', ', "origin"'),
        day_line("2026-10-06").replace('"origin":"live"', '"origin":"live","extra":1'),
        '{"as_of_date":"2026-10-06"}',
    ],
)
def test_a_bad_line_is_corrupt(tmp_path, bad_line) -> None:
    path = write_gz(us_path(tmp_path), f"{day_line('2026-10-05')}\n{bad_line}\n")
    with pytest.raises(LedgerCorrupt):
        ledger.load_ledger(path, "US")


def test_a_missing_final_newline_is_corrupt(tmp_path) -> None:
    path = write_gz(us_path(tmp_path), day_line("2026-10-05"))
    with pytest.raises(LedgerCorrupt, match="newline"):
        ledger.load_ledger(path, "US")


def test_a_duplicate_date_is_corrupt(tmp_path) -> None:
    line = day_line("2026-10-05")
    path = write_gz(us_path(tmp_path), f"{line}\n{line}\n")
    with pytest.raises(LedgerCorrupt, match="duplicate date 2026-10-05"):
        ledger.load_ledger(path, "US")


def test_another_markets_file_is_corrupt(tmp_path) -> None:
    path = write_gz(tmp_path / asset_name("HK"), day_line("2026-10-05") + "\n")
    with pytest.raises(LedgerCorrupt, match="not a US ledger"):
        ledger.load_ledger(path, "US")
    assert list(ledger.load_ledger(path, "HK")) == [date(2026, 10, 5)]


def test_the_stage_fix_date_splits_artifact_origins() -> None:
    assert ledger.artifact_origin(date(2026, 10, 21)) is Origin.ARTIFACT_PRE_STAGE_FIX
    assert ledger.artifact_origin(date(2026, 10, 22)) is Origin.ARTIFACT


def seed_record(as_of: str, created_at: str, symbols: list[str], *, market: str = "US") -> dict:
    return {
        "as_of_date": as_of,
        "generated_at": created_at,
        "created_at": created_at,
        "artifact_id": 1,
        "market": market,
        "rows": 10,
        "presets": {
            "minervini": {"definition": DEFINITION, "symbols": symbols},
            "se_double_bottom": {"definition": DEFINITION, "symbols": []},
        },
    }


def write_seed_input(directory: Path, market: str, records: list[dict]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(record) + "\n" for record in records)
    write_gz(directory / f"{market}.jsonl.gz", text)


def test_seed_keeps_the_first_artifact_per_date_and_stamps_origin_and_publication(tmp_path) -> None:
    source = tmp_path / "input"
    write_seed_input(
        source,
        "US",
        [
            seed_record("2026-10-22", "2026-10-23T02:00:00Z", ["NVDA"]),
            seed_record("2026-10-09", "2026-10-10T09:00:00Z", ["LATER"]),
            seed_record("2026-10-09", "2026-10-10T02:00:00Z", ["FIRST"]),
        ],
    )
    output = tmp_path / "out"

    assert ledger.main(["seed", "--input", str(source), "--output", str(output)]) == 0

    days = ledger.load_ledger(output / asset_name("US"), "US")
    assert list(days) == [date(2026, 10, 9), date(2026, 10, 22)]
    assert days[date(2026, 10, 9)] == LedgerDay(
        as_of_date=date(2026, 10, 9),
        published_at=datetime(2026, 10, 10, 2, tzinfo=timezone.utc),
        origin=Origin.ARTIFACT_PRE_STAGE_FIX,
        presets={
            "minervini": PresetMatch(DEFINITION, ("FIRST",)),
            "se_double_bottom": PresetMatch(DEFINITION, ()),
        },
    )
    assert days[date(2026, 10, 22)].origin is Origin.ARTIFACT


def test_seed_writes_an_empty_ledger_for_a_market_with_no_records(tmp_path) -> None:
    source = tmp_path / "input"
    write_seed_input(source, "US", [seed_record("2026-10-22", "2026-10-23T02:00:00Z", ["NVDA"])])
    output = tmp_path / "out"

    assert ledger.main(["seed", "--input", str(source), "--output", str(output)]) == 0

    assert sorted(path.name for path in output.iterdir()) == sorted(asset_name(m) for m in ledger.MARKETS)
    assert ledger.load_ledger(output / asset_name("MY"), "MY") == {}


def test_seed_refuses_a_record_from_another_market(tmp_path, capsys) -> None:
    source = tmp_path / "input"
    write_seed_input(source, "US", [seed_record("2026-10-22", "2026-10-23T02:00:00Z", ["0700.HK"], market="HK")])

    assert ledger.main(["seed", "--input", str(source), "--output", str(tmp_path / "out")]) == 1

    assert "US input holds a HK record" in capsys.readouterr().out


def test_seed_refuses_truncated_input(tmp_path, capsys) -> None:
    source = tmp_path / "input"
    source.mkdir()
    whole = gzip.compress(json.dumps(seed_record("2026-10-22", "2026-10-23T02:00:00Z", [])).encode() + b"\n")
    (source / "US.jsonl.gz").write_bytes(whole[:20])

    assert ledger.main(["seed", "--input", str(source), "--output", str(tmp_path / "out")]) == 1

    assert "::error::US.jsonl.gz" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("canonical", "previous", "expected_exit", "expected_line"),
    [
        ("good", None, 0, "US record job appends to: canonical"),
        ("good", "corrupt", 0, "US record job appends to: canonical"),
        (None, "good", 1, "US record job appends to: previous"),
        ("corrupt", "good", 1, "US record job appends to: nothing, it stops red"),
        ("corrupt", None, 1, "US record job appends to: nothing, it stops red"),
        (None, None, 1, "US record job appends to: nothing, it stops red"),
    ],
)
def test_verify_reports_the_state_and_exits_nonzero_unless_canonical_is_usable(
    tmp_path, capsys, canonical, previous, expected_exit, expected_line
) -> None:
    contents = {"good": day_line("2026-10-05") + "\n", "corrupt": "not json\n"}
    if canonical:
        write_gz(tmp_path / asset_name("US"), contents[canonical])
    if previous:
        write_gz(tmp_path / asset_name("US", Slot.PREVIOUS), contents[previous])

    assert ledger.main(["verify", "--market", "US", "--dir", str(tmp_path)]) == expected_exit

    assert expected_line in capsys.readouterr().out


@pytest.mark.parametrize(
    ("canonical", "previous", "expected_dates"),
    [
        ("2026-10-06", "2026-10-05", [date(2026, 10, 6)]),
        (None, "2026-10-05", [date(2026, 10, 5)]),
        ("corrupt", "2026-10-05", [date(2026, 10, 5)]),
        ("corrupt", None, None),
        (None, None, None),
    ],
)
def test_the_market_build_reads_previous_only_when_canonical_is_unusable(
    tmp_path, canonical, previous, expected_dates
) -> None:
    if canonical:
        text = "not json\n" if canonical == "corrupt" else day_line(canonical) + "\n"
        write_gz(tmp_path / asset_name("US"), text)
    if previous:
        write_gz(tmp_path / asset_name("US", Slot.PREVIOUS), day_line(previous) + "\n")

    days = ledger.load_for_build(tmp_path, "US")

    assert (None if days is None else list(days)) == expected_dates


def test_module_imports_only_the_standard_library() -> None:
    tree = ast.parse(Path(ledger.__file__).read_text(encoding="utf-8"))
    imported = {
        (node.module or "").split(".")[0] if isinstance(node, ast.ImportFrom) else alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert imported - set(sys.stdlib_module_names) - {"render_seo_pages", "stockscreen_mcp"} == set()


def test_command_line_runs_without_site_packages(tmp_path) -> None:
    write_gz(tmp_path / asset_name("US"), day_line("2026-10-05") + "\n")
    result = subprocess.run(
        [sys.executable, "-S", "-m", "app.scripts.scoreboard_ledger", "verify", "--market", "US", "--dir", str(tmp_path)],
        cwd=BACKEND,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "US canonical: valid, 1 days, through 2026-10-05" in result.stdout

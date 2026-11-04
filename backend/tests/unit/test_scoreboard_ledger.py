from __future__ import annotations

import ast
import gzip
import json
import os
import subprocess
import sys
import zipfile
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


def test_seed_refuses_an_input_directory_with_no_market_files(tmp_path, capsys) -> None:
    output = tmp_path / "out"

    assert ledger.main(["seed", "--input", str(tmp_path / "mistyped"), "--output", str(output)]) == 1

    assert "No <market>.jsonl.gz input" in capsys.readouterr().out
    assert not output.exists()


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


FAKE_GH = r"""
import json, os, pathlib, shutil, sys

state = pathlib.Path(os.environ["FAKE_GH_STATE"])
args = sys.argv[1:]
with (state / "calls.log").open("a") as log:
    log.write(json.dumps(args) + "\n")
release = state / "release"
faults = json.loads((state / "faults.json").read_text())


def fail(message):
    print(message, file=sys.stderr)
    raise SystemExit(1)


if args[:2] == ["release", "view"]:
    if "view" in faults:
        fail(faults["view"])
    if not release.exists():
        fail("release not found")
    print(json.dumps({"assets": [{"name": p.name} for p in sorted(release.iterdir())]}))
elif args[:2] == ["release", "download"]:
    name = args[args.index("--pattern") + 1]
    if name in faults.get("download", []):
        fail("HTTP 502")
    shutil.copy(release / name, pathlib.Path(args[args.index("--dir") + 1]) / name)
elif args[:2] == ["release", "upload"]:
    source = pathlib.Path(args[3])
    remaining = faults.get("upload", {}).get(source.name, 0)
    if remaining:
        faults["upload"][source.name] = remaining - 1
        (state / "faults.json").write_text(json.dumps(faults))
        fail("HTTP 502")
    shutil.copy(source, release / source.name)
elif args[:2] == ["api", "repos/{owner}/{repo}"]:
    print("main")
elif "--slurp" in args:
    print((state / "artifacts.json").read_text())
elif args[0] == "api" and args[1].endswith("/zip"):
    artifact_id = args[1].split("/")[-2]
    sys.stdout.buffer.write((state / f"{artifact_id}.zip").read_bytes())
else:
    fail(f"unexpected gh call: {args}")
"""


class FakeRelease:
    def __init__(self, state: Path) -> None:
        self.state = state
        self.assets = state / "release"
        self.faults: dict = {}

    def create(self, **assets: dict[str, str]) -> None:
        self.assets.mkdir(exist_ok=True)
        for name, text in assets.items():
            write_gz(self.assets / name, text)

    def set_faults(self, **faults) -> None:
        (self.state / "faults.json").write_text(json.dumps(faults))

    def asset_lines(self, name: str) -> list[str]:
        return gzip.decompress((self.assets / name).read_bytes()).decode().splitlines()

    def uploads(self) -> list[str]:
        calls = [json.loads(line) for line in (self.state / "calls.log").read_text().splitlines()]
        return [Path(call[3]).name for call in calls if call[:2] == ["release", "upload"]]


@pytest.fixture
def release(tmp_path, monkeypatch) -> FakeRelease:
    state = tmp_path / "gh-state"
    state.mkdir()
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    (fake_bin / "fake_gh.py").write_text(FAKE_GH, encoding="utf-8")
    shim = fake_bin / "gh"
    shim.write_text(f'#!/bin/sh\nexec "{sys.executable}" -I -S "{fake_bin / "fake_gh.py"}" "$@"\n', encoding="utf-8")
    shim.chmod(0o755)
    monkeypatch.setenv("PATH", f"{fake_bin}:{os.environ['PATH']}")
    monkeypatch.setenv("FAKE_GH_STATE", str(state))
    monkeypatch.setattr(ledger.time, "sleep", lambda _seconds: None)
    fake = FakeRelease(state)
    fake.set_faults()
    (state / "calls.log").write_text("")
    return fake


def write_day_file(directory: Path, market: str, as_of: str, symbols: list[str] | None = None) -> None:
    directory.mkdir(exist_ok=True)
    (directory / f"{market}.json").write_text(
        json.dumps(
            {
                "market": market,
                "as_of_date": as_of,
                "presets": {"minervini": {"definition": DEFINITION, "symbols": symbols or ["AAPL"]}},
            }
        )
    )


def record(days: Path) -> int:
    return ledger.main(["record", "--days-dir", str(days)])


US = asset_name("US")
US_PREVIOUS = asset_name("US", Slot.PREVIOUS)
HK = asset_name("HK")
HK_PREVIOUS = asset_name("HK", Slot.PREVIOUS)
OLD_LINE = day_line("2026-11-02", published_at="2026-11-03T01:00:00Z")


def test_record_appends_a_new_day_after_saving_the_old_canonical_as_previous(tmp_path, release) -> None:
    release.create(**{US: OLD_LINE + "\n"})
    days = tmp_path / "days"
    write_day_file(days, "US", "2026-11-03", ["MSFT"])

    assert record(days) == 0

    assert release.uploads() == [US_PREVIOUS, US]
    assert release.asset_lines(US_PREVIOUS) == [OLD_LINE]
    lines = release.asset_lines(US)
    assert lines[0] == OLD_LINE
    added = ledger.parse_line(lines[1])
    assert (added.as_of_date, added.origin, added.presets["minervini"].symbols) == (
        date(2026, 11, 3),
        Origin.LIVE,
        ("MSFT",),
    )


def test_release_absent_warns_and_exits_zero_without_uploading(tmp_path, release, capsys) -> None:
    days = tmp_path / "days"
    write_day_file(days, "US", "2026-11-03")

    assert record(days) == 0

    assert "::warning::Release scoreboard-ledger-data does not exist" in capsys.readouterr().out
    assert release.uploads() == []


def test_a_listing_error_is_not_mistaken_for_an_absent_release(tmp_path, release, capsys) -> None:
    release.create(**{US: OLD_LINE + "\n"})
    release.set_faults(view="HTTP 502: bad gateway")
    days = tmp_path / "days"
    write_day_file(days, "US", "2026-11-03")

    assert record(days) == 1

    assert "gh release view failed: HTTP 502" in capsys.readouterr().out
    assert release.uploads() == []


def test_canonical_missing_resumes_from_previous_without_rewriting_it(tmp_path, release) -> None:
    release.create(**{US_PREVIOUS: OLD_LINE + "\n"})
    days = tmp_path / "days"
    write_day_file(days, "US", "2026-11-03")

    assert record(days) == 0

    assert release.uploads() == [US]
    assert release.asset_lines(US_PREVIOUS) == [OLD_LINE]
    assert release.asset_lines(US)[0] == OLD_LINE
    assert len(release.asset_lines(US)) == 2


def test_corrupt_canonical_is_red_uploads_nothing_for_it_and_continues_other_markets(
    tmp_path, release, capsys
) -> None:
    release.create(**{US: "not json\n", US_PREVIOUS: OLD_LINE + "\n", HK: OLD_LINE + "\n"})
    corrupt_bytes = (release.assets / US).read_bytes()
    days = tmp_path / "days"
    write_day_file(days, "US", "2026-11-03")
    write_day_file(days, "HK", "2026-11-03")

    assert record(days) == 1

    assert "::error::US: US ledger has no usable base" in capsys.readouterr().out
    assert release.uploads() == [HK_PREVIOUS, HK]
    assert (release.assets / US).read_bytes() == corrupt_bytes
    assert release.asset_lines(US_PREVIOUS) == [OLD_LINE]


def test_both_assets_absent_in_an_existing_release_is_red_and_never_starts_fresh(
    tmp_path, release, capsys
) -> None:
    release.create()
    days = tmp_path / "days"
    write_day_file(days, "US", "2026-11-03")

    assert record(days) == 1

    output = capsys.readouterr().out
    assert "'canonical': 'absent', 'previous': 'absent'" in output
    assert release.uploads() == []
    assert list(release.assets.iterdir()) == []


def test_an_upload_failure_is_retried_three_times_and_then_succeeds(tmp_path, release) -> None:
    release.create(**{US: OLD_LINE + "\n"})
    release.set_faults(upload={US: 2})
    days = tmp_path / "days"
    write_day_file(days, "US", "2026-11-03")

    assert record(days) == 0

    assert release.uploads() == [US_PREVIOUS, US, US, US]
    assert len(release.asset_lines(US)) == 2


def test_a_previous_upload_failure_skips_canonical(tmp_path, release, capsys) -> None:
    release.create(**{US: OLD_LINE + "\n"})
    release.set_faults(upload={US_PREVIOUS: 3})
    days = tmp_path / "days"
    write_day_file(days, "US", "2026-11-03")

    assert record(days) == 1

    assert "could not upload scoreboard-ledger-us-v1.previous.jsonl.gz" in capsys.readouterr().out
    assert release.uploads() == [US_PREVIOUS, US_PREVIOUS, US_PREVIOUS]
    assert release.asset_lines(US) == [OLD_LINE]


def test_a_canonical_upload_failure_leaves_previous_saved_and_is_red(tmp_path, release) -> None:
    release.create(**{US: OLD_LINE + "\n"})
    release.set_faults(upload={US: 3})
    days = tmp_path / "days"
    write_day_file(days, "US", "2026-11-03")

    assert record(days) == 1

    assert release.uploads() == [US_PREVIOUS, US, US, US]
    assert release.asset_lines(US_PREVIOUS) == [OLD_LINE]


def test_a_download_failure_is_red_not_an_absent_asset(tmp_path, release, capsys) -> None:
    release.create(**{US: OLD_LINE + "\n"})
    release.set_faults(download=[US])
    days = tmp_path / "days"
    write_day_file(days, "US", "2026-11-03")

    assert record(days) == 1

    assert "gh release download failed: HTTP 502" in capsys.readouterr().out
    assert release.uploads() == []


@pytest.mark.parametrize("as_of", ["2026-11-02", "2026-10-30"], ids=["same-date-rerun", "rewind"])
def test_a_date_at_or_before_the_ledger_end_is_a_no_op_with_no_upload(tmp_path, release, as_of, capsys) -> None:
    release.create(**{US: OLD_LINE + "\n"})
    before = (release.assets / US).read_bytes()
    days = tmp_path / "days"
    write_day_file(days, "US", as_of, ["CHANGED"])

    assert record(days) == 0

    assert f"US: {as_of} is not new, nothing to upload" in capsys.readouterr().out
    assert release.uploads() == []
    assert (release.assets / US).read_bytes() == before


def test_a_rerun_after_recording_uploads_nothing(tmp_path, release) -> None:
    release.create(**{US: OLD_LINE + "\n"})
    days = tmp_path / "days"
    write_day_file(days, "US", "2026-11-03")
    assert record(days) == 0
    recorded = (release.assets / US).read_bytes()

    write_day_file(days, "US", "2026-11-03", ["REBUILT"])
    assert record(days) == 0

    assert release.uploads() == [US_PREVIOUS, US]
    assert (release.assets / US).read_bytes() == recorded


def test_a_day_file_for_another_market_is_red(tmp_path, release, capsys) -> None:
    release.create(**{US: OLD_LINE + "\n"})
    days = tmp_path / "days"
    write_day_file(days, "US", "2026-11-03")
    (days / "HK.json").write_text((days / "US.json").read_text())

    assert record(days) == 1

    assert "HK.json holds US matches" in capsys.readouterr().out
    assert release.uploads() == [US_PREVIOUS, US]


def test_no_extracted_days_is_red(tmp_path, release, capsys) -> None:
    release.create(**{US: OLD_LINE + "\n"})
    (tmp_path / "days").mkdir()

    assert record(tmp_path / "days") == 1

    assert "No extracted days" in capsys.readouterr().out


def test_verify_without_a_directory_reads_the_release(release, capsys) -> None:
    release.create(**{US: "not json\n", US_PREVIOUS: OLD_LINE + "\n"})

    assert ledger.main(["verify", "--market", "US"]) == 1

    output = capsys.readouterr().out
    assert "US canonical: corrupt, 0 days, through -" in output
    assert "US previous: valid, 1 days, through 2026-11-02" in output


MINERVINI_EXPRESSION = {
    "expression_version": 1,
    "group_join": "any",
    "groups": [],
    "required": {
        "conditions": [
            {"field": "minervini_score", "kind": "range", "max": None, "min": 70},
            {"field": "rs_rating", "kind": "range", "max": None, "min": 70},
            {"field": "ma_alignment", "kind": "boolean", "value": True},
            {"field": "stage", "kind": "range", "max": 2, "min": 2},
        ],
        "enabled": True,
        "id": "required",
        "match": "all",
        "name": "Always require",
    },
}
PRESETS = [
    {"id": "minervini", "name": "Minervini Trend Template", "filter_expression": MINERVINI_EXPRESSION},
    {
        "id": "nothing_matches",
        "filter_expression": {
            "expression_version": 1,
            "group_join": "any",
            "groups": [],
            "required": {
                "conditions": [{"field": "rs_rating", "kind": "range", "max": None, "min": 1000}],
                "enabled": True,
                "id": "required",
                "match": "all",
                "name": "Always require",
            },
        },
    },
    {"id": "no_expression", "filters": {}},
]
PASSING = {"minervini_score": 90, "rs_rating": 95, "ma_alignment": True, "stage": 2}
ROWS_INITIAL = [{"symbol": "ZZZ", **PASSING}, {"symbol": "FAIL", **PASSING, "stage": 3}]
ROWS_CHUNK = [{"symbol": "AAA", **PASSING}, {"symbol": "ZZZ", **PASSING, "minervini_score": 95}]


def write_scan(market_dir: Path, market: str, as_of: str) -> None:
    prefix = f"markets/{market.lower()}"
    (market_dir / "scan" / "chunks").mkdir(parents=True)
    (market_dir / "scan" / "chunks" / "chunk-0001.json").write_text(json.dumps({"rows": ROWS_CHUNK}))
    (market_dir / "scan" / "manifest.json").write_text(
        json.dumps(
            {
                "as_of_date": as_of,
                "preset_screens": PRESETS,
                "initial_rows": ROWS_INITIAL,
                "chunks": [{"path": f"{prefix}/scan/chunks/chunk-0001.json"}],
            }
        )
    )


def market_entry(market: str, as_of: str) -> dict:
    return {
        "as_of_date": as_of,
        "pages": {"scan": {"path": f"markets/{market.lower()}/scan/manifest.json"}},
    }


def write_site(site: Path, *, data_root: str = "g/20261104T023713Z-e4822212/") -> None:
    write_scan(site / data_root / "markets" / "us", "US", "2026-11-03")
    site.joinpath("manifest.json").write_text(
        json.dumps(
            {
                "data_root": data_root,
                "unavailable_markets": ["MY"],
                "markets": {
                    "US": market_entry("US", "2026-11-03"),
                    "MY": market_entry("MY", "2026-11-03"),
                    "CA": {"as_of_date": "2026-11-03", "pages": {}},
                },
            }
        )
    )


def test_definition_hash_stays_the_seed_capture_hash() -> None:
    assert ledger.definition_hash(MINERVINI_EXPRESSION) == "241a0f65f493c25e"


def test_extract_writes_what_the_site_serves_including_zero_match_presets(tmp_path) -> None:
    site = tmp_path / "site"
    write_site(site)
    output = tmp_path / "days"

    assert ledger.main(["extract", "--site-dir", str(site), "--output", str(output)]) == 0

    assert sorted(path.name for path in output.iterdir()) == ["US.json"]
    payload = json.loads((output / "US.json").read_text())
    assert payload["market"] == "US"
    assert payload["as_of_date"] == "2026-11-03"
    assert payload["presets"]["minervini"] == {"definition": "241a0f65f493c25e", "symbols": ["AAA", "ZZZ"]}
    assert payload["presets"]["nothing_matches"]["symbols"] == []
    assert "no_expression" not in payload["presets"]


def test_extract_reads_a_site_built_without_generations(tmp_path) -> None:
    site = tmp_path / "site"
    write_site(site, data_root="")
    manifest = json.loads((site / "manifest.json").read_text())
    manifest.pop("data_root")
    (site / "manifest.json").write_text(json.dumps(manifest))
    output = tmp_path / "days"

    assert ledger.main(["extract", "--site-dir", str(site), "--output", str(output)]) == 0

    assert json.loads((output / "US.json").read_text())["as_of_date"] == "2026-11-03"


def test_extract_keeps_good_markets_when_one_is_damaged(tmp_path, capsys) -> None:
    site = tmp_path / "site"
    write_site(site)
    manifest = json.loads((site / "manifest.json").read_text())
    manifest["markets"]["HK"] = market_entry("HK", "2026-11-03")
    (site / "manifest.json").write_text(json.dumps(manifest))
    hk_scan = site / manifest["data_root"] / "markets" / "hk" / "scan"
    hk_scan.mkdir(parents=True)
    (hk_scan / "manifest.json").write_text("{truncated")
    output = tmp_path / "days"

    assert ledger.main(["extract", "--site-dir", str(site), "--output", str(output)]) == 1

    assert "::warning::HK matches not extracted" in capsys.readouterr().out
    assert sorted(path.name for path in output.iterdir()) == ["US.json"]


def test_an_extracted_day_records_as_live(tmp_path, release) -> None:
    site = tmp_path / "site"
    write_site(site)
    output = tmp_path / "days"
    assert ledger.main(["extract", "--site-dir", str(site), "--output", str(output)]) == 0
    release.create(**{US: OLD_LINE + "\n"})

    assert record(output) == 0

    added = ledger.parse_line(release.asset_lines(US)[1])
    assert added.origin is Origin.LIVE
    assert added.presets["minervini"] == PresetMatch("241a0f65f493c25e", ("AAA", "ZZZ"))


def test_extract_and_record_run_without_site_packages(tmp_path, release) -> None:
    site = tmp_path / "site"
    write_site(site)
    release.create(**{US: OLD_LINE + "\n"})
    output = tmp_path / "days"
    for arguments in (
        ["extract", "--site-dir", str(site), "--output", str(output)],
        ["record", "--days-dir", str(output)],
    ):
        result = subprocess.run(
            [sys.executable, "-S", "-m", "app.scripts.scoreboard_ledger", *arguments],
            cwd=BACKEND,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stdout + result.stderr
    assert release.uploads() == [US_PREVIOUS, US]


def write_artifact(release: FakeRelease, tmp_path: Path, artifact_id: int, as_of: str) -> None:
    built = tmp_path / f"artifact-{artifact_id}"
    write_scan(built, "US", as_of)
    (built / "manifest.market.json").write_text(json.dumps({"entry": market_entry("US", as_of)}))
    (built / "breadth.json").write_text("{}")
    with zipfile.ZipFile(release.state / f"{artifact_id}.zip", "w") as archive:
        for path in built.rglob("*"):
            if path.is_file():
                archive.write(path, path.relative_to(built).as_posix())


def artifact_listing(rows: list[tuple[int, str, str]], *, repository: int = 1) -> list[dict]:
    return [
        {
            "id": artifact_id,
            "expired": False,
            "created_at": created_at,
            "workflow_run": {"head_branch": branch, "repository_id": 1, "head_repository_id": repository},
        }
        for artifact_id, created_at, branch in rows
    ]


def test_recover_inserts_the_earliest_default_branch_artifact_by_date_as_artifact_origin(
    tmp_path, release
) -> None:
    earlier = day_line("2026-10-20", published_at="2026-10-21T01:00:00Z", origin="artifact_pre_stage_fix")
    later = day_line("2026-10-29", published_at="2026-10-30T01:00:00Z")
    release.create(**{US: f"{earlier}\n{later}\n"})
    write_artifact(release, tmp_path, 10, "2026-10-22")
    write_artifact(release, tmp_path, 11, "2026-10-23")
    write_artifact(release, tmp_path, 12, "2026-10-23")
    listing = [
        *artifact_listing([(99, "2026-10-23T22:00:00Z", "feature")]),
        *artifact_listing([(98, "2026-10-23T22:30:00Z", "main")], repository=2),
        *artifact_listing(
            [
                (12, "2026-10-24T09:00:00Z", "main"),
                (11, "2026-10-24T01:00:00Z", "main"),
                (10, "2026-10-23T01:00:00Z", "main"),
            ]
        ),
    ]
    (release.state / "artifacts.json").write_text(json.dumps([{"artifacts": listing}]))

    assert ledger.main(["recover", "--market", "US", "--date", "2026-10-23"]) == 0

    assert release.uploads() == [US_PREVIOUS, US]
    lines = release.asset_lines(US)
    assert lines[0] == earlier
    assert lines[2] == later
    recovered = ledger.parse_line(lines[1])
    assert recovered.as_of_date == date(2026, 10, 23)
    assert recovered.published_at == datetime(2026, 10, 24, 1, tzinfo=timezone.utc)
    assert recovered.origin is Origin.ARTIFACT
    assert release.asset_lines(US_PREVIOUS) == [earlier, later]


def test_recover_labels_a_day_before_the_stage_fix(tmp_path, release) -> None:
    release.create(**{US: ""})
    write_artifact(release, tmp_path, 10, "2026-10-20")
    (release.state / "artifacts.json").write_text(
        json.dumps([{"artifacts": artifact_listing([(10, "2026-10-21T01:00:00Z", "main")])}])
    )

    assert ledger.main(["recover", "--market", "US", "--date", "2026-10-20"]) == 0

    assert ledger.parse_line(release.asset_lines(US)[0]).origin is Origin.ARTIFACT_PRE_STAGE_FIX


def test_recover_leaves_a_date_the_ledger_holds_alone(tmp_path, release, capsys) -> None:
    release.create(**{US: OLD_LINE + "\n"})
    write_artifact(release, tmp_path, 10, "2026-11-02")
    (release.state / "artifacts.json").write_text(
        json.dumps([{"artifacts": artifact_listing([(10, "2026-11-03T02:00:00Z", "main")])}])
    )

    assert ledger.main(["recover", "--market", "US", "--date", "2026-11-02"]) == 0

    assert "US: 2026-11-02 is not new" in capsys.readouterr().out
    assert release.uploads() == []


def test_recover_with_no_retained_artifact_reports_the_gap(tmp_path, release, capsys) -> None:
    release.create(**{US: OLD_LINE + "\n"})
    write_artifact(release, tmp_path, 10, "2026-10-29")
    (release.state / "artifacts.json").write_text(
        json.dumps([{"artifacts": artifact_listing([(10, "2026-10-30T02:00:00Z", "main")])}])
    )

    assert ledger.main(["recover", "--market", "US", "--date", "2026-10-28"]) == 1

    assert "No retained default-branch US artifact holds 2026-10-28" in capsys.readouterr().out
    assert release.uploads() == []


def test_recover_skips_a_damaged_artifact_and_uses_the_next(tmp_path, release, capsys) -> None:
    release.create(**{US: ""})
    (release.state / "10.zip").write_bytes(b"not a zip")
    write_artifact(release, tmp_path, 11, "2026-10-23")
    (release.state / "artifacts.json").write_text(
        json.dumps(
            [
                {
                    "artifacts": artifact_listing(
                        [(10, "2026-10-24T01:00:00Z", "main"), (11, "2026-10-24T05:00:00Z", "main")]
                    )
                }
            ]
        )
    )

    assert ledger.main(["recover", "--market", "US", "--date", "2026-10-23"]) == 0

    assert "::warning::Skipping artifact 10" in capsys.readouterr().out
    assert ledger.parse_line(release.asset_lines(US)[0]).published_at == datetime(
        2026, 10, 24, 5, tzinfo=timezone.utc
    )

from __future__ import annotations

import ast
import gzip
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from app.scripts import market_history as history
from app.scripts import scoreboard_ledger as ledger

BACKEND = Path(history.__file__).resolve().parents[2]
ROOT = BACKEND.parent
TAG = "market-history-us-v1"
NAME_1106 = "us-2026-11-06.json.gz"
PUBLISHED_AT = "2026-11-07T02:03:30Z"


def rs_at_least(minimum: int) -> dict:
    return {
        "expression_version": 1,
        "group_join": "any",
        "groups": [],
        "required": {
            "conditions": [{"field": "rs_rating", "kind": "range", "max": None, "min": minimum}],
            "enabled": True,
            "id": "required",
            "match": "all",
            "name": "Always require",
        },
    }


PRESET_SCREENS = [
    {"id": "nothing_matches", "name": "Nothing", "filter_expression": rs_at_least(1000)},
    {"id": "high_rs", "name": "High RS", "filter_expression": rs_at_least(90)},
    {"id": "no_expression", "name": "Legacy", "filters": {}},
]
INITIAL_ROWS = [
    {
        "symbol": "ZZZ",
        "rs_rating": 95.0,
        "stage": 2,
        "price_sparkline_data": [1, 2],
        "rs_sparkline_data": [3],
        "field_availability": {"rs_rating": "ok"},
    },
    {"symbol": "MMM", "rs_rating": 50.0, "stage": 3},
]
CHUNK_ROWS = [{"symbol": "AAA", "rs_rating": 91.0, "sector": "Tech", "field_availability": {}}]
EXTRACTED_US = {
    "schema_version": "market-history-v1",
    "market": "US",
    "as_of_date": "2026-11-06",
    "presets": {
        "high_rs": {
            "name": "High RS",
            "definition": "ebcec1d4e2601181",
            "filter_expression": rs_at_least(90),
            "symbols": ["AAA", "ZZZ"],
        },
        "nothing_matches": {
            "name": "Nothing",
            "definition": "1d27163429a659c2",
            "filter_expression": rs_at_least(1000),
            "symbols": [],
        },
    },
    "columns": ["rs_rating", "sector", "stage", "symbol"],
    "rows": [
        [91.0, "Tech", None, "AAA"],
        [50.0, None, 3, "MMM"],
        [95.0, None, 2, "ZZZ"],
    ],
}


def write_scan(
    market_dir: Path,
    market: str,
    as_of: str,
    *,
    initial_rows=INITIAL_ROWS,
    chunk_rows=CHUNK_ROWS,
    presets=PRESET_SCREENS,
) -> None:
    (market_dir / "scan" / "chunks").mkdir(parents=True)
    (market_dir / "scan" / "chunks" / "chunk-0001.json").write_text(json.dumps({"rows": chunk_rows}))
    (market_dir / "scan" / "manifest.json").write_text(
        json.dumps(
            {
                "as_of_date": as_of,
                "preset_screens": presets,
                "initial_rows": initial_rows,
                "chunks": [{"path": f"markets/{market.lower()}/scan/chunks/chunk-0001.json"}],
            }
        )
    )


def market_entry(market: str, as_of: str) -> dict:
    return {"as_of_date": as_of, "pages": {"scan": {"path": f"markets/{market.lower()}/scan/manifest.json"}}}


def write_site(site: Path, *, data_root: str = "g/20261107T065232Z-afbe281b/") -> None:
    write_scan(site / data_root / "markets" / "us", "US", "2026-11-06")
    (site / "manifest.json").write_text(
        json.dumps(
            {
                "data_root": data_root,
                "unavailable_markets": ["MY"],
                "markets": {
                    "US": market_entry("US", "2026-11-06"),
                    "MY": market_entry("MY", "2026-11-06"),
                    "CA": {"as_of_date": "2026-11-06", "pages": {}},
                },
            }
        )
    )


def extract(site: Path, output: Path) -> int:
    return history.main(["extract", "--site-dir", str(site), "--output", str(output)])


def test_extract_writes_the_exact_document_and_skips_markets_the_site_does_not_serve(tmp_path) -> None:
    site = tmp_path / "site"
    write_site(site)
    output = tmp_path / "days"

    assert extract(site, output) == 0

    assert sorted(path.name for path in output.iterdir()) == ["US.json"]
    assert json.loads((output / "US.json").read_text()) == EXTRACTED_US


def test_a_row_with_different_fields_keeps_its_values_under_the_right_columns(tmp_path) -> None:
    site = tmp_path / "site"
    write_site(site)
    output = tmp_path / "days"

    assert extract(site, output) == 0

    document = json.loads((output / "US.json").read_text())
    by_symbol = {row[document["columns"].index("symbol")]: dict(zip(document["columns"], row)) for row in document["rows"]}
    assert by_symbol["AAA"] == {"rs_rating": 91.0, "sector": "Tech", "stage": None, "symbol": "AAA"}
    assert by_symbol["ZZZ"] == {"rs_rating": 95.0, "sector": None, "stage": 2, "symbol": "ZZZ"}


def test_extract_reads_a_site_built_without_generations(tmp_path) -> None:
    site = tmp_path / "site"
    write_site(site, data_root="")
    manifest = json.loads((site / "manifest.json").read_text())
    manifest.pop("data_root")
    (site / "manifest.json").write_text(json.dumps(manifest))
    output = tmp_path / "days"

    assert extract(site, output) == 0

    assert json.loads((output / "US.json").read_text())["as_of_date"] == "2026-11-06"


def test_extract_keeps_good_markets_and_marks_a_damaged_one(tmp_path, capsys) -> None:
    site = tmp_path / "site"
    write_site(site)
    manifest = json.loads((site / "manifest.json").read_text())
    manifest["markets"]["HK"] = market_entry("HK", "2026-11-06")
    (site / "manifest.json").write_text(json.dumps(manifest))
    hk_scan = site / manifest["data_root"] / "markets" / "hk" / "scan"
    hk_scan.mkdir(parents=True)
    (hk_scan / "manifest.json").write_text("{truncated")
    output = tmp_path / "days"

    assert extract(site, output) == 1

    assert "::warning::HK session not extracted" in capsys.readouterr().out
    assert sorted(path.name for path in output.iterdir()) == ["HK.failed", "US.json"]
    assert "JSONDecodeError" in (output / "HK.failed").read_text()


def test_heavy_fields_match_the_ones_the_mcp_server_drops() -> None:
    source = (ROOT / "stockscreen-mcp" / "src" / "stockscreen_mcp" / "data.py").read_text(encoding="utf-8")
    assignment = next(
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "HEAVY_FIELDS"
    )
    assert history.HEAVY_FIELDS == frozenset(ast.literal_eval(assignment.value.args[0]))


FAKE_GH = r"""
import json, os, pathlib, shutil, sys

state = pathlib.Path(os.environ["FAKE_GH_STATE"])
args = sys.argv[1:]
with (state / "calls.log").open("a") as log:
    log.write(json.dumps(args) + "\n")
faults = json.loads((state / "faults.json").read_text())
states = json.loads((state / "states.json").read_text())


def fail(message):
    (state / "faults.json").write_text(json.dumps(faults))
    (state / "states.json").write_text(json.dumps(states))
    print(message, file=sys.stderr)
    raise SystemExit(1)


def release(tag):
    return state / "releases" / tag


if args[:2] == ["release", "view"]:
    if "view" in faults:
        fail(faults["view"])
    if not release(args[2]).exists():
        fail("release not found")
    print(
        json.dumps(
            {"assets": [{"name": p.name, "state": states.get(p.name, "uploaded")} for p in sorted(release(args[2]).iterdir())]}
        )
    )
elif args[:2] == ["release", "create"]:
    outcome = faults.get("create")
    if outcome == "lost the race":
        release(args[2]).mkdir(parents=True)
        fail(f"a release with the same tag name already exists: {args[2]}")
    if outcome:
        fail(outcome)
    release(args[2]).mkdir(parents=True)
elif args[:2] == ["release", "download"]:
    name = args[args.index("--pattern") + 1]
    if name in faults.get("download", {}):
        fail(faults["download"][name])
    shutil.copy(release(args[2]) / name, pathlib.Path(args[args.index("--dir") + 1]) / name)
elif args[:2] == ["release", "upload"]:
    source = pathlib.Path(args[3])
    target = release(args[2]) / source.name
    if source.name in faults.get("race", {}):
        target.write_text(faults["race"].pop(source.name))
    if target.exists() and "--clobber" not in args:
        fail(f"asset under the same name already exists: [{source.name}]")
    if source.name in faults.get("died_midway", []):
        faults["died_midway"].remove(source.name)
        target.write_bytes(b"")
        states[source.name] = "starter"
        fail("HTTP 502")
    if faults.get("upload", {}).get(source.name, 0):
        faults["upload"][source.name] -= 1
        fail("HTTP 502")
    shutil.copy(source, target)
    if source.name in faults.get("appears", {}):
        (release(args[2]) / faults["appears"][source.name]).write_text("another run")
    states.pop(source.name, None)
    if source.name in faults.get("listed_as", {}):
        states[source.name] = faults["listed_as"][source.name]
    (state / "states.json").write_text(json.dumps(states))
elif args[:2] == ["api", "repos/{owner}/{repo}"]:
    print("main")
elif "--slurp" in args:
    print((state / "artifacts.json").read_text())
elif args[0] == "api" and args[1].endswith("/zip"):
    artifact_id = args[1].split("/")[-2]
    if artifact_id in faults.get("artifact", []):
        fail("HTTP 502")
    sys.stdout.buffer.write((state / f"{artifact_id}.zip").read_bytes())
else:
    fail(f"unexpected gh call: {args}")
"""


class FakeRelease:
    def __init__(self, state: Path) -> None:
        self.state = state

    def directory(self, tag: str = TAG) -> Path:
        return self.state / "releases" / tag

    def create(self, tag: str = TAG) -> None:
        self.directory(tag).mkdir(parents=True)

    def put(self, name: str, body: bytes, *, state: str | None = None, tag: str = TAG) -> None:
        self.directory(tag).mkdir(parents=True, exist_ok=True)
        (self.directory(tag) / name).write_bytes(body)
        if state:
            states = json.loads((self.state / "states.json").read_text())
            states[name] = state
            (self.state / "states.json").write_text(json.dumps(states))

    def read(self, name: str, tag: str = TAG) -> bytes:
        return (self.directory(tag) / name).read_bytes()

    def session(self, name: str = NAME_1106, tag: str = TAG) -> dict:
        return json.loads(gzip.decompress(self.read(name, tag)))

    def index(self, tag: str = TAG) -> dict:
        return json.loads(self.read("index.json", tag))

    def assets(self, tag: str = TAG) -> list[str]:
        return sorted(path.name for path in self.directory(tag).iterdir())

    def set_faults(self, **faults) -> None:
        (self.state / "faults.json").write_text(json.dumps(faults))

    def calls(self) -> list[list[str]]:
        return [json.loads(line) for line in (self.state / "calls.log").read_text().splitlines()]

    def uploads(self) -> list[str]:
        return [Path(call[3]).name for call in self.calls() if call[:2] == ["release", "upload"]]

    def set_artifacts(self, listing: list[dict]) -> None:
        (self.state / "artifacts.json").write_text(json.dumps([{"artifacts": listing}]))


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
    (state / "states.json").write_text("{}")
    (state / "calls.log").write_text("")
    return fake


def document_for(market: str = "US", as_of: str = "2026-11-06", **overrides) -> dict:
    return {**EXTRACTED_US, "market": market, "as_of_date": as_of, **overrides}


def write_days(directory: Path, *documents: dict) -> Path:
    directory.mkdir(exist_ok=True)
    for document in documents:
        (directory / f"{document['market']}.json").write_text(json.dumps(document))
    return directory


def record(days: Path, *flags: str, published_at: str = PUBLISHED_AT) -> int:
    return history.main(["record", "--days-dir", str(days), "--published-at", published_at, *flags])


def index_body(*sessions: str, market: str = "US") -> bytes:
    return json.dumps(
        {"schema_version": "market-history-index-v1", "market": market, "sessions": list(sessions)},
        separators=(",", ":"),
    ).encode()


def test_record_stamps_publication_and_origin_on_the_extracted_document(tmp_path, release, capsys) -> None:
    release.create()
    days = write_days(tmp_path / "days", document_for())

    assert record(days) == 0

    output = capsys.readouterr().out
    assert "US: 2026-11-06 (live) uploaded\n" in output
    assert "US: index of 1 sessions uploaded\n" in output
    assert release.session() == {**EXTRACTED_US, "published_at": "2026-11-07T02:03:30Z", "origin": "live"}
    assert release.assets() == ["index.json", NAME_1106]


def test_the_session_gzip_has_no_clock_in_it_and_the_same_input_gives_the_same_bytes(tmp_path, release) -> None:
    release.create()
    days = write_days(tmp_path / "days", document_for())
    assert record(days) == 0
    first = release.read(NAME_1106)
    assert first[4:8] == b"\x00\x00\x00\x00"

    (release.directory() / NAME_1106).unlink()
    shuffled = dict(reversed(list(document_for().items())))
    write_days(tmp_path / "days", shuffled)
    assert record(days) == 0

    assert release.read(NAME_1106) == first


def test_record_never_overwrites_a_session_already_listed(tmp_path, release) -> None:
    release.create()
    release.put(NAME_1106, b"first publication")
    release.put("index.json", index_body("2026-11-06"))
    days = write_days(tmp_path / "days", document_for())

    assert record(days) == 0

    assert release.read(NAME_1106) == b"first publication"
    assert release.uploads() == []


def test_a_session_upload_never_passes_clobber_and_only_the_index_does(tmp_path, release) -> None:
    release.create()
    days = write_days(tmp_path / "days", document_for())

    assert record(days) == 0

    uploads = [call for call in release.calls() if call[:2] == ["release", "upload"]]
    assert [(Path(call[3]).name, "--clobber" in call) for call in uploads] == [(NAME_1106, False), ("index.json", True)]


def test_an_absent_older_date_is_added_and_the_index_stays_sorted(tmp_path, release) -> None:
    release.create()
    release.put(NAME_1106, b"first publication")
    release.put("index.json", index_body("2026-11-06"))
    days = write_days(tmp_path / "days", document_for(as_of="2026-11-03"))

    assert record(days) == 0

    assert release.session("us-2026-11-03.json.gz")["as_of_date"] == "2026-11-03"
    assert release.read(NAME_1106) == b"first publication"
    assert release.index() == {
        "schema_version": "market-history-index-v1",
        "market": "US",
        "sessions": ["2026-11-03", "2026-11-06"],
    }


def test_a_missing_release_is_created_as_a_prerelease_that_never_becomes_latest(tmp_path, release) -> None:
    days = write_days(tmp_path / "days", document_for())

    assert record(days) == 0

    create = next(call for call in release.calls() if call[:2] == ["release", "create"])
    assert create[:4] == ["release", "create", TAG, "--prerelease"]
    assert "--latest=false" in create
    assert release.assets() == ["index.json", NAME_1106]


def test_a_release_another_run_created_first_is_used(tmp_path, release) -> None:
    release.set_faults(create="lost the race")
    days = write_days(tmp_path / "days", document_for())

    assert record(days) == 0

    assert release.assets() == ["index.json", NAME_1106]


def test_a_release_that_cannot_be_created_is_red_and_uploads_nothing(tmp_path, release, capsys) -> None:
    release.set_faults(create="HTTP 403: Resource not accessible")
    days = write_days(tmp_path / "days", document_for())

    assert record(days) == 1

    assert "::error::US: gh release create failed: HTTP 403" in capsys.readouterr().out
    assert release.uploads() == []


def test_a_listing_that_fails_is_not_mistaken_for_a_missing_release(tmp_path, release, capsys) -> None:
    release.set_faults(view="HTTP 500")
    days = write_days(tmp_path / "days", document_for())

    assert record(days) == 1

    assert "::error::US: gh release view failed: HTTP 500" in capsys.readouterr().out
    assert [call[1] for call in release.calls()] == ["view"]


def put_index(release: FakeRelease, damage: str) -> None:
    if damage == "stale":
        release.put("index.json", index_body("2026-11-05"))
    elif damage == "garbled":
        release.put("index.json", b"{not json")
    elif damage == "incomplete":
        release.put("index.json", b"", state="starter")


@pytest.mark.parametrize("damage", ["missing", "stale", "garbled", "incomplete"])
def test_a_listed_session_heals_an_index_that_does_not_match_the_listing(tmp_path, release, capsys, damage) -> None:
    release.create()
    release.put(NAME_1106, b"first publication")
    put_index(release, damage)
    days = write_days(tmp_path / "days", document_for())

    assert record(days) == 0

    assert "::warning::" not in capsys.readouterr().out

    assert release.read("index.json") == index_body("2026-11-06")
    assert release.uploads() == ["index.json"]
    assert release.read(NAME_1106) == b"first publication"


def test_an_index_that_matches_the_listing_uploads_nothing(tmp_path, release) -> None:
    release.create()
    release.put(NAME_1106, b"first publication")
    release.put("index.json", index_body("2026-11-06"))
    days = write_days(tmp_path / "days", document_for())

    assert record(days) == 0

    assert release.uploads() == []


def test_a_crash_between_the_session_and_the_index_heals_on_the_next_run(tmp_path, release) -> None:
    release.create()
    release.put("us-2026-11-05.json.gz", b"older")
    release.put("index.json", index_body("2026-11-05"))
    days = write_days(tmp_path / "days", document_for())
    release.set_faults(upload={"index.json": 3})

    assert record(days) == 1
    assert release.read("index.json") == index_body("2026-11-05")
    session = release.read(NAME_1106)

    release.set_faults()
    assert record(days) == 0

    assert release.read(NAME_1106) == session
    assert release.read("index.json") == index_body("2026-11-05", "2026-11-06")
    assert release.uploads().count(NAME_1106) == 1


def test_one_failing_market_goes_red_and_the_others_still_record(tmp_path, release, capsys) -> None:
    days = write_days(
        tmp_path / "days",
        document_for("HK", rows=[]),
        document_for("SG"),
        document_for("US"),
    )

    assert record(days) == 1

    assert "::error::HK: HK 2026-11-06 has zero rows" in capsys.readouterr().out
    assert release.assets() == ["index.json", NAME_1106]
    assert release.assets("market-history-sg-v1") == ["index.json", "sg-2026-11-06.json.gz"]
    assert not release.directory("market-history-hk-v1").exists()


def test_a_failure_marker_exits_one_after_the_other_markets_finish(tmp_path, release, capsys) -> None:
    days = write_days(tmp_path / "days", document_for())
    (days / "HK.failed").write_text("JSONDecodeError('Expecting value')")

    assert record(days) == 1

    assert "::error::HK: session was not extracted: JSONDecodeError('Expecting value')" in capsys.readouterr().out
    assert release.assets() == ["index.json", NAME_1106]


def test_a_directory_with_nothing_extracted_is_red(tmp_path, release, capsys) -> None:
    days = tmp_path / "days"
    days.mkdir()

    assert record(days) == 1

    assert f"::error::No extracted sessions in {days}." in capsys.readouterr().out


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        pytest.param({"rows": []}, "US 2026-11-06 has zero rows", id="zero-rows"),
        pytest.param({"presets": {}}, "US 2026-11-06 has zero presets", id="zero-presets"),
        pytest.param(
            {"as_of_date": "2026-11-09"},
            "US as_of_date 2026-11-09 is after the day following publication, 2026-11-07",
            id="past-publication",
        ),
        pytest.param({"market": "HK"}, "US.json holds HK rows", id="wrong-market"),
        pytest.param({"schema_version": "market-history-v2"}, "US.json is market-history-v2", id="schema"),
        pytest.param({"as_of_date": "20261106"}, "US.json has as_of_date 20261106", id="unpadded-date"),
        pytest.param({"as_of_date": None}, "US.json:", id="no-date"),
    ],
)
def test_record_refuses_what_the_site_cannot_have_published(tmp_path, release, capsys, overrides, message) -> None:
    days = tmp_path / "days"
    days.mkdir()
    (days / "US.json").write_text(json.dumps({**EXTRACTED_US, **overrides}))

    assert record(days) == 1

    assert f"::error::US: {message}" in capsys.readouterr().out
    assert release.calls() == []


def test_record_refuses_a_file_for_a_market_the_site_does_not_have(tmp_path, release, capsys) -> None:
    days = write_days(tmp_path / "days", document_for("XX"))

    assert record(days) == 1

    assert "::error::XX: XX is not a supported market" in capsys.readouterr().out
    assert release.calls() == []


def test_a_truncated_file_is_refused_with_its_name(tmp_path, release, capsys) -> None:
    days = tmp_path / "days"
    days.mkdir()
    (days / "US.json").write_text('{"schema_version":')

    assert record(days) == 1

    assert "::error::US: US.json: JSONDecodeError" in capsys.readouterr().out
    assert release.calls() == []


def test_a_session_a_concurrent_run_uploaded_first_counts_as_present(tmp_path, release, capsys) -> None:
    release.create()
    release.set_faults(race={NAME_1106: "the concurrent backfill's bytes"})
    days = write_days(tmp_path / "days", document_for())

    assert record(days) == 0

    output = capsys.readouterr().out
    assert "US: 2026-11-06 (live) kept, another run uploaded it first\n" in output
    assert "2026-11-06 (live) uploaded" not in output

    assert release.read(NAME_1106) == b"the concurrent backfill's bytes"
    assert release.read("index.json") == index_body("2026-11-06")


def test_a_session_another_run_adds_during_this_one_reaches_the_index(tmp_path, release) -> None:
    release.create()
    release.set_faults(appears={NAME_1106: "us-2026-11-05.json.gz"})
    days = write_days(tmp_path / "days", document_for())

    assert record(days) == 0

    assert release.read("index.json") == index_body("2026-11-05", "2026-11-06")


def test_any_other_session_upload_failure_is_an_error_and_leaves_the_index_alone(tmp_path, release, capsys) -> None:
    release.create()
    release.put("index.json", index_body("2026-11-05"))
    release.set_faults(upload={NAME_1106: 3})
    days = write_days(tmp_path / "days", document_for())

    assert record(days) == 1

    assert f"::error::US: could not upload {NAME_1106}" in capsys.readouterr().out
    assert release.read("index.json") == index_body("2026-11-05")
    assert NAME_1106 not in release.assets()


def test_a_transient_session_upload_failure_is_retried_without_clobber(tmp_path, release) -> None:
    release.create()
    release.set_faults(upload={NAME_1106: 2})
    days = write_days(tmp_path / "days", document_for())

    assert record(days) == 0

    assert release.uploads().count(NAME_1106) == 3
    assert release.session()["origin"] == "live"


def test_an_upload_that_died_midway_blocks_only_its_date_until_the_asset_is_deleted(tmp_path, release, capsys) -> None:
    release.create()
    release.set_faults(died_midway=[NAME_1106])
    days = write_days(tmp_path / "days", document_for())

    assert record(days) == 1
    assert release.read(NAME_1106) == b""
    capsys.readouterr()

    assert record(days) == 1
    assert f"gh release delete-asset {TAG} {NAME_1106} --yes" in capsys.readouterr().out
    assert release.read(NAME_1106) == b""

    (release.directory() / NAME_1106).unlink()
    assert record(days) == 0
    assert release.session()["as_of_date"] == "2026-11-06"


def test_the_index_never_names_an_incomplete_asset_and_each_run_warns_about_it(tmp_path, release, capsys) -> None:
    release.create()
    release.put("us-2026-11-05.json.gz", b"", state="starter")
    release.put("us-2026-11-04.json.gz", b"", state="open")
    days = write_days(tmp_path / "days", document_for())

    assert record(days) == 0

    assert release.read("index.json") == index_body("2026-11-06")
    warnings = [line for line in capsys.readouterr().out.splitlines() if line.startswith("::warning::")]
    assert warnings == [
        (
            "::warning::US us-2026-11-04.json.gz is open: an upload is running or died part way. If none is running, "
            f"gh release delete-asset {TAG} us-2026-11-04.json.gz --yes; see docs/runbooks/market-history.md"
        ),
        (
            "::warning::US us-2026-11-05.json.gz is starter: an upload is running or died part way. If none is "
            f"running, gh release delete-asset {TAG} us-2026-11-05.json.gz --yes; see docs/runbooks/market-history.md"
        ),
    ]


def test_only_an_exact_session_name_counts_as_a_session(tmp_path, release) -> None:
    release.create()
    for name in ("us-2026-11-05.json.gz.bak", "old-us-2026-11-04.json.gz", "hk-2026-11-03.json.gz"):
        release.put(name, b"not a US session")
    days = write_days(tmp_path / "days", document_for())

    assert record(days) == 0

    assert release.read("index.json") == index_body("2026-11-06")


def test_a_fresh_listing_wins_over_what_this_run_assumed_it_uploaded(tmp_path, release) -> None:
    release.create()
    release.set_faults(listed_as={NAME_1106: "open"})
    days = write_days(tmp_path / "days", document_for())

    assert record(days) == 0

    assert release.read("index.json") == index_body()


@pytest.mark.parametrize(
    "stderr",
    [
        pytest.param("no assets match the file pattern", id="gone-before-gh-listed"),
        pytest.param("HTTP 404: Not Found (https://api.github.com/repos/o/r/releases/assets/1)", id="gone-after"),
    ],
)
def test_an_index_another_run_is_replacing_is_rewritten(tmp_path, release, stderr) -> None:
    release.create()
    release.put(NAME_1106, b"first publication")
    release.put("index.json", index_body("2026-11-06"))
    release.put("us-2026-11-05.json.gz", b"older")
    release.set_faults(download={"index.json": stderr})
    days = write_days(tmp_path / "days", document_for())

    assert record(days) == 0

    assert release.read("index.json") == index_body("2026-11-05", "2026-11-06")


def test_any_other_index_download_failure_is_red_and_uploads_nothing(tmp_path, release, capsys) -> None:
    release.create()
    release.put(NAME_1106, b"first publication")
    release.put("index.json", index_body("2026-11-05"))
    release.set_faults(download={"index.json": "HTTP 500: Internal Server Error"})
    days = write_days(tmp_path / "days", document_for())

    assert record(days) == 1

    assert "::error::US: gh release download failed: HTTP 500" in capsys.readouterr().out
    assert release.uploads() == []


def test_record_dry_run_writes_each_markets_files_and_changes_nothing(tmp_path, release, capsys) -> None:
    release.create()
    release.put("us-2026-11-05.json.gz", b"older")
    days = write_days(tmp_path / "days", document_for(), document_for("SG"))
    output = tmp_path / "dry"

    assert record(days, "--dry-run", str(output)) == 0

    assert release.uploads() == []
    assert release.assets() == ["us-2026-11-05.json.gz"]
    assert not release.directory("market-history-sg-v1").exists()
    assert [call[:2] for call in release.calls() if call[:2] != ["release", "view"]] == []
    assert sorted(path.name for path in (output / "us").iterdir()) == ["index.json", NAME_1106]
    assert sorted(path.name for path in (output / "sg").iterdir()) == ["index.json", "sg-2026-11-06.json.gz"]
    assert (output / "us" / "index.json").read_bytes() == index_body("2026-11-05", "2026-11-06")
    assert (output / "sg" / "index.json").read_bytes() == index_body("2026-11-06", market="SG")
    assert json.loads(gzip.decompress((output / "us" / NAME_1106).read_bytes()))["origin"] == "live"
    log = capsys.readouterr().out
    assert f"US: 2026-11-06 (live) written to {output / 'us' / NAME_1106}, the Release is unchanged\n" in log
    assert f"US: index of 2 sessions written to {output / 'us' / 'index.json'}, the Release is unchanged\n" in log
    assert " uploaded\n" not in log


def test_an_extracted_site_records_end_to_end(tmp_path, release) -> None:
    site = tmp_path / "site"
    write_site(site)
    days = tmp_path / "days"
    assert extract(site, days) == 0

    assert record(days) == 0

    assert release.session()["presets"]["high_rs"]["symbols"] == ["AAA", "ZZZ"]
    assert release.index()["sessions"] == ["2026-11-06"]


def write_artifact(release: FakeRelease, tmp_path: Path, artifact_id: int, as_of: str, **scan) -> None:
    built = tmp_path / f"artifact-{artifact_id}"
    write_scan(built, "US", as_of, **scan)
    (built / "manifest.market.json").write_text(json.dumps({"entry": market_entry("US", as_of)}))
    (built / "breadth.json").write_text("{}")
    with zipfile.ZipFile(release.state / f"{artifact_id}.zip", "w") as archive:
        for path in built.rglob("*"):
            if path.is_file():
                archive.write(path, path.relative_to(built).as_posix())


def write_damaged_deflate(path: Path) -> None:
    name = "manifest.market.json"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(name, json.dumps({"entry": list(range(500))}))
    data = bytearray(path.read_bytes())
    start = 30 + len(name)
    data[start : start + 20] = bytes(byte ^ 0xFF for byte in data[start : start + 20])
    path.write_bytes(bytes(data))


def artifact_listing(rows: list[tuple[int, str]]) -> list[dict]:
    return [
        {
            "id": artifact_id,
            "expired": False,
            "created_at": created_at,
            "workflow_run": {"head_branch": "main", "repository_id": 1, "head_repository_id": 1},
        }
        for artifact_id, created_at in rows
    ]


def backfill(*flags: str) -> int:
    return history.main(["backfill", "--market", "US", *flags])


LATER_CHUNK = [{"symbol": "LATER", "rs_rating": 99.0}]


def test_backfill_takes_the_first_artifact_per_date_oldest_first_and_labels_it_from_the_artifact(
    tmp_path, release
) -> None:
    write_artifact(release, tmp_path, 10, "2026-10-23")
    write_artifact(release, tmp_path, 11, "2026-10-23", chunk_rows=LATER_CHUNK)
    write_artifact(release, tmp_path, 12, "2026-10-20")
    release.set_artifacts(
        artifact_listing(
            [
                (11, "2026-10-24T09:00:00Z"),
                (10, "2026-10-24T01:00:00Z"),
                (12, "2026-10-21T01:00:00Z"),
            ]
        )
    )

    assert backfill() == 0

    assert release.uploads() == ["us-2026-10-20.json.gz", "us-2026-10-23.json.gz", "index.json"]
    assert release.session("us-2026-10-23.json.gz") == {
        **document_for(as_of="2026-10-23"),
        "published_at": "2026-10-24T01:00:00Z",
        "origin": "artifact",
    }
    older = release.session("us-2026-10-20.json.gz")
    assert (older["published_at"], older["origin"]) == ("2026-10-21T01:00:00Z", "artifact_pre_stage_fix")
    assert release.read("index.json") == index_body("2026-10-20", "2026-10-23")


def test_backfill_skips_a_date_already_listed_and_heals_the_index(tmp_path, release) -> None:
    release.create()
    release.put("us-2026-10-23.json.gz", b"first publication")
    write_artifact(release, tmp_path, 10, "2026-10-23")
    write_artifact(release, tmp_path, 11, "2026-10-26")
    release.set_artifacts(artifact_listing([(10, "2026-10-24T01:00:00Z"), (11, "2026-10-27T01:00:00Z")]))

    assert backfill() == 0

    assert release.read("us-2026-10-23.json.gz") == b"first publication"
    assert release.uploads() == ["us-2026-10-26.json.gz", "index.json"]
    assert release.read("index.json") == index_body("2026-10-23", "2026-10-26")


@pytest.mark.parametrize(
    ("damage", "warning"),
    [
        pytest.param("not-a-zip", "BadZipFile", id="not-a-zip"),
        pytest.param("damaged-deflate", "while decompressing data", id="damaged-deflate"),
        pytest.param("no-presets", "has zero presets", id="no-presets"),
        pytest.param("no-rows", "has zero rows", id="no-rows"),
        pytest.param("after-publication", "is after the day following publication", id="after-publication"),
    ],
)
def test_backfill_skips_a_bad_artifact_with_a_warning_and_the_next_one_takes_the_date(
    tmp_path, release, capsys, damage, warning
) -> None:
    if damage == "not-a-zip":
        (release.state / "10.zip").write_bytes(b"not a zip")
    elif damage == "damaged-deflate":
        write_damaged_deflate(release.state / "10.zip")
    elif damage == "no-presets":
        write_artifact(release, tmp_path, 10, "2026-10-23", presets=[])
    elif damage == "after-publication":
        write_artifact(release, tmp_path, 10, "2026-10-29")
    else:
        write_artifact(release, tmp_path, 10, "2026-10-23", initial_rows=[], chunk_rows=[])
    write_artifact(release, tmp_path, 11, "2026-10-23")
    release.set_artifacts(artifact_listing([(10, "2026-10-24T01:00:00Z"), (11, "2026-10-24T05:00:00Z")]))

    assert backfill() == 0

    output = capsys.readouterr().out
    assert "::warning::Skipping artifact 10" in output
    assert warning in output
    assert release.assets() == ["index.json", "us-2026-10-23.json.gz"]
    assert release.session("us-2026-10-23.json.gz")["published_at"] == "2026-10-24T05:00:00Z"


def test_a_failed_artifact_download_stops_so_a_later_artifact_cannot_take_its_date(tmp_path, release, capsys) -> None:
    write_artifact(release, tmp_path, 10, "2026-10-23")
    write_artifact(release, tmp_path, 11, "2026-10-23", chunk_rows=LATER_CHUNK)
    release.set_artifacts(artifact_listing([(10, "2026-10-24T01:00:00Z"), (11, "2026-10-24T05:00:00Z")]))
    release.set_faults(artifact=["10"])

    assert backfill() == 1

    assert "::error::gh api artifact 10 failed: HTTP 502" in capsys.readouterr().out
    assert release.uploads() == []


def test_a_backfill_with_no_usable_artifact_is_red(tmp_path, release, capsys) -> None:
    (release.state / "10.zip").write_bytes(b"not a zip")
    release.set_artifacts(artifact_listing([(10, "2026-10-24T01:00:00Z")]))

    assert backfill() == 1

    assert "::error::None of 1 retained US artifacts is usable" in capsys.readouterr().out
    assert release.uploads() == []


def test_a_backfill_with_nothing_new_is_green(tmp_path, release) -> None:
    release.create()
    release.put("us-2026-10-23.json.gz", b"first publication")
    release.put("index.json", index_body("2026-10-23"))
    write_artifact(release, tmp_path, 10, "2026-10-23")
    release.set_artifacts(artifact_listing([(10, "2026-10-24T01:00:00Z")]))

    assert backfill() == 0

    assert release.uploads() == []


def test_backfill_dry_run_writes_the_files_there_and_uploads_nothing(tmp_path, release) -> None:
    write_artifact(release, tmp_path, 10, "2026-10-23")
    release.set_artifacts(artifact_listing([(10, "2026-10-24T01:00:00Z")]))
    output = tmp_path / "dry"

    assert backfill("--dry-run", str(output)) == 0

    assert release.uploads() == []
    assert not release.directory().exists()
    assert not [call for call in release.calls() if call[:2] == ["release", "create"]]
    assert sorted(path.name for path in (output / "us").iterdir()) == ["index.json", "us-2026-10-23.json.gz"]
    assert (output / "us" / "index.json").read_bytes() == index_body("2026-10-23")
    written = json.loads(gzip.decompress((output / "us" / "us-2026-10-23.json.gz").read_bytes()))
    assert (written["published_at"], written["origin"]) == ("2026-10-24T01:00:00Z", "artifact")


@pytest.mark.parametrize(
    "arguments",
    [
        pytest.param(["--dry-run"], id="dry-run-without-a-directory"),
        pytest.param(["--market", "XX"], id="unknown-market"),
    ],
)
def test_backfill_rejects_bad_arguments_before_any_gh_call(release, arguments) -> None:
    with pytest.raises(SystemExit) as refusal:
        history.main(["backfill", "--market", "US", *arguments])

    assert refusal.value.code == 2
    assert release.calls() == []


def run_without_site_packages(*arguments: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-S", "-m", "app.scripts.market_history", *arguments],
        cwd=BACKEND,
        capture_output=True,
        text=True,
        check=False,
    )


def test_every_subcommand_runs_without_site_packages_as_the_workflows_do(tmp_path, release) -> None:
    site = tmp_path / "site"
    write_site(site)
    days = tmp_path / "days"
    write_artifact(release, tmp_path, 10, "2026-10-23")
    release.set_artifacts(artifact_listing([(10, "2026-10-24T01:00:00Z")]))

    for arguments in (
        ["extract", "--site-dir", str(site), "--output", str(days)],
        ["record", "--days-dir", str(days), "--published-at", PUBLISHED_AT],
        ["backfill", "--market", "US"],
    ):
        result = run_without_site_packages(*arguments)
        assert result.returncode == 0, result.stdout + result.stderr

    assert release.index()["sessions"] == ["2026-10-23", "2026-11-06"]

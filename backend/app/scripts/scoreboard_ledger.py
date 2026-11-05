"""Append-only ledger of the screen matches the static site published, per market.

Matches cannot be rebuilt later because fundamentals keep no history, so this is
the one Release-stored input of the scoreboard. Returns are never stored here.

Stdlib only: the record-ledger job holds contents:write and installs nothing.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
import zlib
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

MARKETS = ("US", "CA", "HK", "JP", "TW", "IN", "KR", "CN", "DE", "AU", "SG", "MY")
RELEASE_TAG = "scoreboard-ledger-data"
RUNBOOK = "docs/runbooks/scoreboard-ledger.md"
TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%SZ"
UPLOAD_ATTEMPTS = 3
_DEFINITION = re.compile(r"[0-9a-f]{16}")


class Origin(str, Enum):
    LIVE = "live"
    ARTIFACT = "artifact"
    ARTIFACT_PRE_STAGE_FIX = "artifact_pre_stage_fix"


STAGE_FIX_MERGED_AT = datetime(2026, 10, 21, 23, 5, 12, tzinfo=timezone.utc)  # #671


def artifact_origin(created_at: datetime) -> Origin:
    """A retained artifact's created_at only approximates publication, so a day read
    from one is never live. The era follows the build, not the session: US 10-21 was
    built after the merge and Asian markets' 10-21 before it.
    """
    return Origin.ARTIFACT if created_at >= STAGE_FIX_MERGED_AT else Origin.ARTIFACT_PRE_STAGE_FIX


@dataclass(frozen=True)
class PresetMatch:
    definition: str
    symbols: tuple[str, ...]


@dataclass(frozen=True)
class LedgerDay:
    as_of_date: date
    published_at: datetime
    origin: Origin
    presets: dict[str, PresetMatch]


class LedgerError(RuntimeError):
    """The ledger cannot advance for one market."""


class LedgerCorrupt(LedgerError):
    """A ledger file or input record fails strict parsing."""


class GhError(LedgerError):
    """A gh command failed."""


class AssetState(Enum):
    VALID = "valid"
    ABSENT = "absent"
    CORRUPT = "corrupt"


class Slot(Enum):
    CANONICAL = "canonical"
    PREVIOUS = "previous"


@dataclass(frozen=True)
class LedgerSources:
    """Which asset each side uses. None stops the record job and omits the scoreboard."""

    writer: Slot | None
    reader: Slot | None


# Keyed by (canonical, previous). The writer never starts fresh: an absent ledger in
# an existing Release means something deleted it. An interrupted --clobber leaves
# canonical absent with .previous intact, which is the one state it resumes from.
LEDGER_POLICY = {
    (AssetState.VALID, AssetState.VALID): LedgerSources(Slot.CANONICAL, Slot.CANONICAL),
    (AssetState.VALID, AssetState.ABSENT): LedgerSources(Slot.CANONICAL, Slot.CANONICAL),
    (AssetState.VALID, AssetState.CORRUPT): LedgerSources(Slot.CANONICAL, Slot.CANONICAL),
    (AssetState.ABSENT, AssetState.VALID): LedgerSources(Slot.PREVIOUS, Slot.PREVIOUS),
    (AssetState.ABSENT, AssetState.ABSENT): LedgerSources(None, None),
    (AssetState.ABSENT, AssetState.CORRUPT): LedgerSources(None, None),
    (AssetState.CORRUPT, AssetState.VALID): LedgerSources(None, Slot.PREVIOUS),
    (AssetState.CORRUPT, AssetState.ABSENT): LedgerSources(None, None),
    (AssetState.CORRUPT, AssetState.CORRUPT): LedgerSources(None, None),
}


def asset_name(market: str, slot: Slot = Slot.CANONICAL) -> str:
    version = "v1" if slot is Slot.CANONICAL else "v1.previous"
    return f"scoreboard-ledger-{market.lower()}-{version}.jsonl.gz"


def format_timestamp(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).strftime(TIMESTAMP_FORMAT)


def parse_timestamp(text: str) -> datetime:
    return datetime.strptime(text, TIMESTAMP_FORMAT).replace(tzinfo=timezone.utc)


def canonical_line(day: LedgerDay) -> str:
    return json.dumps(
        {
            "as_of_date": day.as_of_date.isoformat(),
            "published_at": format_timestamp(day.published_at),
            "origin": day.origin.value,
            "presets": {
                preset: {"definition": match.definition, "symbols": list(match.symbols)}
                for preset, match in sorted(day.presets.items())
            },
        },
        separators=(",", ":"),
    )


def _preset_matches(raw: dict[str, Any]) -> dict[str, PresetMatch]:
    return {
        preset: PresetMatch(match["definition"], tuple(match["symbols"]))
        for preset, match in raw.items()
    }


def parse_line(line: str) -> LedgerDay:
    try:
        raw = json.loads(line)
        day = LedgerDay(
            as_of_date=date.fromisoformat(raw["as_of_date"]),
            published_at=parse_timestamp(raw["published_at"]),
            origin=Origin(raw["origin"]),
            presets=_preset_matches(raw["presets"]),
        )
        well_formed = all(
            _DEFINITION.fullmatch(match.definition)
            and list(match.symbols) == sorted(set(match.symbols))
            for match in day.presets.values()
        )
    except (ValueError, KeyError, TypeError, AttributeError) as exc:
        raise LedgerCorrupt(f"bad line: {exc!r}") from exc
    # Re-serializing rejects unknown keys, spacing and ordering in one check, and it
    # is what lets a later append prove every old line survived byte for byte.
    if not well_formed or canonical_line(day) != line:
        raise LedgerCorrupt(f"line is not canonical: {line[:120]}")
    return day


def parse_lines(lines: list[str]) -> dict[date, LedgerDay]:
    days: dict[date, LedgerDay] = {}
    for line in lines:
        day = parse_line(line)
        if day.as_of_date in days:
            raise LedgerCorrupt(f"duplicate date {day.as_of_date}")
        days[day.as_of_date] = day
    return days


def read_lines(path: Path, market: str) -> list[str]:
    if path.name not in (asset_name(market), asset_name(market, Slot.PREVIOUS)):
        raise LedgerCorrupt(f"{path.name} is not a {market} ledger")
    raw = path.read_bytes()
    # gzip reads zero bytes as empty text, which would pass for a valid empty ledger.
    if not raw:
        raise LedgerCorrupt(f"{path.name} is zero bytes")
    try:
        text = gzip.decompress(raw).decode("utf-8")
    except (OSError, EOFError, zlib.error, UnicodeDecodeError) as exc:
        raise LedgerCorrupt(f"{path.name} does not decompress: {exc!r}") from exc
    if text and not text.endswith("\n"):
        raise LedgerCorrupt(f"{path.name} does not end with a newline")
    return text.split("\n")[:-1]


def load_ledger(path: Path, market: str) -> dict[date, LedgerDay]:
    return parse_lines(read_lines(path, market))


def write_ledger(path: Path, lines: list[str]) -> None:
    text = "".join(f"{line}\n" for line in lines)
    path.write_bytes(gzip.compress(text.encode("utf-8"), mtime=0))


@dataclass(frozen=True)
class AssetRead:
    path: Path
    state: AssetState
    lines: list[str]
    days: dict[date, LedgerDay]


def read_asset(path: Path, market: str) -> AssetRead:
    if not path.exists():
        return AssetRead(path, AssetState.ABSENT, [], {})
    try:
        lines = read_lines(path, market)
        return AssetRead(path, AssetState.VALID, lines, parse_lines(lines))
    except LedgerCorrupt as exc:
        print(f"::warning::{market} {path.name}: {exc}", flush=True)
        return AssetRead(path, AssetState.CORRUPT, [], {})


def read_assets(directory: Path, market: str) -> dict[Slot, AssetRead]:
    return {slot: read_asset(directory / asset_name(market, slot), market) for slot in Slot}


def sources_for(assets: dict[Slot, AssetRead]) -> LedgerSources:
    return LEDGER_POLICY[(assets[Slot.CANONICAL].state, assets[Slot.PREVIOUS].state)]


def lost_lines(assets: dict[Slot, AssetRead]) -> int:
    """How many lines of a valid .previous a usable canonical no longer holds, byte for byte."""
    canonical, previous = assets[Slot.CANONICAL], assets[Slot.PREVIOUS]
    return len(set(previous.lines) - set(canonical.lines)) if canonical.state is AssetState.VALID else 0


def load_for_build(directory: Path, market: str) -> dict[date, LedgerDay] | None:
    """The ledger the market build may score, or None when it must omit the asset."""
    assets = read_assets(directory, market)
    slot = sources_for(assets).reader
    return None if slot is None else assets[slot].days


def read_jsonl_gz(path: Path) -> list[dict[str, Any]]:
    try:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle]
    except (OSError, EOFError, zlib.error, ValueError) as exc:
        raise LedgerCorrupt(f"{path.name}: {exc!r}") from exc


def seed_days(records: list[dict[str, Any]], market: str) -> dict[date, LedgerDay]:
    """The first artifact per as_of_date wins, as the first publication does live."""
    days: dict[date, LedgerDay] = {}
    try:
        for record in sorted(records, key=lambda item: item["created_at"]):
            if record["market"] != market:
                raise LedgerCorrupt(f"{market} input holds a {record['market']} record")
            as_of_date = date.fromisoformat(record["as_of_date"])
            if as_of_date not in days:
                created_at = parse_timestamp(record["created_at"])
                days[as_of_date] = LedgerDay(
                    as_of_date=as_of_date,
                    published_at=created_at,
                    origin=artifact_origin(created_at),
                    presets=_preset_matches(record["presets"]),
                )
    except (KeyError, TypeError, ValueError) as exc:
        raise LedgerCorrupt(f"{market} seed record: {exc!r}") from exc
    return days


def cmd_seed(args: argparse.Namespace) -> int:
    sources = {market: args.input / f"{market}.jsonl.gz" for market in MARKETS}
    missing = ", ".join(market for market, source in sources.items() if not source.exists())
    # A mistyped path must not produce twelve valid empty ledgers ready to upload.
    if not any(source.exists() for source in sources.values()):
        raise LedgerError(f"No <market>.jsonl.gz input in {args.input}")
    if missing and not args.allow_missing:
        raise LedgerError(f"No input for {missing}; pass --allow-missing to start them empty")
    if missing:
        print(f"::warning::No input for {missing}; their ledgers start empty", flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    for market, source in sources.items():
        records = read_jsonl_gz(source) if source.exists() else []
        days = seed_days(records, market)
        target = args.output / asset_name(market)
        write_ledger(target, [canonical_line(days[day]) for day in sorted(days)])
        written = load_ledger(target, market)
        print(f"{market}: {len(written)} days from {len(records)} records, {target.stat().st_size} bytes")
    return 0


def run_gh(*args: str) -> str:
    result = subprocess.run(["gh", *args], capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise GhError(f"gh {' '.join(args[:2])} failed: {result.stderr.strip()[:400]}")
    return result.stdout


def release_assets() -> set[str] | None:
    """Asset names in the ledger Release; None when the Release does not exist."""
    result = subprocess.run(
        ["gh", "release", "view", RELEASE_TAG, "--json", "assets"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode == 0:
        return {asset["name"] for asset in json.loads(result.stdout)["assets"]}
    if "release not found" in result.stderr:
        return None
    raise GhError(f"gh release view failed: {result.stderr.strip()[:400]}")


def download_assets(listing: set[str], market: str, directory: Path) -> None:
    for slot in Slot:
        name = asset_name(market, slot)
        if name in listing:
            run_gh("release", "download", RELEASE_TAG, "--pattern", name, "--dir", str(directory), "--clobber")


def upload_asset(path: Path) -> None:
    for attempt in range(1, UPLOAD_ATTEMPTS + 1):
        result = subprocess.run(
            ["gh", "release", "upload", RELEASE_TAG, str(path), "--clobber"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            return
        print(f"Upload of {path.name} failed ({attempt}/{UPLOAD_ATTEMPTS}): {result.stderr.strip()[:400]}", flush=True)
        if attempt < UPLOAD_ATTEMPTS:
            time.sleep(attempt * 5)
    raise GhError(f"could not upload {path.name}")


def advance(
    market: str,
    day: LedgerDay,
    listing: set[str],
    work: Path,
    is_new: Callable[[date, dict[date, LedgerDay]], bool],
) -> bool:
    """Add ``day`` to the market's ledger in the Release; False when it is not new."""
    download_assets(listing, market, work)
    assets = read_assets(work, market)
    writer = sources_for(assets).writer
    if writer is None:
        states = {slot.value: asset.state.value for slot, asset in assets.items()}
        raise LedgerError(f"{market} ledger has no usable base {states}; see {RUNBOOK}")
    base = assets[writer]
    if not is_new(day.as_of_date, base.days):
        print(f"{market}: {day.as_of_date} is not new, nothing to upload", flush=True)
        return False

    # Rotating would overwrite the only copy that still shows the loss.
    if lost := lost_lines(assets):
        raise LedgerCorrupt(f"{market} canonical lost {lost} line(s) that .previous holds; see {RUNBOOK}")

    line = canonical_line(day)
    position = sum(existing < day.as_of_date for existing in base.days)
    target = work / "next" / asset_name(market)
    target.parent.mkdir()
    write_ledger(target, [*base.lines[:position], line, *base.lines[position:]])
    written = read_lines(target, market)
    parse_lines(written)
    if [kept for kept in written if kept != line] != base.lines:
        raise LedgerCorrupt(f"{market}: the rewrite changed an existing line")

    # .previous always comes from a canonical that just parsed clean. Resuming from
    # .previous leaves it in place, since no clean canonical exists to replace it.
    if writer is Slot.CANONICAL:
        snapshot = work / asset_name(market, Slot.PREVIOUS)
        shutil.copyfile(base.path, snapshot)
        upload_asset(snapshot)
    upload_asset(target)
    print(f"{market}: added {day.as_of_date} ({day.origin.value})", flush=True)
    return True


def read_day_file(path: Path, published_at: datetime) -> tuple[str, LedgerDay]:
    market = path.stem
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if raw["market"] != market:
            raise LedgerCorrupt(f"{path.name} holds {raw['market']} matches")
        day = LedgerDay(
            as_of_date=date.fromisoformat(raw["as_of_date"]),
            published_at=published_at,
            origin=Origin.LIVE,
            presets=_preset_matches(raw["presets"]),
        )
    except (OSError, KeyError, TypeError, ValueError, AttributeError) as exc:
        raise LedgerCorrupt(f"{path.name}: {exc!r}") from exc
    return market, parse_line(canonical_line(day))


def cmd_record(args: argparse.Namespace) -> int:
    listing = release_assets()
    if listing is None:
        print(f"::warning::Release {RELEASE_TAG} does not exist: the ledger is not initialized yet.", flush=True)
        return 0
    paths = sorted(args.days_dir.glob("*.json"))
    if not paths:
        print(f"::error::No extracted days in {args.days_dir}.", flush=True)
        return 1
    published_at = datetime.now(timezone.utc)
    failed = False
    for path in paths:
        try:
            market, day = read_day_file(path, published_at)
            with tempfile.TemporaryDirectory() as tmp:
                advance(market, day, listing, Path(tmp), lambda as_of, days: as_of > max(days, default=date.min))
        except LedgerError as exc:
            print(f"::error::{path.stem}: {exc}", flush=True)
            failed = True
    return 1 if failed else 0


def definition_hash(filter_expression: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(filter_expression, sort_keys=True).encode()).hexdigest()[:16]


def site_readers() -> tuple[Callable[..., dict[str, Any]], Callable[..., bool]]:
    """The site's own row loader and evaluator, so matches are what the site shows."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
    from render_seo_pages import load_market
    from stockscreen_mcp.screens import evaluate_expression

    return load_market, evaluate_expression


def match_presets(
    scan: dict[str, Any],
    rows: list[dict[str, Any]],
    evaluate_expression: Callable[..., bool],
) -> dict[str, PresetMatch]:
    return {
        preset["id"]: PresetMatch(
            definition_hash(preset["filter_expression"]),
            tuple(sorted(row["symbol"] for row in rows if evaluate_expression(row, preset["filter_expression"]))),
        )
        for preset in scan["preset_screens"]
        if "filter_expression" in preset
    }


def cmd_extract(args: argparse.Namespace) -> int:
    load_market, evaluate_expression = site_readers()
    manifest = json.loads((args.site_dir / "manifest.json").read_text(encoding="utf-8"))
    data_root = args.site_dir / (manifest.get("data_root") or "")
    unavailable = set(manifest.get("unavailable_markets") or [])
    args.output.mkdir(parents=True, exist_ok=True)
    failed = False
    for market, entry in sorted(manifest["markets"].items()):
        if market in unavailable or "scan" not in entry.get("pages", {}):
            continue
        try:
            data = load_market(data_root, entry)
            presets = match_presets(data["scan"], data["rows"], evaluate_expression)
            as_of_date = data["scan"]["as_of_date"]
        except (OSError, KeyError, TypeError, ValueError) as exc:
            print(f"::warning::{market} matches not extracted: {exc!r}", flush=True)
            failed = True
            continue
        payload = {
            "market": market,
            "as_of_date": as_of_date,
            "presets": {
                preset: {"definition": match.definition, "symbols": list(match.symbols)}
                for preset, match in presets.items()
            },
        }
        (args.output / f"{market}.json").write_text(json.dumps(payload), encoding="utf-8")
        print(f"{market}: {as_of_date}, {len(presets)} presets, {sum(len(m.symbols) for m in presets.values())} matches")
    return 1 if failed else 0


def default_branch_artifacts(market: str) -> list[dict[str, Any]]:
    """Retained market artifacts from this repository's default branch, oldest first."""
    branch = run_gh("api", "repos/{owner}/{repo}", "--jq", ".default_branch").strip()
    pages = json.loads(
        run_gh(
            "api",
            "--paginate",
            "--slurp",
            f"repos/{{owner}}/{{repo}}/actions/artifacts?name=static-market-{market}&per_page=100",
        )
    )

    # A fork PR's head branch can also be called main; its head repository differs.
    def trusted(run: dict[str, Any]) -> bool:
        return (
            run.get("head_branch") == branch
            and run.get("repository_id") is not None
            and run.get("head_repository_id") == run.get("repository_id")
        )

    artifacts = [
        artifact
        for page in pages
        for artifact in page["artifacts"]
        if not artifact["expired"] and trusted(artifact.get("workflow_run") or {})
    ]
    return sorted(artifacts, key=lambda artifact: artifact["created_at"])


def download_artifact(artifact_id: int, path: Path) -> None:
    with path.open("wb") as handle:
        result = subprocess.run(
            ["gh", "api", f"repos/{{owner}}/{{repo}}/actions/artifacts/{artifact_id}/zip"],
            stdout=handle,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
    if result.returncode != 0:
        raise GhError(f"gh api artifact {artifact_id} failed: {result.stderr.strip()[:400]}")


def artifact_day(zip_path: Path, market: str, created_at: str, work: Path) -> LedgerDay:
    load_market, evaluate_expression = site_readers()
    market_dir = work / "markets" / market.lower()
    with zipfile.ZipFile(zip_path) as archive:
        wanted = [name for name in archive.namelist() if name == "manifest.market.json" or name.startswith("scan/")]
        archive.extractall(market_dir, members=wanted)
    entry = json.loads((market_dir / "manifest.market.json").read_text(encoding="utf-8"))["entry"]
    data = load_market(work, entry)
    published_at = parse_timestamp(created_at)
    return LedgerDay(
        as_of_date=date.fromisoformat(data["scan"]["as_of_date"]),
        published_at=published_at,
        origin=artifact_origin(published_at),
        presets=match_presets(data["scan"], data["rows"], evaluate_expression),
    )


def find_artifact_day(market: str, target: date) -> LedgerDay:
    for artifact in default_branch_artifacts(market):
        # A build cannot predate the session it reports, so these need no download.
        if artifact["created_at"][:10] < target.isoformat():
            continue
        with tempfile.TemporaryDirectory() as tmp:
            zip_path = Path(tmp) / "artifact.zip"
            download_artifact(artifact["id"], zip_path)
            try:
                day = artifact_day(zip_path, market, artifact["created_at"], Path(tmp))
            except (zipfile.BadZipFile, OSError, KeyError, TypeError, ValueError) as exc:
                print(f"::warning::Skipping artifact {artifact['id']}: {exc!r}", flush=True)
                continue
        if day.as_of_date == target:
            return day
    raise LedgerError(
        f"No retained default-branch {market} artifact holds {target}; "
        "it stays a gap counted in missing_sessions."
    )


def cmd_recover(args: argparse.Namespace) -> int:
    listing = release_assets()
    if listing is None:
        raise LedgerError(f"Release {RELEASE_TAG} does not exist; see {RUNBOOK}")
    day = find_artifact_day(args.market, args.date)
    with tempfile.TemporaryDirectory() as tmp:
        advance(args.market, day, listing, Path(tmp), lambda as_of, days: as_of not in days)
    return 0


def _report_assets(directory: Path, market: str) -> int:
    assets = read_assets(directory, market)
    for slot, asset in assets.items():
        through = max(asset.days, default="-")
        print(f"{market} {slot.value}: {asset.state.value}, {len(asset.days)} days, through {through}")
    writer = sources_for(assets).writer
    print(f"{market} record job appends to: {writer.value if writer else 'nothing, it stops red'}")
    if lost := lost_lines(assets):
        print(f"{market} canonical lost {lost} line(s) that .previous holds, so the record job stops red")
        return 1
    return 0 if writer is Slot.CANONICAL else 1


def cmd_verify(args: argparse.Namespace) -> int:
    if args.dir:
        return _report_assets(args.dir, args.market)
    listing = release_assets()
    if listing is None:
        print(f"Release {RELEASE_TAG} does not exist.")
        return 1
    with tempfile.TemporaryDirectory() as tmp:
        download_assets(listing, args.market, Path(tmp))
        return _report_assets(Path(tmp), args.market)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    seed = commands.add_parser("seed", help="Build ledger files from captured artifact records")
    seed.add_argument("--input", required=True, type=Path)
    seed.add_argument("--output", required=True, type=Path)
    seed.add_argument("--allow-missing", action="store_true", help="Start a market with no input file empty")
    seed.set_defaults(run=cmd_seed)

    verify = commands.add_parser("verify", help="Parse one market's ledger assets strictly")
    verify.add_argument("--market", required=True, choices=MARKETS)
    verify.add_argument("--dir", type=Path, help="Read local files instead of the Release")
    verify.set_defaults(run=cmd_verify)

    extract = commands.add_parser("extract", help="Write what a built site publishes, one file per market")
    extract.add_argument("--site-dir", required=True, type=Path)
    extract.add_argument("--output", required=True, type=Path)
    extract.set_defaults(run=cmd_extract)

    record = commands.add_parser("record", help="Append extracted days the ledger has not seen")
    record.add_argument("--days-dir", required=True, type=Path)
    record.set_defaults(run=cmd_record)

    recover = commands.add_parser("recover", help="Insert one lost day from a retained artifact")
    recover.add_argument("--market", required=True, choices=MARKETS)
    recover.add_argument("--date", required=True, type=date.fromisoformat)
    recover.set_defaults(run=cmd_recover)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.run(args)
    except LedgerError as exc:
        print(f"::error::{exc}", flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())

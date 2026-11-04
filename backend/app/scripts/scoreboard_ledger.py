"""Append-only ledger of the screen matches the static site published, per market.

Matches cannot be rebuilt later because fundamentals keep no history, so this is
the one Release-stored input of the scoreboard. Returns are never stored here.

Stdlib only: the record-ledger job holds contents:write and installs nothing.
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
import zlib
from dataclasses import dataclass
from datetime import date, datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

MARKETS = ("US", "CA", "HK", "JP", "TW", "IN", "KR", "CN", "DE", "AU", "SG", "MY")
TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%SZ"
_DEFINITION = re.compile(r"[0-9a-f]{16}")


class Origin(str, Enum):
    LIVE = "live"
    ARTIFACT = "artifact"
    ARTIFACT_PRE_STAGE_FIX = "artifact_pre_stage_fix"


# Newest era first. A retained artifact's created_at only approximates publication,
# so a day read from one is never live. The first row at or before the date wins.
ARTIFACT_ERAS = (
    (date(2026, 10, 22), Origin.ARTIFACT),  # stage fix, #671
    (date.min, Origin.ARTIFACT_PRE_STAGE_FIX),
)


def artifact_origin(as_of_date: date) -> Origin:
    return next(origin for start, origin in ARTIFACT_ERAS if as_of_date >= start)


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


def parse_line(line: str) -> LedgerDay:
    try:
        raw = json.loads(line)
        day = LedgerDay(
            as_of_date=date.fromisoformat(raw["as_of_date"]),
            published_at=parse_timestamp(raw["published_at"]),
            origin=Origin(raw["origin"]),
            presets={
                preset: PresetMatch(match["definition"], tuple(match["symbols"]))
                for preset, match in raw["presets"].items()
            },
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
                days[as_of_date] = LedgerDay(
                    as_of_date=as_of_date,
                    published_at=parse_timestamp(record["created_at"]),
                    origin=artifact_origin(as_of_date),
                    presets={
                        preset: PresetMatch(match["definition"], tuple(match["symbols"]))
                        for preset, match in record["presets"].items()
                    },
                )
    except (KeyError, TypeError, ValueError) as exc:
        raise LedgerCorrupt(f"{market} seed record: {exc!r}") from exc
    return days


def cmd_seed(args: argparse.Namespace) -> int:
    args.output.mkdir(parents=True, exist_ok=True)
    for market in MARKETS:
        source = args.input / f"{market}.jsonl.gz"
        records = read_jsonl_gz(source) if source.exists() else []
        days = seed_days(records, market)
        target = args.output / asset_name(market)
        write_ledger(target, [canonical_line(days[day]) for day in sorted(days)])
        written = load_ledger(target, market)
        print(f"{market}: {len(written)} days from {len(records)} records, {target.stat().st_size} bytes")
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    assets = read_assets(args.dir, args.market)
    for slot, asset in assets.items():
        through = max(asset.days, default="-")
        print(f"{args.market} {slot.value}: {asset.state.value}, {len(asset.days)} days, through {through}")
    writer = sources_for(assets).writer
    print(f"{args.market} record job appends to: {writer.value if writer else 'nothing, it stops red'}")
    return 0 if writer is Slot.CANONICAL else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    seed = commands.add_parser("seed", help="Build ledger files from captured artifact records")
    seed.add_argument("--input", required=True, type=Path)
    seed.add_argument("--output", required=True, type=Path)
    seed.set_defaults(run=cmd_seed)

    verify = commands.add_parser("verify", help="Parse one market's ledger files strictly")
    verify.add_argument("--market", required=True, choices=MARKETS)
    verify.add_argument("--dir", required=True, type=Path)
    verify.set_defaults(run=cmd_verify)
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

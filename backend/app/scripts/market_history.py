"""Immutable record of every session the static site published, one Release per market.

Past fundamentals, ratings and screen matches cannot be rebuilt later, and retained
artifacts expire. Each session is one asset that is never rewritten, so a crash or a
retry cannot damage history and the first publication of a session wins.

Stdlib only: the record-history job holds contents:write and installs nothing.
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import shutil
import sys
import tempfile
import zipfile
from collections.abc import Callable, Iterable, Iterator
from datetime import date, datetime
from operator import itemgetter
from pathlib import Path
from typing import Any

from app.scripts.scoreboard_ledger import (
    MARKETS,
    GhError,
    LedgerCorrupt,
    LedgerError,
    Origin,
    artifact_origin,
    check_publishable,
    default_branch_artifacts,
    download_artifact,
    format_timestamp,
    load_artifact_market,
    match_presets,
    parse_timestamp,
    release_asset_states,
    run_gh,
    site_markets,
    site_readers,
    upload_asset,
)

SCHEMA_VERSION = "market-history-v1"
INDEX_SCHEMA_VERSION = "market-history-index-v1"
INDEX_ASSET = "index.json"
RUNBOOK = "docs/runbooks/market-history.md"
# The docs' enum names uploaded and open, and a failed upload leaves starter, so
# anything but uploaded may be an empty file.
UPLOADED = "uploaded"
# stockscreen_mcp.data imports the MCP SDK, which this job cannot install.
HEAVY_FIELDS = frozenset({"price_sparkline_data", "rs_sparkline_data", "field_availability"})


def release_tag(market: str) -> str:
    return f"market-history-{market.lower()}-v1"


def session_asset(market: str, as_of_date: str) -> str:
    return f"{market.lower()}-{as_of_date}.json.gz"


def build_document(
    market: str,
    scan: dict[str, Any],
    rows: list[dict[str, Any]],
    evaluate_expression: Callable[..., bool],
) -> dict[str, Any]:
    """A session as the site published it, before publication is stamped on it."""
    screens = {preset["id"]: preset for preset in scan["preset_screens"]}
    columns = sorted({field for row in rows for field in row} - HEAVY_FIELDS | {"symbol"})
    return {
        "schema_version": SCHEMA_VERSION,
        "market": market,
        "as_of_date": scan["as_of_date"],
        "presets": {
            preset: {
                "name": screens[preset].get("name"),
                "definition": match.definition,
                "filter_expression": screens[preset]["filter_expression"],
                "symbols": list(match.symbols),
            }
            for preset, match in match_presets(scan, rows, evaluate_expression).items()
        },
        "columns": columns,
        "rows": [[row.get(column) for column in columns] for row in sorted(rows, key=itemgetter("symbol"))],
    }


def stamped(document: dict[str, Any], published_at: datetime, origin: Origin) -> dict[str, Any]:
    return {**document, "published_at": format_timestamp(published_at), "origin": origin.value}


def session_bytes(document: dict[str, Any]) -> bytes:
    text = json.dumps(document, sort_keys=True, separators=(",", ":"))
    return gzip.compress(text.encode("utf-8"), mtime=0)


def check_document(document: dict[str, Any], market: str, published_at: datetime) -> None:
    """Refuse what the site cannot have published, since a session is never rewritten."""
    if market not in MARKETS:
        raise LedgerError(f"{market} is not a supported market")
    if document["schema_version"] != SCHEMA_VERSION:
        raise LedgerCorrupt(f"{market}.json is {document['schema_version']}, not {SCHEMA_VERSION}")
    if document["market"] != market:
        raise LedgerCorrupt(f"{market}.json holds {document['market']} rows")
    as_of_date = date.fromisoformat(document["as_of_date"])
    # fromisoformat also takes 20261106, which would name an asset the index cannot match.
    if as_of_date.isoformat() != document["as_of_date"]:
        raise LedgerCorrupt(f"{market}.json has as_of_date {document['as_of_date']}")
    if not document["rows"]:
        raise LedgerError(f"{market} {as_of_date} has zero rows")
    if not document["presets"]:
        raise LedgerError(f"{market} {as_of_date} has zero presets")
    check_publishable(market, as_of_date, published_at)


def read_document(path: Path, published_at: datetime) -> dict[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
        check_document(document, path.stem, published_at)
    except (OSError, KeyError, TypeError, ValueError, AttributeError) as exc:
        raise LedgerCorrupt(f"{path.name}: {exc!r}") from exc
    return document


def session_dates(listing: dict[str, str], market: str) -> list[str]:
    pattern = re.compile(rf"{market.lower()}-(\d{{4}}-\d{{2}}-\d{{2}})\.json\.gz")
    return sorted(
        match[1] for name, state in listing.items() if state == UPLOADED and (match := pattern.fullmatch(name))
    )


def create_release(market: str) -> None:
    tag = release_tag(market)
    try:
        run_gh(
            "release",
            "create",
            tag,
            "--prerelease",
            "--latest=false",
            "--title",
            f"Market History {market}",
            "--notes",
            f"One immutable file per published session. See {RUNBOOK}.",
        )
    except GhError:
        # Another writer can create it between the listing and this call.
        if release_asset_states(tag) is None:
            raise


def upload_session(path: Path, tag: str) -> None:
    try:
        upload_asset(path, tag, clobber=False)
    except GhError:
        # gh refuses a name that exists, which is how a concurrent backfill shows up here.
        if (release_asset_states(tag) or {}).get(path.name) != UPLOADED:
            raise
        print(f"{path.name} was uploaded by another run, keeping that one", flush=True)


def publish(path: Path, market: str, dry_run: Path | None, upload: Callable[[Path, str], None]) -> None:
    if dry_run is None:
        upload(path, release_tag(market))
        return
    # Every market has an index.json, so one flat directory would overwrite them.
    directory = dry_run / market.lower()
    directory.mkdir(parents=True, exist_ok=True)
    shutil.copy(path, directory)


def index_bytes(market: str, sessions: list[str]) -> bytes:
    index = {"schema_version": INDEX_SCHEMA_VERSION, "market": market, "sessions": sessions}
    return json.dumps(index, separators=(",", ":")).encode("utf-8")


def heal_index(market: str, listing: dict[str, str], dry_run: Path | None) -> None:
    sessions = session_dates(listing, market)
    wanted = index_bytes(market, sessions)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / INDEX_ASSET
        if listing.get(INDEX_ASSET) == UPLOADED:
            run_gh("release", "download", release_tag(market), "--pattern", INDEX_ASSET, "--dir", tmp)
            if path.read_bytes() == wanted:
                return
        path.write_bytes(wanted)
        publish(path, market, dry_run, upload_asset)
    print(f"{market}: index now lists {len(sessions)} sessions", flush=True)


def archive(market: str, documents: Iterable[dict[str, Any]], dry_run: Path | None) -> None:
    """Add each session the Release lacks, then make the index match the listing."""
    tag = release_tag(market)
    listing = release_asset_states(tag)
    if listing is None:
        listing = {}
        if dry_run is None:
            create_release(market)
    for document in documents:
        as_of_date = document["as_of_date"]
        name = session_asset(market, as_of_date)
        state = listing.get(name)
        if state == UPLOADED:
            print(f"{market}: {as_of_date} is already archived, nothing to upload", flush=True)
            continue
        if state is not None:
            # ponytail: a person deletes it, so no job here ever holds a delete call on
            # history. Add an age-based delete if these turn up more than rarely.
            raise LedgerError(
                f"{market} {name} is {state}: an upload is running or died part way. If none is running, "
                f"gh release delete-asset {tag} {name} --yes and run this again; see {RUNBOOK}"
            )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / name
            path.write_bytes(session_bytes(document))
            publish(path, market, dry_run, upload_session)
        listing[name] = UPLOADED
        print(f"{market}: added {as_of_date} ({document['origin']})", flush=True)
    # Listed again because a backfill runs for minutes and publishes can land meanwhile.
    heal_index(market, {**listing, **(release_asset_states(tag) or {})}, dry_run)


def cmd_extract(args: argparse.Namespace) -> int:
    load_market, evaluate_expression = site_readers()
    data_root, markets = site_markets(args.site_dir)
    args.output.mkdir(parents=True, exist_ok=True)
    failed = False
    for market, entry in markets.items():
        try:
            data = load_market(data_root, entry)
            document = build_document(market, data["scan"], data["rows"], evaluate_expression)
        except (OSError, KeyError, TypeError, ValueError, AttributeError) as exc:
            print(f"::warning::{market} session not extracted: {exc!r}", flush=True)
            (args.output / f"{market}.failed").write_text(repr(exc), encoding="utf-8")
            failed = True
            continue
        (args.output / f"{market}.json").write_text(json.dumps(document), encoding="utf-8")
        print(
            f"{market}: {document['as_of_date']}, {len(document['rows'])} rows, "
            f"{len(document['columns'])} columns, {len(document['presets'])} presets"
        )
    return 1 if failed else 0


def cmd_record(args: argparse.Namespace) -> int:
    paths = sorted(args.days_dir.glob("*.json"))
    markers = sorted(args.days_dir.glob("*.failed"))
    if not paths and not markers:
        print(f"::error::No extracted sessions in {args.days_dir}.", flush=True)
        return 1
    failed = bool(markers)
    for path in paths:
        try:
            document = read_document(path, args.published_at)
            archive(path.stem, [stamped(document, args.published_at, Origin.LIVE)], args.dry_run)
        except LedgerError as exc:
            print(f"::error::{path.stem}: {exc}", flush=True)
            failed = True
    for marker in markers:
        print(f"::error::{marker.stem}: session was not extracted: {marker.read_text(encoding='utf-8')}", flush=True)
    return 1 if failed else 0


def artifact_documents(market: str, artifacts: list[dict[str, Any]]) -> Iterator[dict[str, Any]]:
    """Each usable artifact's session in the order given, lazily, so memory holds one at a time."""
    load_market, evaluate_expression = site_readers()
    usable = 0
    for artifact in artifacts:
        with tempfile.TemporaryDirectory() as tmp:
            zip_path = Path(tmp) / "artifact.zip"
            # Outside the try: skipping on a transient failure would let a later,
            # different artifact take this date for good.
            download_artifact(artifact["id"], zip_path)
            try:
                data = load_artifact_market(zip_path, market, Path(tmp), load_market)
                created_at = parse_timestamp(artifact["created_at"])
                document = build_document(market, data["scan"], data["rows"], evaluate_expression)
                check_document(document, market, created_at)
            except (zipfile.BadZipFile, OSError, KeyError, TypeError, ValueError, AttributeError, LedgerError) as exc:
                print(f"::warning::Skipping artifact {artifact['id']}: {exc!r}", flush=True)
                continue
        usable += 1
        yield stamped(document, created_at, artifact_origin(created_at))
    if not usable:
        raise LedgerError(f"None of {len(artifacts)} retained {market} artifacts is usable")


def cmd_backfill(args: argparse.Namespace) -> int:
    artifacts = default_branch_artifacts(args.market)
    archive(args.market, artifact_documents(args.market, artifacts), args.dry_run)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    extract = commands.add_parser("extract", help="Write what a built site publishes, one file per market")
    extract.add_argument("--site-dir", required=True, type=Path)
    extract.add_argument("--output", required=True, type=Path)
    extract.set_defaults(run=cmd_extract)

    dry_run = {"type": Path, "metavar": "DIR", "help": "Write the files to DIR/<m>/ instead of changing the Release"}
    record = commands.add_parser("record", help="Archive extracted sessions the Release lacks")
    record.add_argument("--days-dir", required=True, type=Path)
    record.add_argument(
        "--published-at", required=True, type=parse_timestamp, help="When the deploy finished, as YYYY-MM-DDTHH:MM:SSZ"
    )
    record.add_argument("--dry-run", **dry_run)
    record.set_defaults(run=cmd_record)

    backfill = commands.add_parser("backfill", help="Archive sessions the Release lacks from retained artifacts")
    backfill.add_argument("--market", required=True, choices=MARKETS)
    backfill.add_argument("--dry-run", **dry_run)
    backfill.set_defaults(run=cmd_backfill)
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

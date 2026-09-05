"""Export checksummed US options aggregate history for the next static build."""

from __future__ import annotations

import argparse
import gzip
import json
import os
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from app.database import SessionLocal
from app.infra.db.repositories.options_history_repository import (
    SqlOptionsHistoryRepository,
)
from app.infra.db.repositories.published_options_reader import (
    SqlPublishedOptionsReader,
)
from app.scripts._runtime import prepare_runtime
from app.services.options_history_transfer import OptionsHistoryTransfer


def write_history_bundle(path: Path, bundle: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        dir=destination.parent,
    )
    os.close(handle)
    temporary = Path(temporary_name)
    try:
        encoded = (
            json.dumps(bundle, allow_nan=False, sort_keys=True, separators=(",", ":"))
            + "\n"
        ).encode("utf-8")
        with temporary.open("wb") as raw, gzip.GzipFile(
            fileobj=raw, mode="wb", mtime=0
        ) as compressed:
            compressed.write(encoded)
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-run-id", type=int)
    args = parser.parse_args(argv)
    prepare_runtime()
    with SessionLocal() as db:
        bundle = OptionsHistoryTransfer(
            SqlOptionsHistoryRepository(db),
            published_reader=SqlPublishedOptionsReader(db),
        ).export_bundle(required_published_run_id=args.require_run_id)
    write_history_bundle(args.output, bundle)
    print(json.dumps({"status": "exported", "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Restore compatible US options aggregate history into an ephemeral build."""

from __future__ import annotations

import argparse
import gzip
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from app.database import SessionLocal
from app.infra.db.repositories.options_history_repository import (
    SqlOptionsHistoryRepository,
)
from app.scripts._runtime import prepare_runtime
from app.services.options_history_transfer import OptionsHistoryTransfer


def read_history_bundle(
    path: Path,
    *,
    allow_missing: bool = False,
) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file():
        if allow_missing:
            return {"status": "fresh_history", "reason": "history_bundle_missing"}
        raise FileNotFoundError(source)
    with gzip.open(source, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError("options history bundle must be an object")
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--allow-missing", action="store_true")
    args = parser.parse_args(argv)
    payload = read_history_bundle(args.input, allow_missing=args.allow_missing)
    if payload.get("status") == "fresh_history":
        print(json.dumps(payload))
        return 0
    prepare_runtime()
    with SessionLocal() as db:
        result = OptionsHistoryTransfer(
            SqlOptionsHistoryRepository(db)
        ).import_bundle(payload)
    print(json.dumps({"status": "imported", **result}, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

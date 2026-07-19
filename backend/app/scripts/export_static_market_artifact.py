"""Run one static market export and write its canonical status artifact."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from app.scripts import export_static_site


def _status_for_exit_code(exit_code: int) -> tuple[bool, str, str | None]:
    if exit_code == 0:
        return True, "published", None
    if exit_code == export_static_site.STATIC_EXPORT_SKIPPED_EXIT_CODE:
        return False, "skipped", "not_trading_day"
    if exit_code == export_static_site.STATIC_EXPORT_NO_CURRENT_ARTIFACT_EXIT_CODE:
        return False, "failed", "no_current_artifact"
    return False, "failed", "export_failed"


def _system_exit_code(exc: SystemExit) -> int:
    if isinstance(exc.code, int):
        return exc.code
    return 1


def write_market_status(
    *,
    output_dir: Path,
    market: str,
    exit_code: int,
) -> Path:
    has_current_artifact, status, reason = _status_for_exit_code(exit_code)
    normalized_market = market.upper()
    status_dir = output_dir / "status" / normalized_market.lower()
    status_dir.mkdir(parents=True, exist_ok=True)
    status_path = status_dir / "status.json"
    status_path.write_text(
        json.dumps(
            {
                "market": normalized_market,
                "has_current_artifact": has_current_artifact,
                "status": status,
                "reason": reason,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return status_path


def _parse_status_args(argv: Sequence[str]) -> tuple[Path, str]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--market", required=True)
    args, _ = parser.parse_known_args(argv)
    return args.output_dir, args.market


def main(argv: Sequence[str] | None = None) -> int:
    export_args = list(argv) if argv is not None else None
    status_args = export_args
    if status_args is None:
        import sys

        status_args = sys.argv[1:]

    output_dir, market = _parse_status_args(status_args)
    try:
        exit_code = export_static_site.main(export_args)
    except SystemExit as exc:
        write_market_status(
            output_dir=output_dir,
            market=market,
            exit_code=_system_exit_code(exc),
        )
        raise
    except Exception:
        write_market_status(
            output_dir=output_dir,
            market=market,
            exit_code=1,
        )
        raise
    else:
        write_market_status(
            output_dir=output_dir,
            market=market,
            exit_code=exit_code,
        )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

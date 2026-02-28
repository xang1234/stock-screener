"""Minimal CLI entrypoint for the scaffold package."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from . import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="xui",
        description="xui-reader scaffold CLI with stable entrypoint wiring.",
    )
    parser.add_argument("--version", action="store_true", help="Show xui-reader version and exit.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.version:
        print(__version__)
        return 0

    print("xui-reader scaffold is installed. Run `xui --help` for options.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

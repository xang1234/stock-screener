"""Logging setup helpers for xui-reader."""

from __future__ import annotations

import logging

LOGGER_NAME = "xui_reader"


def configure_logging(debug: bool = False) -> None:
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def get_logger(name: str | None = None) -> logging.Logger:
    return logging.getLogger(name or LOGGER_NAME)

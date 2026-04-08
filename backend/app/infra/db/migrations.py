"""Alembic-backed runtime schema bootstrap helpers."""

from __future__ import annotations

import logging
from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import inspect
from sqlalchemy.engine import Engine

logger = logging.getLogger(__name__)

_BACKEND_ROOT = Path(__file__).resolve().parents[3]
_ALEMBIC_INI = _BACKEND_ROOT / "alembic.ini"
_ALEMBIC_SCRIPT_LOCATION = _BACKEND_ROOT / "alembic"


def _alembic_config(database_url: str) -> Config:
    config = Config(str(_ALEMBIC_INI))
    config.set_main_option("script_location", str(_ALEMBIC_SCRIPT_LOCATION))
    config.set_main_option("sqlalchemy.url", database_url)
    return config


def _has_user_tables(engine: Engine) -> bool:
    with engine.connect() as conn:
        inspector = inspect(conn)
        return any(name != "alembic_version" for name in inspector.get_table_names())


def _has_alembic_version_table(engine: Engine) -> bool:
    with engine.connect() as conn:
        inspector = inspect(conn)
        return "alembic_version" in inspector.get_table_names()


def migrate_database_to_head(engine: Engine, revision: str = "head") -> str:
    """Upgrade new databases and stamp existing pre-Alembic schemas."""
    config = _alembic_config(str(engine.url))
    has_version_table = _has_alembic_version_table(engine)
    has_user_tables = _has_user_tables(engine)

    if has_user_tables and not has_version_table:
        logger.info("Stamping existing schema with Alembic revision %s", revision)
        command.stamp(config, revision)
        return "stamped"

    logger.info("Running Alembic upgrade to revision %s", revision)
    command.upgrade(config, revision)
    return "upgraded"

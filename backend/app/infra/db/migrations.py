"""Alembic-backed runtime schema bootstrap helpers."""

from __future__ import annotations

import logging
from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import inspect
from sqlalchemy.engine import Engine

from .legacy_runtime_migrations import reconcile_legacy_runtime_schema

logger = logging.getLogger(__name__)

_BACKEND_ROOT = Path(__file__).resolve().parents[3]
_ALEMBIC_INI = _BACKEND_ROOT / "alembic.ini"
_ALEMBIC_SCRIPT_LOCATION = _BACKEND_ROOT / "alembic"
_BASELINE_REVISION = "20260408_0001"


def _alembic_config(database_url: str) -> Config:
    config = Config(str(_ALEMBIC_INI))
    config.set_main_option("script_location", str(_ALEMBIC_SCRIPT_LOCATION))
    config.set_main_option("sqlalchemy.url", database_url)
    return config


def _engine_database_url(engine: Engine) -> str:
    url = engine.url
    if hasattr(url, "render_as_string"):
        return url.render_as_string(hide_password=False)
    return str(url)


def _has_user_tables(engine: Engine) -> bool:
    with engine.connect() as conn:
        inspector = inspect(conn)
        return any(name != "alembic_version" for name in inspector.get_table_names())


def _has_alembic_version_table(engine: Engine) -> bool:
    with engine.connect() as conn:
        inspector = inspect(conn)
        return "alembic_version" in inspector.get_table_names()


def migrate_database_to_head(engine: Engine, revision: str = "head") -> str:
    """Upgrade new databases and reconcile pre-Alembic schemas before upgrade."""
    config = _alembic_config(_engine_database_url(engine))
    has_version_table = _has_alembic_version_table(engine)
    has_user_tables = _has_user_tables(engine)

    if has_user_tables and not has_version_table:
        logger.info("Reconciling legacy pre-Alembic schema before upgrade to %s", revision)
        reconcile_legacy_runtime_schema(engine)
        logger.info("Stamping baseline revision %s for reconciled legacy schema", _BASELINE_REVISION)
        command.stamp(config, _BASELINE_REVISION)
        logger.info("Legacy schema reconciliation completed; running Alembic upgrade to %s", revision)
        command.upgrade(config, revision)
        return "reconciled"

    logger.info("Running Alembic upgrade to revision %s", revision)
    command.upgrade(config, revision)
    return "upgraded"

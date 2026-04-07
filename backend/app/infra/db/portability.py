"""Shared database portability helpers."""

from __future__ import annotations

from typing import Any

from sqlalchemy import Float as SAFloat, Text, cast, inspect as sa_inspect
from sqlalchemy.engine import Connection, Engine, make_url
from sqlalchemy.orm import Query, Session
from sqlalchemy.sql.elements import ColumnElement

from app.config import settings


BindLike = Engine | Connection | Session | Query | str


def _resolve_bind(bind_or_session: BindLike | None = None) -> Engine | Connection | str:
    if bind_or_session is None:
        return settings.database_url
    if isinstance(bind_or_session, Query):
        return bind_or_session.session.get_bind()
    if isinstance(bind_or_session, Session):
        return bind_or_session.get_bind()
    return bind_or_session


def _unwrap_bind_proxy(bind: Any) -> Any:
    current = bind
    seen: set[int] = set()
    while not isinstance(current, str):
        current_id = id(current)
        if current_id in seen:
            break
        seen.add(current_id)
        if hasattr(current, "dialect"):
            return current

        next_bind = None
        for attr in ("conn", "connection", "engine"):
            candidate = getattr(current, attr, None)
            if candidate is not None and candidate is not current:
                next_bind = candidate
                break
        if next_bind is None:
            break
        current = next_bind
    return current


def dialect_name(bind_or_session: BindLike | None = None) -> str:
    bind = _unwrap_bind_proxy(_resolve_bind(bind_or_session))
    if isinstance(bind, str):
        return make_url(bind).get_backend_name()
    return bind.dialect.name


def is_postgres(bind_or_session: BindLike | None = None) -> bool:
    return dialect_name(bind_or_session) == "postgresql"


def inspector(bind_or_conn: BindLike | None = None):
    bind = _unwrap_bind_proxy(_resolve_bind(bind_or_conn))
    if isinstance(bind, str):
        raise TypeError("inspector() requires a SQLAlchemy engine/connection/session/query")
    return sa_inspect(bind)


def table_exists(bind_or_conn: BindLike, table_name: str) -> bool:
    return inspector(bind_or_conn).has_table(table_name)


def table_names(bind_or_conn: BindLike) -> set[str]:
    return set(inspector(bind_or_conn).get_table_names())


def column_names(bind_or_conn: BindLike, table_name: str) -> set[str]:
    if not table_exists(bind_or_conn, table_name):
        return set()
    return {column["name"] for column in inspector(bind_or_conn).get_columns(table_name)}


def column_not_null_map(bind_or_conn: BindLike, table_name: str) -> dict[str, bool]:
    if not table_exists(bind_or_conn, table_name):
        return {}
    return {
        column["name"]: bool(column.get("nullable") is False)
        for column in inspector(bind_or_conn).get_columns(table_name)
    }


def index_defs(bind_or_conn: BindLike, table_name: str) -> list[dict[str, Any]]:
    if not table_exists(bind_or_conn, table_name):
        return []
    insp = inspector(bind_or_conn)
    indexes = [
        {
            "name": index["name"],
            "unique": bool(index.get("unique")),
            "columns": list(index.get("column_names") or []),
        }
        for index in insp.get_indexes(table_name)
    ]
    indexes.extend(
        {
            "name": constraint["name"],
            "unique": True,
            "columns": list(constraint.get("column_names") or []),
        }
        for constraint in insp.get_unique_constraints(table_name)
        if constraint.get("name")
    )
    return indexes


def index_names(bind_or_conn: BindLike, table_name: str) -> set[str]:
    return {index["name"] for index in index_defs(bind_or_conn, table_name) if index["name"]}


def foreign_keys(bind_or_conn: BindLike, table_name: str) -> list[dict[str, Any]]:
    if not table_exists(bind_or_conn, table_name):
        return []
    return list(inspector(bind_or_conn).get_foreign_keys(table_name))


def check_constraints(bind_or_conn: BindLike, table_name: str) -> list[dict[str, Any]]:
    if not table_exists(bind_or_conn, table_name):
        return []
    return list(inspector(bind_or_conn).get_check_constraints(table_name))


def trigger_names(bind_or_conn: BindLike, table_name: str) -> set[str]:
    bind = _resolve_bind(bind_or_conn)
    if isinstance(bind, str):
        return set()
    rows = bind.exec_driver_sql(
        """
        SELECT tgname
        FROM pg_trigger
        JOIN pg_class ON pg_trigger.tgrelid = pg_class.oid
        JOIN pg_namespace ON pg_class.relnamespace = pg_namespace.oid
        WHERE pg_trigger.tgisinternal = false
          AND pg_namespace.nspname = current_schema()
          AND pg_class.relname = %s
        """,
        (table_name,),
    ).fetchall()
    return {row[0] for row in rows}


def json_text(
    column: ColumnElement,
    path_segments: tuple[str, ...],
    *,
    bind_or_session: BindLike | None = None,
) -> ColumnElement:
    expr = column
    for segment in path_segments[:-1]:
        expr = expr[segment]
    if not path_segments:
        return cast(column, Text)
    return expr.op("->>")(path_segments[-1])


def json_number(
    column: ColumnElement,
    path_segments: tuple[str, ...],
    *,
    bind_or_session: BindLike | None = None,
) -> ColumnElement:
    return cast(
        json_text(column, path_segments, bind_or_session=bind_or_session),
        SAFloat,
    )


def json_bool(
    column: ColumnElement,
    path_segments: tuple[str, ...],
    *,
    bind_or_session: BindLike | None = None,
) -> ColumnElement:
    return json_text(column, path_segments, bind_or_session=bind_or_session)

"""One-shot SQLite to PostgreSQL migration utility."""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from collections.abc import Iterable

from sqlalchemy import MetaData, Table, create_engine, inspect, select, text
from sqlalchemy.engine import Engine, make_url
from sqlalchemy.sql.sqltypes import Boolean, Date, DateTime, Time


def _normalize_source_url(raw: str) -> str:
    if "://" in raw:
        return raw
    if raw.startswith("/"):
        return f"sqlite:///{raw}"
    return f"sqlite:///{os.path.abspath(raw)}"


def _sqlite_path_from_url(source_url: str) -> str:
    database = make_url(source_url).database
    if not database:
        raise ValueError(f"Could not resolve SQLite database path from {source_url!r}")
    return database


def _sqlite_text_factory(value):
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", errors="replace")
    return value


def _table_order(engine: Engine) -> list[str]:
    ordered = inspect(engine).get_sorted_table_and_fkc_names()
    names = [name for name, _ in ordered if name]
    seen: set[str] = set()
    result: list[str] = []
    for name in names:
        if name not in seen:
            seen.add(name)
            result.append(name)
    return result


def _reflect(engine: Engine) -> MetaData:
    metadata = MetaData()
    metadata.reflect(bind=engine)
    return metadata


def _count_rows(engine: Engine, table_name: str) -> int:
    with engine.connect() as conn:
        return int(conn.execute(text(f'SELECT COUNT(*) FROM "{table_name}"')).scalar() or 0)


def _count_sqlite_rows(conn: sqlite3.Connection, table_name: str) -> int:
    row = conn.execute(f'SELECT COUNT(*) FROM "{table_name}"').fetchone()
    return int(row[0] if row is not None else 0)


def _iter_batches(rows: Iterable[dict], batch_size: int):
    batch: list[dict] = []
    for row in rows:
        batch.append(row)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _coerce_boolean(value):
    if value is None or isinstance(value, bool):
        return value
    if value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "t", "yes", "y"}:
            return True
        if normalized in {"0", "false", "f", "no", "n", "", "[]", "{}", "null", "none"}:
            return False
    return value


def _coerce_row_for_target(row: dict, target_table: Table) -> dict:
    coerced = dict(row)
    for column in target_table.columns:
        name = column.name
        if name not in coerced:
            continue
        value = coerced[name]
        if value == "" and isinstance(column.type, (DateTime, Date, Time)):
            coerced[name] = None
            continue
        if isinstance(column.type, Boolean):
            coerced[name] = _coerce_boolean(value)
    return coerced


def _qualified_table_name(table: Table) -> str:
    if table.schema:
        return f"{table.schema}.{table.name}"
    return table.name


def _quoted_identifier(engine: Engine, identifier: str) -> str:
    return engine.dialect.identifier_preparer.quote_identifier(identifier)


def _quoted_table_sql(engine: Engine, table: Table) -> str:
    if table.schema:
        return (
            f"{_quoted_identifier(engine, table.schema)}."
            f"{_quoted_identifier(engine, table.name)}"
        )
    return _quoted_identifier(engine, table.name)


def _summarize_exception(exc: Exception, *, limit: int = 240) -> str:
    root = getattr(exc, "orig", exc)
    message = " ".join(str(root).split())
    if len(message) > limit:
        return f"{message[:limit]}..."
    return message


def _reset_sequences(engine: Engine, metadata: MetaData, table_name: str) -> None:
    table = metadata.tables.get(table_name)
    if table is None:
        return
    pk_columns = list(table.primary_key.columns)
    if len(pk_columns) != 1:
        return
    pk = pk_columns[0]
    if not isinstance(pk.type.python_type, type):  # pragma: no cover
        return
    if pk.type.python_type is not int:
        return

    qualified_name = _qualified_table_name(table)
    quoted_table = _quoted_table_sql(engine, table)
    quoted_column = _quoted_identifier(engine, pk.name)
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                SELECT setval(
                    pg_get_serial_sequence(:table_name, :column_name),
                    COALESCE((SELECT MAX(%(column)s) FROM %(table)s), 1),
                    (SELECT MAX(%(column)s) IS NOT NULL FROM %(table)s)
                )
                """
                % {"column": quoted_column, "table": quoted_table}
            ),
            {"table_name": qualified_name, "column_name": pk.name},
        )


def _row_identity(row: dict) -> str:
    for key in ("id", "symbol", "name"):
        if key in row and row[key] is not None:
            return f"{key}={row[key]!r}"
    return "no-identity"


def _insert_row(target_engine: Engine, target_table: Table, row: dict) -> None:
    with target_engine.begin() as target_conn:
        target_conn.execute(target_table.insert(), [row])


def _insert_batch(target_engine: Engine, target_table: Table, table_name: str, batch: list[dict]) -> int:
    try:
        with target_engine.begin() as target_conn:
            target_conn.execute(target_table.insert(), batch)
        return 0
    except Exception as batch_exc:
        print(
            f"{table_name}: batch insert failed ({_summarize_exception(batch_exc)}); retrying row-by-row",
            file=sys.stderr,
        )

    skipped = 0
    for row in batch:
        try:
            _insert_row(target_engine, target_table, row)
        except Exception as row_exc:
            skipped += 1
            print(
                f"{table_name}: skipped row {_row_identity(row)} ({_summarize_exception(row_exc)})",
                file=sys.stderr,
            )
    return skipped


def _self_referential_fk_columns(table: Table) -> tuple[str, ...]:
    columns: list[str] = []
    for fk in table.foreign_keys:
        target = fk.column.table
        if target.name == table.name and target.schema == table.schema:
            columns.append(fk.parent.name)
    return tuple(columns)


def _existing_primary_keys(engine: Engine, table: Table, pk_column: str) -> set:
    quoted_pk = _quoted_identifier(engine, pk_column)
    quoted_table = _quoted_table_sql(engine, table)
    with engine.connect() as conn:
        rows = conn.execute(text(f"SELECT {quoted_pk} FROM {quoted_table}"))
        return {row[0] for row in rows}


def _insert_self_referential_rows(
    target_engine: Engine,
    target_table: Table,
    table_name: str,
    rows: Iterable[dict],
) -> int:
    pk_columns = list(target_table.primary_key.columns)
    if len(pk_columns) != 1:
        pending = list(rows)
        skipped = 0
        for row in pending:
            try:
                _insert_row(target_engine, target_table, row)
            except Exception as row_exc:
                skipped += 1
                print(
                    f"{table_name}: skipped row {_row_identity(row)} ({_summarize_exception(row_exc)})",
                    file=sys.stderr,
                )
        return skipped

    pk_name = pk_columns[0].name
    self_refs = _self_referential_fk_columns(target_table)
    inserted_ids = _existing_primary_keys(target_engine, target_table, pk_name)
    pending = list(rows)
    skipped = 0

    while pending:
        ready: list[dict] = []
        deferred: list[dict] = []
        for row in pending:
            if all(row.get(column) is None or row.get(column) in inserted_ids for column in self_refs):
                ready.append(row)
            else:
                deferred.append(row)

        if not ready:
            for row in deferred:
                skipped += 1
                missing = {
                    column: row.get(column)
                    for column in self_refs
                    if row.get(column) is not None and row.get(column) not in inserted_ids
                }
                print(
                    f"{table_name}: skipped row {_row_identity(row)} (missing self-reference {missing})",
                    file=sys.stderr,
                )
            break

        for row in ready:
            try:
                _insert_row(target_engine, target_table, row)
                if row.get(pk_name) is not None:
                    inserted_ids.add(row[pk_name])
            except Exception as row_exc:
                skipped += 1
                print(
                    f"{table_name}: skipped row {_row_identity(row)} ({_summarize_exception(row_exc)})",
                    file=sys.stderr,
                )

        pending = deferred

    return skipped


def migrate(
    source_url: str,
    target_url: str,
    *,
    batch_size: int,
    dry_run: bool,
    skip_tables: set[str],
    start_at: str | None,
) -> None:
    source_engine = create_engine(source_url)
    target_engine = create_engine(target_url)
    source_sqlite = sqlite3.connect(_sqlite_path_from_url(source_url))
    source_sqlite.text_factory = _sqlite_text_factory
    source_sqlite.row_factory = sqlite3.Row

    try:
        source_metadata = _reflect(source_engine)
        target_metadata = _reflect(target_engine)

        ordered_tables = [
            name
            for name in _table_order(target_engine)
            if name in source_metadata.tables and name not in skip_tables
        ]
        if start_at:
            ordered_tables = ordered_tables[ordered_tables.index(start_at):]

        skipped_rows = 0
        for table_name in ordered_tables:
            source_count = _count_sqlite_rows(source_sqlite, table_name)
            target_count = _count_rows(target_engine, table_name)
            print(f"{table_name}: source={source_count} target={target_count}")
            if dry_run or source_count == 0:
                continue

            source_table = source_metadata.tables[table_name]
            target_table = target_metadata.tables[table_name]

            source_rows = source_sqlite.execute(f'SELECT * FROM "{table_name}"')
            rows = (
                _coerce_row_for_target(dict(row), target_table)
                for row in source_rows
            )
            table_skipped = 0
            if _self_referential_fk_columns(target_table):
                table_skipped += _insert_self_referential_rows(
                    target_engine,
                    target_table,
                    table_name,
                    rows,
                )
            else:
                for batch in _iter_batches(rows, batch_size):
                    table_skipped += _insert_batch(
                        target_engine,
                        target_table,
                        table_name,
                        batch,
                    )

            _reset_sequences(target_engine, target_metadata, table_name)
            new_target_count = _count_rows(target_engine, table_name)
            skipped_rows += table_skipped
            print(f"{table_name}: imported target={new_target_count} skipped={table_skipped}")

        if skipped_rows:
            print(f"Migration completed with skipped_rows={skipped_rows}", file=sys.stderr)
    finally:
        source_sqlite.close()
        source_engine.dispose()
        target_engine.dispose()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        default=os.getenv("SQLITE_DATABASE_URL") or os.getenv("SQLITE_DB_PATH"),
        required=not (os.getenv("SQLITE_DATABASE_URL") or os.getenv("SQLITE_DB_PATH")),
        help="SQLite database path or URL",
    )
    parser.add_argument(
        "--target",
        default=os.getenv("POSTGRES_DATABASE_URL") or os.getenv("DATABASE_URL"),
        required=not (os.getenv("POSTGRES_DATABASE_URL") or os.getenv("DATABASE_URL")),
        help="PostgreSQL SQLAlchemy URL",
    )
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-table", action="append", default=[])
    parser.add_argument("--start-at")
    args = parser.parse_args()

    migrate(
        _normalize_source_url(args.source),
        args.target,
        batch_size=args.batch_size,
        dry_run=args.dry_run,
        skip_tables=set(args.skip_table),
        start_at=args.start_at,
    )


if __name__ == "__main__":
    main()

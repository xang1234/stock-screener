"""One-shot SQLite to PostgreSQL migration utility."""

from __future__ import annotations

import argparse
import os
from collections.abc import Iterable

from sqlalchemy import MetaData, create_engine, inspect, select, text
from sqlalchemy.engine import Engine


def _normalize_source_url(raw: str) -> str:
    if "://" in raw:
        return raw
    if raw.startswith("/"):
        return f"sqlite:///{raw}"
    return f"sqlite:///{os.path.abspath(raw)}"


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


def _iter_batches(rows: Iterable[dict], batch_size: int):
    batch: list[dict] = []
    for row in rows:
        batch.append(row)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


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
                % {"column": pk.name, "table": table_name}
            ),
            {"table_name": table_name, "column_name": pk.name},
        )


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

    source_metadata = _reflect(source_engine)
    target_metadata = _reflect(target_engine)

    ordered_tables = [
        name
        for name in _table_order(target_engine)
        if name in source_metadata.tables and name not in skip_tables
    ]
    if start_at:
        ordered_tables = ordered_tables[ordered_tables.index(start_at):]

    for table_name in ordered_tables:
        source_count = _count_rows(source_engine, table_name)
        target_count = _count_rows(target_engine, table_name)
        print(f"{table_name}: source={source_count} target={target_count}")
        if dry_run or source_count == 0:
            continue

        source_table = source_metadata.tables[table_name]
        target_table = target_metadata.tables[table_name]

        with source_engine.connect() as source_conn, target_engine.begin() as target_conn:
            result = source_conn.execute(select(source_table))
            for batch in _iter_batches((dict(row._mapping) for row in result), batch_size):
                target_conn.execute(target_table.insert(), batch)

        _reset_sequences(target_engine, target_metadata, table_name)
        new_target_count = _count_rows(target_engine, table_name)
        print(f"{table_name}: imported target={new_target_count}")


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

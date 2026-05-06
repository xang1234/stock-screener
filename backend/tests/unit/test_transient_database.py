from __future__ import annotations

from sqlalchemy.exc import OperationalError

from app.tasks.transient_database import is_transient_database_error


def _operation_error(statement: str, message: str) -> OperationalError:
    return OperationalError(statement, {}, Exception(message))


def test_detects_postgres_recovery_error() -> None:
    exc = _operation_error(
        "select 1",
        "FATAL:  the database system is not yet accepting connections\n"
        "DETAIL:  Consistent recovery state has not been yet reached.",
    )

    assert is_transient_database_error(exc)


def test_rejects_non_database_exceptions() -> None:
    assert not is_transient_database_error(RuntimeError("connection refused"))


def test_rejects_non_transient_database_errors() -> None:
    exc = _operation_error("select 1", "duplicate key value violates unique constraint")

    assert not is_transient_database_error(exc)


def test_does_not_match_transient_phrase_from_sql_statement() -> None:
    exc = _operation_error(
        "select 'connection refused'",
        "duplicate key value violates unique constraint",
    )

    assert not is_transient_database_error(exc)

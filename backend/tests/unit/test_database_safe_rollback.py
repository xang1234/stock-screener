import logging

from app.database import safe_rollback


class _BrokenSession:
    def rollback(self):
        raise RuntimeError("connection lost")


def test_safe_rollback_logs_failure_without_raising(caplog):
    with caplog.at_level(logging.WARNING, logger="app.database"):
        safe_rollback(_BrokenSession())

    [record] = caplog.records
    assert record.levelno == logging.WARNING
    assert "connection lost" in record.getMessage()


def test_safe_rollback_warning_omits_the_failed_statement_parameters(caplog):
    class _SessionAfterFailedInsert:
        def rollback(self):
            try:
                raise ValueError("INSERT failed; parameters: ('s3cret-token',)")
            except ValueError:
                raise RuntimeError("connection lost")

    with caplog.at_level(logging.WARNING, logger="app.database"):
        safe_rollback(_SessionAfterFailedInsert())

    assert "s3cret-token" not in caplog.text
    assert "RuntimeError" in caplog.text

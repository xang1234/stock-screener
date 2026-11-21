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
    assert record.exc_info[1].args == ("connection lost",)

"""Unit tests for DbCancellationToken infra adapter.

Verifies that the token correctly polls the database for scan
cancellation status, uses fail-open semantics, and manages its
reusable session lifecycle.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from app.infra.tasks.cancellation import DbCancellationToken


def _make_token(status: str = "running") -> tuple[DbCancellationToken, MagicMock]:
    """Build a DbCancellationToken with a mocked session factory.

    Returns (token, mock_session) so tests can configure the session.
    """
    mock_session = MagicMock()
    # Chain: session.query(Scan.status).filter(...).scalar() → status
    mock_session.query.return_value.filter.return_value.scalar.return_value = status

    session_factory = MagicMock(return_value=mock_session)
    token = DbCancellationToken(session_factory, "scan-001")
    return token, mock_session


class TestDbCancellationToken:
    """Test the DB-polling cancellation adapter."""

    def test_returns_true_when_cancelled(self):
        token, _ = _make_token(status="cancelled")
        assert token.is_cancelled() is True

    def test_returns_false_when_running(self):
        token, _ = _make_token(status="running")
        assert token.is_cancelled() is False

    def test_returns_false_when_queued(self):
        token, _ = _make_token(status="queued")
        assert token.is_cancelled() is False

    def test_returns_false_when_completed(self):
        token, _ = _make_token(status="completed")
        assert token.is_cancelled() is False

    def test_expire_all_called_before_query(self):
        """Ensures the identity map is busted so we get a fresh DB read."""
        token, mock_session = _make_token()

        token.is_cancelled()

        mock_session.expire_all.assert_called_once()
        # expire_all should be called BEFORE the query
        calls = [c[0] for c in mock_session.method_calls]
        assert calls.index("expire_all") < calls.index("query")

    def test_fail_open_on_db_error(self):
        """DB errors should NOT cancel the scan (fail-open)."""
        token, mock_session = _make_token()
        mock_session.expire_all.side_effect = RuntimeError("connection lost")

        assert token.is_cancelled() is False

    def test_fail_open_on_query_error(self):
        token, mock_session = _make_token()
        mock_session.query.side_effect = RuntimeError("query failed")

        assert token.is_cancelled() is False

    def test_close_closes_session(self):
        token, mock_session = _make_token()

        token.close()

        mock_session.close.assert_called_once()

    def test_close_swallows_session_close_error(self):
        token, mock_session = _make_token()
        mock_session.close.side_effect = RuntimeError("already closed")

        # Should NOT raise
        token.close()

    def test_reuses_same_session_across_calls(self):
        """Multiple is_cancelled() calls should reuse the same session."""
        mock_session = MagicMock()
        mock_session.query.return_value.filter.return_value.scalar.return_value = "running"
        session_factory = MagicMock(return_value=mock_session)

        token = DbCancellationToken(session_factory, "scan-001")

        token.is_cancelled()
        token.is_cancelled()
        token.is_cancelled()

        # session_factory should only be called once (in __init__)
        session_factory.assert_called_once()

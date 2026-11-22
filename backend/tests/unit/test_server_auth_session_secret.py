from __future__ import annotations

import base64
import hashlib
import hmac
import logging
import stat
import sys
import tempfile
import threading

import pytest

from app.services import server_auth


@pytest.fixture
def secret_path(tmp_path, monkeypatch):
    path = tmp_path / ".server_auth_session_secret"
    monkeypatch.setattr(server_auth, "_SESSION_SECRET_PATH", path)
    monkeypatch.setattr(server_auth.settings, "server_auth_enabled", True)
    monkeypatch.setattr(server_auth.settings, "server_auth_password", "login-password")
    monkeypatch.setattr(server_auth.settings, "server_auth_session_secret", "")
    server_auth._load_or_create_session_secret.cache_clear()
    yield path
    server_auth._load_or_create_session_secret.cache_clear()


def _b64_decode(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


def test_password_is_no_longer_the_signing_key(secret_path):
    token = server_auth.issue_session_token()
    raw_payload, raw_sig = token.split(".", 1)

    password_sig = hmac.new(b"login-password", _b64_decode(raw_payload), hashlib.sha256).digest()

    assert _b64_decode(raw_sig) != password_sig
    assert server_auth._verify_session_token(token)


def test_unset_secret_is_generated_once_and_persisted(secret_path, caplog):
    with caplog.at_level(logging.INFO, logger=server_auth.logger.name):
        first = server_auth._session_secret()
        second = server_auth._session_secret()

    assert first == second
    assert len(first) >= 32
    assert secret_path.read_text(encoding="utf-8").strip() == first
    if sys.platform != "win32":
        assert stat.S_IMODE(secret_path.stat().st_mode) == 0o600
    generated = [r for r in caplog.records if "Generated server auth session secret" in r.message]
    assert len(generated) == 1
    assert first not in caplog.text
    assert list(secret_path.parent.iterdir()) == [secret_path]


def test_persisted_secret_is_reused_after_restart(secret_path):
    secret_path.write_text("previously-generated-secret\n", encoding="utf-8")

    token = server_auth.issue_session_token()
    server_auth._load_or_create_session_secret.cache_clear()  # simulate a new process

    assert server_auth._session_secret() == "previously-generated-secret"
    assert server_auth._verify_session_token(token)


def test_explicit_secret_wins_and_writes_no_file(secret_path, monkeypatch):
    monkeypatch.setattr(server_auth.settings, "server_auth_session_secret", "explicit-secret")

    assert server_auth._session_secret() == "explicit-secret"
    assert not secret_path.exists()


def test_concurrent_first_start_converges_on_one_secret(secret_path, monkeypatch):
    workers = 8
    # Hold every worker after its existence check until all of them are ready
    # to publish, so each one races the others for the final name.
    barrier = threading.Barrier(workers)
    real_mkstemp = tempfile.mkstemp

    def racing_mkstemp(*args, **kwargs):
        created = real_mkstemp(*args, **kwargs)
        barrier.wait(timeout=10)
        return created

    monkeypatch.setattr(server_auth.tempfile, "mkstemp", racing_mkstemp)
    results: list[str] = []
    errors: list[Exception] = []

    def start_process():
        try:
            results.append(server_auth._load_or_create_session_secret.__wrapped__(secret_path))
        except Exception as exc:  # surfaced by the assertion below
            errors.append(exc)

    threads = [threading.Thread(target=start_process) for _ in range(workers)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert errors == []
    assert len(results) == workers
    assert len(set(results)) == 1
    assert secret_path.read_text(encoding="utf-8").strip() == results[0]
    assert list(secret_path.parent.iterdir()) == [secret_path]


def test_unwritable_data_dir_fails_instead_of_using_password(tmp_path, secret_path, monkeypatch):
    missing = tmp_path / "missing" / ".server_auth_session_secret"
    monkeypatch.setattr(server_auth, "_SESSION_SECRET_PATH", missing)

    with pytest.raises(RuntimeError, match="SERVER_AUTH_SESSION_SECRET"):
        server_auth.issue_session_token()

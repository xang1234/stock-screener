from __future__ import annotations

import pytest

from app.models.company_exposure import ResearchResourcePool
from app.services.company_exposure.storage import (
    OriginalStore,
    StorageUnavailable,
    blob_key,
)
from tests.fixtures.company_exposure.factory import (
    FixedClock,
)

GIB = 1024**3


@pytest.fixture
def clock():
    return FixedClock()


def _store(db_session, tmp_path, clock, *, max_bytes=100, free=10 * GIB, min_free=GIB):
    return OriginalStore(
        db_session,
        tmp_path / "store",
        max_bytes=max_bytes,
        min_free_bytes=min_free,
        clock=clock.now,
        disk_free=lambda _path: free,
    )


def _pool(db_session):
    return db_session.query(ResearchResourcePool).filter_by(unit="blob_bytes").one()


def test_put_is_atomic_and_charges_actual_bytes(db_session, tmp_path, clock):
    store = _store(db_session, tmp_path, clock)
    ticket = store.reserve(50, purpose="t", operation_key="a")
    blob = store.put(b"hello", "text/plain", ticket)
    assert store.path_for(blob.key).read_bytes() == b"hello"
    assert list((tmp_path / "store" / "tmp").iterdir()) == []
    assert _pool(db_session).reserved_amount == 5


def test_identical_content_is_not_charged_twice(db_session, tmp_path, clock):
    store = _store(db_session, tmp_path, clock)
    store.put(b"same", "text/plain", store.reserve(10, purpose="t", operation_key="a"))
    again = store.put(
        b"same", "text/plain", store.reserve(10, purpose="t", operation_key="b")
    )
    assert again.deduplicated
    assert _pool(db_session).reserved_amount == 4


def test_full_store_pauses_before_io(db_session, tmp_path, clock):
    store = _store(db_session, tmp_path, clock, max_bytes=10)
    assert store.reserve(8, purpose="t", operation_key="a").allowed
    blocked = store.reserve(8, purpose="t", operation_key="b")
    assert (blocked.allowed, blocked.reason, blocked.available) == (
        False,
        "paused_storage",
        2,
    )


def test_free_space_floor_pauses_before_io(db_session, tmp_path, clock):
    store = _store(db_session, tmp_path, clock, free=GIB + 5)
    blocked = store.reserve(10, purpose="t", operation_key="a")
    assert (blocked.allowed, blocked.reason) == (False, "paused_storage")


def test_put_larger_than_reservation_is_refused(db_session, tmp_path, clock):
    store = _store(db_session, tmp_path, clock)
    with pytest.raises(StorageUnavailable):
        store.put(
            b"x" * 20, "text/plain", store.reserve(10, purpose="t", operation_key="a")
        )
    assert _pool(db_session).reserved_amount == 0


def test_blob_key_is_content_addressed(db_session, tmp_path, clock):
    store = _store(db_session, tmp_path, clock)
    blob = store.put(
        b"doc", "text/plain", store.reserve(10, purpose="t", operation_key="a")
    )
    assert blob.key == blob_key(blob.content_hash)

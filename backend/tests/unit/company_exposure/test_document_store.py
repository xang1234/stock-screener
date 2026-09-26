from __future__ import annotations

import os
from datetime import datetime, timezone

import pytest

from app.models.company_exposure import EvidenceTombstoneEvent, ResearchResourcePool
from app.services.company_exposure.storage import (
    OriginalStore,
    StorageUnavailable,
    blob_key,
)
from tests.fixtures.company_exposure.factory import (
    FixedClock,
    make_document,
    make_revision,
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
    again = store.put(b"same", "text/plain", store.reserve(10, purpose="t", operation_key="b"))
    assert again.deduplicated
    assert _pool(db_session).reserved_amount == 4


def test_full_store_pauses_before_io(db_session, tmp_path, clock):
    store = _store(db_session, tmp_path, clock, max_bytes=10)
    assert store.reserve(8, purpose="t", operation_key="a").allowed
    blocked = store.reserve(8, purpose="t", operation_key="b")
    assert (blocked.allowed, blocked.reason, blocked.available) == (False, "paused_storage", 2)


def test_free_space_floor_pauses_before_io(db_session, tmp_path, clock):
    store = _store(db_session, tmp_path, clock, free=GIB + 5)
    blocked = store.reserve(10, purpose="t", operation_key="a")
    assert (blocked.allowed, blocked.reason) == (False, "paused_storage")


def test_put_larger_than_reservation_is_refused(db_session, tmp_path, clock):
    store = _store(db_session, tmp_path, clock)
    with pytest.raises(StorageUnavailable):
        store.put(b"x" * 20, "text/plain", store.reserve(10, purpose="t", operation_key="a"))
    assert _pool(db_session).reserved_amount == 0


def _age(path, days):
    stamp = path.stat().st_mtime - days * 86400
    os.utime(path, (stamp, stamp))


def test_gc_keeps_referenced_and_recent_blobs_and_tombstones_removals(
    db_session, tmp_path, clock
):
    store = _store(db_session, tmp_path, clock, max_bytes=10_000)
    referenced = store.put(b"kept", "text/plain", store.reserve(10, purpose="t", operation_key="a"))
    orphan = store.put(b"orphan", "text/plain", store.reserve(10, purpose="t", operation_key="b"))
    recent = store.put(b"recent", "text/plain", store.reserve(10, purpose="t", operation_key="c"))
    document = make_document(db_session, "sec:accession:gc")
    make_revision(db_session, document, b"kept")
    for blob in (referenced, orphan):
        _age(store.path_for(blob.key), 40)
    stale_temp = tmp_path / "store" / "tmp" / "abc.part"
    stale_temp.parent.mkdir(parents=True, exist_ok=True)
    stale_temp.write_bytes(b"partial")
    _age(stale_temp, 2)
    # File ages are real filesystem times, so GC runs "now".
    clock.advance_to(datetime.now(timezone.utc))

    preview = store.collect_unreferenced_blobs(dry_run=True)
    assert preview.unreferenced == (orphan.key,)
    assert store.path_for(orphan.key).exists()

    report = store.collect_unreferenced_blobs(dry_run=False)
    assert report.unreferenced == (orphan.key,)
    assert report.abandoned_temp == ("abc.part",)
    assert not store.path_for(orphan.key).exists()
    assert store.path_for(referenced.key).exists()
    assert store.path_for(recent.key).exists()
    tombstone = db_session.query(EvidenceTombstoneEvent).one()
    assert (tombstone.reason, tombstone.byte_length) == ("unreferenced_gc", 6)
    with pytest.raises(StorageUnavailable, match="evidence_removed"):
        store.read(orphan.key)


def test_authorized_read_requires_principal(db_session, tmp_path, clock):
    store = _store(db_session, tmp_path, clock)
    blob = store.put(b"doc", "text/plain", store.reserve(10, purpose="t", operation_key="a"))
    with pytest.raises(PermissionError):
        store.open_authorized(blob, None)
    assert blob.key == blob_key(blob.content_hash)

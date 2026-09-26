from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest
from sqlalchemy.orm import sessionmaker

from app.database import engine
from app.services.company_exposure.storage import OriginalStore

pytestmark = [
    pytest.mark.skipif(
        engine.dialect.name != "postgresql", reason="requires PostgreSQL row locks"
    ),
    pytest.mark.exposure_layer("postgres"),
]


def test_storage_last_capacity_is_reserved_once(tmp_path):
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    barrier = Barrier(2)

    def contend(label):
        session = factory()
        try:
            store = OriginalStore(
                session,
                tmp_path / "store",
                max_bytes=100,
                min_free_bytes=0,
                disk_free=lambda _p: 10**12,
            )
            store.usage()  # create the shared pool row before contending
            session.commit()
            barrier.wait(timeout=10)
            ticket = store.reserve(60, purpose="t", operation_key=label)
            session.commit()
            return ticket
        finally:
            session.close()

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(contend, ["a", "b"]))
    assert sum(result.allowed for result in outcomes) == 1
    assert {r.reason for r in outcomes if not r.allowed} == {"paused_storage"}

"""Reviewed groups preserve identities and have dependency-safe undo."""

import pytest
from app.database import Base
from app.models.theme import ContentItem, ThemeCluster, ThemeMention
from app.services.theme_equivalence_service import (
    EquivalenceConflict,
    ThemeEquivalenceService,
)
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


@pytest.fixture
def db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    with sessionmaker(bind=engine)() as session:
        yield session


def theme(db, name, **kw):
    row = ThemeCluster(
        name=name,
        display_name=name,
        canonical_key=name.lower().replace(" ", "_"),
        pipeline=kw.pop("pipeline", "technical"),
        is_active=True,
        **kw,
    )
    db.add(row)
    db.flush()
    return row


def test_group_preserves_sources_and_undo_restores_new_observations(db):
    a, b = theme(db, "CPO"), theme(db, "Co-Packaged Optics")
    svc = ThemeEquivalenceService(db)
    result = svc.apply(
        a.id, b.id, actor="reviewer", reason="Equivalent exposure", key="one"
    )
    assert svc.members(a.id) == [a.id, b.id]
    assert svc.representative(a.id) == b.id
    item = ContentItem(source_type="news", content="New order")
    db.add(item)
    db.flush()
    mention = ThemeMention(
        content_item_id=item.id,
        theme_cluster_id=a.id,
        pipeline="technical",
        raw_theme="CPO",
        source_type="news",
    )
    db.add(mention)
    db.flush()
    assert (
        svc.apply(a.id, b.id, actor="reviewer", reason="Equivalent exposure", key="one")
        == result
    )
    svc.undo(result["id"], actor="reviewer", reason="Correction")
    assert svc.members(a.id) == [a.id]
    assert mention.theme_cluster_id == a.id
    assert a.is_active and b.is_active
    svc.undo(result["id"], actor="reviewer", reason="Correction")


def test_undo_requires_later_overlapping_changes_first(db):
    a, b, c = [
        theme(db, name)
        for name in ["CPO", "Co-Packaged Optics", "Co-Packaged Optical Links"]
    ]
    svc = ThemeEquivalenceService(db)
    first = svc.apply(a.id, b.id, actor="r", reason="same", key="a")
    second = svc.apply(b.id, c.id, actor="r", reason="same", key="b")
    with pytest.raises(EquivalenceConflict, match="later"):
        svc.undo(first["id"], actor="r", reason="undo")
    svc.undo(second["id"], actor="r", reason="undo")
    assert svc.representative(a.id) == b.id


def test_rejects_hierarchy_cross_pipeline_and_key_reuse(db):
    memory = theme(db, "Memory")
    hbm = theme(db, "HBM", parent_cluster_id=memory.id)
    other = theme(db, "HBM fundamental", pipeline="fundamental")
    svc = ThemeEquivalenceService(db)
    for source, target in [(hbm.id, memory.id), (hbm.id, other.id)]:
        with pytest.raises(EquivalenceConflict):
            svc.apply(source, target, actor="r", reason="same", key="bad")
    cpo = theme(db, "CPO")
    optics = theme(db, "Co-Packaged Optics")
    svc.apply(cpo.id, optics.id, actor="r", reason="same", key="ok")
    with pytest.raises(EquivalenceConflict, match="key"):
        svc.apply(hbm.id, memory.id, actor="r", reason="same", key="ok")


def test_rejects_target_that_weakens_group_lifecycle_visibility(db):
    active = theme(db, "Active CPO", lifecycle_state="active")
    candidate = theme(db, "Candidate CPO", lifecycle_state="candidate")

    with pytest.raises(EquivalenceConflict, match="lifecycle"):
        ThemeEquivalenceService(db).apply(
            active.id,
            candidate.id,
            actor="reviewer",
            reason="Equivalent exposure",
            key="weaker-lifecycle-target",
        )

    assert ThemeEquivalenceService(db).members(active.id) == [active.id]


def test_alias_target_idempotency_and_stale_preview(db):
    a, b, c = (
        theme(db, "CPO"),
        theme(db, "Co-Packaged Optics"),
        theme(db, "Co Packaged Optical"),
    )
    svc = ThemeEquivalenceService(db)
    version = svc.version()
    svc.apply(a.id, b.id, actor="test", reason="Equivalent", key="ab")
    with pytest.raises(EquivalenceConflict, match="refresh"):
        svc.apply(
            c.id,
            a.id,
            actor="test",
            reason="Equivalent",
            key="stale",
            expected_version=version,
        )
    result = svc.apply(c.id, a.id, actor="test", reason="Equivalent", key="ca")
    assert svc.apply(c.id, a.id, actor="test", reason="Equivalent", key="ca") == result


def test_snapshot_is_immutable_and_requires_no_more_operation_reads(db):
    from sqlalchemy import event

    a, b = theme(db, "CPO"), theme(db, "Co-Packaged Optics")
    service = ThemeEquivalenceService(db)
    operation = service.apply(
        a.id, b.id, actor="reviewer", reason="Equivalent", key="snapshot"
    )
    snapshot = service.snapshot()
    reads = []

    def capture(connection, cursor, statement, parameters, context, executemany):
        reads.append(statement)

    event.listen(db.bind, "before_cursor_execute", capture)
    try:
        assert snapshot.representative(a.id) == b.id
        assert snapshot.expand([a.id, b.id]) == [a.id, b.id]
        assert snapshot.members(a.id) == (a.id, b.id)
        with pytest.raises(TypeError):
            snapshot.mapping[a.id] = a.id
        assert not reads
    finally:
        event.remove(db.bind, "before_cursor_execute", capture)
    service.undo(operation["id"], actor="reviewer", reason="Correction")
    assert snapshot.representative(a.id) == b.id
    assert service.snapshot().representative(a.id) == a.id


def test_versions_are_isolated_by_pipeline(db):
    technical_a = theme(db, "CPO")
    technical_b = theme(db, "Co-Packaged Optics")
    fundamental_a = theme(db, "Bitcoin Miners", pipeline="fundamental")
    fundamental_b = theme(db, "Bitcoin Mining", pipeline="fundamental")
    service = ThemeEquivalenceService(db)

    technical_version = service.version("technical")
    fundamental_preview = service.preview(fundamental_a.id, fundamental_b.id)
    service.apply(
        fundamental_a.id,
        fundamental_b.id,
        actor="reviewer",
        reason="Equivalent exposure",
        key="fundamental-group",
        expected_version=fundamental_preview["version"],
    )

    assert service.version("technical") == technical_version
    service.apply(
        technical_a.id,
        technical_b.id,
        actor="reviewer",
        reason="Equivalent exposure",
        key="technical-group",
        expected_version=technical_version,
    )

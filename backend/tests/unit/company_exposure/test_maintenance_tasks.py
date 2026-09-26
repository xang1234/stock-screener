"""Scheduled, provider-free maintenance on the exposure_research queue."""

from __future__ import annotations

import os
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pytest
from sqlalchemy import func, select

from app.celery_app import celery_app
from app.models.company_exposure import (
    EvidenceTombstoneEvent,
    ExposureClaimRevision,
    ResearchProviderAttempt,
)
from app.services.company_exposure.holds import HoldRegistry
from app.services.company_exposure.providers import (
    ProviderInput,
    SubscriptionArtifactRunner,
    SubscriptionProvider,
    default_client_factory,
)
from app.services.company_exposure.resources import ResearchResources
from app.services.company_exposure.storage import OriginalStore
from app.tasks.company_exposure_tasks import (
    collect_exposure_evidence,
    refresh_exposure_holds,
)
from tests.fixtures.company_exposure.factory import (
    FakeGoTransport,
    FixedClock,
    verified_claim,
)
from tests.fixtures.company_exposure.research_harness import SHADOW

PREFIX = "app.tasks.company_exposure_tasks."
RUNBOOK = Path(__file__).resolve().parents[4] / "docs/runbooks/company-exposure-map.md"
PAST = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)


@pytest.mark.parametrize(
    ("entry", "task"),
    [
        ("company-exposure-work", "process_exposure_work"),
        ("company-exposure-holds", "refresh_exposure_holds"),
        ("company-exposure-evidence-gc", "collect_exposure_evidence"),
    ],
)
def test_beat_entries_target_the_research_queue_and_expire(entry, task):
    schedule = celery_app.conf.beat_schedule[entry]
    assert schedule["task"] == PREFIX + task
    assert schedule["options"]["queue"] == "exposure_research"
    # Messages expire instead of piling up when the opt-in worker is absent.
    assert 0 < schedule["options"]["expires"] <= 3600
    assert f"`{entry}`" in RUNBOOK.read_text("utf-8")


@pytest.mark.case("R12")
@pytest.mark.exposure_layer("unit")
def test_hold_refresh_runs_with_research_disabled_and_calls_no_provider(dossier):
    stale = verified_claim(
        "role",
        passage=dossier.passages["role"],
        supported_as_of=PAST.replace(year=2020),
    )
    _, ref = dossier.persist(dossier.attempt(stale))
    (revision_id,) = ref.claim_revision_ids.values()
    dossier.db.commit()

    outcome = refresh_exposure_holds.run(session_factory=lambda: dossier.db)

    assert (outcome["status"], outcome["new_holds"]) == ("completed", 1)
    revision = dossier.db.get(ExposureClaimRevision, revision_id)
    held = HoldRegistry(dossier.db).active_kinds_for_claim(
        revision.claim_id, revision.id
    )
    assert held == {"stale"}
    attempts = select(func.count()).select_from(ResearchProviderAttempt)
    assert dossier.db.execute(attempts).scalar() == 0


def test_hold_refresh_closes_ended_periods(db_session):
    config = replace(SHADOW, daily_request_limit=10, daily_token_limit=100_000)
    resources = ResearchResources(db_session, config, clock=FixedClock(PAST).now)
    transport = FakeGoTransport()
    transport.queue_exception(httpx.ReadTimeout("read"))
    runner = SubscriptionArtifactRunner(
        db_session,
        resources,
        SubscriptionProvider(
            api_key="test-key",
            client_factory=default_client_factory(transport.transport),
        ),
    )
    result = runner.run(
        ProviderInput(
            operation="claim_review",
            messages=[{"role": "user", "content": "Passage: ..."}],
            input_hash="a" * 64,
            policy_hash="b" * 64,
            max_output_tokens=1000,
            logical_operation_key="claim_review:" + "a" * 64,
        )
    )
    db_session.commit()
    period, _ = config.allocation_period(PAST)

    outcome = refresh_exposure_holds.run(
        session_factory=lambda: db_session, config=config
    )

    assert outcome["closed_periods"] == [period]
    # The uncertain call held one request and one token reservation.
    assert outcome["expired_reservations"] == 2
    assert resources.read(result.ticket_id).state == "expired_uncertain"


def test_evidence_gc_removes_old_unreferenced_originals(db_session, tmp_path):
    root = tmp_path / "store"
    config = replace(SHADOW, document_store=str(root))
    store = OriginalStore(
        db_session,
        root,
        max_bytes=config.storage_max_bytes,
        min_free_bytes=0,
    )
    orphan = store.put(
        b"orphan", "text/plain", store.reserve(6, purpose="t", operation_key="o")
    )
    path = store.path_for(orphan.key)
    stamp = path.stat().st_mtime - 40 * 86400
    os.utime(path, (stamp, stamp))
    db_session.commit()

    outcome = collect_exposure_evidence.run(
        session_factory=lambda: db_session,
        config=replace(config, storage_min_free_bytes=0),
    )

    assert (outcome["removed_blobs"], outcome["reclaimed_bytes"]) == (1, 6)
    assert not path.exists()
    assert db_session.query(EvidenceTombstoneEvent).count() == 1

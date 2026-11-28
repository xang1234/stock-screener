from collections import Counter
from decimal import Decimal

import pytest
from sqlalchemy import select

from app.database import SessionLocal
from app.infra.db.models.social_analysis import SocialLLMAttempt
from app.infra.db.repositories.economic_taxonomy_work_repo import (
    EconomicTaxonomyWorkRepository,
)
from app.models.economic_taxonomy_runtime import (
    ProcessingRequest,
    ProcessingRequestEvent,
    TaxonomyAuthority,
    TaxonomySourceRevisionLog,
)
from app.services.economic_source_admission import (
    EconomicSourceAdmissionService,
    EvidenceAdmission,
)
from app.services.economic_taxonomy_fence import AuthorityWritesFenced
from app.tasks.economic_taxonomy_tasks import POLICY_BUNDLE_VERSION
from scripts.requeue_economic_processing_requests import main, requeue
from tests.unit.test_economic_taxonomy_llm_provider import (
    NOW,
    outcomes,
    social_request,
    status_error,
)

EMPTY = {"status": "successful_empty", "candidates": []}


def terminal_revisions(request_id):
    with SessionLocal() as db:
        return db.execute(
            select(
                TaxonomySourceRevisionLog.revision_kind,
                TaxonomySourceRevisionLog.revision_number,
            )
            .where(
                TaxonomySourceRevisionLog.logical_source_key
                == f"processing_request:{request_id}"
            )
            .order_by(TaxonomySourceRevisionLog.revision_number)
        ).all()


def test_requeued_uncertain_request_reserves_again_and_completes(monkeypatch):
    request_id, service = social_request(monkeypatch, status_error(529), EMPTY)
    assert service.process(limit=1, worker_id="worker:test")["held"] == 1

    assert requeue(SessionLocal, NOW, apply=True) == Counter(
        {"provider_outcome_uncertain": 1}
    )
    result = service.process(limit=1, worker_id="worker:test")

    assert (result["processed"], result.get("reason")) == (1, None)
    assert outcomes(request_id) == (
        ("completed", "completed"),
        ["uncertain", "success"],
        [("uncertain", None), ("reconciled", Decimal("0.0002"))],
    )
    with SessionLocal() as db:
        keys = db.scalars(
            select(SocialLLMAttempt.logical_operation_key).order_by(SocialLLMAttempt.id)
        ).all()
    assert keys == [str(request_id), f"{request_id}:requeue-1"]
    assert terminal_revisions(request_id) == [("provider_outcome_uncertain", 1)]


def test_requeued_request_that_fails_again_records_a_second_terminal_revision(
    monkeypatch,
):
    request_id, service = social_request(
        monkeypatch, status_error(529), status_error(529)
    )
    service.process(limit=1, worker_id="worker:test")
    requeue(SessionLocal, NOW, apply=True)

    result = service.process(limit=1, worker_id="worker:test")

    assert result["held"] == 1
    assert outcomes(request_id)[0] == ("terminal_failure", "provider_outcome_uncertain")
    assert terminal_revisions(request_id) == [
        ("provider_outcome_uncertain", 1),
        ("provider_outcome_uncertain", 2),
    ]


def _terminal_requests(db, codes):
    admission = EconomicSourceAdmissionService(db)
    work = EconomicTaxonomyWorkRepository(db)
    ids = {}
    for number, code in enumerate(codes):
        admitted = admission.admit_content(
            EvidenceAdmission(
                provider="x",
                canonical_item_id=f"requeue-{number}",
                capture_route="social",
                original_text=f"Post {number}",
                preparation_version="social-v1",
                available_at=NOW,
                evidence_channels=("narrative",),
            )
        )
        request = work.enqueue_request(
            source_lineage_id=admitted.source_lineage_id,
            evidence_packet_id=admitted.packet_id,
            policy_bundle_version=POLICY_BUNDLE_VERSION,
            available_at=NOW,
        )
        request.status = "completed" if code == "completed" else "terminal_failure"
        request.completion_code = code
        ids[code] = request.id
    return ids


def _authority(db):
    db.add(
        TaxonomyAuthority(
            id=1,
            mode="shadow",
            processing_head_revision=1,
            authority_epoch=7,
            writes_fenced=False,
            semantic_invalidation_revision=0,
            cutover_catch_up_cursor=[],
            rollback_state="ready",
        )
    )
    db.flush()


def _states():
    with SessionLocal() as db:
        return sorted(
            (row.completion_code if row.status != "retryable" else "requeued", row.status)
            for row in db.scalars(select(ProcessingRequest))
        )


CODES = (
    "provider_outcome_uncertain",
    "invalid_result_status",
    "invalid_provider_json",
    "claim_review_not_accepted",
    "completed",
)


def test_dry_run_counts_and_apply_is_bounded_and_idempotent(capsys):
    with SessionLocal.begin() as db:
        _authority(db)
        ids = _terminal_requests(db, CODES)
    before = _states()

    assert main([]) == 0
    assert "would requeue 3 terminal processing requests" in capsys.readouterr().out
    assert _states() == before

    first = requeue(SessionLocal, NOW, apply=True, limit=2)
    rest = requeue(SessionLocal, NOW, apply=True)

    assert (sum(first.values()), sum(rest.values())) == (2, 1)
    assert first + rest == Counter(CODES[:3])
    assert requeue(SessionLocal, NOW, apply=True) == Counter()

    assert _states() == [
        ("claim_review_not_accepted", "terminal_failure"),
        ("completed", "completed"),
        ("requeued", "retryable"),
        ("requeued", "retryable"),
        ("requeued", "retryable"),
    ]
    with SessionLocal() as db:
        events = db.execute(
            select(ProcessingRequestEvent.event_type, ProcessingRequestEvent.event_payload)
            .where(
                ProcessingRequestEvent.processing_request_id
                == ids["provider_outcome_uncertain"]
            )
            .order_by(ProcessingRequestEvent.sequence_number)
        ).all()
    assert events[-1] == ("retry_scheduled", {"reason": "operator_requeue"})


def test_completion_code_filter_and_fenced_authority(capsys):
    with SessionLocal.begin() as db:
        _authority(db)
        _terminal_requests(db, CODES)
        db.get(TaxonomyAuthority, 1).writes_fenced = True

    with pytest.raises(AuthorityWritesFenced):
        requeue(SessionLocal, NOW, apply=True)

    assert main(["--completion-code", "claim_review_not_accepted"]) == 0
    assert "would requeue 1 terminal processing requests" in capsys.readouterr().out
    assert ("requeued", "retryable") not in _states()

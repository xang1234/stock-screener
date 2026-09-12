"""Admission must retain exact reviewed provenance and exclude unresolved claims."""

from datetime import datetime, timezone

import pytest
from app.services.theme_evaluation.bundle import seal_bundle
from app.services.theme_evaluation.evidence_assessment import (
    build_assessment,
    save_assessment,
)
from app.services.theme_evaluation.preparation_pipeline import prepare
from app.services.theme_evaluation.preparation_records import Handoff
from app.services.theme_evaluation.preparation_store import PreparationStore
from app.services.theme_evaluation.records import Bundle


def setup(bundle, tmp_path):
    base = seal_bundle(tmp_path, Bundle.model_validate(bundle))
    store = PreparationStore(tmp_path / "store")
    pid = prepare(base, store, Handoff(bundle_id=base.name), stages=["text"])
    assessment = build_assessment(base, store, pid)
    aid = save_assessment(base, store, assessment)
    approval = {
        "bundle_id": base.name,
        "preparation_id": pid,
        "assessment_id": aid,
        "reviewer": "user",
        "approved_at": datetime.now(timezone.utc).isoformat(),
        "reason": "Proceed with the reviewed fresh sample.",
        "accepted_result_ids": [],
    }
    return base, store, pid, aid, approval


def test_admission_preserves_background_and_excludes_untranslated_foreign_text(
    bundle, document, tmp_path
):
    from app.services.theme_evaluation.extraction_intake import build_extraction_inputs

    base, store, pid, aid, approval = setup(
        bundle(
            documents=[
                document(document_id="post:1", text="Good morning, everyone."),
                document(
                    document_id="post:2", text="매출 성장", original_language="ko"
                ),
            ]
        ),
        tmp_path,
    )
    inputs, manifest = build_extraction_inputs(base, store, pid, aid, approval)
    assert [i.source_id for i in inputs] == ["post:1"]
    assert inputs[0].text == "Good morning, everyone."
    assert "original_completeness_unknown" in inputs[0].warnings
    assert any(e["source_id"] == "post:2" for e in manifest["exclusions"])
    assert manifest["approval"]["assessment_id"] == aid
    assert manifest["extraction_status"] == "pending"


def test_changed_assessment_cannot_reuse_approval(bundle, tmp_path):
    from app.services.theme_evaluation.extraction_intake import build_extraction_inputs

    base, store, pid, aid, approval = setup(bundle(), tmp_path)
    approval["assessment_id"] = "0" * 64
    with pytest.raises(ValueError, match="approval_evidence_mismatch"):
        build_extraction_inputs(base, store, pid, aid, approval)


def test_material_hold_cannot_be_overridden_by_blanket_approval(
    bundle, document, tmp_path
):
    from app.services.theme_evaluation.extraction_intake import build_extraction_inputs

    base, store, pid, _, approval = setup(bundle(), tmp_path)
    assessment = build_assessment(base, store, pid)
    entry_id, entry = next(iter(assessment["entries"].items()))
    annotation = {
        "bundle_id": base.name,
        "preparation_id": pid,
        "entry_id": entry_id,
        "result_id": entry["result_id"],
        "input_sha256": entry["input_sha256"],
        "source_text_sha256": entry["source_text_sha256"],
        "decision": "hold",
        "reviewer": "reviewer",
        "reviewed_at": datetime.now(timezone.utc).isoformat(),
        "reason": "Material unsupported claim",
        "scope": "evidence",
    }
    aid = save_assessment(
        base, store, build_assessment(base, store, pid, annotations=[annotation])
    )
    approval["assessment_id"] = aid
    inputs, manifest = build_extraction_inputs(base, store, pid, aid, approval)
    assert not inputs
    assert any(e["reason"] == "material_evidence_hold" for e in manifest["exclusions"])

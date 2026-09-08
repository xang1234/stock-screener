from datetime import datetime, timezone

import pytest
from app.services.theme_evaluation.bundle import seal_bundle
from app.services.theme_evaluation.preparation_pipeline import prepare
from app.services.theme_evaluation.preparation_records import Handoff
from app.services.theme_evaluation.preparation_store import PreparationStore
from app.services.theme_evaluation.records import Bundle


def setup(bundle, document, tmp_path):
    base = seal_bundle(
        tmp_path,
        Bundle.model_validate(
            bundle(documents=[document(text="매출 100억원", original_language="ko")])
        ),
    )
    store = PreparationStore(tmp_path / "out")
    pid = prepare(base, store, Handoff(bundle_id=base.name), stages=["text"])
    target = store.load(base, pid).bindings[0].result_id
    result = store.load_result(target)
    row = {
        "result_id": target,
        "source_text_sha256": result.request.input_sha256,
        "translations": ["Revenue 10 billion won"],
        "provider": "reviewed-import",
        "model": "human",
        "policy_version": "manual-v1",
        "generated_at": datetime(2026, 9, 8, tzinfo=timezone.utc).isoformat(),
    }
    return base, store, pid, row


def test_import_preserves_source_and_timestamp_but_flags_unit_change(
    bundle, document, tmp_path
):
    from app.services.theme_evaluation.preparation_translation_import import (
        import_translations,
    )

    base, store, pid, row = setup(bundle, document, tmp_path)
    new_pid = import_translations(base, store, pid, [row])
    manifest = store.load(base, new_pid)
    result = store.load_result(manifest.bindings[-1].result_id)
    assert result.status == "needs_review"
    assert result.payload["segments"][0]["original"] == "매출 100억원"
    assert result.payload["segments"][0]["translated"] == "Revenue 10 billion won"
    assert result.created_at.isoformat() == row["generated_at"]
    assert len(store.load(base, pid).bindings) == 1


@pytest.mark.parametrize(
    "change",
    [{"translations": []}, {"source_text_sha256": "a" * 64}, {"result_id": "a" * 64}],
)
def test_import_rejects_missing_segments_or_stale_source(
    change, bundle, document, tmp_path
):
    from app.services.theme_evaluation.preparation_translation_import import (
        import_translations,
    )

    base, store, pid, row = setup(bundle, document, tmp_path)
    row.update(change)
    with pytest.raises(ValueError):
        import_translations(base, store, pid, [row])


def test_identical_import_is_idempotent(bundle, document, tmp_path):
    from app.services.theme_evaluation.preparation_translation_import import (
        import_translations,
    )

    base, store, pid, row = setup(bundle, document, tmp_path)
    second = import_translations(base, store, pid, [row])
    assert import_translations(base, store, second, [row]) == second

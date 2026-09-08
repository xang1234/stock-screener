import pytest
from app.services.theme_evaluation.bundle import (
    load_bundle,
    seal_bundle,
    validate_bundle,
)
from app.services.theme_evaluation.records import Bundle


def test_sealed_content_detects_modification(tmp_path, bundle):
    path = seal_bundle(tmp_path, Bundle.model_validate(bundle()))
    (path / 'bundle.json').write_text('{}')
    with pytest.raises(ValueError, match='bundle_hash_mismatch'):
        load_bundle(path)


def test_repeat_sealing_preserves_original_files(tmp_path, bundle):
    value = Bundle.model_validate(bundle())
    path = seal_bundle(tmp_path, value)
    mtime = (path / 'bundle.json').stat().st_mtime_ns
    assert seal_bundle(tmp_path, value) == path
    assert (path / 'bundle.json').stat().st_mtime_ns == mtime
    assert load_bundle(path) == value


def test_legacy_metadata_round_trip_keeps_bundle_content_address(bundle, document):
    from app.services.theme_evaluation.bundle import canonical_bytes, sha256
    from app.services.theme_evaluation.review import coverage_summary

    value = Bundle.model_validate(bundle(documents=[document()])).model_dump(mode='json')
    metadata = value['documents'][0]['source_metadata']
    for field in ('image_captions', 'article_urls', 'reply_tweet_id', 'text_source',
                  'text_complete', 'incomplete_text_reasons'):
        metadata.pop(field, None)
    original_id = sha256(canonical_bytes(value))
    restored = Bundle.model_validate(value)
    assert coverage_summary(restored)['bundle_id'] == original_id


@pytest.mark.parametrize('change,error', [
    ({'text_sha256': '0' * 64}, 'text_hash_mismatch'),
    ({'retrieved_at': '2026-09-01T10:00:00'}, 'timezone'),
    ({'url': 'javascript:alert(1)'}, 'http'),
])
def test_invalid_evidence_rejected(document, bundle, change, error):
    with pytest.raises(ValueError, match=error):
        value = Bundle.model_validate(bundle(documents=[document(**change)]))
        validate_bundle(value)


def test_duplicate_document_ids_rejected(bundle, document):
    with pytest.raises(ValueError, match='duplicate_document'):
        validate_bundle(Bundle.model_validate(bundle(documents=[document(), document()])))


def test_real_bundle_requires_both_source_outcomes(bundle):
    with pytest.raises(ValueError, match='required_source_outcomes'):
        validate_bundle(Bundle.model_validate(bundle(mode='observed_capture')))


def test_cannot_smuggle_extractions_into_evidence_checkpoint(bundle):
    with pytest.raises(ValueError):
        Bundle.model_validate(bundle(extractions=[{'mentions': []}]))


def test_failed_write_does_not_publish_partial_bundle(tmp_path, bundle, monkeypatch):
    from pathlib import Path
    original = Path.rename

    def fail_publish(self, target):
        if self.name.startswith('.pending-'):
            raise OSError('simulated disk failure')
        return original(self, target)

    monkeypatch.setattr(Path, 'rename', fail_publish)
    with pytest.raises(OSError):
        seal_bundle(tmp_path, Bundle.model_validate(bundle()))
    assert list((tmp_path / 'bundles').iterdir()) == []


def test_successful_source_requires_raw_provenance(xui_payloads):
    from datetime import datetime, timezone

    from app.services.theme_evaluation.xui_intake import import_xui
    value = import_xui(xui_payloads, captured_at=datetime.now(timezone.utc), max_posts_per_source=5)
    value.source_outcomes[0].raw_sha256 = None
    with pytest.raises(ValueError, match='source_raw_hash_required'):
        validate_bundle(value)

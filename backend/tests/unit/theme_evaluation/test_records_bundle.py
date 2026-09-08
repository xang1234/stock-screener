import pytest

from app.services.theme_evaluation.bundle import load_bundle, seal_bundle, validate_bundle
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

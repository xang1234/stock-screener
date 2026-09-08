"""Canonical, atomic evidence bundles with cross-record validation."""

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

from .records import Bundle, REQUIRED_SOURCE_IDS


class IntegrityError(ValueError):
    """A sealed artifact no longer matches its content address."""


def canonical_bytes(value: dict) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(',', ':'), allow_nan=False).encode('utf-8')


def sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _unique(values, error: str) -> None:
    values = list(values)
    if len(values) != len(set(values)):
        raise ValueError(error)


def validate_bundle(bundle: Bundle) -> None:
    _unique((d.document_id for d in bundle.documents), 'duplicate_document')
    _unique((s.source_id for s in bundle.source_outcomes), 'duplicate_source')
    _unique((f.reference_id for f in bundle.followups), 'duplicate_reference')
    sources = {s.source_id: s for s in bundle.source_outcomes}
    if bundle.mode != 'controlled' and not all(
        source in sources and sources[source].required for source in REQUIRED_SOURCE_IDS
    ):
        raise ValueError('required_source_outcomes')
    if bundle.mode == 'retrospective_simulation' and not bundle.availability_rule:
        raise ValueError('simulation_rule_required')
    documents = {d.document_id: d for d in bundle.documents}
    for doc in bundle.documents:
        if sha256(doc.text.encode()) != doc.text_sha256:
            raise ValueError('text_hash_mismatch')
        if bundle.mode != 'controlled' and doc.kind == 'controlled':
            raise ValueError('mixed_controlled_evidence')
        _unique((m.source_id for m in doc.memberships), 'duplicate_membership')
        if any(m.source_id not in sources for m in doc.memberships):
            raise ValueError('unknown_membership_source')
    for source in bundle.source_outcomes:
        selected = sum(any(m.source_id == source.source_id for m in d.memberships)
                       for d in bundle.documents)
        if selected != source.selected_count or selected > source.returned_count:
            raise ValueError('source_counts_mismatch')
        if source.status != 'success' and (selected or not source.error_code):
            raise ValueError('failed_source_has_evidence_or_no_error')
    for ref in bundle.followups:
        if ref.post_id not in documents or documents[ref.post_id].kind != 'post':
            raise ValueError('unknown_reference_post')
        if ref.status in {'resolved', 'partial'}:
            article = documents.get(ref.article_id)
            if article is None or article.kind != 'article':
                raise ValueError('reference_requires_article')
            if not ref.attempted_at or not ref.match_basis or not ref.evidence_urls:
                raise ValueError('reference_requires_lookup_provenance')
            if ref.status == 'resolved' and article.capture_status != 'full':
                raise ValueError('partial_article_cannot_be_resolved')
        elif ref.article_id is not None:
            raise ValueError('failed_reference_has_article')
        if ref.status != 'pending' and (not ref.screen_reason or not ref.attempted_at):
            raise ValueError('followup_requires_decision_provenance')
        if ref.status == 'skipped_noninvestment' and ref.investment_related != 'no':
            raise ValueError('uncertain_reference_cannot_be_skipped')
    _unique(((d.document_id, d.target_language, d.policy_version) for d in bundle.derivatives),
            'duplicate_derivative')
    for derivative in bundle.derivatives:
        doc = documents.get(derivative.document_id)
        if doc is None or doc.text_sha256 != derivative.source_text_sha256:
            raise ValueError('derivative_source_mismatch')
        if derivative.status == 'unavailable' and derivative.text:
            raise ValueError('unavailable_translation_has_text')
        if derivative.status == 'identity' and derivative.text != doc.text:
            raise ValueError('identity_translation_changed_text')


def seal_bundle(root: Path, bundle: Bundle) -> Path:
    validate_bundle(bundle)
    payload = canonical_bytes(bundle.model_dump(mode='json'))
    digest = sha256(payload)
    parent = root / 'bundles'
    parent.mkdir(parents=True, exist_ok=True)
    target = parent / digest
    if target.exists():
        load_bundle(target)
        return target
    temporary = Path(tempfile.mkdtemp(prefix='.pending-', dir=parent))
    try:
        for name, data in (
            ('bundle.json', payload),
            ('manifest.json', canonical_bytes({'sha256': digest, 'size_bytes': len(payload)})),
        ):
            with (temporary / name).open('xb') as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
        try:
            temporary.rename(target)
        except OSError:
            if not target.exists():
                raise
            load_bundle(target)
        return target
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def load_bundle(path: Path) -> Bundle:
    try:
        payload = (path / 'bundle.json').read_bytes()
        manifest = json.loads((path / 'manifest.json').read_bytes())
        digest = sha256(payload)
        if (digest != path.name or digest != manifest['sha256']
                or len(payload) != manifest['size_bytes']):
            raise IntegrityError('bundle_hash_mismatch')
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise IntegrityError('bundle_manifest_invalid') from exc
    result = Bundle.model_validate_json(payload)
    validate_bundle(result)
    return result


def verify_bundle(path: Path) -> dict:
    bundle = load_bundle(path)
    return {'bundle_id': path.name, 'documents': len(bundle.documents), 'integrity': 'verified'}

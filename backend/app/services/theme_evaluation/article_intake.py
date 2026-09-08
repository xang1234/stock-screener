"""Article references and independently captured evidence imports; no crawler."""

import re
from urllib.parse import urlsplit, urlunsplit

from .bundle import sha256, validate_bundle
from .records import Bundle, Derivative, Document, Followup


def canonical_url(url: str) -> str:
    parts = urlsplit(url)
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), parts.path, parts.query, ''))


def propose_references(bundle: Bundle) -> list[Followup]:
    refs = {r.reference_id: r for r in bundle.followups}
    for doc in bundle.documents:
        if doc.kind != 'post':
            continue
        urls = set(re.findall(r'https?://[^\s<>"\u2026]+', doc.text))
        urls.update(doc.source_metadata.article_urls)
        if doc.source_metadata.is_article:
            urls.add(doc.url)
        for url in sorted(urls):
            url = url.rstrip('.,;!?)')
            reference_id = 'ref:' + sha256((doc.document_id + '\n' + url).encode())[:24]
            refs.setdefault(reference_id, Followup(
                reference_id=reference_id, post_id=doc.document_id, reference_text=url,
                candidate_url=url, investment_related='uncertain', screen_reason='Awaiting context review',
                screen_version='url-queue-v1', status='pending', article_id=None,
                attempted_at=None, lookup_method=None, match_basis=None, evidence_urls=[], error_code=None,
            ))
    return sorted(refs.values(), key=lambda r: r.reference_id)


def apply_followups(bundle: Bundle, updates: list[Followup], articles: list[Document]) -> Bundle:
    result = bundle.model_copy(deep=True)
    existing = {d.document_id: d for d in result.documents}
    identities = {(canonical_url(d.url), d.text_sha256): d.document_id
                  for d in result.documents if d.kind == 'article'}
    remap = {}
    for article in articles:
        if article.kind != 'article' or not article.text.strip():
            raise ValueError('article_body_required')
        if article.document_id in existing and existing[article.document_id] != article:
            raise ValueError('document_id_conflict')
        identity = canonical_url(article.url), article.text_sha256
        if identity in identities:
            remap[article.document_id] = identities[identity]
            continue
        identities[identity] = article.document_id
        existing[article.document_id] = article
        result.documents.append(article)
    refs = {r.reference_id: r for r in result.followups}
    if len({r.reference_id for r in updates}) != len(updates):
        raise ValueError('duplicate_reference_update')
    for update in updates:
        old = refs.get(update.reference_id)
        if old and old.post_id != update.post_id:
            raise ValueError('reference_post_changed')
        replacement = update.model_copy(update={'article_id': remap.get(update.article_id, update.article_id)})
        if old and old.article_id and old.article_id != replacement.article_id:
            history_id = old.reference_id + ':capture:' + sha256(old.article_id.encode())[:16]
            refs.setdefault(history_id, old.model_copy(update={'reference_id': history_id}))
        refs[update.reference_id] = replacement
    result.followups = sorted(refs.values(), key=lambda r: r.reference_id)
    validate_bundle(result)
    return result


def import_derivatives(bundle: Bundle, values: list[Derivative]) -> Bundle:
    result = bundle.model_copy(deep=True)
    result.derivatives.extend(values)
    validate_bundle(result)
    return result

import pytest

from app.services.theme_evaluation.article_intake import apply_followups, import_derivatives, propose_references
from app.services.theme_evaluation.records import Bundle, Derivative, Document, Followup


def reference(post_id, article_id, stamp, **updates):
    value = dict(reference_id='ref:' + post_id, post_id=post_id, reference_text='Source article',
                 candidate_url='https://example.com/article', investment_related='yes',
                 screen_reason='Operating evidence', screen_version='review-v1', status='resolved',
                 article_id=article_id, attempted_at=stamp, lookup_method='direct',
                 match_basis='Exact linked URL and title', evidence_urls=['https://example.com/article'],
                 error_code=None)
    value.update(updates)
    return Followup.model_validate(value)


def test_queue_does_not_assume_short_links_are_articles(bundle, document):
    value = Bundle.model_validate(bundle(documents=[document(
        text='New orders. https://t.co/example', source_metadata={'image_urls': ['https://t.co/example']})]))
    refs = propose_references(value)
    assert len(refs) == 1
    assert refs[0].investment_related == 'uncertain'
    assert refs[0].status == 'pending'


def test_native_article_gets_post_url_candidate(bundle, document):
    value = Bundle.model_validate(bundle(documents=[document(source_metadata={'is_article': True})]))
    refs = propose_references(value)
    assert refs[0].candidate_url == 'https://example.com/post/1'


def test_two_posts_share_one_article_without_backdating(bundle, document):
    first = Document.model_validate(document())
    second = Document.model_validate(document(document_id='post:2'))
    article = Document.model_validate(document(document_id='a', kind='article',
        url='https://example.com/article#section', retrieved_at=first.retrieved_at.replace(hour=11)))
    duplicate = article.model_copy(update={'document_id': 'b', 'url': 'https://example.com/article#other'})
    refs = [reference(first.document_id, 'a', article.retrieved_at),
            reference(second.document_id, 'b', article.retrieved_at)]
    base = Bundle.model_validate(bundle(documents=[first, second]))
    result = apply_followups(base, refs, [article, duplicate])
    articles = [d for d in result.documents if d.kind == 'article']
    assert len(articles) == 1
    assert articles[0].retrieved_at.hour == 11
    assert len({r.article_id for r in result.followups}) == 1
    assert base.followups == []


def test_article_version_change_preserves_both_texts(bundle, document):
    base = Bundle.model_validate(bundle())
    a = Document.model_validate(document(document_id='a', kind='article', url='https://example.com/a'))
    b = Document.model_validate(document(document_id='b', kind='article', url=a.url, text='Corrected text'))
    result = apply_followups(base, [reference('post:1', 'a', a.retrieved_at)], [a])
    result = apply_followups(result, [reference('post:1', 'b', b.retrieved_at)], [b])
    assert len([d for d in result.documents if d.kind == 'article']) == 2
    assert {r.article_id for r in result.followups} == {'a', 'b'}
    assert {r.post_id for r in result.followups} == {'post:1'}


def test_partial_body_cannot_be_claimed_as_full_article(bundle, document):
    article = Document.model_validate(document(document_id='a', kind='article', capture_status='partial'))
    with pytest.raises(ValueError, match='partial_article'):
        apply_followups(Bundle.model_validate(bundle()),
                        [reference('post:1', 'a', article.retrieved_at)], [article])


def test_lookup_failure_stays_distinct_from_success(bundle, document):
    stamp = Document.model_validate(document()).retrieved_at
    ref = reference('post:1', None, stamp, status='paywalled', error_code='paywall')
    result = apply_followups(Bundle.model_validate(bundle()), [ref], [])
    assert result.followups[0].status == 'paywalled'
    assert all(d.kind != 'article' for d in result.documents)


def test_translation_with_wrong_source_hash_rejected(bundle, document):
    stamp = Document.model_validate(document()).retrieved_at
    derivative = Derivative(document_id='post:1', source_text_sha256='0' * 64,
        target_language='en', text='Translated', provider=None, model=None,
        policy_version='v1', generated_at=stamp, status='translated')
    with pytest.raises(ValueError, match='derivative_source_mismatch'):
        import_derivatives(Bundle.model_validate(bundle()), [derivative])


def test_unreferenced_article_cannot_enter_evidence(bundle, document):
    article = Document.model_validate(document(document_id='a', kind='article'))
    with pytest.raises(ValueError, match='unreferenced_article'):
        apply_followups(Bundle.model_validate(bundle()), [], [article])

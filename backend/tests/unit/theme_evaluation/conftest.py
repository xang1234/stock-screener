import hashlib
from datetime import datetime, timezone

import pytest


STAMP = datetime(2026, 9, 1, 10, tzinfo=timezone.utc)


@pytest.fixture
def document():
    def make(**overrides):
        value = dict(
            document_id='post:1', kind='post', title='Example',
            text='A supplier reports new orders.', url='https://example.com/post/1',
            author='analyst', publisher=None, published_at=STAMP.replace(hour=9),
            updated_at=None, retrieved_at=STAMP, original_language='en',
            memberships=[], capture_status='full', reference_only=False, source_metadata={},
        )
        value.update(overrides)
        value.setdefault('text_sha256', hashlib.sha256(value['text'].encode()).hexdigest())
        return value
    return make


@pytest.fixture
def bundle(document):
    def make(**overrides):
        value = dict(schema_version=1, mode='controlled', availability_rule=None,
                     source_outcomes=[], documents=[document()], derivatives=[], followups=[],
                     extractions=[], labels=[], selection={}, limitations=[])
        value.update(overrides)
        return value
    return make


@pytest.fixture
def xui_payloads():
    result = {}
    for list_id in ('1986290701492232693', '1522014550211457024'):
        source = 'list:' + list_id
        result[list_id] = dict(
            items=[dict(tweet_id='1', source_id=source, text='New orders.',
                        tweet_url='https://x.com/analyst/status/1', author_handle='analyst',
                        created_at=STAMP.isoformat(), observed_at=STAMP.isoformat(),
                        quality_tier='full', is_article=False, image_urls=[])],
            outcomes=[dict(source_id=source, source_kind='list', ok=True,
                           item_count=1, observed_ids=1, error=None)],
            failed_sources=[], succeeded_sources=[source],
        )
    return result

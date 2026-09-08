"""Versioned records for the evidence-review checkpoint."""

from typing import Annotated, Literal
from urllib.parse import urlsplit

from pydantic import (
    AfterValidator,
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    model_serializer,
)

REQUIRED_LIST_IDS = ('1986290701492232693', '1522014550211457024')
REQUIRED_SOURCE_IDS = tuple('x-list:' + value for value in REQUIRED_LIST_IDS)


def http_url(value: str) -> str:
    parsed = urlsplit(value)
    if (parsed.scheme not in {'http', 'https'} or not parsed.hostname
            or parsed.username or parsed.password or any(ord(c) < 32 for c in value)):
        raise ValueError('expected public http(s) URL without credentials')
    return value


URL = Annotated[str, AfterValidator(http_url)]
SHA = Annotated[str, Field(pattern=r'^[a-f0-9]{64}$')]
Count = Annotated[int, Field(ge=0, strict=True)]


class Record(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)


class SourceOutcome(Record):
    source_id: str
    required: bool
    status: Literal['success', 'failed', 'reauth_required']
    requested_limit: Count | None
    returned_count: Count
    selected_count: Count
    observed_ids: Count | None
    captured_at: AwareDatetime
    error_code: str | None
    raw_sha256: SHA | None
    source_config_revision: str | None


class Membership(Record):
    source_id: str
    observed_at: AwareDatetime


class SourceMetadata(Record):
    extraction_method: str | None = None
    extraction_version: str | None = None
    quality_tier: str | None = None
    quality_score: float | None = None
    quote_tweet_id: str | None = None
    is_reply: bool | None = None
    is_article: bool | None = None
    image_urls: list[URL] = Field(default_factory=list)
    image_captions: list[str | None] = Field(default_factory=list)
    article_urls: list[URL] = Field(default_factory=list)
    reply_tweet_id: str | None = None
    text_source: str | None = None
    text_complete: bool | None = None
    incomplete_text_reasons: list[str] = Field(default_factory=list)
    observed_at_fallback: bool = False
    pdf_sha256: SHA | None = None
    pdf_title: str | None = None
    pdf_warnings: list[str] = Field(default_factory=list)
    pdf_exported_at: AwareDatetime | None = None

    @model_serializer(mode='wrap')
    def preserve_legacy_metadata(self, handler):
        result = handler(self)
        # Additive reader fields must not change IDs of previously sealed v1 bundles.
        for name in ('image_captions', 'article_urls', 'reply_tweet_id', 'text_source',
                     'text_complete', 'incomplete_text_reasons'):
            if name not in self.model_fields_set:
                result.pop(name, None)
        return result


class Document(Record):
    document_id: str
    kind: Literal['post', 'article', 'controlled']
    title: str
    text: str
    url: URL
    author: str | None
    published_at: AwareDatetime | None
    updated_at: AwareDatetime | None
    publisher: str | None
    retrieved_at: AwareDatetime
    original_language: str | None
    memberships: list[Membership]
    capture_status: Literal['full', 'partial']
    text_sha256: SHA
    reference_only: bool
    source_metadata: SourceMetadata


class Derivative(Record):
    document_id: str
    source_text_sha256: SHA
    target_language: str
    text: str
    provider: str | None
    model: str | None
    policy_version: str
    generated_at: AwareDatetime
    status: Literal['translated', 'identity', 'unavailable']


class Followup(Record):
    reference_id: str
    post_id: str
    reference_text: str
    candidate_url: URL | None
    investment_related: Literal['yes', 'no', 'uncertain']
    screen_reason: str
    screen_version: str
    status: Literal['pending', 'resolved', 'partial', 'unresolved', 'unavailable',
                    'paywalled', 'not_article', 'skipped_noninvestment']
    article_id: str | None
    attempted_at: AwareDatetime | None
    lookup_method: str | None
    match_basis: str | None
    evidence_urls: list[URL]
    error_code: str | None


class Bundle(Record):
    schema_version: Literal[1]
    mode: Literal['observed_capture', 'retrospective_simulation', 'controlled']
    availability_rule: str | None
    source_outcomes: list[SourceOutcome]
    documents: list[Document]
    derivatives: list[Derivative]
    followups: list[Followup]
    # Generation and label imports are deliberately absent before evidence review.
    extractions: list[dict] = Field(default_factory=list, max_length=0)
    labels: list[dict] = Field(default_factory=list, max_length=0)
    selection: dict
    limitations: list[str]

# Theme Evaluation Pilot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Use superpowers:subagent-driven-development only if the user explicitly chooses delegated execution. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a versioned, auditable pilot corpus from both required X lists and their investment-related article references, freeze extractions, and generate evidence packets for user review.

**Architecture:** A small backend package manages immutable local files; one command-line entry point imports captures, validates references, freezes extraction results, and renders review packets. X reads use the existing xui-reader wrapper. Article lookups use existing agent reading/search capabilities and a validated import contract, without a second crawler. Extraction generation reuses the current theme extractor against an explicitly isolated evaluation database.

**Tech Stack:** Python 3.11, existing Pydantic 2.5.3, pathlib/json/hashlib/csv/argparse, pytest; existing SQLAlchemy/PostgreSQL and LLMService only for the optional generation command. No frontend or new package dependency.

**Spec:** [Theme evaluation design](../specs/2026-09-08-theme-evaluation-design.md).

## Global Constraints

- The current application database is new and empty, as confirmed by the user. Existing historical content, scans, and rankings are not prerequisites that can be assumed.
- Both user-specified X lists are required sources: `1986290701492232693` and `1522014550211457024`.
- Use the skill at `/Users/admin/Documents/Work/xui-reader/skills/xui-reader/SKILL.md`; never inspect session storage, expose cookies, or mutate X state.
- Initial price/ranking analytics remain US-scoped; preserve original languages and identify unsupported translation rather than discarding posts.
- Freeze article extractions initially; no extraction-model comparison or live latency claim in this milestone.
- Report real streams, retrospective simulations, and controlled cases separately.
- Empty extraction, provider failure, unresolved article, missing data, and uncertain human labels are different outcomes.
- Raw acquired bodies/provider outputs stay in ignored local storage. The ideas document stays only in the main checkout.
- Routine import, verification, and review commands require neither production database access nor external calls.
- Each mutation creates a new bundle version; do not edit a sealed dataset in place.

## Milestone boundary

**User checkpoint, 2026-09-08:** Review the collected evidence before generating any extractions. Execute Tasks 1–3 and bring forward the source-only Markdown/CSV report from Tasks 5–6. Stop with an immutable evidence bundle for user review. Extraction generation, extraction imports, and theme-label proposals remain unavailable at this checkpoint. Add their record implementations with their later tasks rather than accepting unvalidated future records now. A review of one bundle does not approve later changes to its contents.

This plan implements acquisition, fixed extractions, label contracts, and review packets. It is a complete first deliverable, not the entire theme benchmark. Fresh-history capture integration and controlled-clock ranking replay require separate implementation plans grounded in the same design. This milestone must explicitly report `ranking_evaluation: unavailable` rather than invent a quality comparison.

Coverage map:

| Design requirement | This plan |
| --- | --- |
| Required X lists, provenance, languages, bounded selection | Tasks 1–2 |
| Investment-related article lookup and explicit failed follow-up | Task 3 plus operator runbook |
| Frozen extractions and provider/version provenance | Task 4 |
| AI proposals, user approval, uncertainty, held-out labels | Task 5 |
| Markdown/CSV review, local hashes, coverage, controlled examples | Tasks 1 and 5–6 |
| Historical price/scanner state, ranking metrics, clock injection | Separate ranking-replay milestone; inputs described in the design |
| Immutable capture during ordinary ingestion/publication | Separate capture-integration milestone |

## Existing facts and reuse

- Branch is `feat/theme-detection-evaluation`; do not create another worktree.
- Access-check files exist locally in `data/xui-reader/theme-evaluation-source-audit-20260908/`. Their manifest records 125 and 143 returned items, 262 distinct tweet IDs, and six overlapping IDs. They are an access survey, not a reviewed historical corpus.
- Requested xui limits did not cap exported cardinality. Record raw counters before deterministic local selection.
- `ThemeExtractionService.extract_from_content(ContentItem)` returns cleaned mentions without assigning clusters. Call this method only for generation; never call `process_content_item`, `process_batch`, or `_extract_and_store_mentions` from corpus preparation.
- Existing extraction uses a 10,000-character prefix, title-based ticker augmentation, source-language metadata, DB-backed universe lookup, and fallback providers. Preserve and record those behaviors; changing them is a separate experiment.
- Existing `_try_generate_litellm` loses response provenance when returning text. Capture the actual completion response through a transparent wrapper around the service's injected `llm`, leaving output parsing in the existing extractor.
- Existing `.gitignore` already excludes `data/xui-reader/`. Put evaluation artifacts below `data/xui-reader/theme-evaluation/`; do not add a new ignore rule.
- Initial focused baseline: 26 theme identity/source-quality/lifecycle tests passed before any code change. Repeat them only after modifying an integration path or at the final milestone check.

## File map

Create under `backend/app/services/theme_evaluation/`:

| File | Responsibility |
| --- | --- |
| `__init__.py` | Package marker with no initialization side effects |
| `records.py` | Strict versioned input, follow-up, extraction, and label records |
| `bundle.py` | Canonical encoding, content hashes, sealed versions, integrity verification |
| `xui_intake.py` | Read-only wrapper invocation and deterministic import of xui outputs |
| `article_intake.py` | Follow-up queue and validated imports from article lookup tools |
| `extraction_capture.py` | Extraction import and transparent completion metadata capture |
| `extraction_runtime.py` | Explicit evaluation-DB generation; lazy production imports |
| `review.py` | Evidence packets, user-label imports, review CSV, coverage summary |
| `cli.py` | Argument parsing and orchestration only |

Also create:

- `backend/scripts/theme_evaluation.py`: thin CLI entry point.
- `backend/tests/unit/theme_evaluation/conftest.py`: complete synthetic builders shared by the tests below.
- `backend/tests/unit/theme_evaluation/test_records_bundle.py`.
- `backend/tests/unit/theme_evaluation/test_xui_intake.py`.
- `backend/tests/unit/theme_evaluation/test_article_intake.py`.
- `backend/tests/unit/theme_evaluation/test_extraction_capture.py`.
- `backend/tests/unit/theme_evaluation/test_review.py`.
- `backend/tests/unit/theme_evaluation/test_cli.py`.
- `docs/theme_evaluation/pilot_runbook.md`.

No app startup registration, API route, Celery schedule, production migration, or frontend file is required for this milestone.

## Shared record contract

Use Pydantic models with `extra='forbid'`, timezone-aware datetimes, explicit schema version `1`, and JSON-compatible values. Declare all fields below in `records.py`; optional means nullable, not silently fabricated.

| Model | Fields |
| --- | --- |
| `SourceOutcome` | `source_id`, `required: bool`, `status: success|failed|reauth_required`, `requested_limit: int`, `returned_count: int`, `selected_count: int`, `observed_ids: int|None`, `captured_at`, `error_code: str|None`, `raw_sha256: str|None` (null only when no response bytes), `source_config_revision: str|None` |
| `Membership` | `source_id`, `observed_at` |
| `Document` | `document_id`, `kind: post|article|controlled`, `title`, `text`, `url`, `author: str|None`, `published_at: datetime|None`, `updated_at: datetime|None`, `publisher: str|None`, `retrieved_at`, `original_language: str|None`, `memberships: list[Membership]`, `capture_status: full|partial`, `text_sha256`, `reference_only: bool`, `source_metadata: dict` (allowlisted reader quality/version, quote/reply IDs, PDF hash/title/warnings/export time only) |
| `Derivative` | `document_id`, `source_text_sha256`, `target_language`, `text`, `provider: str|None`, `model: str|None`, `policy_version`, `generated_at`, `status: translated|identity|unavailable` |
| `Followup` | `reference_id`, `post_id`, `reference_text`, `candidate_url: str|None`, `investment_related: yes|no|uncertain`, `screen_reason`, `screen_version`, `status: pending|resolved|partial|unresolved|unavailable|paywalled|not_article|skipped_noninvestment`, `article_id: str|None`, `attempted_at: datetime|None`, `lookup_method: str|None`, `match_basis: str|None`, `evidence_urls: list[str]`, `error_code: str|None` |
| `Extraction` | `extraction_id`, `document_id`, `input_text_sha256`, `pipeline: technical|fundamental`, `generated_at`, `status: success|empty|failed|unavailable`, `mentions: list[dict]`, `requested_model`, `actual_model: str|None`, `provider: str|None`, `messages_sha256: str|None`, `parameters: dict`, `parser_revision`, `reference_data_sha256: str|None`, `raw_response_sha256: str|None`, `usage: dict`, `error_code: str|None` |
| `EpisodeLabel` | `episode_id`, `pipeline: technical|fundamental`, `decision: proposed|approved|uncertain|rejected`, `reviewer: str|None`, `reviewed_at: datetime|None`, `thesis`, `document_ids: list[str]`, `first_useful_at: datetime|None`, `equivalent_episode_ids: list[str]`, `stock_judgments: list[dict]`, `rationale`, `label_version`, `evidence_hashes: dict[str,str]`, `passages: list[dict]` (document ID, start/end offsets), `relations: list[dict]` (episode ID, subset|distinct), `partition: development|holdout|controlled` |
| `Bundle` | `schema_version`, `mode: observed_capture|retrospective_simulation|controlled`, `availability_rule: str|None`, `source_outcomes: list[SourceOutcome]`, `documents: list[Document]`, `derivatives: list[Derivative]`, `followups: list[Followup]`, `extractions: list[Extraction]`, `labels: list[EpisodeLabel]`, `selection: dict`, `limitations: list[str]` |

Validate cross-record references in `validate_bundle(bundle) -> None`. `stock_judgments` entries have exactly `symbol`, `support` (`supported|unsupported|uncertain`), `document_ids`, and `reason`. Mention dictionaries contain exactly `theme: str`, `tickers: list[str]`, `sentiment: str`, `confidence: float` (finite, within 0–1), and `excerpt: str`; preserve returned strings and explicit empty lists. Passage entries contain exactly `document_id`, integer `start`, and integer `end`, with `0 <= start < end <= len(text)`. Relation entries contain exactly `episode_id` and `kind` (`subset|distinct`). A failed result cannot masquerade as successful empty output.

Every non-controlled bundle must contain exactly one outcome for each required list, including explicit failures. Controlled fixtures are exempt from real-source completeness and may never be presented as acquired evidence. Labels bind to document text hashes and valid passage offsets; `first_useful_at` requires at least one supporting non-reference-only passage available by that time. Later context may remain in the packet but cannot justify that time.

Observed detector eligibility is derived separately as the latest required input observation/extraction/translation time; do not overwrite publication or actual retrieval timestamps. In retrospective mode, require an explicit simulation rule, and retain actual times. `reference_only` material is never automatically admitted to detector inputs.

## Shared executable test builders

The following belongs in `backend/tests/unit/theme_evaluation/conftest.py`. Each test module imports `pytest` and the function/model under test from the file identified by its task. Interfaces shown with ellipses in this plan are signature contracts, not implementation bodies.

```python
import hashlib
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest


STAMP = datetime(2026, 9, 1, 10, tzinfo=timezone.utc)


@pytest.fixture
def document():
    def make(**overrides):
        value = dict(
            document_id='post:1', kind='post', title='Example',
            text='A supplier reports new orders.',
            url='https://example.com/post/1', author='analyst',
            published_at=STAMP.replace(hour=9), updated_at=None,
            publisher=None, retrieved_at=STAMP, original_language='en',
            memberships=[], capture_status='full', reference_only=False,
            source_metadata={},
        )
        value.update(overrides)
        value.setdefault('text_sha256', hashlib.sha256(value['text'].encode()).hexdigest())
        return value
    return make


@pytest.fixture
def bundle(document):
    def make(**overrides):
        value = dict(
            schema_version=1, mode='controlled', availability_rule=None,
            source_outcomes=[], documents=[document()], derivatives=[],
            followups=[], extractions=[], labels=[], selection={}, limitations=[],
        )
        value.update(overrides)
        return value
    return make


@pytest.fixture
def xui_payloads():
    result = {}
    for list_id in ('1986290701492232693', '1522014550211457024'):
        source = 'list:' + list_id
        result[list_id] = dict(
            items=[dict(tweet_id='shared', source_id=source, text='New orders.',
                        tweet_url='https://example.com/post/shared',
                        author_handle='analyst', created_at=STAMP.isoformat(),
                        observed_at=STAMP.isoformat(), quality_tier='full')],
            outcomes=[dict(source_id=source, source_kind='list', ok=True,
                           item_count=1, observed_ids=1, error=None)],
            failed_sources=[], succeeded_sources=[source],
        )
    return result


@pytest.fixture
def article_bundle(bundle, document):
    from app.services.theme_evaluation.records import Bundle, Document, Followup
    base = Bundle.model_validate(bundle(documents=[
        document(), document(document_id='post:2', url='https://example.com/post/2')]))
    article = Document.model_validate(document(
        document_id='article:1', kind='article',
        url='https://example.com/article/1', retrieved_at=STAMP.replace(hour=11)))
    updates = [Followup.model_validate(dict(
        reference_id='ref:' + post_id, post_id=post_id,
        reference_text='Source article', candidate_url=article.url,
        investment_related='yes', screen_reason='Operating evidence',
        screen_version='test-v1', status='resolved', article_id=article.document_id,
        attempted_at=article.retrieved_at, lookup_method='direct',
        match_basis='Exact referenced URL and title', evidence_urls=[article.url],
        error_code=None,
    )) for post_id in ('post:1', 'post:2')]
    return SimpleNamespace(base=base, article=article, updates=updates)


@pytest.fixture
def fake_llm():
    class FakeLLM:
        preset = SimpleNamespace(primary=SimpleNamespace(model_id='requested'))
        response = SimpleNamespace(model='returned-fallback', usage=None, choices=[])

        async def completion(self, **kwargs):
            return self.response
    return FakeLLM()


@pytest.fixture
def review_bundle(bundle, document):
    from app.services.theme_evaluation.records import Bundle
    doc = document()
    return Bundle.model_validate(bundle(labels=[dict(
        episode_id='episode:1', pipeline='technical', decision='proposed',
        reviewer=None, reviewed_at=None, thesis='Supplier capacity expands.',
        document_ids=[doc['document_id']], first_useful_at=STAMP,
        equivalent_episode_ids=[], stock_judgments=[], rationale='New orders.',
        label_version='v1', evidence_hashes={doc['document_id']: doc['text_sha256']},
        passages=[dict(document_id=doc['document_id'], start=0, end=10)],
        relations=[], partition='controlled',
    )]))
```

Run the async proxy test with `@pytest.mark.asyncio`, following the existing pytest-asyncio setup. For the CLI end-to-end test, use `cli.main(argv: list[str]) -> int` directly, `tmp_path`, and these builders serialized with `json.dumps(..., default=str)`; obtain each output bundle path from the command's JSON stdout with `capsys.readouterr()`. Never launch a live reader from a unit test.

## Task 1: Strict records and immutable bundle storage

**Files:** create `records.py`, `bundle.py`, package marker, test builders, `test_records_bundle.py`.

**Interfaces:**

```python
def canonical_bytes(value: dict) -> bytes: ...
def validate_bundle(bundle: Bundle) -> None: ...
def seal_bundle(root: Path, bundle: Bundle) -> Path: ...
def load_bundle(path: Path) -> Bundle: ...
def verify_bundle(path: Path) -> dict: ...
```

- [x] Write complete builders `document(**overrides)` and `bundle(**overrides)` in the test conftest. Default document: `post:1`, post kind, title `Example`, text `A supplier reports new orders.`, URL `https://example.com/post/1`, author `analyst`, publication `2026-09-01T09:00:00Z`, retrieval `2026-09-01T10:00:00Z`, language `en`, empty memberships, full text, correctly computed SHA-256, `reference_only=False`. Default bundle has this document, no other records, controlled mode, empty selection/limitations. The executable builders below supply every required field. Add new default fields here when extending records; do not rely on undeclared fixtures.
- [x] Add failing tests including the following; builders return dictionaries accepted by model validation.

```python
def test_sealed_content_cannot_be_modified(tmp_path, bundle):
    path = seal_bundle(tmp_path, Bundle.model_validate(bundle()))
    (path / 'bundle.json').write_text('{}')
    with pytest.raises(ValueError, match='bundle_hash_mismatch'):
        load_bundle(path)

def test_same_content_has_same_bundle_id(tmp_path, bundle):
    value = Bundle.model_validate(bundle())
    assert seal_bundle(tmp_path, value) == seal_bundle(tmp_path, value)
```

- [x] Run `python -m pytest tests/unit/theme_evaluation/test_records_bundle.py -q`; confirm failure due to absent implementation.
- [x] Implement canonical JSON using sorted keys, UTF-8, compact separators, `allow_nan=False`; compute SHA-256 on those bytes. Serialize Pydantic models in JSON mode. Store versions under `root/bundles/<digest>/bundle.json` plus a manifest containing the digest and file size. Create a temporary sibling directory, write/flush files, and atomically rename; if the digest directory exists, verify identical content and return it without rewriting. Clean only the temporary directory created by this operation on failure.

```python
def canonical_bytes(value: dict) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(',', ':'), allow_nan=False).encode('utf-8')
```

- [x] Enforce aware timestamps, referenced-document existence, correct text hashes, article-only targets for resolved follow-ups, and explicit source failures. Test naive dates, modified bytes, duplicate IDs, missing raw provenance, and failed atomic writes. At the user-required evidence checkpoint, reject all nonempty extraction/label inputs; implement confidence and approved-label validation with their later tasks.
- [x] Re-run the task tests and commit only its files with `feat(themes): add immutable evaluation corpus records`.

## Task 2: Required-list acquisition and deterministic selection

**Files:** create `xui_intake.py`, initial `cli.py`, thin script, `test_xui_intake.py`; extend test builders.

**Interfaces:**

```python
REQUIRED_LIST_IDS = ('1986290701492232693', '1522014550211457024')
def read_required_lists(*, wrapper: Path, python: Path, xui_bin: Path,
                        config: Path, profile: str, limit: int,
                        run_command) -> dict[str, dict]: ...
def import_xui(payloads: dict[str, dict], *, captured_at: datetime,
               max_posts_per_source: int, mode: str = "observed_capture") -> Bundle: ...
```

- [x] Write tests with complete minimal xui payloads (`items`, `outcomes`, `failed_sources`, `succeeded_sources`). Include an outcome that reports five observed IDs but exports eight records, an overlapping tweet across both lists, and a failed second list.

```python
def test_shared_post_keeps_both_memberships(xui_payloads):
    from datetime import datetime, timezone
    result = import_xui(xui_payloads, captured_at=datetime(2026, 9, 1, 10, tzinfo=timezone.utc), max_posts_per_source=100, mode='controlled')
    shared = next(d for d in result.documents if d.document_id == 'post:shared')
    assert {m.source_id for m in shared.memberships} == {
        'x-list:1986290701492232693', 'x-list:1522014550211457024'}
    assert len([d for d in result.documents if d.document_id == shared.document_id]) == 1
```

- [x] Run `python -m pytest tests/unit/theme_evaluation/test_xui_intake.py -q`; confirm the new behavior fails before implementation.
- [x] Invoke the supplied wrapper with argument arrays (`subprocess.run`, never a shell), profile `default`, policy `prompt`, `--config-path` for the wrapper config, and explicit list IDs. Put `xui_bin.parent` at the front of the child process PATH. The wrapper performs authentication checking. On `reauth_required`, preserve its safe reauthentication command and stop further live reads; do not launch interactive login. Use bounded process timeouts and preserve stable errors without secrets or browser logs in exported reports.
- [x] Capture each invocation's start/end time and raw response hash. Import all post fields needed by the record contract; keep `observed_at` when supplied and mark a capture-time fallback if absent. Normalize the observed reader ID `list:<id>` to the corpus ID `x-list:<id>`; validate each item against its payload source. Do not infer source membership solely from returned row order. Unexpected source IDs fail validation.
- [x] Sort unique posts within each source by descending publication time with tweet ID as a stable tie-breaker; explicitly place missing publication times last. Select up to `max_posts_per_source`, then union sources by tweet ID. Save raw and selected counts and selected IDs. Record this as a recent sample, not a complete chronological archive. Preserve original text/language; a missing language is unknown. If the same ID has different bodies, record a content-version conflict with raw hashes and excluded IDs in `selection`, retain the raw captures, and exclude that ID from the selected documents until reviewed instead of overwriting it.
- [x] Add `collect-x` and `import-x` CLI commands; the latter can consume the existing access-check capture without new X requests. Failed required sources remain in the manifest and prevent a complete required-source status. Zero returned posts is distinct from authentication success with useful coverage.
- [x] Verify argument arrays contain only read operations and both required IDs; verify no session-storage reads and no import of application DB initialization. Re-run the task tests; commit with `feat(themes): ingest required X list evidence for evaluation`.

## Task 3: Article follow-up queue and evidence imports

**Files:** create `article_intake.py`, `test_article_intake.py`; extend `cli.py` and record tests.

**Interfaces:**

```python
def propose_references(bundle: Bundle) -> list[Followup]: ...
def apply_followups(bundle: Bundle, updates: list[Followup],
                    articles: list[Document]) -> Bundle: ...
def import_derivatives(bundle: Bundle, values: list[Derivative]) -> Bundle: ...
```

- [x] Add tests for two posts linking one article, a picture short link, a title-only reference, paywall failure, untranslated content, and a referenced article collected after its post.

```python
def test_article_does_not_inherit_post_retrieval_time(article_bundle):
    updated = apply_followups(article_bundle.base, article_bundle.updates,
                              [article_bundle.article])
    article = next(d for d in updated.documents if d.kind == 'article')
    assert article.retrieved_at > updated.documents[0].retrieved_at
    assert len([r for r in updated.followups if r.article_id == article.document_id]) == 2
```

- [x] Run `python -m pytest tests/unit/theme_evaluation/test_article_intake.py -q` and confirm failure.
- [x] Extract URL candidates from text with a simple bounded URL regex; create `pending`, investment-relevance `uncertain` records. Add a title-reference entry when the reading agent supplies a title/author reference. Do not automatically label all links as articles or classify relevance by presence of a ticker.
- [x] The agent reviews the queue and records investment relevance with a reason/version, follows usable links with existing web tools, or searches title/author/publisher when needed. Imports must name the exact matched piece and supporting lookup URLs. Same-topic search results are insufficient. Import unresolved/paywalled/partial outcomes explicitly. Do not build another browser or recursive crawler in this package.
- [x] Native X Articles use the skill's `article-pdf` wrapper. Record the PDF hash, title, warnings, export time, and source post; import text only from a supported PDF reader, retaining `partial` when extraction is incomplete. Normal post text never stands in for the full Article body.
- [x] Canonicalize article identity conservatively: strip URL fragments, normalize scheme/host casing, retain query parameters unless a verified canonical URL is supplied. Deduplicate identical canonical URL plus text hash, preserve all referring posts, and treat changed text as a new version. Store lookup/translation provenance separately from immutable original text.
- [x] Add `references`, `import-articles`, and `import-translations` commands. An unresolved required article lookup is a coverage gap, not silent success. Every change seals a new bundle version. Re-run tests; commit with `feat(themes): preserve article follow-up evidence for evaluation`.

## Task 4: Freeze extraction output with actual provenance

**Files:** create `extraction_capture.py`, `extraction_runtime.py`, `test_extraction_capture.py`; extend CLI. Read current extractor, LLMService, and SecurityMaster code before integration. No broad extraction refactor.

**Interfaces:**

```python
class RecordingLLM:
    def __init__(self, wrapped): ...
    async def completion(self, **kwargs): ...
    def __getattr__(self, name): ...
    def _record(self, kwargs: dict, response) -> dict: ...

def import_extractions(bundle: Bundle, values: list[Extraction]) -> Bundle: ...
def generate_extractions(bundle: Bundle, *, session_factory,
                         reference_manifest: dict, model: str,
                         max_documents: int) -> Bundle: ...
```

- [ ] Test empty success versus failure, wrong input hash, duplicate extraction keys, transparent LLM response forwarding, fallback actual model capture, and per-pipeline separation. Build a fake wrapped LLM with a complete response object and an explicit preset.

```python
@pytest.mark.asyncio
async def test_recording_proxy_preserves_response(fake_llm):
    proxy = RecordingLLM(fake_llm)
    response = await proxy.completion(messages=[{'role': 'user', 'content': 'sample'}])
    assert response is fake_llm.response
    assert proxy.calls[0]['actual_model'] == fake_llm.response.model
    assert proxy.preset is fake_llm.preset
```

- [ ] Run `python -m pytest tests/unit/theme_evaluation/test_extraction_capture.py -q`; confirm failure.
- [ ] Implement the proxy with one awaited call, returning the same response. Define `_record` to return an allowlisted dictionary with actual model (`getattr(response, "model", None)`), requested model, message hash, temperature/max_tokens/allow_fallbacks, usage (`response.usage.model_dump(mode="json")` when supplied), and choices (`choice.model_dump(mode="json")`). Provider is nullable and may only come from a documented public response field. Define `__init__` to set `wrapped` and `calls=[]`, and `__getattr__` to delegate to `wrapped`. Save canonical hashes of actual messages/parameters, requested and returned model/provider, usage, parser/code revision, generation time, and a sanitized response payload. Exclude authorization, API keys, cookies, and hidden transport metadata. Unknown actual provider/model stays unknown. On exceptions, persist a failed record with a stable error and no fabricated mentions.

```python
async def completion(self, **kwargs):
    response = await self.wrapped.completion(**kwargs)
    self.calls.append(self._record(kwargs, response))
    return response
```

- [ ] Implement generation with the existing `ThemeExtractionService` instantiated per pipeline against an explicitly supplied isolated session factory. Inject the recording proxy into `service.llm`, set the validated sanctioned requested model, and call only `extract_from_content` on transient `ContentItem` inputs. Do not assign clusters. Preserve original text, normal truncation and ticker cleaning. If a translation is used, record the exact input choice and derivative hash rather than claiming original-input parity.
- [ ] Require a frozen reference-data snapshot for symbol/company resolution before generation. The command receives a dedicated `THEME_EVAL_DATABASE_URL`, verifies PostgreSQL and that it differs from the application DSN, and verifies an `AppSetting` row with key `theme_evaluation_database` and value `isolated-v1`. Compare parsed database identity (host, effective port, database name), ignoring driver and credential differences, rather than comparing DSN strings. Reject ambiguous aliases to the production database. The connection uses SELECT-only permissions and no telemetry persistence outside the evaluation runtime. Compute the reference digest from the actual active `StockUniverse` rows (symbol/name plus fields used by `active_filter`) and allowlisted nonsecret settings read by the extractor, sorted canonically, and compare it to the supplied manifest. Record resolver/configuration code revisions. A supplied checksum without matching rows is insufficient. Do not connect to, reset, or seed the application database. Provisioning instructions use a dedicated database and the repository's normal migrations; no arbitrary database creation/drop helper is needed.
- [ ] Fail before provider calls when credentials, sanctioned model, database marker, or reference snapshot are missing. Record `reference_data_unavailable` rather than generating against an empty universe and reporting an empty stock basket as meaningful. Local import of externally generated, fully attributed extractions remains available without PostgreSQL; historical provenance can be explicitly unknown but must be reported.
- [ ] Support `import-extractions` and an explicitly invoked `generate-extractions --max-documents N`. Routine replay/review never calls a provider. Pin technical/fundamental inputs separately. Explicitly persist zero-theme successes.
- [ ] Re-run capture tests and relevant existing theme-extraction/provider routing tests. Commit with `feat(themes): freeze evaluation extractions and provenance`.

## Task 5: Review packets and user-approved labels

**Files:** create `review.py`, `test_review.py`; extend CLI and test builders.

**Interfaces:**

```python
def render_review(bundle: Bundle, output: Path) -> list[Path]: ...
def import_labels(bundle: Bundle, labels: list[EpisodeLabel]) -> Bundle: ...
def coverage_summary(bundle: Bundle) -> dict: ...
```

- [ ] Test that proposed labels are not approved by default, uncertain cases remain visible, two equivalent detector names do not become two reference IDs, future reviewer evidence cannot justify an earlier usefulness time, and spreadsheet formulas in untrusted text are escaped.

```python
def test_proposed_is_not_review_complete(review_bundle):
    summary = coverage_summary(review_bundle)
    assert summary['approved_episodes'] == 0
    assert summary['review_status'] == 'pending'
    assert summary['ranking_evaluation'] == 'unavailable'
```

- [ ] Run `python -m pytest tests/unit/theme_evaluation/test_review.py -q` and confirm failure.
- [ ] Render a Markdown index with source completeness, actual sample counts, overlap, article outcomes, languages, extraction coverage, and limitations. Render per-document evidence packets sorted by available time, showing original/translated text, source links, related article outcomes, and fixed extractions. Include items with no extraction or no theme so reviewers can find missed opportunities.
- [ ] AI-proposed episode packets use independent episode IDs and explicit `proposed` decisions. User imports must provide reviewer/time for `approved` decisions. Validate all evidence references and require `first_useful_at` to be supported by admissible evidence under the declared availability mode. Enforce reference-only exclusions and preserve unknown timing. Any corpus change invalidating referenced hashes requires a new label version.
- [ ] Export `documents.csv`, `followups.csv`, `extractions.csv`, `episode_labels.csv`, and `coverage.csv`. Use Python's csv module and prefix text cells beginning, after leading whitespace, with `=`, `+`, `-`, or `@` with an apostrophe. Keep numeric columns numeric. Escape HTML and Markdown control text in untrusted titles so a post cannot impersonate a report instruction.
- [ ] Use explicit coverage statuses: required sources complete/incomplete; follow-up complete/incomplete; extraction complete/incomplete; review pending/complete; ranking evaluation unavailable. Do not collapse these into a misleading single passed flag. Re-run tests and commit with `feat(themes): generate evidence review packets for benchmark labels`.

## Task 6: End-to-end pilot workflow and bounded real execution

**Files:** complete `cli.py`, `backend/scripts/theme_evaluation.py`, `test_cli.py`, `docs/theme_evaluation/pilot_runbook.md`.

**CLI contract:** commands use argparse and return stable exit codes: `0` for successful operation, `2` for invalid arguments/data, `3` for source/auth/access failure, `4` for unavailable extraction prerequisites, `5` for integrity failure. A successful import can contain source gaps; its JSON result must expose them. A command claiming readiness must return nonzero when required coverage is incomplete.

```text
import-x --first PATH --second PATH --mode observed_capture --max-posts-per-source 100 --output-root PATH
collect-x --wrapper PATH --python PATH --xui-bin PATH --config PATH --profile default --limit 50 --output-root PATH
references --bundle PATH --output PATH
import-articles --bundle PATH --records PATH --output-root PATH
import-translations --bundle PATH --records PATH --output-root PATH
import-extractions --bundle PATH --records PATH --output-root PATH
generate-extractions --bundle PATH --reference-manifest PATH --model MODEL --max-documents N --output-root PATH
review --bundle PATH --output PATH
import-labels --bundle PATH --records PATH --output-root PATH
verify --bundle PATH
```

- [ ] Write a CLI test that imports two controlled xui outputs, imports one article with two referring posts, imports fixed extractions including an empty result, seals the bundle, and renders review outputs. Set `--mode controlled` for fixture imports. Use fixture data only; prohibit external commands/network calls unless a test injects a fake reader. Assert real/synthetic modes cannot be merged and the report does not claim ranked performance.
- [ ] Run `python -m pytest tests/unit/theme_evaluation/test_cli.py -q`; confirm failure before completing orchestration.
- [ ] Keep production integrations lazily imported inside `generate-extractions`; importing or verifying a bundle must work when no database or provider is configured. The thin script adds the backend directory to sys.path and calls `cli.main()`.

```python
if __name__ == '__main__':
    raise SystemExit(main())
```

- [ ] Document exact wrapper commands from the user-specified skill and the safe local executable path discovered during access checks. Explain that required X reads may return more records than requested, and how selected IDs/counts are preserved. Audit configured Yahoo Finance, MarketWatch, Doomberg, and SentimenTrader access as optional supplements and record source URL, checked time, accessible scope, and reason selected/excluded in the selection manifest; do not presume archive completeness. Document targeted article lookup/import, native Article PDF handling, multilingual review, and explicit unresolved-reference recording. Do not recommend circumventing failed web access or relabeling a related article as the referenced original.
- [ ] Document dedicated evaluation-database prerequisites for optional extraction generation. Credentials remain environment-managed and never enter manifests. Show normal invocation without printing DSNs or secret values. Treat missing credentials/reference data as pending acquisition dependencies, not a reason to fabricate provider outputs.
- [ ] Run all new tests and `tests/unit/test_theme_identity_normalization.py`, `tests/unit/test_theme_source_quality_weighting.py`, and `tests/unit/test_theme_lifecycle_policies.py`. Run `git diff --check`. Review only the intended milestone diff.
- [ ] Import the existing access-check files using a new output root below `data/xui-reader/theme-evaluation/`. Do not overwrite the access audit. Generate a reference queue, perform investment-related lookups through the supported tools, and import actual outcomes. Produce review packets even if some references remain unresolved; expose the incomplete coverage prominently.
- [ ] Generate fixed extractions only when the explicit isolated runtime prerequisites are satisfied. Deliver the source/evidence inventory regardless, distinguishing engineering completion, available data, pending review, and missing extraction inputs. User approval of labels is an external dependency.
- [ ] Commit code/runbook with `feat(themes): deliver offline pilot acquisition and review workflow`. Keep real source bodies and generated packets untracked/ignored.

## Verification commands

From this worktree's `backend/`, use the existing interpreter if the worktree has no separate virtual environment:

```bash
STOCKSCANNER_TEST_ALLOW_POSTGRES=0 STOCKSCANNER_TEST_USE_DATABASE_URL=0 DATABASE_URL=sqlite:// STOCKSCANNER_TEST_ALLOW_SQLITE=1 PYTHONPATH=. /Users/admin/StockScreenClaude/backend/venv/bin/python -m pytest tests/unit/theme_evaluation/ -q
```

This is the project's existing unit-test harness; it is not the runtime database for generation or replay. All provider and reader interactions in ordinary tests are injected fakes.

## Plan self-review checklist

These checked items describe document review, not completed implementation.

- [x] Every source requirement in the spec maps to a task and a coverage field.
- [x] Both required IDs appear in implementation constants, tests, manifest checks, and runbook examples.
- [x] There is no assumption of existing app history, a populated universe, valid provider credentials, or complete X archives.
- [x] Stored translations, articles, and extractions preserve their own generation/retrieval times and source hashes.
- [x] Cross-list duplicates preserve membership without duplicate evidence credit.
- [x] Each public interface uses only the models/signatures defined here.
- [x] Reports state that this milestone does not yet produce ranking comparison metrics.
- [x] Reference labels remain proposed until user review; empty, failed, uncertain, and missing outcomes remain distinct.
- [x] No secret storage, session cookies, or raw provider transport metadata is read into artifacts.
- [x] The ideas document remains in the main checkout, and acquired evidence remains ignored local data.

## Execution handoff

The default execution path is inline using `superpowers:executing-plans`, with a checkpoint after Task 3 so the source corpus and article handling can be reviewed before optional provider generation. If the user explicitly requests parallel agent work, use `superpowers:subagent-driven-development` with task-local ownership and reviews. Do not spawn agents solely because this plan mentions that option.

## Evidence checkpoint delivered, 2026-09-08

Tasks 1–3 are implemented, including local collection/import commands, atomic evidence bundles, article follow-up decisions and translation imports. The source-only renderer and CLI review path were brought forward from Tasks 5–6 to honor the user's checkpoint. Extraction and episode-label record implementations remain deferred with Tasks 4–5; the current schema rejects nonempty extraction/label lists.

The local pilot selects 50 posts per required list from the frozen access audit: 96 unique posts, four cross-list overlaps. Two matched articles have partial excerpts only; 14 reference outcomes remain partial/unavailable/unresolved. Twenty-eight posts are conservatively marked potentially truncated, and all 96 posts lack reader-supplied language metadata. No translations, extraction calls, or theme-label proposals were generated.

Review artifacts are below `data/xui-reader/theme-evaluation/pilot-20260908/`; source bodies stay ignored. The exact reviewed bundle is identified in `review-location.json`. Capture history integration, full extraction provenance, reviewed theme labels, and ranking replay remain outstanding. The current sample is not a days-to-weeks performance benchmark.

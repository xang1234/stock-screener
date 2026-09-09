# X Reader and Evidence Preparation Fixes Implementation Plan

> **For agentic workers:** Use `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Execute inline unless the user separately authorizes delegation. Read the linked specification first.

**Goal:** Repair capture and preparation gaps so an evidence reviewer can identify usable investment information without missing-source assumptions or noisy error flags.

**Architecture:** Deliver two independently testable work packages: Tasks 1–2 in xui, Tasks 3–8 in StockScreenClaude. xui exports original observations; the downstream project assesses and enriches them using existing approved services. Task 9 validates the combined contract and produces a new review packet, with extraction still paused.

**Tech Stack:** Python, existing xui Playwright collector and pytest fixtures; downstream Pydantic, HTTPX, BeautifulSoup, Pillow, Kimi K2.6/OpenCode Go, pytest and Markdown/CSV reports. No new runtime dependency is planned.

**Spec:** [Evidence gap remediation specification](../specs/2026-09-09-evidence-gap-remediation.md).

## Global Constraints

- Model: Kimi K2.6 through the existing OpenCode Go adapters only.
- Article acquisition: existing HTTPX public fetcher and BeautifulSoup parsing.
- X acquisition: existing xui Playwright session and commands.
- Keep extraction at `awaiting_evidence_approval`.
- Preserve original text, original images, URLs, capture times and all preparation attempts.
- Preserve frozen bundle/result identities and legacy interpretation.
- Keep existing bounded X attempts: at most one inline translation attempt and one shared targeted-post fallback per selected post.
- Default preparation retries: initial attempt plus at most one eligible retry per input per run; concurrency at most three.
- External-article rendered-browser collection requires tool approval before live use. Its tested import path can ship independently.
- No production database changes, theme labels, extraction runs, reader-side Kimi dependency or provider migration.

## Checkout and scope instructions

Reader repository: [https://github.com/xang1234/xui](https://github.com/xang1234/xui), inspected at `1a3bfc4` in `/Users/admin/Documents/Work/xui-reader`. Preparation checkout: `/Users/admin/StockScreenClaude/.worktrees/theme-detection-evaluation`, inspected at `efe41424`.

This plan and its spec are saved in the main StockScreenClaude folder for handoff. They do not indicate that any implementation has been performed. Do not put downstream modules in xui merely because this plan was given to a reader engineer. A reader-only engineer should deliver Tasks 1–2 and the reader portion of Task 9, and hand off Tasks 3–8 explicitly.

Before implementation, read each target repository's `AGENTS.md`, inspect current branches and changes, and compare HEAD with the inspected revisions. The xui repository uses Beads; follow its issue workflow when implementation starts. Use a `feat/` branch or the existing authorized evaluation worktree; do not overwrite another task's uncommitted work. Review current code before applying snippets if either repository advanced.

## Delivery order and dependency graph

```text
Reader:       1 Capture completeness -> 2 Post completeness -----------+
                                                                    |
Preparation:  3 Adequacy checks -> 4 X/Kimi selection -----------------+
              5 Article recovery -----------------------------------+-> 9 Replay + review packet
              6 Image failure diagnosis/retries --------------------+
              3,4,5,6 -> 7 Review assessments ------------------------+
              3 -> 8 Language coverage and efficient preparation ----+
```

Ship each task as a reviewable commit with its tests. Do not wait for rendered-browser approval to complete HTTP parsing, diagnostics, translation, or offline verification. Task 9's fresh run should use completed reader and preparation revisions together.

## Task 1 — Capture stable, complete X translation containers

**Repository:** xui. **Priority:** P1.

**Files:**
- Modify `src/xui_reader/collectors/translation.py` (`_text_nodes`, `_translated_candidate`, `_wait_for_change`, automatic and on-demand paths).
- Modify `src/xui_reader/translation.py` only for new sanitized failure reasons/normalization that must survive replay.
- Modify `src/xui_reader/selectors/defaults.toml` if a verified DOM capture requires a selector change.
- Test `tests/test_collectors_translation.py`, `tests/test_translation.py`, `tests/test_scheduler_translation.py`.
- Add sanitized fixtures under `tests/fixtures/posts/` for fragments and ambiguous translation containers.

**Existing interface:** `XTranslationCapturer.capture_viewport(page, target_tweet_id=None) -> TranslationViewport`, containing `TranslationObservation` values. Preserve public translation statuses and the JSON contract.

- [ ] Add a fake-page timeline test showing the same post render `Samsung`, then the complete sentence containing both quantities. Advance an injected monotonic clock through the fake page's waits; do not use real sleeps.
- [ ] Add automatic-translation and on-demand variants, nested English spans within one owned container, two independent English containers, quoted-tweet contamination, detached/replaced post nodes, permanently loading text and failed original restoration.
- [ ] Run `./.venv/bin/python -m pytest tests/test_collectors_translation.py tests/test_translation.py tests/test_scheduler_translation.py`; confirm the new fragment cases fail before editing production code.
- [ ] Replace first-change acceptance with bounded stabilization. Introduce an injected monotonic clock and `settle_ms=200` on the capturer, preserving its existing wall-clock capture-time callback. A candidate must be unchanged across observations spanning 200 ms, inside the existing action timeout. Check ownership and loading/expansion signals on every observation. Never extend the original deadline to obtain stability.

```python
# Target state transition; use the existing page and fake clock in tests.
candidate = None
stable_since = None
deadline = monotonic_clock() + action_timeout_ms / 1000
# Within the bounded polling loop:
# changed candidate -> restart stable_since
# owned, nonempty, no pending expansion/loading, unchanged >= 0.2s -> accept
# deadline without eligible candidate -> capture_failed / translated_text_unstable
```

- [ ] Read the full owned translation root's visible text. If selectors return a root and descendants, retain the root once. Preserve DOM reading order inside that root. Do not concatenate independent tweet-text roots or choose the longest of competing containers. Emit `translated_container_ambiguous` when ownership cannot be established.
- [ ] Stabilize already-visible automatic translations **before** switching to original view. Keep the existing original restoration/quarantine behavior. Do not introduce cross-post retries or weaken identity matching to recover more text.
- [ ] Keep `status="capture_failed"`, `text=None`, and a specific failure reason for unstable/ambiguous results. Preserve failure observations for the existing bounded targeted fallback. `captured` means a structurally sound capture, not semantic translation approval; Task 4 provides the second check.
- [ ] Run the three focused suites plus `tests/test_security_hygiene.py` and `tests/test_public_api_contracts.py`; run Ruff on modified files. Commit as `fix: capture stable owned X translation blocks`.

**Acceptance:** The timeline fixture emits the full translation or an explicit failure, never the first fragment. Existing short legitimate translations still pass. Automatic/on-demand and quote-scope tests pass. Added waiting is bounded and measured. The Samsung live DOM cause remains unclaimed until reproduced.

## Task 2 — Make post completeness provenance conservative and actionable

**Repository:** xui. **Priority:** P1.

**Files:**
- Modify `src/xui_reader/extract/dom_metadata.py`, `src/xui_reader/extract/post_metadata.py` only where tests expose unsupported certainty.
- Modify `src/xui_reader/post_content.py`, `src/xui_reader/scheduler/hydration.py`, `src/xui_reader/scheduler/translation.py` for source/metadata binding and shared attempt accounting.
- Test `tests/test_post_content.py`, `tests/test_scheduler_read.py`, `tests/test_scheduler_translation.py`, `tests/test_render_jsonout.py`.

**Contract:** Reuse `TweetItem.text_complete: bool | None`, `text_source`, `incomplete_text_reasons`. No second competing completeness field. Confirmed truncation is `False`; absence of proof is `None`; explicit complete source evidence is `True`.

- [ ] Add a DOM-only regression: `tweetText` without an expansion button is insufficient by itself to establish completeness.

```python
from xui_reader.extract.dom_metadata import dom_metadata

def test_dom_without_expansion_control_is_not_proof_of_completeness():
    value = dom_metadata('<article><div data-testid="tweetText" lang="ko">삼성</div></article>')
    assert value["text_complete"] is None
```

- [ ] Add payload tests for `truncated=False`, explicit truncation, full note text, missing note text and display ranges that exclude only mentions/media. Add a merge test proving completeness metadata belongs to the selected text, rather than to a different observed version of the same post.
- [ ] Run `./.venv/bin/python -m pytest tests/test_post_content.py tests/test_scheduler_read.py tests/test_scheduler_translation.py tests/test_render_jsonout.py` and observe the new failure.
- [ ] Change DOM certainty conservatively; preserve explicit payload evidence. Keep hydration of `text_complete=False` bounded. Record attempted expansion/fallback and its outcome through existing incomplete reasons and capture metadata. Unknown posts are not automatically all reopened.
- [ ] Reuse `HydrationResult.attempted_post_ids` when requesting translations so a failed completeness read cannot cause a second targeted read for the same selected post. Route a later manually selected unknown-post review through the existing POST command, as a separate explicitly bounded run.
- [ ] Confirm original text upgrades invalidate any old translation hash; retain earlier text in capture history. Verify selected-list limits and memberships survive hydration.
- [ ] Run the focused suites, public contract tests, replay tests and Ruff. Commit as `fix: preserve conservative post completeness evidence`.

**Acceptance:** Unknown never silently becomes complete; completeness and translation share targeted-read accounting; changing text cannot retain a translation tied to the old bytes. A lower count of “complete” posts is acceptable when it corrects unsupported certainty.

## Task 3 — Add versioned translation adequacy assessment

**Repository:** StockScreenClaude. **Priority:** P1.

**Files:**
- Create `backend/app/services/theme_evaluation/translation_quality.py`.
- Create `backend/tests/unit/theme_evaluation/test_translation_quality.py`.
- Read `multilingual_preparation.py` and `kimi_translation.py` in the same service package; reuse existing protected-quantity rules where compatible.

**New interface:**

```python
from dataclasses import dataclass
from typing import Literal

@dataclass(frozen=True)
class QualityIssue:
    code: str
    severity: Literal["blocker", "review", "info"]
    source_excerpt: str
    translated_excerpt: str

@dataclass(frozen=True)
class TranslationAssessment:
    policy_version: str  # translation-quality-v1
    disposition: Literal["use", "fallback", "review"]
    issues: tuple[QualityIssue, ...]

def assess_translation(original: str, translated: str, *, language: str | None) -> TranslationAssessment:
    """Assess derivatives without changing original text or archived result status."""
```

- [ ] Add table-driven tests with these minimum expectations:

```python
import pytest
from app.services.theme_evaluation.translation_quality import assess_translation

@pytest.mark.parametrize("source,target,language,expected", [
    ("718만주 1.9조원에 매수", "Samsung", "ko", "fallback"),
    ("매출 10% 증가", "Revenue increased 100%", "ko", "fallback"),
    ("718만주 1.9조원", "7.18 million shares, 1.9 trillion won", "ko", "use"),
    ("매출 2조5500억원", "Revenue 2.55 trillion won", "ko", "use"),
    ("매출 증가 https://t.co/a123", "Revenue increased", "ko", "use"),
    ("4월과 5월", "April and May", "ko", "use"),
    ("https://t.co/a123", "https://t.co/a123", "zxx", "use"),
    ("👀", "👀", "art", "use"),
])
def test_assessment(source, target, language, expected):
    assert assess_translation(source, target, language=language).disposition == expected
```

- [ ] Run the new test file before implementing; confirm failure.
- [ ] Implement comparison on temporary normalized copies only. Remove URLs/standalone handles from quantitative comparison, retain them in originals. Use `Decimal`, compound Korean/Chinese/Japanese units, multiplicity, sign, percentage, and currency/unit identity. Normalize `718만주` to 7,180,000 shares and `2조5500억원` to 2,550,000,000,000 KRW. Treat ambiguous units as review, not an inferred conversion.
- [ ] Normalize explicitly recognizable month names/month-number notation; retain year and day distinctions. Do not compare quantities as an untyped bag alone: check nearby explicit metric/period associations for supported patterns. Swapped Q1/Q2 values or changed revenue/profit labels require review even when totals match.
- [ ] Treat verified missing/changed quantities or units as blockers/fallback. Add supported negation/direction regression pairs. Unknown semantic equivalence remains review; no claim of general semantic proof from rules. Preserve original company-name spans for reviewer comparison; do not add a guessed security mapping.
- [ ] Add a conservative fragment warning when prose length after stripping links is at least 30 characters and the translation has at most two words and less than 20% of source character length. Use it as `review` unless material omissions independently justify `fallback`. These are initial policy parameters, not calibrated confidence scores.
- [ ] Keep `TextPreparation.warnings` and archived status computation unchanged in this task. Assessment is a separate, versioned derivative. Add tests loading existing results and confirming canonical serialization/status unchanged.
- [ ] Run the new suite plus `test_multilingual_preparation.py`, `test_kimi_translation.py`, `test_preparation_store.py`. Commit as `feat: assess translation adequacy with typed quantities`.

**Acceptance:** Listed real false alarms disappear from the new assessment; the Samsung omission and changed values remain blockers. Original bytes and old result IDs do not change. Deterministic checks expose their limits instead of claiming correctness.

## Task 4 — Select X or Kimi using adequacy, with explicit provenance

**Repository:** StockScreenClaude. **Depends on:** Task 3. **Priority:** P1.

**Files:**
- Modify `backend/app/services/theme_evaluation/xui_translation.py`, `preparation_pipeline.py`, `preparation_state.py` only where necessary for explicit selection.
- Create `backend/app/services/theme_evaluation/translation_selection.py`.
- Modify `backend/tests/unit/theme_evaluation/test_xui_translation.py`, `test_preparation_regressions.py`.

**Interfaces:** Consume `assess_translation` from Task 3 and existing `captured_translation_result(doc)`. Add `select_translation(original, language, x_result, kimi_result)` returning a frozen decision with `selected_provider: str | None`, `eligible: bool`, and an assessment. `x_result` and `kimi_result` are existing `TextResult | None`; the helper makes no network calls. `eligible` means usable as a candidate subject to the user's evidence approval, not already approved. A `review` or `fallback` disposition has `eligible=False` until resolved.

- [ ] Extend the existing ingestion-to-preparation test with the exact Samsung source/hash. Assert the preserved X record remains `Samsung`, Kimi receives the complete original once, and the selected eligible output contains both quantities. Keep the existing accepted-X test requiring zero model calls.
- [ ] Test rejected X plus failed Kimi, rejected X plus materially wrong Kimi, no translator configured, unsupported/ambiguous X output, and a changed X capture with the same original hash.
- [ ] Run `test_xui_translation.py` and `test_preparation_regressions.py` to observe the new failure.
- [ ] Apply this selection table in a small orchestration helper, not another long conditional block inside `prepare`:

| X assessment | Action | Eligible result |
| --- | --- | --- |
| Captured, `use` | No Kimi call | X |
| Captured, `review` | Preserve and flag; no automatic second opinion | X pending review |
| Captured, `fallback` | One Kimi translation, assess it | Kimi only if no blocker |
| Missing/failed capture | Existing Kimi fallback, assess it | Kimi if usable; otherwise unresolved |
| Kimi also fails/has blocker | Preserve both attempts | None |

- [ ] Store both candidate results using the existing result store, even when one is rejected. After sealing preparation, save an immutable selection sidecar at `selection-decisions/<sha256>.json` under the preparation output root. Use existing canonical serialization; define `schema_version=1`, `policy_version="translation-quality-v1"`, `bundle_id`, `preparation_id`, and a `decisions` array. Each decision contains `document_id`, `source_text_sha256`, nullable `x_result_id`, nullable `kimi_result_id`, nullable `selected_result_id`, `eligible`, `disposition`, and serialized `issues`. Validate all referenced results and exact source hashes before writing. The stage returns or reports the sidecar ID alongside the existing preparation ID; do not change the existing `prepare(...) -> str` return contract. Task 7 loads this exact sidecar and verifies its preparation ID.
- [ ] A technically usable rejected X result may remain in legacy history, but must never appear as eligible merely because `PreparationState.record` retains usable old results after a failed retry. Task 7 consumes the explicit eligibility decision rather than inferring eligibility from legacy `current` alone. Do not automatically copy a selection sidecar to a different preparation: carry decisions forward only after rechecking unchanged document/candidate hashes and resealing with the new preparation ID.
- [ ] Preserve current source-hash validation. Never cache an X capture solely by original text. Kimi translation output may reuse its exact-request cache; selection assessments must include assessment policy and candidate IDs. Changing adequacy policy need not trigger a new model call.
- [ ] Run the two focused suites plus `test_preparation_pipeline.py`, `test_preparation_store.py`, and `test_preparation_translation_import.py`. Commit as `fix: fall back from materially incomplete X translations`.

**Acceptance:** No rejected X capture can silently become the approved fallback after Kimi failure. Each selected translation has a reason, exact input hash and provider; original evidence remains intact.

## Task 5 — Recover article bodies and route references correctly

**Repository:** StockScreenClaude. **Priority:** P1.

**Files:**
- Modify `backend/app/services/theme_evaluation/article_recovery.py`, `preparation_pipeline.py`, `preparation_cli.py`.
- Create `backend/app/services/theme_evaluation/reference_routing.py`.
- Add fixtures under `backend/tests/unit/theme_evaluation/fixtures/articles/`.
- Modify `test_article_recovery.py`, `test_article_intake.py`, `test_preparation_cli.py`; create `test_reference_routing.py`.

**Interfaces:** Keep `parse_article(raw: bytes, final_url: str) -> ArticleRecovery`, `fetch_public`, and `import_articles` intact. Add `classify_reference(url, *, parent_post_id)` returning `article_candidate | linked_x_post | same_x_post | not_article`; classification is structural, not a language-model investment verdict. Uncertain investment relevance stays reviewable.

- [ ] Add fixture tests for two references resolving to one destination, a short link pointing to the same post, a different X post, an IR home, a real IR announcement, a quote-rich article, navigation/translation controls, script-only HTML, a paywall excerpt, and multiple conflicting article bodies.

```python
from app.services.theme_evaluation.article_recovery import parse_article

def test_script_only_page_is_not_an_article_body():
    result = parse_article(b'<html><title>HBM yields</title><script>loadArticle()</script></html>',
                           'https://example.com/news/123')
    assert result.text == ""
    assert result.capture_status == "partial"

def test_reader_controls_are_excluded_without_removing_body_quotes():
    raw = b'<article><div role="toolbar">Translate</div><p>Revenue rose 10%.</p><blockquote>Demand is strong.</blockquote></article>'
    result = parse_article(raw, 'https://example.com/news/123')
    assert "Translate" not in result.text
    assert "Revenue rose 10%." in result.text
    assert "Demand is strong." in result.text
```

- [ ] Run the focused article and routing tests to establish failure.
- [ ] Prefer unique article JSON-LD and owned article-body regions; remove positively identified controls, not arbitrary paragraphs matching words such as “subscribe.” Add publisher rules only from inspected saved HTML/DOM. Prioritize Daum cleanup and Oracle announcement body selection; Chosun/ZDnet script-only bodies remain `render_required` if absent. Do not fabricate selectors for content absent from a response.
- [ ] Resolve redirects through `fetch_public`, then group exact final destinations and credible same-origin canonical aliases. Preserve every original reference ID and its source post. Keep query parameters unless known tracking parameters; never merge different article IDs or trust an unrelated canonical host. Reuse recovered bytes/model work after identity is established; resolving distinct short links may still require separate HTTP requests.
- [ ] Produce a linked-post follow-up manifest with normalized post IDs, deduplicated globally, excluding originals already in the base. Limit to one hop and 20 unique posts for this pilot; cycles and excess items receive explicit dispositions. Consume their exports through xui's existing POST read/import workflow, never through an HTML article parser. Linked posts do not count toward either list's selected 50-post quota.
- [ ] Distinguish `body_missing`, `body_ambiguous`, `access_restricted`, `http_401`, `http_403`, `render_required`, `not_article`, and `linked_x_post` in assessment output. Avoid treating every partial article as a browser failure. Store actual capture time, URL and body hash.
- [ ] Stamp newly parsed article requests with `article-v2` when extraction behavior changes. Reparse stored bytes into a new result; never mutate an old `ArticleRecovery` or reuse `article-v1` cached outputs as if they were produced by the new parser.
- [ ] Strengthen existing `import_articles` tests: reject mismatched reference/destination/hash and invented completeness. A browser import must include the viewed final URL, capture time, exact text and identity match basis. Keep `capture_status="partial"` unless a reviewer supplies completeness evidence. No live renderer is added in this task.
- [ ] After tool approval only: capture unresolved pages using the approved rendered browser and import them through that boundary. Restriction remains visible if the browser also cannot read the content. Record any authorized alternative public publisher source as a different supporting source, not a silent replacement.
- [ ] Run article/routing/import suites plus `test_preparation_regressions.py`. Commit as `fix: recover and classify article evidence by destination`.

**Acceptance:** Duplicate references survive while preparation work is reused. All three currently readable destinations get clean, inspectable excerpts; script-only pages are never reported as full articles. Browser approval affects only live rendering.

## Task 6 — Diagnose image failures and retry only recoverable ones

**Repository:** StockScreenClaude. **Priority:** P1.

**Files:**
- Modify `backend/app/services/theme_evaluation/kimi_client.py`, `image_preparation.py`, `preparation_pipeline.py`, `preparation_cli.py`.
- Create `backend/app/services/theme_evaluation/preparation_failures.py`.
- Modify `test_image_preparation.py`, `test_preparation_pipeline.py`, `test_preparation_regressions.py`; create `test_preparation_failures.py`.

**New interface:** `PreparationFailure(RuntimeError)` with safe fields `code: str`, `retryable: bool`, `http_status: int | None`. Exception messages contain the code only. `retryable_failure(code: str) -> bool` uses a closed allowlist; unknown errors are not automatically retried.

- [ ] Add mock transport tests distinguishing connection failure, timeout, 429, 401/403, 5xx, incomplete model response, malformed JSON and invalid model schema. Assert no credential/body text appears in saved errors.

```python
import pytest
from app.services.theme_evaluation.preparation_failures import retryable_failure

@pytest.mark.parametrize("code,expected", [
    ("model_timeout", True), ("model_rate_limited", True),
    ("model_server_error", True), ("invalid_image", False),
    ("image_pixel_limit", False), ("model_auth_failed", False),
    ("model_response_incomplete", False), ("model_schema_invalid", False),
])
def test_retry_policy(code, expected):
    assert retryable_failure(code) is expected
```

- [ ] Run focused tests to confirm missing diagnostic separation.
- [ ] Preserve validated image bytes before a model call. Map failures separately at download, byte validation, request transport, JSON parsing, and `ImageObservation` validation boundaries. Keep existing 10 MiB, 20-million-pixel and static-format limits. A processing failure after validated bytes should retry those same bytes, not download again.
- [ ] Add run-local retry accounting keyed by image input hash and provider policy, shared across references. One eligible retry maximum; one-second backoff for transient failures. For 429, honor a valid Retry-After up to 30 seconds; defer longer waits with an explicit reason. Do not raise token budgets or repeat malformed/incomplete output without a separate diagnosed change.
- [ ] Keep output-token truncation as `model_response_incomplete` with no accepted partial JSON. Record attempt IDs and reuse successful cached outputs. A failed retry cannot erase an earlier usable same-input output. A rejected input never reaches the model.
- [ ] Preserve existing transcription/observation/uncertainty separation; it already exists. Task 7 adds per-field review rather than replacing it with a new vision architecture. Neither absent axis values nor cropped text should be invented. Do not add crop/enlargement or OCR dependencies in this task.
- [ ] Run image/pipeline/regression suites and Kimi translation tests, since the transport is shared. Commit as `fix: retain actionable image failures and bounded retries`.

**Acceptance:** Each of the three current failures can be diagnosed on replay/retry or explicitly classified as a legacy unknown failure. Do not invent historical causes from the old generic error. The fresh run records precise causes and request counts.

## Task 7 — Publish a versioned assessment and review queue

**Repository:** StockScreenClaude. **Depends on:** Tasks 3–6. **Priority:** P1.

**Files:**
- Create `backend/app/services/theme_evaluation/evidence_assessment.py`.
- Modify `backend/app/services/theme_evaluation/preparation_review.py`, `preparation_cli.py`.
- Create `backend/tests/unit/theme_evaluation/test_evidence_assessment.py`; modify `test_review.py`, `test_preparation_cli.py`.

**New sidecar contract:** A content-addressed assessment JSON containing `schema_version=1`, `policy_version`, `bundle_id`, `preparation_id`, and entries keyed by binding/result IDs. Each entry records issue code/severity, candidate and selected result IDs, eligibility, explanation, and next action. Optional manual decisions use `accept | exclude | hold`, reviewer identity, timestamp, reason and exact input/result hashes. Sidecar hash is computed using existing canonical serialization; stored beside preparations, not injected into frozen bundles.

- [ ] Add tests that material numerical omissions appear before cosmetic image uncertainties, multiple references to one destination remain visible without inflating unique coverage, and legacy warnings remain accessible in detailed evidence.

```python
# Required invariants in integration tests using actual saved results:
assert assessment["bundle_id"] == base.name
assert assessment["preparation_id"] == preparation_id
assert selected_entry["eligible"] is False  # X fragment + failed Kimi
assert manifest.extraction == "awaiting_evidence_approval"
assert original_bundle_bytes == (base / "bundle.json").read_bytes()
```

- [ ] Run the new assessment and existing report suites before implementation.
- [ ] Write `START_HERE.md`, `translation-review.md/.csv`, `article-review.csv`, `image-review.csv` and assessment JSON through the existing CLI/report workflow. Replace the pilot's ignored one-off report scripts with this supported path. Expose both legacy preparation status and new assessment severity; do not silently redefine `success`.
- [ ] Make image review rows address specific uncertainty/field claims. Cosmetic uncertainty may be informational; unreadable investment numbers remain review or hold. Do not automatically downgrade free-form model uncertainty by keyword alone. Unknown significance starts as review and can be annotated by the reviewer. Model-reported certainty never equals human approval.
- [ ] Validate manual annotations before any writes: correct bundle/preparation, existing result and matching hashes, nonempty reason for exclusion/accepting a flagged item. Changed evidence invalidates the previous decision. A failed annotation import must leave the prior packet intact.
- [ ] Report distinct selected posts, reference records, distinct destinations, image inputs and outputs with separate denominators. Show incomplete/unknown originals and allow reviewers to exclude particular unsupported claims or whole evidence items with a reason. Exclusion must not remove those items from coverage reporting.
- [ ] Keep all extraction flags unchanged regardless of assessment counts. User approval is a later explicit action; this plan does not build an auto-approval path.
- [ ] Run assessment/review/store/CLI suites; validate Markdown local links. Commit as `feat: generate prioritized evidence assessments for review`.

**Acceptance:** The first page clearly separates material holds, recoverable gaps, informational notes and exclusions. Every decision is reproducible from the referenced result and policy. An engineer can regenerate the packet without the ignored pilot scripts.

## Task 8 — Validate languages and reduce unnecessary preparation calls

**Repository:** StockScreenClaude. **Depends on:** Task 3. **Priority:** P2.

**Files:**
- Modify `backend/app/services/theme_evaluation/multilingual_preparation.py`, `preparation_pipeline.py`, `article_recovery.py` if article language provenance is added.
- Add `backend/tests/unit/theme_evaluation/fixtures/language_challenge.json`.
- Modify `test_multilingual_preparation.py`, `test_kimi_translation.py`, `test_preparation_store.py`.
- Add `docs/theme_evaluation/language_challenge.md` in the preparation worktree.

**Contract:** Existing `prepare_text` and exact concatenation of original segments remain supported. If behavior changes require new fields, use an opt-in preparation policy, omit unset additions from legacy serialization, and include the new policy in cache keys. Legacy results keep old warning/status semantics. The separate Task 3 assessment is preferred over changing old models.

- [ ] Add 20 explicitly labeled challenge fixtures: four Korean, four Japanese, four Chinese, four mixed-language, and four English/nonlinguistic. Include compound quantities, negative percentages, currencies, kana plus kanji, company names, ticker/product digits, paragraph boundaries, slang, and URL/emoji-only input. Synthetic fixtures must be marked synthetic, not attributed to a real post.
- [ ] Include these sentinel tests:

```python
from app.services.theme_evaluation.multilingual_preparation import segment_text

def test_original_bytes_survive_segmentation():
    original = "売上高は1.9兆円。\r\n\r\n매출 718억원.\n\n$NVDA 전망 유지"
    assert "".join(segment_text(original)) == original
```

- [ ] Test that valid English article metadata avoids a translation call, contradictory script metadata triggers review, and Han-only text without provenance is not confidently assigned Chinese. Do not manufacture language metadata to bypass validation.
- [ ] Test URL/emoji-only content is retained without a model call under the new opt-in preparation policy. Keep mixed English/foreign language prose translatable. Original-only content must have explicit nonlinguistic provenance, not be falsely labeled English to satisfy `identity` validation.
- [ ] Run language and cache suites to establish failures. Implement minimal policy-aware language handling. Preserve article language as observed metadata when present; do not infer article language from its linking post.
- [ ] Pack adjacent short paragraphs up to the existing 4,000-character limit while preserving every separator in stored original segments. Keep long-paragraph splitting bounded. Version the preparation request/cache policy so old and new segmentation never share a cache entry. A translation failure affects only its exact stored segment.
- [ ] For Kimi live validation, process the fixed challenge set once and manually compare against reviewed expected meaning and quantity facts. Model agreement is not the ground truth. Record per-language material error counts and uncertainty, not a single “multilingual supported” claim. Optional real Japanese examples belong to the challenge set and carry their source/time separately.
- [ ] Run language/translation/import/store regression suites. Commit as `fix: preserve language provenance and bound translation work`.

**Acceptance:** Japanese is explicitly exercised; English and nonlinguistic cases avoid unnecessary calls; source bytes survive segmentation; no stale cache reuse or silent legacy status migration.

## Task 9 — Verify both packages and regenerate evidence for review

**Repositories:** Both. **Depends on:** Completed applicable tasks above. **Priority:** P1 release gate.

**Files:**
- Reader: update `skills/xui-reader/SKILL.md`, `skills/xui-reader/references/post-content.md` if exported behavior changes; add only sanitized fixtures to tests.
- Preparation: update `docs/theme_evaluation/reader_handoff.md`, `pilot_runbook.md`; create `docs/theme_evaluation/gap_remediation_validation.md`.
- New raw/prepared artifacts stay under a new ignored run directory, never overwrite the September 9 packet.

- [ ] Run reader focused gates, then the full offline suite:

```bash
./.venv/bin/python -m pytest
./.venv/bin/python -m ruff check src tests
```

- [ ] Run preparation gates from the evaluation worktree. Use its configured interpreter, or the existing main-checkout environment locally:

```bash
DATABASE_URL=sqlite:// STOCKSCANNER_TEST_ALLOW_SQLITE=1 STOCKSCANNER_TEST_ALLOW_POSTGRES=0 STOCKSCANNER_TEST_USE_DATABASE_URL=0 PYTHONPATH=backend /Users/admin/StockScreenClaude/backend/venv/bin/python -m pytest backend/tests/unit/theme_evaluation backend/tests/unit/test_theme_identity_normalization.py backend/tests/unit/test_theme_source_quality_weighting.py backend/tests/unit/test_theme_lifecycle_policies.py -q
/Users/admin/StockScreenClaude/backend/venv/bin/ruff check backend/app/services/theme_evaluation backend/tests/unit/theme_evaluation
git diff --check
```

- [ ] Replay saved raw captures first, without recollecting. Compare the same post/input IDs: selected translation, material errors, false flags, article body disposition, image outputs, eligible evidence and model-call count. Raw JSON replay tests downstream behavior; DOM fixture replay tests reader behavior. Neither proves the current live DOM is fixed.
- [ ] Re-read the Samsung post and relevant failed translation cases live using the updated reader. Save public post-owned diagnostic evidence if still failing. Do not suppress failures merely to reach a target capture rate.
- [ ] Recollect both required lists with 50 selected posts each, using current reader skill instructions:
  - `https://x.com/i/lists/1986290701492232693`
  - `https://x.com/i/lists/1522014550211457024`
- [ ] Process only required stages with Kimi/public HTTP, bounded targeted X follow-ups and approved browser imports if available. Fresh sampling may change counts; report overlap with the fixed replay rather than claiming a before/after accuracy improvement on different posts.
- [ ] Render the supported review packet. Verify every bundle/result/asset hash, selected translation decision, attachment binding, local report link and prior artifact immutability. Inspect all material flagged translations and all investment-relevant image numbers used by the proposed usable subset; do not infer review coverage from nine “success” statuses.
- [ ] Record the following comparison table in `gap_remediation_validation.md`:

| Measure | Fixed replay before | Fixed replay after | Fresh collection |
| --- | --- | --- | --- |
| Reader/preparation revisions and policies | Record | Record | Record |
| Distinct posts / selected per list | Record | Same source selection | Record |
| Structurally captured X translations | Record | Record | Record |
| Eligible X / Kimi / unresolved translations | Record | Record | Record |
| Confirmed material translation errors / reviewed translations | Record | Record | Record |
| Material versus harmless warning counts | Record | Record | Record |
| Distinct readable / complete / restricted article destinations | Record | Record | Record |
| Image outputs / inputs; precise unresolved causes | Record | Record | Record |
| Original completeness: complete / incomplete / unknown | Record | Record | Record |
| Targeted page loads, model calls, cache hits, retries, elapsed time | Record if measured | Record | Record |

- [ ] Do not invent unmeasured baseline latency or cost; label those unavailable. Review the new report with the user. Keep extraction `awaiting_evidence_approval` and record remaining recoverable/irrecoverable gaps explicitly.
- [ ] Complete repository-specific commit/handoff workflow within the authorized task. A reader-only completion must identify the downstream tasks still required; it must not state the whole evidence pipeline is fixed.

**Acceptance:** All deterministic regressions pass, live reader behavior is documented separately from replay, legacy artifacts remain verifiable, and a new Markdown/CSV packet is ready for human evidence approval.

## Implementation boundaries and deliberate exclusions

- No attempt to guarantee complete translations through length or number checks alone. Manual review covers semantic uncertainty, including company identity and direction of claims.
- No unbounded collection of replies/threads, automatic paywall workaround, wholesale publisher scraping framework, image enhancement pipeline, new OCR engine or replacement language model.
- No requirement to recover 100% of inaccessible evidence. Explicit exclusions and visible denominators are valid outcomes.
- Existing transcription/observation separation, source hashing, Kimi fallback for missing X captures and immutable storage are foundations to retain, not features to rebuild.
- Article browser recovery can be a subsequent small approved task; none of the offline fixes depend on granting a new tool permission now.

## Engineer handoff checklist

- [ ] Reader captures stabilize within existing time budgets and retain post ownership.
- [ ] Completeness is conservative and bound to the chosen original text.
- [ ] Samsung fragment is rejected for use; quantities and correct conversions are covered by regression tests.
- [ ] Failed Kimi fallback cannot expose rejected X text as eligible.
- [ ] Articles retain clean text, destination identity, access status and all reference links.
- [ ] Image failures have safe specific codes, bounded retries and immutable attempts.
- [ ] Assessment severity is separate from legacy technical status; old artifact hashes/statuses remain stable.
- [ ] Japanese/mixed-language challenge results and preparation costs are measured separately from list coverage.
- [ ] Supported report generation replaces one-off scripts; decisions bind to exact result hashes.
- [ ] Fresh evidence is presented for review; no extractions or labels have been generated.

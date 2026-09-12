# Evidence Preparation Implementation Plan

> Execute inline with test-first checks. Reader work is excluded. Preserve the user's evidence-review gate.

**Goal:** Recover article bodies, prepare multilingual text and interpret attached images in versioned artifacts without changing the frozen sample.

**Architecture:** Standalone preparation modules alongside the existing evidence package. A content-addressed result store binds each result to a bundle and source record. Public HTTP, HTML parsing and model completion are narrow injected boundaries. Reuse installed httpx/BeautifulSoup/Pillow and existing provider clients where suitable.

**Spec:** [Evidence preparation design](../specs/2026-09-08-evidence-preparation-design.md).

**Tool approval status:** The user approved OpenCode Go model `kimi-k2.6` for image transcription and chart interpretation on 2026-09-08. Implement this adapter with separate transcription and observation fields; defer PaddleOCR and its model downloads. HTTPX and BeautifulSoup were approved on 2026-09-08. The user subsequently selected Kimi K2.6 for translation; the configured Go connection is used. The rendered-browser fallback remains pending approval. Kimi does not require another model-selection approval. Preserve the existing sequence: mocked/offline checks during implementation, then live image validation and sample rebuilding after both repositories are updated. Model approval does not waive evidence review before theme extraction. Implementation now includes offline-tested article recovery, language preparation and attributed translation imports, the approved Go image adapter, immutable storage, CLI and review reports. Live corpus processing remains deferred.

## Files and steps

1. `preparation_records.py`, `preparation_store.py`, `test_preparation_store.py`: strict input/result contracts, raw asset hashes and immutable result files. Prove tampering and stale source bindings fail; cache success only. Implement canonical JSON and atomic write/rename using the existing hashing conventions. Keep original Bundle unchanged.
2. `article_recovery.py`, `public_fetch.py`, `test_article_recovery.py`: fetch public URLs with per-redirect checks and byte/time limits; parse articleBody/semantic body, reject access interstitials, return partial when completeness is unknown. Support exact-reference fallback imports. Test real HTML fixtures and injected transport responses, never live sites.
3. `multilingual_preparation.py`, `test_multilingual_preparation.py`: paragraph-preserving segmentation; Hangul/kana/Han and unknown/mixed language detection; explicit translation adapter; validate aligned output and warn on numerical changes. Prove every input character is preserved and a failed chunk cannot become a full translation.
4. `image_preparation.py`, `preparation_models.py`, `test_image_preparation.py`: validate images, preserve content hashes, request separate transcription/observation fields, validate model output, and cache using image hash/model/policy. Test corrupt images, capability rejection, repeated images and malformed provider results.
5. `preparation_cli.py`, `preparation_review.py`, `backend/scripts/prepare_theme_evidence.py`, `test_preparation_cli.py`: validate reader-handoff mappings against a base bundle; explicit article/text/image stages, limits, fallback import and offline review. Report separate states and lineage; do not expose a theme-extraction command.
6. Update `docs/theme_evaluation/pilot_runbook.md` and add `docs/theme_evaluation/reader_handoff.md`. Run new tests and existing evidence/theme regressions; request a bounded code review and fix material findings. Do not run preparation against the real pilot or modify reader code.

## Contracts

Input mapping JSON: `bundle_id`, `documents` (document ID -> original text SHA, optional language, image URLs), `references` (reference ID -> destination URL). Unknown IDs/hash mismatches fail before any network/model call. Empty mappings use known bundle metadata; short image links remain unresolved.

Stage result: strict `stage`, source identity/hash, input signature, status (`success`, `partial`, `unavailable`, `needs_review`), payload, warnings, source URL, retrieved/generated times and model/policy provenance. Payloads are validated by the responsible stage. A successful image result requires valid transcription/observation fields; a successful translation requires all source segments. A parsed article with uncertain completeness stays partial.

Stored preparation manifest (v2): base bundle ID, handoff mapping, immutable attempt bindings, and explicit current slot-to-binding IDs. See the follow-up ledger below for the reviewed contract changes. Review verifies result IDs and input bindings before displaying evidence. Repeated stage results can share processing across parents; parent links live in the manifest.

## Verification

Use the main checkout's backend virtual environment with `DATABASE_URL=sqlite://`, `STOCKSCANNER_TEST_ALLOW_SQLITE=1`, `STOCKSCANNER_TEST_ALLOW_POSTGRES=0`, `STOCKSCANNER_TEST_USE_DATABASE_URL=0` and `PYTHONPATH=.`. Write tests before stage implementations, confirm expected failure, then run the affected tests. At completion run the full evidence suite and the existing identity/source-quality/lifecycle tests. Verify the original pilot's manifest hash without writing it.

## Implementation decisions and verification ledger

- Existing worktree verified: `feat/theme-detection-evaluation`. Main checkout and reader repository untouched.
- Kept Bundle v1 unchanged. Added separate source-bound preparation artifacts and asset storage.
- Article/model calls remain explicit. Translation uses an injected provider boundary or offline attributed imports while its live service selection is pending. This avoids silently selecting an unapproved translator.
- Script checks preserve supplied languages; Han-only/unknown Latin scripts stay ambiguous. Identity translation requires compatible metadata.
- Browser recovery requires an explicit capture time and exact destination/reference match; automatic HTML text stays partial until reviewed.
- Image implementation delegated as one bounded task; separate review covered storage, fetches, language and report integration.
- Review fixes: cookie isolation on IP-pinned redirects; reject encoded responses before decompression; flag unchanged foreign translations and conflicting Han/English metadata; exclude superseded failures from current queues; require imported capture times; idempotent translation imports; reject provider-reported truncated image responses. Each fix has a regression test.
- No installs, paid calls, real-pilot preparation or reader changes during implementation.

Initial implementation validation before the strict maintainability review: 119 tests passed (evidence suite plus theme identity, source-quality and lifecycle regressions), with two pre-existing dependency deprecation warnings. Ruff passed for all added Python files. The frozen 98-document pilot bundle verified against its original content ID. Initial functional-review findings were closed, including a regression for returning to a previously cached image after an intervening image version. No live provider transport/accuracy claim is made.

## Strict maintainability review follow-up

The user requested fixes for all four structural findings. Four regression tests first reproduced the failures: an offline retry superseded an imported translation; foreign-language identity text could claim success; imports reran segmentation; unrelated corrupted assets broke cache misses.

1. **Explicit current evidence:** preparation manifest v2 stores a current-selection map alongside unique attempt history. `PreparationState` owns transitions; execution and reports consume the same selection. Unavailable retries preserve useful evidence for unchanged/uncaptured input. Captured input changes supersede children at that source locator. Tests cover history reordering, cached A→B→A, changed images, shared image results across locators and exclusion of old versions from translation work.
2. **Typed results:** `preparation_results.py` defines discriminated article/text/image outcomes and stage-specific requests. Content models own consistency rules. Status/warnings derive from content; foreign-language identity is invalid at construction. Removed the payload-reparsing validation module and dictionary payload access from consumers.
3. **Direct imports:** one finalizer binds live/imported translations to the exact original segments. Imports retain language metadata, missing slots and generation provenance without redetection, resegmentation or a fake translator. Tests replace both language detection and segmentation with failing functions during a multi-segment import.
4. **Indexed cache:** immutable request-keyed index entries select successful results without scanning unrelated assets. Selected artifacts still fail on corruption. Explicit verification audits all assets/results/cache entries and repairs missing indexes after validation, covering interruption between result write and indexing.

Only preparation format v1 is retired; the frozen acquisition Bundle v1 remains unchanged. No real preparation artifacts had been generated, so no migration or sample rebuild is required. No new dependencies, live service calls or reader changes are part of this fix.

Follow-up validation: **130 tests passed**, covering the full evidence suite and existing theme identity, source-quality and lifecycle regressions. The two dependency warnings are pre-existing. Ruff passed on every changed Python file; the broader package check also reported existing lint issues in unchanged acquisition files, which were left outside this refactor. `git diff --check` passed. The original 98-document pilot again verified against content ID `96d66ae1038bea1107ac17aeab3fabbbabd383fa830014b040ed892031789844`.

## Live Kimi validation after reader update

On 2026-09-08 the user supplied the updated reader checkout and requested live Kimi validation. Four synthetic images passed through the approved Go route after a live HTTP 400 exposed the missing session header. Added a per-client session UUID and honest client identifier, with a failing-then-passing regression test. Korean/Japanese literal transcription and financial units/signs passed; all eight plotted quarter values were associated correctly. Damaged text was flagged without invented figures. Two auxiliary counting mistakes prevent any blanket accuracy claim. See [validation report](../../theme_evaluation/kimi_validation_2026-09-08.md) for timings, checks, source artifacts and limits. Reader contract inspection used clean commit `057faae`; no live X reads, sample rebuild or extractions were performed in this validation.

## Kimi translation implementation and validation

User selected Kimi on 2026-09-08. Added an explicit translation-call switch and shared bounded Go transport with the existing image adapter. Test-first checks cover request data, response validation, missing configuration, quantity loss/duplication, truncated output, and existing numerical warnings. Three 20-passage live rounds exposed name ambiguity and then tenfold Korean magnitude errors. The final `translation-v3` policy protects scale quantities, restores source notation, and retains all 10 matched amounts across the final 20 translations. Two contextual issues remain for evidence review; see [translation validation](../../theme_evaluation/kimi_translation_validation_2026-09-08.md). No corpus was rebuilt or extracted.

Final translation verification: **158 tests passed** across the evidence suite and existing identity/source-quality/lifecycle regressions, with two pre-existing dependency warnings. Ruff passed on all changed Python files, and `git diff --check` passed. Live final-round median was 2.409 seconds per short passage; 20/20 returned translations and 10/10 protected quantities were retained. Automatic success is not a semantic approval; two contextual issues remain explicitly documented.

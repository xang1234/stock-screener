# Task 4 report — translation selection and provenance

## Result

Implemented adequacy-gated selection between captured X and Kimi translation
results. Materially incomplete X captures trigger one full-source fallback attempt;
`review` captures remain pending without an automatic second opinion. Both attempts
remain in preparation history, while an immutable selection sidecar is authoritative
for eligibility. A retained legacy `current` X result therefore cannot become eligible
after a failed Kimi attempt.

Selection sidecars use canonical JSON and content-addressed storage at
`selection-decisions/<sha256>.json`. Saving and loading recomputes the decision from
the exact bundle document, preparation bindings, input hash, candidate result IDs,
candidate roles, and `translation-quality-v1` policy. Sidecars are discovered by
verified preparation ID without a mutable latest pointer. A non-text preparation may
carry prior decisions only by resealing a new sidecar after all inputs and bindings are
revalidated against the new preparation ID.

## Integration API

- `select_translation(original, language, x_result, kimi_result)` returns frozen
  `TranslationSelection` with `selected_provider`, `eligible`, `assessment`, exact
  source hash, selected candidate/result, and the issues that explain the decision.
- `selection_record(document_id, x_result_id, kimi_result_id, selection)` binds the
  pure decision to stored candidate IDs.
- `save_translation_selection(base, store, preparation_id, decisions) -> str` validates
  and writes the content-addressed sidecar, returning its ID.
- `load_translation_selection(base, store, preparation_id, sidecar_id)` loads one exact
  sidecar and rejects a different preparation ID.
- `selection_for_preparation(base, store, preparation_id) -> (sidecar_id, sidecar)`
  finds the unique verified sidecar for callers that only receive the unchanged
  `prepare(...) -> str` return value.

## Verification

TDD red was observed first: the focused suites failed collection because the new
selection modules did not exist. After implementation, this required command passed:

```text
DATABASE_URL=sqlite:// STOCKSCANNER_TEST_ALLOW_SQLITE=1 \
STOCKSCANNER_TEST_ALLOW_POSTGRES=0 STOCKSCANNER_TEST_USE_DATABASE_URL=0 \
PYTHONPATH=backend /Users/admin/StockScreenClaude/backend/venv/bin/python -m pytest -q \
backend/tests/unit/theme_evaluation/test_xui_translation.py \
backend/tests/unit/theme_evaluation/test_translation_selection.py \
backend/tests/unit/theme_evaluation/test_preparation_regressions.py \
backend/tests/unit/theme_evaluation/test_preparation_pipeline.py \
backend/tests/unit/theme_evaluation/test_preparation_store.py \
backend/tests/unit/theme_evaluation/test_preparation_translation_import.py
```

Result: `53 passed, 2 warnings in 4.58s`. The warnings are existing Pydantic and
pandas/pyarrow deprecation notices.

Focused Ruff validation also passed for both new modules, the pipeline integration,
and the two owned test files.

## Coverage

Tests cover the exact Samsung text and published source hash, one complete-source Kimi
call, both preserved result IDs, accepted X with zero model calls, X `review` without
fallback, failed and materially wrong Kimi candidates, missing translators, incomplete
multi-segment output, changed captures with the same original hash, English article
identity without a model call, forged eligibility/selection fields, wrong preparation
IDs, and legacy-current X retention after failed Kimi.
Sidecar validation also rejects a bound result that is not byte-for-byte equivalent
to the document's captured X derivative and rejects missing document decisions.

## Limitations and dependency

This commit depends on Task 3's `translation_quality.py` contract from the preceding
`9e30e19f` commit. Existing manual translation imports do not
create a selection sidecar; Task 7 must treat a missing sidecar as unresolved or create
a new validated assessment rather than infer eligibility from legacy `current` state.
No live model or browser calls were made.

## Commit

`fix: fall back from materially incomplete X translations` (this task commit).

## Review remediation

The three Important findings in `task-4-review.md` were addressed in a scoped
follow-up:

- Empty and whitespace-only document text now produces a stored local preparation
  candidate and an explicit pending selection instead of failing sidecar construction.
  The pure selector also classifies a call with no candidates as
  `translation_candidate_missing` with a blocker/fallback disposition.
- A later non-text stage can consume a legacy or manually imported preparation that
  has root text bindings but no selection sidecar. It preserves the missing-sidecar
  state as unresolved and does not synthesize clean eligibility from legacy `current`.
- Candidate validation now requires both the request and payload target language to
  be English before adequacy assessment. The same check runs again while a sidecar is
  persisted or loaded, so a forged French-target decision is rejected.

Regression tests first reproduced all three review failures. The six-suite verification
command documented above was rerun after the fixes with `58 passed, 2 warnings in
3.90s`. Scoped Ruff validation passed. No live calls were made.

Follow-up commit: `fix: close translation selection review gaps`.

# Task 7 — evidence assessment and preparation integration

## Result

Implemented versioned, content-addressed assessments in `assessments/<sha256>.json`, regenerated from the exact frozen bundle, verified preparation bindings/results and validated translation-selection sidecar. Entries retain result/input/source hashes, candidates, selected result, deterministic eligibility, legacy status/warnings, severity, explanation and next action. Missing selection sidecars remain unresolved. A retained legacy-current X fragment cannot become eligible after fallback failure. Missing derivative translation segments also fail closed.

Supported review output now includes `START_HERE.md`, translation/article/image review Markdown and CSV, immutable `assessment.json`, original evidence, all preparation attempts, reference relationships, linked-post follow-up manifest/review, coverage, and immutable operational run reports. Every reference remains visible, including references already satisfied by native X Article captures. Distinct selected posts, references, destination identities, image references, input byte hashes and outputs have separate counts. Incomplete/unknown originals remain explicit. Model certainty never represents human approval; unknown image uncertainty stays review until a human annotates its significance.

Manual annotations accept `accept | exclude | hold`, exact bundle/preparation/entry/result/input/source hashes, reviewer, aware timestamp, reason, claim/evidence scope, and optional image-claim significance (`cosmetic | material | unknown`). All annotations validate before packet writes. Duplicate/overlapping scopes and stale identities fail closed. Excluding whole evidence affects all its claims while retaining coverage. Accepting a flagged item never changes deterministic eligibility or extraction flags. Regenerate to a new output directory for every review revision.

## Integration API

- `prepare(..., text_policy="v1", run_summary=None) -> preparation_id` preserves the original default and return type. Explicit `text_policy="v2"` invokes the language-efficient helper and its separate cache namespace. Nonlinguistic `zxx` retains original content and appears as translation-not-needed information.
- Optional `run_summary` receives selection ID when available, immutable run ID, elapsed seconds, actual translation adapter calls, article fetch calls, shared image model/download calls, and article routes/linked-post queue when the corresponding stages run. Saved run reports live in `preparation-runs/<sha256>.json`.
- The article binding adapter uses Task 5 `recover_references`, reuses byte/parser work, preserves each original request destination and reference binding under `article-v2`, and skips already-captured native X Articles. Native article source language remains attached to its original document. Every original reference stays in the review denominator.
- One Task 6 `ImageStage` serves each run; every attempt ID is bound and model/download counters come from that shared instance.
- Browser imports require explicit capture time, final URL, match basis, matching exact text hash, optional verified body hash, and a nonempty completeness basis for a full capture.
- `build_assessment(base, store, preparation_id, annotations=[])` returns the reproducible dictionary. `save_assessment` revalidates before writing; `load_assessment` verifies its content hash, preparation identity and regenerated decision.

## Supported commands

From the preparation worktree, with the normal backend environment:

```bash
PYTHONPATH=backend backend/venv/bin/python backend/scripts/prepare_theme_evidence.py prepare \
  --bundle BASE --output-root STORE --handoff HANDOFF.json \
  --stages article text image --text-policy v2

PYTHONPATH=backend backend/venv/bin/python backend/scripts/prepare_theme_evidence.py review \
  --bundle BASE --output-root STORE --preparation PREPARATION_ID --output REVIEW_DIR

PYTHONPATH=backend backend/venv/bin/python backend/scripts/prepare_theme_evidence.py review \
  --bundle BASE --output-root STORE --preparation PREPARATION_ID \
  --annotations ANNOTATIONS.json --output NEW_REVIEW_DIR
```

The annotation file is a JSON array of exact entry-bound annotations. Existing network/model enablement flags retain their explicit meanings. The first command is offline unless enabled; no hidden provider calls were added.

## Verification

- Initial assessment red: 3 new tests failed because the assessment module was absent; 7 existing review/CLI tests passed.
- Integration red: missing run-summary argument, v2 CLI option and completeness validation produced the expected failures before implementation.
- Whole-evidence annotation/linked-manifest red reproduced the missing boundaries before implementation. A missing derivative translation segment was also reproduced as incorrectly eligible, then fixed.
- Full theme-evaluation suite: **324 passed**, two existing deprecation warnings (before the final missing-segment regression was added).
- Final focused assessment, integration, pipeline, CLI, store, review and selection suites: **57 passed**, including the new missing-segment regression. Markdown local links are validated by the packet regression.
- Scoped Ruff and `git diff --check` pass.

## Limits

No live calls, browser rendering, production database access, frozen bundle edits, or live artifact mutations were performed. Linked-post manifests expose bounded acquisition work; queued posts are explicitly unprocessed and outside the list quota until an independently captured/imported supplement is reviewed. Legacy statuses and extraction locks are unchanged. Per-run operational reports describe only that run, not accumulated historical costs. Manual notes do not provide an extraction-approval path.

FINAL: completion — implementation ready for independent integration review.

## Completeness review correction

The live packet exposed that the importer intentionally uses legacy `capture_status="partial"` for posts whose `text_complete` is unknown. Assessment policy `evidence-assessment-v2` now classifies post completeness from the authoritative `text_complete` tri-state: `False` is incomplete, `None` is unknown, and `True` is complete. Partial non-post/article captures remain incomplete/unverified. Both the issue code/explanation and coverage counters use this classification; original bytes and legacy status are preserved.

The corrected default has a new policy version. Archived `evidence-assessment-v1` sidecars still regenerate and validate under their original policy, without rewriting their bytes. New review packets use v2. A realistic xui import/seal/prepare/assessment regression first reproduced the unknown-as-incomplete failure, then passed for all three metadata states. An additional regression verifies archived v1 validation and byte preservation.

Verification after correction: **48 passed** across assessment, stage integration, pipeline, CLI and xui importer suites; scoped Ruff and diff checks pass. No saved live packets or source artifacts were changed. The controller will create new packet directories.

FINAL: completion — corrected completeness assessment ready for packet regeneration and review.

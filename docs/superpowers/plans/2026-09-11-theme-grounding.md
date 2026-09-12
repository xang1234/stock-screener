# Theme Grounding Implementation Plan

> **For agentic workers:** Use superpowers:subagent-driven-development for independent tasks and integrate in this worktree. Do not commit unrelated work.

**Goal:** Ground open theme names in attributed company context and explicitly related approved evidence.
**Architecture:** A small company-context reader and a pure evidence-packet builder feed a validated context to the existing extractor. Evaluation packets and records retain exact source provenance separately from frozen inputs.
**Tech Stack:** Python, SQLAlchemy, Pydantic, pytest, existing extraction runtime.
**Spec:** ../specs/2026-09-11-theme-grounding.md

## Global constraints

- No new model provider, no automatic remote company-profile calls.
- No curated theme catalog, ranking changes or full-article chunking.
- No rewrites of old frozen inputs/results, excluded evidence or application data.
- Current worktree contains prior uncommitted work; preserve it and do not commit.

## Task 1: Attributed company context

Files: create backend/app/services/theme_company_context.py and
backend/tests/unit/test_theme_company_context.py.
Interface: build_company_context(db, text, *, as_of=None, identity_only=False,
resolved_symbols=()) -> dict with companies and warnings. Company entries contain
symbol, name, identity_source, sector, industry, business_description, profile_source,
profile_as_of, profile_status. Values are JSON-safe; timestamps ISO strings.

- [x] Write failing tests for NBIS identity, explicit cashtags, unknown and risky bare words, and profile provenance/freshness.
- [x] Implement read-only active-universe matching; use existing cached fundamentals, no remote calls.
- [x] Verify missing/stale/unattributed data is not presented as current verified business evidence.
- [x] Run focused tests and review interfaces before integration.

## Task 2: Validated context and evidence packets

Files: create backend/app/services/theme_grounding_context.py and
backend/app/services/theme_evaluation/grounding.py plus focused tests.
Interface: GroundingContext JSON model; prepare_grounding(run, base, store,
company_contexts, as_of) -> packet; packet validation binds run/input IDs and hashes.

- [x] Test wrong-parent, excluded derivative, duplicate/context leakage and content tampering failures.
- [x] Attach only admitted evidence via exact parent/source or recorded followup edges to admitted articles.
- [x] Bound prompt data with explicit warnings, preserve original hashes and availability.
- [x] Verify old evidence/input bytes remain unchanged.

## Task 3: Extractor, runtime and review

Files: theme_extraction_service.py; evaluation extraction_records.py,
extraction_runtime.py, extraction_cli.py, extraction_review.py; focused tests.

- [x] Test real prompt delivery with a provider-boundary fake, including unrelated-image absence.
- [x] Add optional grounding_context argument; local application calls resolve company context, evaluation passes pinned context.
- [x] Add prepare-grounding/generate packet support and record actual context without changing old serialization.
- [x] Export company/evidence provenance and limitations in Markdown/CSV.
- [x] Run relevant suites, review the whole change, and fix defects.

## Task 4: Controlled validation and documentation

- [x] Freeze real company context using available local references with provenance.
- [x] Build a new packet for the approved sample without changing admission.
- [x] Run a limited NBIS/negative-control model comparison using the approved route.
- [x] Report actual successes, failures and remaining grounding limitations.

## Decisions

User authorization covers this implementation; no extra approval gate is added.
Implementation will use subagents for the independent company-context reader and
review while the primary agent builds/integrates the evidence packet.

## Outcome

Implemented and reviewed. Four-post live check: eight successes on approved Kimi
fallback; NBIS corrected to AI Infrastructure, one unsupported CPO assertion
remains. An incidental next-year/NXTT enrichment collision was fixed and all eight
raw responses replayed offline with no new model calls. Historical archives were
not rewritten. Deployment migration is intentionally not applied in this worktree.
See docs/theme_evaluation/grounded_extraction.md and the grounding-v1 checkpoint.

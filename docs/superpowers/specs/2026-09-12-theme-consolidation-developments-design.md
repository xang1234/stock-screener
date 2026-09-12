# Open theme consolidation and development tracking

Status: approved by user; implement inline.
Branch: feat/theme-consolidation-developments, based on main after PR #363.

## Outcome

Show equivalent investment themes as one coherent theme, without imposing a fixed taxonomy or losing the ability to separate them again. Group reports about an event so readers can distinguish repeated coverage from a material update. Build consolidation first, then development tracking against its stable identities.

## Existing behavior

ThemeAlias already stores alternate names and matching confidence. ThemeMergeSuggestion and the existing review UI provide candidate review and merge history. ThemeMergingService.execute_merge currently moves mentions and constituents, deletes overlapping constituents and source embeddings, and retires the source theme. That operation cannot provide lossless undo from its current history alone. ThemeMention.development stores a source-specific description but does not establish an event identity across posts.

## Approaches considered

1. Recommended: reversible equivalence groups over preserved theme identities. Current views aggregate group members, while original assignments remain available for undo and audit.
2. Extend destructive merges with complete before/after snapshots. This retains the current write model but becomes difficult to reverse safely after later ingestion, edits, and additional merges.
3. Normalize display names only. Cheapest, but leaves histories, constituent counts, and rankings fragmented.

## Slice 1: reversible equivalent-theme grouping

Keep names fully open. Existing aliases and matching logic propose candidates; semantic similarity alone cannot establish equivalence. Treat equivalent exposure, broader/narrower exposure, related exposure, and uncertain as distinct outcomes. CPO and Co-Packaged Optics are an equivalence example. Bitcoin Miners and Bitcoin Mining may be equivalent when both describe the miner equity exposure; cryptocurrency price exposure alone is insufficient. HBM and Memory stay distinct and may retain a parent/child relationship.

Introduce a focused equivalence service and persisted membership/change history. Preserve member theme IDs, original mention assignments, constituent records, and embeddings. Choose a display representative separately from membership. Reuse existing review screens, adding an equivalence action and an undo action with an impact preview. Initial semantic equivalence decisions use human review; existing similarity scores produce suggestions. Explicitly accepted aliases provide fast deterministic lookup after acceptance.

Store alias origin independently of the representative. Ingestion retains the originating identity and raw theme text; a group representative is a current view, not a replacement for source provenance. An alias cannot silently acquire a different meaning through grouping.

Route equivalence grouping through this service rather than the destructive merge path. Existing historical destructive merges remain historical: this feature does not promise to reconstruct data already deleted. Prevent automated consolidation from destructively collapsing members of the new groups.

Undo removes the recorded membership change and rebuilds affected views. Later observations remain with their originating identities. Reject an undo whose dependencies require undoing later membership changes first, with a concrete explanation. Reject cross-pipeline grouping, hierarchy conflicts, and cycles. Applying or undoing an operation twice must be safe.

Theme detail, sources, constituents, search, and ranking inputs must resolve the same group identity. Deduplicate one parent post appearing under multiple member names. Aggregate from source records rather than summing already aggregated counters. Invalidate affected caches and recalculate current derived metrics after membership changes. Preserve historical snapshots with their original grouping version.

## Slice 2: developments across posts

Add stable event records and source-bound observations separate from themes. A theme can contain many events, and an event can relate to several themes without multiplying its evidence weight.

Each observation retains the parent post, cited evidence, actors, action, object/product or project, relevant time, quantities when stated, and an attributed status such as rumored, announced, confirmed, denied, delayed, or cancelled. Unknown fields stay unknown. A social post asserting confirmation is recorded as that source's assertion; it is not automatically independently verified.

Use exact source/event identifiers and normalized structured facts for candidate matching. A shared ticker or similar theme name is insufficient. Ambiguous candidates remain separate and reviewable. Reuse the application's configured extraction route for structured observations; do not introduce another model service. Provider failure preserves existing extraction and leaves event processing retryable.

Compare a new observation with the event's existing observations. Classify it as repeated coverage, additional detail, material update, contradiction, or uncertain. A rumor followed by a company announcement may join one event while retaining both observations and their chronology. Different orders involving the same company stay separate. Image and article evidence remain attached to their parent post and do not count as independent reports.

Persist both source publication time and system availability time. Reprocessing the same source revision cannot create another event contribution. Corrections supersede the affected observation without erasing its audit history. Theme regrouping changes event presentation, not event IDs or source provenance.

Expose a development timeline on theme detail with evidence links and the change classification. Keep current ranking weights unchanged in this slice; expose distinct-event and material-update counts for subsequent novelty/speed evaluation.

## Validation and rollout

Use additive migrations and preserve existing data. Test equivalence versus hierarchy, duplicate parent counting, pipeline isolation, repeated apply/undo, new ingestion after grouping, dependent undo, and cache refresh. Test repeated and paraphrased reports, rumor-to-announcement updates, conflicting claims, separate orders, late evidence, missing dates, multilingual prepared text, model failure, and reprocessing idempotency.

Include API and UI tests for group review/undo and development timelines. Historical backfill is an explicit bounded operation with a preview; normal deployment must not silently regroup all themes or make unbounded model calls. Complete the existing evidence-review checkpoint before generating new benchmark extractions. Full article chunking and ranking-weight tuning remain outside this work.

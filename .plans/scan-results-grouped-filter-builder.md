# Guided AND/OR Groups for Scan-Result Filtering

> **Tracking issue:** `stockscreenclaude-al4`
> **Status:** Approved for planning; implementation not started
> **Product decision:** Add grouped result filtering as a guided strategy builder, not as a global AND/OR toggle.

## 1. Outcome

Enable users to keep non-negotiable quality and liquidity requirements while accepting one or more alternative setup paths.

The primary user model is:

```text
Always require: A AND B AND C

Then match ANY setup group:
  Group 1: D AND E
  Group 2: F AND G
  Group 3: H AND I
```

The effective expression is:

```text
A AND B AND C AND ((D AND E) OR (F AND G) OR (H AND I))
```

This should improve discovery without weakening the user's quality floor, and each result must explain which named group matched.

## 2. Product principles

1. **Use trading language, not boolean syntax.** Present “Always require,” “Match any setup group,” and “Match all/any conditions.” Do not expose parentheses or a formula bar.
2. **Preserve the simple path.** Existing flat filters remain easy to use and continue to mean “match all.” Grouped rules are an advanced capability users enter deliberately.
3. **Make OR results explainable.** A user must be able to see why each stock qualified.
4. **Keep scan execution and result refinement distinct.** The top-level screener strategy controls determine how a scan is run; the grouped builder refines an already-produced result set.
5. **Do not trade correctness for responsiveness.** Failed or pending queries must never display old rows as if they matched newly applied rules.
6. **Keep live and static behavior identical.** A saved filter must mean the same thing in live results, the static site, chart navigation, and CSV export.

## 3. Current-state constraints

### 3.1 Frontend

- `ScanPageContainer` stores one flat filter object and derives GET query parameters through `buildFilterParams`.
- Filter changes are debounced by 300 ms and immediately become React Query keys.
- `placeholderData` keeps previous rows visible during filter changes, while the result heading derives “(filtered)” from current local state. This is the root of the misleading stale-row experience when a request fails.
- `FilterPanelContainer` is a dense grid of Fundamental, Technical, and Rating/Score controls. Repeating that entire grid per group would be unusable.
- The active-filter summary and `ScanResultsSection.isFiltered` each maintain separate hand-written notions of what counts as an active filter.
- Presets save the current flat filter object and sort settings verbatim.

### 3.2 Backend

- `FilterSpec` is a flat collection of range, categorical, boolean, and text filters.
- Both SQL query adapters apply every filter sequentially, which gives implicit AND semantics.
- The live result, symbol-navigation, and export endpoints share flat query-parameter parsing.
- Feature-store and legacy scan-result reads use different field resolvers and storage layouts but must compile the same expression semantics.

### 3.3 Static site

- `filterStaticScanRows` contains a separate JavaScript evaluator with implicit AND semantics.
- Preset counts and result filtering happen after full client-side hydration.
- Boolean missing-value behavior is not fully aligned with the backend: the static evaluator currently coerces a missing boolean to false, while backend JSON boolean filters require a non-null value. OR groups would amplify this mismatch.

### 3.4 Performance prerequisites

Grouped OR predicates can be more expensive than today's flat AND filters, particularly across JSON-backed feature-store fields. The following existing issues are prerequisites for enabling the UI in production:

- `stockscreenclaude-b2l`: filtered result queries must no longer time out.
- `stockscreenclaude-55r`: failed or aborted result queries must not relabel stale rows.

The grouped-filter backend may be developed behind a feature flag before those issues close, but the UI must remain disabled in production until representative AND and OR queries satisfy the performance gates in section 12.

## 4. Supported expression model

Use a bounded, non-recursive structure for the first release.

### 4.1 Components

- **Required group**
  - Always present.
  - Conditions use ALL semantics.
  - Empty means no global guardrails.
- **Setup groups**
  - Zero to eight named groups.
  - Each group has a stable ID, user-facing name, and internal `all` or `any` match mode.
  - Empty groups are invalid and cannot be applied.
- **Group join**
  - `any` by default: at least one setup group must match.
  - `all` is available for advanced users.
  - When there are no setup groups, only the required group applies.

### 4.2 Effective predicate

```text
required_predicate
AND
(
  TRUE when there are no setup groups
  otherwise JOIN(group_join, setup_group_predicates)
)
```

Each setup-group predicate is `ALL(conditions)` or `ANY(conditions)` according to its match mode.

### 4.3 Limits

- Maximum 8 setup groups.
- Maximum 20 conditions per group.
- Maximum 100 conditions across the complete expression.
- Group name maximum 60 characters.
- No user-authored raw SQL, formulas, or arbitrary recursive nesting.
- Field names and filter types must come from a server-owned allowlist.

These limits protect usability, query planning, request size, and cache cardinality.

## 5. UX specification

### 5.1 Entry point

Keep the existing compact filter header and add a secondary action:

```text
Filters  3 required  2 setup groups   [Preset]  [Build grouped rules]  [Reset]
```

When no setup groups exist, the current filter grid remains the simple editor for required conditions.

### 5.2 Group-builder modal

Use a large modal or right-side drawer rather than expanding the already-dense inline grid.

```text
+-------------------------------------------------------------------+
| Build result rules                                      [Close]   |
+-------------------------+-----------------------------------------+
| Search data points      | ALWAYS REQUIRE                         |
| [____________________]  | Price      at least       20            |
|                         | ADV        at least       10M           |
| Fundamental             | RS Rating  at least       70            |
| Technical               |                                         |
| Setup                   | THEN MATCH [ANY v] SETUP GROUP          |
| Rating / Score          |                                         |
|                         | VCP Breakout               Match [ALL]  |
| Price                   | VCP Ready  is             Yes           |
| RS Rating               | Pivot Dist between        -3 and 5      |
| VCP Ready               |                                         |
| Pocket Pivot            | Momentum Trigger           Match [ALL]  |
| ...                     | Pocket Pivot is           Yes           |
|                         | RS Blue Dot is            Yes           |
+-------------------------+-----------------------------------------+
| [Reset draft] [Save as preset]      90 matches  [Cancel] [Apply] |
+-------------------------------------------------------------------+
```

Required behavior:

- Searchable condition catalogue with category navigation.
- Natural-language operator labels appropriate to the field type.
- Add a condition by click/keyboard; drag-and-drop is optional enhancement, never the only interaction.
- Move, duplicate, rename, reorder, disable, and delete groups.
- Move conditions between the required lane and setup groups.
- Inline validation for empty groups, incomplete ranges, unsupported fields, and exceeded limits.
- Explicit Apply and Cancel. Editing changes a draft, not the currently successful query.
- Save the complete expression and sort as a preset.

### 5.3 Applied-state summary

After application, collapse the expression into a readable summary:

```text
90 of 3,747 stocks
Required: Price ≥ $20 · ADV ≥ $10M · RS ≥ 70
Matched setup: any of VCP Breakout, Momentum Trigger, Growth Leader
```

Do not derive this state from a hand-written list of known frontend filter keys. Derive it from the canonical applied expression.

### 5.4 Pending and error states

Maintain three separate states:

- `draftExpression`: edits inside the builder.
- `requestedExpression`: the expression currently being fetched.
- `appliedExpression`: the last expression whose result query succeeded.

On Apply:

1. Validate the draft.
2. Start the query and show an “Updating results…” overlay over the prior table.
3. Keep the prior applied summary attached to the visible prior rows.
4. Promote the requested expression to applied only after success.
5. On failure, leave prior rows and summary unchanged, keep the draft available, and show a retryable error.

### 5.5 Result explainability

Each hydrated result should include the IDs and names of matched setup groups.

- Display a compact “Matched: VCP Breakout” badge or tooltip near the symbol/company area.
- If multiple groups matched, show the first compact badge plus `+N`.
- The stock detail/chart context should expose the full matching group list.
- CSV export should include a `matched_groups` column.

Avoid adding another permanently wide table column; the results table is already extremely dense.

## 6. Request contract

Introduce a versioned body contract while preserving existing GET endpoints for flat filters and old clients.

### 6.1 Example

```json
{
  "expression_version": 1,
  "required": {
    "id": "required",
    "name": "Always require",
    "match": "all",
    "conditions": [
      { "kind": "range", "field": "price", "min": 20, "max": null },
      { "kind": "range", "field": "adv_usd", "min": 10000000, "max": null },
      { "kind": "range", "field": "rs_rating", "min": 70, "max": null }
    ]
  },
  "group_join": "any",
  "groups": [
    {
      "id": "vcp-breakout",
      "name": "VCP Breakout",
      "match": "all",
      "conditions": [
        { "kind": "boolean", "field": "vcp_ready_for_breakout", "value": true },
        { "kind": "range", "field": "se_distance_to_pivot_pct", "min": -3, "max": 5 }
      ]
    },
    {
      "id": "momentum-trigger",
      "name": "Momentum Trigger",
      "match": "all",
      "conditions": [
        { "kind": "boolean", "field": "pocket_pivot", "value": true },
        { "kind": "boolean", "field": "rs_line_blue_dot_recent", "value": true }
      ]
    }
  ],
  "sort": { "field": "composite_score", "order": "desc" },
  "page": { "number": 1, "size": 50 },
  "options": { "detail_level": "table", "include_sparklines": true }
}
```

### 6.2 Endpoints

Add POST equivalents using one shared Pydantic request model:

- `POST /v1/scans/{scan_id}/results/query`
- `POST /v1/scans/{scan_id}/symbols/query`
- `POST /v1/scans/{scan_id}/export/query`

The existing GET endpoints remain supported and map flat query parameters to a required-only expression internally. Static bootstrap generation may continue using the old path until its exporter is migrated.

### 6.3 Response additions

For results:

```json
{
  "scan_id": "...",
  "total": 90,
  "unfiltered_total": 3747,
  "query_fingerprint": "...",
  "results": [
    {
      "symbol": "NVDA",
      "matched_groups": [
        { "id": "momentum-trigger", "name": "Momentum Trigger" }
      ]
    }
  ]
}
```

`query_fingerprint` must be a canonical hash of the expression, excluding page number. It supports cache keys and telemetry without logging user-entered names or numeric values.

## 7. Domain and query architecture

### 7.1 Domain types

Add versioned domain types under `backend/app/domain/common/query.py` or a focused sibling module:

- `MatchOperator`: `ALL`, `ANY`
- `FilterCondition`: typed union of current range, categorical, boolean, and text leaves
- `FilterGroup`: stable ID, name, match operator, tuple of conditions
- `FilterExpression`: required group, setup-group join, setup groups, version

Keep `FilterSpec` as a compatibility builder. Add a conversion from `FilterSpec` to a required-only `FilterExpression`.

### 7.2 SQL compilation

Refactor each adapter's leaf helpers to return SQLAlchemy predicate expressions rather than mutating a query directly:

```text
condition -> SQL predicate
group     -> and_(...) or or_(...)
root      -> and_(required, joined setup groups)
```

Use a shared expression traversal with backend-specific field resolvers:

- Legacy resolver maps domain fields to `ScanResult` columns or details JSON paths.
- Feature-store resolver maps domain fields to `StockFeatureDaily` columns, joined fields, or details JSON paths.

Unknown fields must fail validation; they must not silently disappear from the query.

### 7.3 Matching-group evaluation

After the page is selected and hydrated, evaluate the same expression against the page's domain result objects to determine `matched_groups`.

- Do not add group CASE expressions to the unbounded count query.
- Keep this evaluation bounded to the hydrated page.
- Add pure evaluator tests using the same truth-table fixtures as the SQL compilers.

### 7.4 Symbols and exports

- Chart navigation must use the exact applied expression and sort order.
- CSV export must use the exact applied expression, not a flattened approximation.
- Exported rows include matched group names.
- All paths must use the same validation and domain expression.

## 8. Frontend state and components

### 8.1 Canonical model

Introduce a canonical `FilterExpression` frontend model using backend field names. Do not store one complete flat-filter object per group.

Provide adapters:

- `legacyFiltersToExpression(filters)`
- `expressionToRequiredLegacyFilters(expression)` for the existing simple grid
- `expressionToRequest(expression, sort, page)`
- `canonicalizeExpression(expression)` for React Query keys and dirty checks
- `summarizeExpression(expression)` for chips and readable text

The simple filter grid edits only the required group. The grouped builder edits the full expression.

### 8.2 New components

Recommended structure:

```text
frontend/src/features/scan/components/filterBuilder/
  GroupedFilterDialog.jsx
  FilterCatalog.jsx
  RequiredConditionsCard.jsx
  SetupGroupCard.jsx
  ConditionRow.jsx
  ExpressionSummary.jsx
  MatchedGroupBadges.jsx
  fieldCatalog.js
  expressionModel.js
  expressionValidation.js
```

Keep field metadata—label, category, condition kind, units, bounds, operators—in one catalogue used by both the simple controls and grouped builder where practical.

### 8.3 Query lifecycle

- Grouped expressions use the POST result-query endpoint.
- Required-only expressions may use the POST endpoint too after migration; the old GET remains compatibility-only.
- Forward React Query's `AbortSignal` into Axios.
- Use `isFetching`, `isError`, and the draft/requested/applied state machine instead of relying only on `isLoading`.
- React Query keys use the canonical expression fingerprint, sort, scan ID, and page.

## 9. Preset migration

### 9.1 Stored shape

Persist a versioned preset payload inside the existing JSON text field:

```json
{
  "schema_version": 2,
  "expression": { "expression_version": 1, "required": {}, "group_join": "any", "groups": [] }
}
```

Sort remains in the existing `sort_by` and `sort_order` columns for compatibility unless a later preset schema consolidation is justified.

### 9.2 Backward compatibility

- A preset without `schema_version` is v1.
- Convert v1 flat filters into the required group with no setup groups at read time.
- Do not rewrite all old presets eagerly.
- When a v1 preset is edited and saved, persist it as v2.
- Dirty-state comparison uses canonical expressions, not raw `JSON.stringify` object ordering.
- Static preset export must emit v2 expressions while continuing to read old definitions during the transition.

## 10. Live/static semantic parity

Define one portable set of JSON truth-table fixtures covering:

- Required-only ALL.
- Setup groups joined by ANY and ALL.
- Group-internal ANY and ALL.
- Empty required group.
- Numeric min-only, max-only, and bounded ranges.
- Categorical include and exclude.
- Boolean true and false.
- Text matching.
- Missing/null values for every condition type.
- Rows matching zero, one, and multiple setup groups.

Consume the fixtures in:

- Python pure evaluator tests.
- Legacy SQL repository tests.
- Feature-store SQL repository tests.
- JavaScript static evaluator tests.
- API result/symbol/export integration tests.

### 10.1 Missing-value policy

Use these initial rules:

- Missing values never satisfy positive range, include, text, or boolean conditions.
- Boolean false means a known false value, not missing.
- Categorical exclusion retains today's behavior: missing values pass an exclusion condition.
- The UI labels exclusion explicitly and warns when an exclusion is placed in an ANY group because it may make the branch broad.

Pin these decisions in tests before implementing OR compilation.

## 11. Implementation phases

### Phase 0 — Prerequisite correctness and benchmarks

1. Complete or explicitly re-verify `stockscreenclaude-b2l` and `stockscreenclaude-55r` on live US, HK, and JP scans.
2. Capture baseline result totals and wall-clock timings for representative flat presets.
3. Add benchmark expressions for:
   - Indexed-column AND.
   - Indexed-column OR.
   - Mixed indexed and JSON-backed OR.
   - Nested setup-engine JSON conditions.
4. Confirm the UI never marks a requested expression applied before its query succeeds.

Exit criterion: existing flat filters are correct, failure-visible, and fast enough to serve as the grouped-query baseline.

### Phase 1 — Expression contract and pure semantics

1. Add backend request schemas and bounded domain types.
2. Implement validation and canonicalization.
3. Implement a pure Python expression evaluator.
4. Create cross-runtime truth-table fixtures.
5. Add v1 `FilterSpec` to required-only expression conversion.

Exit criterion: expression semantics and missing-value behavior are fully test-pinned without changing production queries.

### Phase 2 — SQL compilation and API paths

1. Refactor legacy and feature-store leaf filters into predicate builders.
2. Compile group and root predicates with SQLAlchemy `and_`/`or_`.
3. Add POST result, symbol, and export endpoints.
4. Add matched-group evaluation for hydrated pages and exports.
5. Add query fingerprinting and bounded telemetry.
6. Run correctness and performance comparisons against equivalent flat filters.

Exit criterion: backend endpoints return correct, explainable grouped results with acceptable query plans on both repositories.

### Phase 3 — Frontend state and simple-mode compatibility

1. Add the canonical frontend expression model and adapters.
2. Convert the existing flat panel into an editor for required conditions.
3. Add draft/requested/applied state separation.
4. Forward abort signals and surface pending/error states.
5. Replace hand-maintained “filtered” detection with expression-derived counts and summaries.
6. Update chart navigation and export calls to use the applied expression.

Exit criterion: existing simple filters and presets produce unchanged totals while using the new lifecycle safely.

### Phase 4 — Guided grouped-rule builder

1. Build the searchable condition catalogue and condition editors.
2. Build required and setup-group cards.
3. Add group join and internal match controls.
4. Add keyboard-accessible move, duplicate, disable, rename, and delete actions.
5. Add validation, Apply/Cancel, reset-draft, undo, and save-preset flows.
6. Add expression summary and matched-group badges/tooltips.

Exit criterion: a user can build the example in section 1 without learning boolean notation and can identify why each returned stock matched.

### Phase 5 — Presets, static mode, and rollout

1. Add lazy v1-to-v2 preset conversion.
2. Update static preset export and client-side evaluation.
3. Run the shared parity fixture suite in Python and JavaScript.
4. Add the feature flag and deploy backend support first.
5. Enable the UI for internal use, then progressively for production.
6. Document grouped filtering and provide two or three starter grouped presets.

Exit criterion: live and static results, navigation, exports, and saved presets are semantically identical.

## 12. Performance gates

Before production enablement:

- Flat-filter result totals remain identical to the current implementation.
- Representative grouped queries over approximately 10,000 rows:
  - p50 under 1 second.
  - p95 under 2 seconds for indexed/common conditions.
  - Hard acceptance ceiling of 5 seconds for supported mixed JSON expressions.
- Count queries do not hydrate row blobs or calculate matched-group labels.
- One Apply action produces one result request; it does not issue automatic per-group count queries.
- Query cancellation reaches Axios and the backend/database cancellation path where supported.
- A slow or failed query never changes the applied expression summary or “X of Y” count.

If mixed JSON OR queries cannot meet the ceiling, restrict the initial field catalogue for grouped rules to indexed fields and surface unsupported fields as unavailable rather than shipping unpredictable latency.

## 13. Test plan

### Backend unit tests

- Schema validation and limits.
- Canonicalization/fingerprinting stability.
- Truth tables for all match combinations.
- Missing-value policy.
- V1 `FilterSpec` conversion.
- Pure matched-group evaluation.

### Backend repository/integration tests

- Legacy and feature-store SQL return the same symbols as the pure evaluator.
- Results, symbols, and export use identical expressions and ordering.
- Multiple matching groups are returned correctly.
- Unknown or unsupported fields fail with a 422 response.
- Count/hydration query shape remains lean.
- PostgreSQL expression-index plans remain usable for representative OR predicates.

### Frontend tests

- Existing simple controls edit required conditions.
- Builder keyboard and pointer interactions.
- Draft cancellation and successful application.
- Failed application preserves prior rows and summary.
- Preset v1 load, v2 save, update, rename, and dirty state.
- Applied summary and matched-group badges.
- Export and chart navigation receive the applied expression, not the draft.

### Static tests

- Shared truth-table fixtures.
- Preset match counts.
- Boolean false versus missing.
- Multiple matched groups and navigation ordering.

### End-to-end checks

- Build and apply the section 1 example on live US, HK, and JP scans.
- Verify exact totals against direct API calls.
- Open charts and navigate only through the filtered symbol order.
- Export and compare symbols plus matched-group labels.
- Repeat an equivalent preset on the static site and compare totals.

## 14. Rollout and telemetry

Use a runtime feature flag such as `grouped_scan_filters_enabled`.

Record operational telemetry without logging user-entered group names or condition values:

- Repository path: legacy or feature store.
- Expression version.
- Group count and condition count.
- Condition field names and kinds.
- Group join mode.
- Query fingerprint.
- Duration, result count, cancellation, timeout, and error category.

Rollout sequence:

1. Backend endpoints dark-launched with tests and benchmarks.
2. Internal UI enablement.
3. One starter grouped preset per major workflow.
4. Production enablement after one week of acceptable latency/error telemetry.
5. Revisit automatic group preview counts only after production query performance is proven.

## 15. Non-goals for the first release

- Arbitrary recursive boolean trees.
- Formula text entry.
- Natural-language rule generation.
- Automatic optimizer suggestions based on historical performance.
- Unlimited per-group live counts.
- Sharing/public URLs unless already supported safely by preset infrastructure.
- Changing how the underlying scan itself scores or passes stocks.

## 16. Definition of done

The feature is complete when:

1. Existing flat filters and presets preserve their result sets.
2. Users can express required guardrails plus named alternative setup paths without boolean syntax.
3. Every result explains which setup group or groups matched.
4. Live results, static results, chart navigation, and CSV export use identical semantics and ordering.
5. Missing values behave consistently across Python, SQL, and JavaScript.
6. Failed, cancelled, or slow queries cannot relabel stale rows.
7. Representative grouped queries satisfy the documented performance gates.
8. Presets migrate lazily and remain backward compatible.
9. Backend, frontend, static, integration, and end-to-end quality gates pass.
10. The feature is deployed behind a flag, observed, and then enabled based on telemetry.

## 17. Critical files

Backend:

- `backend/app/domain/common/query.py`
- `backend/app/api/v1/scan_filter_params.py`
- `backend/app/api/v1/scans.py`
- `backend/app/infra/query/scan_result_query.py`
- `backend/app/infra/query/feature_store_query.py`
- `backend/app/infra/db/repositories/scan_result_repo.py`
- `backend/app/infra/db/repositories/feature_store_repo.py`
- `backend/app/schemas/filter_preset.py`
- `backend/app/use_cases/scanning/get_scan_results.py`
- `backend/app/use_cases/scanning/get_scan_symbols.py`
- `backend/app/use_cases/scanning/export_scan_results.py`

Frontend/live:

- `frontend/src/features/scan/pages/ScanPageContainer.jsx`
- `frontend/src/features/scan/components/FilterPanelContainer.jsx`
- `frontend/src/features/scan/components/ScanResultsSection.jsx`
- `frontend/src/features/scan/hooks/useScanFilterPresets.js`
- `frontend/src/components/Scan/ResultsTable.jsx`
- `frontend/src/components/Scan/filters/`
- `frontend/src/utils/filterUtils.js`
- `frontend/src/api/scans.js`

Frontend/static:

- `frontend/src/static/scanClient.js`
- `frontend/src/static/pages/StaticScanPage.jsx`
- `frontend/src/static/hooks/usePresetScreens.js`
- `backend/app/services/static_site_export_service.py`

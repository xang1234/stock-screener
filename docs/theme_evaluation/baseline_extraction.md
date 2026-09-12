# Frozen baseline extraction stage

The subsequent [theme/development separation](theme_development_separation.md) adds an optional source-specific summary while preserving fully open theme names and legacy records. Its separate first-ten treatment produced 16 successful outcomes, four provider failures and 12 mentions. Review under `baseline-extraction/theme-development-v1/` flags unsupported NBIS/nuclear inference; this result is not an accuracy endorsement. The 499 relevant backend checks and frontend build passed.

The next stage now admits reviewed evidence into immutable extraction input runs and supports explicit generation/import, verification, and Markdown/CSV review. It reuses `ThemeExtractionService.extract_from_content` for both technical and fundamental pipelines without creating clusters or changing production tables. It does not implement ranking replay or reference theme labels yet.

The user authorized proceeding with the September 10 fresh packet after reviewing preparation and approved MiniMax M2.7 as primary with an appropriate OpenCode Go fallback. Kimi K2.6 is the selected fallback; the expired Z.AI route is disabled for this runtime. MiniMax uses the application's metered single-attempt path (all retry layers disabled), followed by at most one Kimi fallback. Each actual attempt has independent model, request hashes, effective parameters, public response payload and usage provenance. Unknown returned model/provider remains unknown. Empty imported call metadata means unknown provenance, not proof of zero provider calls.

## Evidence admission

`prepare-inputs` requires exact bundle, preparation and assessment IDs plus an attributed approval file. It loads and verifies the archived assessment, so later policy/selection changes cannot silently change an approval. Material holds and exclusions win over blanket approval. Known English original text is admitted with completeness limitations. Other languages need an eligible selected English translation. Approved image transcription and eligible recovered article derivatives require suitable English preparation; chart interpretation prose is not substituted for transcription.

Background and non-investment posts are retained among admitted originals. Unresolved translations, missing derivatives, material holds and ambiguous normalized quantities remain explicit exclusions. Original versus normalized text, numerical values, currencies and source spans are bound into the frozen input ID. Images and articles keep result IDs and parent relationships; they are not counted as independent corroboration.

Existing acquisition bundles and preparation stores remain unchanged. The new sidecars live in `extraction-runs/<content-hash>/`. Successful empty extraction, failed extraction and pending generation remain separate. Strict mention and call schemas reject malformed imports.

## Commands

From `backend/`, use `python scripts/extract_theme_evidence.py`:

```text
prepare-inputs --bundle PATH --store PATH --preparation-id ID --assessment-id ID --approval FILE --output-root PATH
verify --run PATH
review --run PATH --bundle PATH --store PATH --output NEW_DIRECTORY
import-extractions --run PATH --records FILE --output-root PATH
generate --run PATH --reference-manifest FILE --env-file FILE --code-revision REVISION --max-documents 10 --allow-model-calls --output-root PATH
```

Generation defaults to both pipelines. A bounded ten-input run can produce twenty extraction outcomes and at most forty provider attempts if every primary fails. Routine review/verify/import never calls a model. An import document contains `input_run_id` and `records`; its reviewed approval must match the frozen run. Generation requires an unprocessed approved input run. The CLI reports an incomplete run as partial rather than treating unprocessed inputs as successful empty results.

Credentials are read only from the process environment or an explicitly supplied environment file; they are not command argument values or artifact contents. The dedicated PostgreSQL reference runtime is described in [extraction_runtime.md](extraction_runtime.md). Missing prerequisites return exit code 4 before provider calls. Integrity failures return 5, invalid data 2.

For evidence-rich exclusions, supply the original bundle and preparation store together. The renderer verifies them against the run's frozen assessment and exports full source text, saved English translation candidates and their status, source URL, parent post context, and readable exclusion findings. Missing article/image text is explicitly marked; parent post text stays in separate columns. Candidate translations are review evidence, not new approvals. Omitting both context arguments remains supported but produces exclusions without source text. Review never generates translations or changes admission decisions.

## September 10 admission and preflight

The fresh sample produced **91 admitted inputs: 89 originals, one selected English translation, and one image transcription**. This is a limited input set, not all captured article/image context. Two source records with material holds remain excluded. Eight root records have unresolved English preparation; derivative exclusions are listed separately in the CSV and must not be added to the post denominator.

The local packet is under `data/xui-reader/theme-evaluation/gap-remediation-20260910/baseline-extraction/`. `location.json` identifies the exact content-addressed input run. `input-review/extractions.md` and its CSVs show all admitted inputs and exclusions.

The original preflight stopped because the evaluation database URL was absent. That dependency is now resolved: the existing local Docker PostgreSQL server hosts a separate `theme_evaluation_20260910` database with a SELECT-only role. Its required reference-table schemas were restored from the migrated application database, and a read-only snapshot of 17,211 active US/HK/JP stock references was imported. Extraction reads only this isolated snapshot.

The first ten inputs have now produced twenty successful outcomes, thirteen mention objects and twelve empty outcomes, all via the approved Kimi fallback. MiniMax's token-plan allowance was exhausted. The initial failed attempt is retained; a subsequent retry succeeded after adding OpenCode Go's documented stable session header. Standalone extraction now binds an isolated runtime-service context rather than depending on application startup.

`first-batch-review/FIRST_BATCH_REVIEW.md` and `theme-mentions.csv` compare source text, raw model tickers and final tickers. Review confirmed thirteen spurious `POST` additions from generic post titles, two `PRTH` additions from “priority”, and one `3650.HK` addition from “keep”. These are deterministic company-name matching artifacts, not raw model hallucinations. Foreign listing removals are separately identified as reference-coverage effects. The remaining 81 inputs await checkpoint review; there are no reviewer-approved accuracy labels or ranking scores.

## Validation

447 tests passed across theme evaluation, theme identity/lifecycle/source weighting and MiniMax routing. New extraction code/tests pass Ruff. Independent scoped review findings were reproduced and fixed; its final 30-test extraction suite passed. Existing dependency deprecation warnings remain. The real admission run verifies with 91 inputs and zero extraction records.

Subsequent startup/export verification passed 36 extraction tests. Session-header fixes passed 71 focused runtime, image and translation tests with independent review. The successful ten-input result run verifies with 91 bound inputs and 20 recorded outcomes. All 487 original frozen evidence/input files and all 106 exclusion rows are unchanged. The runtime source manifest and archive retain the code used for the successful retry.

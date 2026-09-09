# Theme evaluation: evidence review checkpoint

Collect and inspect evidence before generating extractions. This CLI has no extraction or theme-label command. It works without the application database, provider keys, or running services.

The required sources are [list 1986290701492232693](https://x.com/i/lists/1986290701492232693) and [list 1522014550211457024](https://x.com/i/lists/1522014550211457024). Their presence does not prove historical completeness.

## Existing review packet

In the evaluation worktree, open `data/xui-reader/theme-evaluation/pilot-20260908/review-final/START_HERE.md`. The packet includes:

- `evidence.md`: captured posts, source timestamps, article references and lookup outcomes.
- `documents.csv`: original text, document IDs, hashes and source memberships.
- `followups.csv`: all article/media reference decisions, reasons and lookup provenance.
- `coverage.csv` and `coverage.json`: counts and unavailable stages.
- `../review-location.json`: exact immutable bundle path and ID.

These files contain real source content and stay in ignored local storage. The ideas document remains in the main checkout under `docs/research/`.

The pilot includes 96 unique posts from the latest 50 returned per list, with four shared posts. It spans approximately four hours. Two matched articles have short partial excerpts; their full bodies are not frozen. No translations, extractions or theme-label proposals have been generated. User review is pending.

## Reuse the frozen list captures

Run from the worktree root. This command makes no network requests and prints the resulting `bundle_path`:

```bash
/Users/admin/StockScreenClaude/backend/venv/bin/python backend/scripts/theme_evaluation.py import-x \
  --first data/xui-reader/theme-evaluation-source-audit-20260908/list-1986290701492232693.json \
  --second data/xui-reader/theme-evaluation-source-audit-20260908/list-1522014550211457024.json \
  --requested-limit 5 --captured-at 2026-09-08T03:15:21.552933Z \
  --max-posts-per-source 50 --output-root data/xui-reader/theme-evaluation/reimport
```

The requested reader limit was five, but the files contain 125 and 143 rows. Selection occurs locally, before any theme judgments: publication descending, tweet ID ascending for ties, unknown dates last. Conflicting bodies for one tweet ID are excluded and recorded. Shared posts retain both memberships and observation times.

Use the printed path with `review --bundle PATH --output NEW_DIRECTORY`, `references --bundle PATH --output NEW_FILE`, or `verify --bundle PATH`. Each command follows the same interpreter and script prefix above. Review output paths must be new; existing evidence is not overwritten.

## Fresh X reads

Use the [xui-reader skill](/Users/admin/Documents/Work/xui-reader/skills/xui-reader/SKILL.md). The supplied wrapper checks authentication before reading. Example for one list:

```bash
PATH=/Users/admin/Documents/Work/xui-reader/.venv/bin:$PATH \
/Users/admin/Documents/Work/xui-reader/.venv/bin/python \
  /Users/admin/Documents/Work/xui-reader/skills/xui-reader/scripts/xui_read.py \
  list 1986290701492232693 --profile default --login-policy prompt --limit 50
```

The package's `collect-x` reads both required lists sequentially. Supply `--wrapper`, `--python`, `--xui-bin`, `--config`, `--profile`, `--limit`, `--max-posts-per-source` and `--output-root`; the local executables are those shown above, and the config defaults to `/Users/admin/.config/xui-reader/config.toml` when using the wrapper directly. If authentication fails, stop and follow the reader's reauthentication guidance. Never read session storage or copy cookies into a corpus.

## Article and translation imports

Review URLs and title-only references in context. Attached images, newsletter signup pages and quoted posts are not automatically articles. Follow investment-related article references using the existing web tools; search author/title/publisher if necessary. Keep an unresolved result when the exact piece cannot be verified. Search results from a different date or merely a similar topic do not establish a match.

`import-articles --bundle PATH --records FILE --output-root ROOT` accepts a JSON object with exactly `bundle_id`, `mode`, `followups` and `articles`. `bundle_id` and `mode` must match the supplied frozen bundle. Records use the strict `Followup` and `Document` models in `backend/app/services/theme_evaluation/records.py`; the local `article-review-import.json` is an actual example. Each update seals a new version. Articles must have captured text and a referring-post relationship; corrections preserve earlier captured versions and their links.

`partial` means only some of the original body is captured. A discovered URL or a paraphrase must not be imported as a full original article. Record article retrieval separately from post observation; never backdate it to publication. If only a calendar publication date is known, retain the date in match evidence and leave the exact timestamp unknown.

Native long-form X Articles use the skill's `article-pdf` wrapper and a deterministic output path. Preserve the returned PDF hash, title, warnings and export time in `source_metadata`. Read the resulting PDF through a supported reader before importing text. Record export failure explicitly; a normal post is not the full Article body.

`import-translations` accepts exactly `bundle_id`, `mode` and `derivatives`. Preserve original text, bind each translation to its source-text hash, and record translator/model, policy version and actual generation time. Unsupported translations remain explicit. Do not silently exclude non-English posts.

## Review and next stage

Review source mix, truncated content, article gaps, multilingual coverage and the time window. Ordinary background posts remain included so a later detector cannot be evaluated only on handpicked investment successes.

Approval must identify the reviewed bundle. Do not run extraction on a changed dataset under an earlier approval. Subsequent implementation will add extraction against an isolated evaluation database with a frozen reference universe; the new/empty application database is not assumed to supply either historical content or ticker resolution.

Exit codes: `0` completed operation (inspect coverage separately), `2` invalid data/path/arguments, `3` live source/auth failure, `5` bundle integrity failure. A successful review command does not mean the evidence is complete or approved.

## Preparation after reader updates

Implementation lives in the evaluation worktree. The original pilot is frozen. Rebuild acquisition only after both repositories are updated; keep evidence review before theme extraction.

The preparation commands use existing HTTPX, BeautifulSoup and Pillow. Kimi K2.6 via OpenCode Go is approved for original-language image transcription and chart interpretation. PaddleOCR is deferred. HTTPX and BeautifulSoup are approved for article recovery. Kimi K2.6 is now selected for translation through the configured Go connection. The rendered-browser fallback still needs its recorded approval before use.

From `backend`, using the configured project Python environment:

```bash
python scripts/prepare_theme_evidence.py prepare \
  --bundle <bundle-directory> --output-root <preparation-root> \
  --handoff <reader-handoff.json> --stages article text image
```

This defaults to offline operation. Missing downloads/models/translations produce visible unavailable outcomes. It never reads X lists, writes production tables, generates theme extractions, or changes the base bundle. See [reader handoff](reader_handoff.md) for mappings and imports.

After the reader update and applicable tool approval, `--allow-network` enables public article/image downloads. `--allow-model-calls` separately enables the approved Go image request, using `OPENCODE_GO_API_KEY` from the environment. Do not put keys in command arguments or handoff files. The model is fixed to `kimi-k2.6`; there is no silent provider fallback. No new local OCR package is required.

The [2026-09-08 live Kimi validation](kimi_validation_2026-09-08.md) passed image transport, configured parameters, Korean/Japanese financial transcription and tested chart associations across four controlled fixtures. The adapter now supplies Go’s required session header and an explicit client name. Two minor commentary counting errors were observed. Real-corpus accuracy remains unmeasured; retain original images for evidence review. The current output cap is 2,048 tokens; provider-reported truncation fails visibly instead of becoming a successful transcription. Dense images may need a later approved higher cap or crop strategy based on fixture results.

Every preparation command prints a preparation ID. Review and verify it with:

```bash
python scripts/prepare_theme_evidence.py review \
  --bundle <bundle-directory> --output-root <preparation-root> \
  --preparation <preparation-id> --output <new-review-directory>

python scripts/prepare_theme_evidence.py verify \
  --bundle <bundle-directory> --output-root <preparation-root> \
  --preparation <preparation-id>
```

Review outputs:

- `START_HERE.md`: material holds, review work, informational notes, exclusions and coverage.
- `translation-review.md` / `.csv`, `article-review.md` / `.csv`, `image-review.md` / `.csv`: prioritized findings with exact result/source hashes and next actions.
- `assessment.json`: immutable, reproducible assessment; separate from legacy processing statuses.
- `linked-post-manifest.json`, `linked-post-review.md`, `reference-manifest.csv`: all reference relationships and bounded follow-up work, separate from selected-post counts.
- `preparation-runs.json`: measured request counts and elapsed time for recorded runs.
- `evidence.md`: original evidence link, images, transcription, visual observations, original/translated segments, provenance and gaps.
- `preparations.csv`: detailed payloads, parent/source links, model/policy, timestamps, warnings, and current/superseded status.
- `article_followups.csv`: only current unresolved/partial article versions, ready for assisted recovery.
- `coverage.json`: current gaps, missing image metadata, and pending evidence approval.
- `original/`: the unchanged base-bundle review packet.

Keep the review directory together when sharing it: image links are relative. No images load from remote hosts during offline review. The content-addressed store contains `assets`, `results`, and `preparations`, plus a request-keyed `cache` index. Ordinary lookup verifies only the selected result and its assets. The explicit `verify` command checks all stored assets/results and cache entries, including unbound attempts, then verifies the requested preparation and its source bindings. It rebuilds missing index entries only after the asset/result audit passes; interrupted indexing otherwise causes a harmless cache miss. Place it under ignored local `data/`, never source control. Successful identical image inputs reuse processing across posts; original processing times remain intact. Failed/partial/uncertain results stay retryable.

Use `--prior-preparation <id>` on prepare/import-articles to retain prior results in a new preparation. It requires the same base and handoff. The manifest explicitly records current evidence separately from attempt history; list order never selects the current version. An unavailable retry retains a usable prior result when its input is unchanged or could not be recaptured. Changed captured inputs select a new version even if processing fails, and supersede translations of the old article/image at that source locator. The text stage processes only current article/image versions. Other completed results become current, including results carrying review warnings. All attempts remain visible in the report. If source text or handoff metadata changes, start a new appropriately bound preparation. Re-running HTTP recovery fetches the URL again because web content can change.

```bash
python scripts/prepare_theme_evidence.py import-articles \
  --bundle <bundle-directory> --output-root <preparation-root> \
  --handoff <reader-handoff.json> --prior-preparation <id> --records <article-import.json>

python scripts/prepare_theme_evidence.py import-translations \
  --bundle <bundle-directory> --output-root <preparation-root> \
  --preparation <id> --records <translation-import.json>
```

To prepare recovered article text or image transcription for translation, run the text stage with the resulting prior preparation. Translation imports create another immutable preparation; repeated identical imports are idempotent. All evidence versions still require user review before extraction.

### Preparation format version 2

Preparation manifests now use schema version 2 with `bindings` for unique attempt history and `current` mapping slot hashes to selected binding hashes. A slot identifies source kind/ID, stage, parent result and input locator. Article, text and image results have typed requests and payloads. Status and warnings derive from validated content rather than duplicated stored status fields. This changes only preparation artifacts; acquisition Bundle v1 and its frozen content IDs remain unchanged.

The earlier development-only preparation format is intentionally rejected rather than silently migrated. It was used in temporary tests, not on the frozen pilot. If an external scratch preparation used that format, rebuild it in a new output directory. Keep old evidence for inspection.

### Kimi translation

Enable translation explicitly with `--stages text --allow-translation-calls`, using `OPENCODE_GO_API_KEY`. It can be combined with article/image stages; `--allow-network` controls article/image downloads and `--allow-model-calls` controls image interpretation separately. The translation flag sends source text to Go even if public downloads are disabled. Without it, text preparation remains offline.

The [live validation report](kimi_translation_validation_2026-09-08.md) records 20 final-round translations, 10/10 preserved large quantities and two contextual issues. Policy `translation-v3` retains written CJK large quantities in source notation; it does not ask Kimi to convert them into billions. Each protected token must return exactly once before original quantities are restored. Invalid/truncated output remains unavailable. Name, supplier/customer direction and speaker attribution still need evidence review.

The adapter bounds each segment to 4,000 characters and each output to 4,096 tokens, with a 45-second read timeout. This is a short-passage validation, not a measured long-article throughput or accuracy guarantee. The configured route remains Go; a direct API route has not been provisioned.

### Evidence quality and language policy

New preparation assesses X output before using it. Missing text or supported material quantity/polarity discrepancies trigger Kimi fallback when explicitly enabled. Ambiguous meaning remains review work; the checks are not a semantic correctness guarantee. Original X candidates, failed Kimi attempts and their provenance remain inspectable. Use the selected-result sidecar rather than a legacy `success` or `current` field to determine deterministic eligibility.

Opt into language preparation v2 with:

```bash
python scripts/prepare_theme_evidence.py prepare \
  --bundle <bundle-directory> --output-root <preparation-root> \
  --handoff <reader-handoff.json> --prior-preparation <preparation-id> \
  --stages text --text-policy v2 --allow-translation-calls
```

V2 preserves observed metadata separately from its conservative language decision, avoids translating URL/emoji-only content, and packs adjacent paragraphs into bounded segments without losing source characters. Article language comes from that article. Legacy behavior remains the default for compatibility; the v2 cache namespace prevents incompatible reuse. See [language challenge](language_challenge.md) and [remediation validation](gap_remediation_validation.md).

Image processing records specific sanitized failure codes. One run permits at most one retry for an eligible transient failure per input/provider/policy, reusing validated bytes. Malformed, truncated, authentication and deterministic validation failures do not trigger blind retries. Every attempt is preserved, and actual download/model calls are reported separately from reused outcomes.

The review command accepts `--annotations <annotations.json>` containing an array of evidence decisions. Copy exact `entry_id`, `result_id`, `input_sha256`, `source_text_sha256`, bundle and preparation IDs from the assessment, then add `decision` (`accept`, `exclude`, `hold`), `reviewer`, `reviewed_at` with timezone, and a nonblank `reason`. Scope defaults to one `claim`; `evidence` covers that source/result/stage. Image uncertainty can additionally be classified as `cosmetic`, `material` or `unknown`. Stale hashes, conflicting scopes and malformed annotations are rejected before writing a packet. Always render to a new directory. An annotation never changes original artifacts, deterministic selection, or the extraction approval gate.

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

The preparation commands use existing HTTPX, BeautifulSoup and Pillow. Kimi K2.6 via OpenCode Go is approved for original-language image transcription and chart interpretation. PaddleOCR is deferred. The proposed live translation service and article tool choices still need their recorded approval before actual use; neither was called during implementation.

From `backend`, using the configured project Python environment:

```bash
python scripts/prepare_theme_evidence.py prepare \
  --bundle <bundle-directory> --output-root <preparation-root> \
  --handoff <reader-handoff.json> --stages article text image
```

This defaults to offline operation. Missing downloads/models/translations produce visible unavailable outcomes. It never reads X lists, writes production tables, generates theme extractions, or changes the base bundle. See [reader handoff](reader_handoff.md) for mappings and imports.

After the reader update and applicable tool approval, `--allow-network` enables public article/image downloads. `--allow-model-calls` separately enables the approved Go image request, using `OPENCODE_GO_API_KEY` from the environment. Do not put keys in command arguments or handoff files. The model is fixed to `kimi-k2.6`; there is no silent provider fallback. No new local OCR package is required.

Before processing a real corpus, run a small controlled image fixture through Go to verify image transport, non-thinking parameters, Korean/Japanese text, numbers/units and chart labels. Provider parameters and actual evidence accuracy have not been validated live. The current output cap is 2,048 tokens; provider-reported truncation fails visibly instead of becoming a successful transcription. Dense images may need a later approved higher cap or crop strategy based on fixture results.

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

- `evidence.md`: original evidence link, images, transcription, visual observations, original/translated segments, provenance and gaps.
- `preparations.csv`: detailed payloads, parent/source links, model/policy, timestamps, warnings, and current/superseded status.
- `article_followups.csv`: only current unresolved/partial article versions, ready for assisted recovery.
- `coverage.json`: current gaps, missing image metadata, and pending evidence approval.
- `original/`: the unchanged base-bundle review packet.

Keep the review directory together when sharing it: image links are relative. No images load from remote hosts during offline review. The content-addressed store contains `assets`, `results`, and `preparations`; integrity checks cover all referenced assets/results and source bindings. Place it under ignored local `data/`, never source control. Successful identical image inputs reuse processing across posts; original processing times remain intact. Failed/partial/uncertain results stay retryable.

Use `--prior-preparation <id>` on prepare/import-articles to retain prior results in a new preparation. It requires the same base and handoff. Latest attempts are current; earlier versions remain visible as superseded history. If source text or handoff metadata changes, start a new appropriately bound preparation. Re-running HTTP recovery fetches the URL again because web content can change.

```bash
python scripts/prepare_theme_evidence.py import-articles \
  --bundle <bundle-directory> --output-root <preparation-root> \
  --handoff <reader-handoff.json> --prior-preparation <id> --records <article-import.json>

python scripts/prepare_theme_evidence.py import-translations \
  --bundle <bundle-directory> --output-root <preparation-root> \
  --preparation <id> --records <translation-import.json>
```

To prepare recovered article text or image transcription for translation, run the text stage with the resulting prior preparation. Translation imports create another immutable preparation; repeated identical imports are idempotent. All evidence versions still require user review before extraction.

# Evidence gap remediation specification

## Objective and ownership

Make collected investment evidence reviewable and reliably usable before theme extraction, while bounding collection latency and model calls. This work repairs acquisition and preparation; it does not implement theme extraction or change the production database.

The reader repository is [xang1234/xui](https://github.com/xang1234/xui). Its inspected local checkout is `/Users/admin/Documents/Work/xui-reader`, revision `1a3bfc4`. The Git remote matches the supplied repository URL. GitHub's web page could not be fetched during planning, so this design is grounded in that local revision, not a claim about the latest remote branch.

Downstream preparation lives in `/Users/admin/StockScreenClaude/.worktrees/theme-detection-evaluation`, branch `feat/theme-detection-evaluation`, inspected revision `efe41424`. Paths beginning `backend/` below belong there, not in xui.

| Owner | Responsibilities |
| --- | --- |
| xui | Original X text, complete translated DOM capture, post identity, bounded hydration, completeness metadata, linked-post reads through its existing post command |
| StockScreenClaude | Evaluate translation adequacy, Kimi fallback, public article recovery, image processing/retries, evidence assessments, reproducible review packet |

## Observed baseline

The September 9 packet contains 99 distinct posts selected from 50 posts per required list, 33 images, and 54 reference records. Thirteen X translations were captured, 18 were unavailable (`control_missing`), and three failed (two `post_identity_mismatch`, one `translated_text_missing`). Kimi fallback exists for missing/failed captures. Seventy-nine post completeness values remain unknown; 20 are explicitly complete. Unknown does not establish truncation.

Three image results remain unavailable under the generic `image_validation_or_processing_failed` code. Thirty have outputs: nine without reported uncertainty, 21 with uncertainty. One numeric table was manually spot-checked; the other images have not been comprehensively audited. The nine successful statuses are not nine correctness verdicts.

Three article destinations have readable but unverified partial text: two Daum articles and one Substack post. Other references include short/expanded duplicates, X posts, non-articles, script-only publisher pages and a Reuters HTTP 401. No full article is certified complete. There are seven linked-X-post reference records, not necessarily seven unique missing posts.

The strongest failure example is post `2097617015238500778`:

```text
Original: 이재용 삼성전자 회장, 母홍라희 보유 718만주 1.9조원에 매수 - 조선비즈 https://t.co/yZWIyQI9gK
Captured X translation: Samsung
Original-text hash: sha256:82390db58687dbb27615c8f67acd5580a90da628a7a90d9179cf8a652a27b241
```

The identity/hash matches but the transaction and quantities are absent. The DOM cause has not been reproduced. Selecting the first English candidate and accepting the first changed frame are code risks to test, not proven causes of this particular failure.

## Requirements

1. Preserve original text, original images, URLs, capture times and all preparation attempts. Translation is a derivative, never a replacement for canonical evidence.
2. Accept X translations only after identity checks and adequacy assessment. Material omissions trigger one Kimi fallback. Suspicion without proof triggers review, not invented corrections.
3. Normalize financial quantities and dates conservatively. Ignore URL tokens in numeric comparisons; do not ignore digits in product names or stock identifiers. Never infer an unstated currency or exchange rate.
4. Prefer public HTTP article recovery and deterministic parsing. Distinguish no body, partial body, access restriction, non-article, and linked X post. Preserve source-reference relationships when deduplicating work.
5. Keep image transcription, visual observations and uncertainty separate. Retry diagnosed transient failures only. Unreadable values remain unknown.
6. Publish original/translation/image/article evidence and assessments in Markdown and CSV. Review severity must distinguish material blockers, review items, and informational notes.
7. Validate Japanese, Korean, Chinese, mixed-language and nonlinguistic cases in a separate challenge set. Do not represent that set as naturally sampled list coverage.
8. Keep extraction at `awaiting_evidence_approval`. Preparing a clean packet must never imply user approval.

## Approved services and limits

- Model: Kimi K2.6 through the existing OpenCode Go adapters only.
- Article acquisition: existing HTTPX public fetcher and BeautifulSoup parsing.
- X acquisition: existing xui Playwright session and commands.
- No PaddleOCR, additional model provider, translation API, new paid scraper or production service is required.
- A proposed external-article rendered-browser collector requires the user's tool approval before live use. Implement and test the existing browser-import boundary now; live rendering is a conditional follow-up, not a blocker for the rest of this plan.
- Never export or log cookies, authorization headers, credentials or browser storage state. Preserve existing per-hop public-address checks and size limits.
- Keep existing bounded X attempts: at most one inline translation attempt and one shared targeted-post fallback per selected post. Do not independently double the budget for completeness and translation.
- Default preparation retries: initial attempt plus at most one eligible retry per input per run; concurrency at most three; no model retry for deterministic validation errors.
- Preserve frozen bundle/result identities and legacy interpretation. A new assessment is a versioned derivative; it must not silently reinterpret archived statuses or rewrite archived JSON.

## Evidence handoff

Local evidence root, relative to the preparation worktree:

`data/xui-reader/theme-evaluation/reader-update-20260909/`

Start at `review/START_HERE.md`; inspect `review/translation-review.csv`, `review/article-review.csv`, `review/preparations.csv`, and `reader-gap.json`. These files are ignored local artifacts and will not appear merely by cloning either repository. An engineer without them must use the embedded Samsung reproduction and checked-in sanitized fixtures, then obtain a fresh authorized capture for live confirmation. Never substitute invented DOM for a claimed real capture.

Base bundle: `853b19ceb1f980afda8249825929b9de3f1a025dcb2aa5a5f6b75f6eb36efc90`.

Preparation: `ceb89077c0b4a6d3244d50af45f9c40478df6eab4bac8514ccba501419a5fff0`.

## Success criteria

- The Samsung fragment is never an eligible selected translation; its original capture remains inspectable.
- Correct date/unit conversions and dropped short links do not cause material mismatch findings; changed quantities, currency units, polarity or identity do.
- Every unresolved article/image failure has a specific disposition and next action.
- Review decisions bind to exact evidence/result hashes and become stale when those inputs change.
- Deterministic replay and a fresh collection are reported separately, with denominators, elapsed time, targeted reads and model-call counts.
- All required regression tests pass. The user can approve a clearly enumerated usable subset without pretending inaccessible evidence was recovered.

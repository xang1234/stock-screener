# Kimi K2.6 live validation — 2026-09-08

**Verdict: suitable for the next evidence-review pilot, with original images retained for checking.** The approved OpenCode Go route accepted image input and returned valid structured output for all four controlled fixtures after fixing required client/session headers. Financial transcription and tested chart associations passed. Visual commentary still contained two minor factual errors. These four synthetic cases do not establish real-corpus accuracy or reliability.

## Results

| Fixture | Text / financial checks | Interpretation review | End-to-end time |
| --- | --- | --- | ---: |
| Korean financial summary, 1200×620 | 8/8 original lines; 6/6 critical strings | Labels, decimals, negative debt, percentages, currency pair and 억원 units preserved. Commentary incorrectly counted 7 lines rather than 8. | 8.49 s |
| Japanese results table, 1300×850 | 14/14 original lines; 19/19 critical strings | Prior/current columns, negative cash flow, 億円/万株 units and per-share 円 exception preserved. Commentary incorrectly counted 10 metrics rather than 9. | 12.76 s |
| Dual-axis financial chart, 1300×850 | 14/14 critical strings; manually verified 8/8 quarter values | Revenue bars correctly read as 80/100/90/120 USD million; margin line as 10/12/8/15%. Correct left/right axes. No investment recommendation. | 12.03 s |
| Damaged/cropped excerpt, 1200×620 | 3/3 critical strings | Flagged damaged margin and cropped footer; invented no missing financial value. Transcribed an embedded instruction without following it. | 10.94 s |

Line checks compare exact characters after removing whitespace. Critical-string checks measure presence, not numerical reasoning, reading order or alignment; the latter were inspected manually against the fixture images. The Japanese fixture's draft type was `document`; visual review correctly treats it as a table. That draft-label correction is recorded in `manual_review.json`, and no automated type score is claimed.

There were **4 successful model requests and 2 HTTP 400 rejections**. Median successful wall time was **11.49 seconds**, range **8.49–12.76 seconds**. These are four single trials, not a latency distribution or repeatability benchmark. Successful response usage metadata was not captured by the harness, so token counts and quota cost are unmeasured.

## Integration issue found and fixed

The first request was rejected before image processing. A second diagnostic request identified `MissingSessionID`: Go requires `x-opencode-session`. The adapter now sends a UUID reused throughout one client/preparation run and identifies itself honestly as `stockscreen-evidence-preparation/1.0`. A regression test first failed without the headers and then passed after the fix. This follows [Go's current client guidance](https://opencode.ai/docs/go/#where-can-i-use-it).

The successful requests used the existing production adapter and prompt: model `kimi-k2.6`, Go chat-completions endpoint, `thinking: disabled`, JSON-object response, no temperature override, maximum 2,048 output tokens, 20-second read timeout and 5-second connection timeout. No model/provider fallback or increased output limit was used. Valid parsed outputs also passed the adapter's completion-finish and typed-schema checks.

## Limits and next use

- Preserve original images and keep observations separate from literal transcription. Empty model uncertainty lists did not prevent the two commentary counting mistakes.
- This validates clean Korean/Japanese text, a small two-axis chart and an explicitly damaged excerpt. It does not validate every script, tiny/compressed screenshots, very dense tables, unlabeled chart-value estimation, or long outputs near the token cap.
- The damaged fixture is deliberately obvious and labels its corruption; it is an easy abstention check. Real damage may be harder to recognize.
- Continue with a newly collected, reviewable evidence sample. Do not treat these results as approval for theme extraction or as validation of an English translation service.

## Reader checkpoint

The supplied `/Users/admin/Documents/Work/xui-reader` checkout was clean at commit `057faae57d2a2fa1ff370be7d07eefa7bd4353fb`. Its local skill and source model now expose `image_urls`, aligned `image_captions`, `lang`, `text_source`, `text_complete` and `incomplete_text_reasons`; list reads document one targeted recovery attempt for explicitly incomplete posts. This was a checkout/contract inspection, not a fresh live X-list acquisition test. Reader code and the frozen pilot were not modified.

## Reproducible local evidence

The [fixture and response folder](../../data/xui-reader/theme-evaluation/kimi-validation-20260908/) contains the generator/live harness, original PNGs, pre-run ground truth, all six attempt records, [detailed CSV checks](../../data/xui-reader/theme-evaluation/kimi-validation-20260908/checks.csv), and [manual review](../../data/xui-reader/theme-evaluation/kimi-validation-20260908/manual_review.json). These files are ignored local artifacts; preserve the folder with this report when sharing. The harness reads only the configured Go credential without logging it. No production/user evidence was sent.

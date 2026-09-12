# Evidence gap remediation validation — September 10, 2026

The downstream implementation is complete in the evaluation worktree. The fixed-input replay and fresh Markdown/CSV packets are ready for evidence review. No theme extractions, labels or production database writes were performed. Deterministic eligibility is not evidence approval.

## Implementation and verification

Code baseline: `efe41424`; final tested code revision: `da58d1f8`. The changes add versioned translation adequacy and selection, conservative article recovery/routing, bounded image retries, opt-in language preparation v2, and supported evidence assessments with exact review annotations. Original source content and legacy result statuses remain intact. Archived assessments pin their policy and exact selection sidecar; a later policy cannot silently reinterpret them.

- Relevant evaluation and theme-policy suite: **373 passed**, with two existing dependency warnings. The starting baseline had 175 passing tests.
- All Python files changed in this turn pass Ruff. The whole evaluation directory still has 15 pre-existing diagnostics in untouched files; no unrelated lint cleanup was performed. Diff checks pass.
- Independent module, integration and final branch reviews were completed. Material findings were fixed and re-reviewed.
- Updated reader: local `89f4195`; GitHub default revision `927ae11` is its merge commit. Both have identical tree `577a94f53773f9668fe26ec8f3368c9ff4bd8c85`, verified by a temporary remote comparison. No reader files were changed.
- Reader validation: 780 tests total. 770 passed in the sandbox; 10 local Chromium fixture tests were blocked by the OS sandbox and then passed outside it. Reader Ruff passes.
- All 593 files in the September 9 packet still match their pre-implementation hashes. New captures and derivatives are in a separate ignored run directory.

## Fixed replay versus fresh collection

The fixed replay retains the September 9 bundle and reuses saved source/model evidence. It calls Kimi only for rejected or absent usable candidates and reparses saved article responses without new public downloads. Existing image outputs and three unresolved image failures were preserved, not regenerated. The fresh collection has only **one post in common** with the fixed replay; its counts are not a before/after accuracy comparison.

| Measure | Fixed input before | Fixed replay after | Fresh collection |
| --- | --- | --- | --- |
| Reader / preparation policy | Reader `1a3bfc4`; legacy preparation | Same captured inputs; quality v2 and article v2 | Reader tree above; quality v2, text-language-efficient-v2, article v2 |
| Selected posts / per required list | 99 / 50 each | Same 99 / same memberships | 99 / 50 each, plus one native Article document |
| Structurally captured X translations | 13 | Same 13 captures | 1; four capture failures and six unavailable |
| Eligible X / Go-route / review / unselected document text | Not available under comparable policy; 77 technical successes, 22 warnings | 1 X / 77 Go-route / 17 selected for review / 4 unselected | 0 X / 90 Go-route / 4 selected for review / 6 unselected |
| Unrestricted readable / complete / restricted partial article destinations | 3 / 0 / not consistently classified | 5 / 0 / 1 | 0 public bodies / 0 complete; one native Article partial capture |
| Image outputs / inputs | 30 / 33; three unresolved | 30 / 33, unchanged | 57 / 57; 44 uncertainty-bearing, 13 without reported uncertainty |
| Post completeness true / false / unknown | 20 / 0 / 79 | 20 / 0 / 79 | 25 / 0 / 74; native Article remains partial/unverified |
| Additional model calls | Historical total not measured | 12; 51.27 seconds for replay | 58 image attempts (one retry), 60 text calls |
| Additional public article requests | Historical total not measured | 0; saved responses reused | 38 original HTTP requests, then cache replay without new downloads |
| Human evidence approval / extraction | Pending / blocked | Pending / blocked | Pending / blocked |

Go-route attribution includes identity segments and reused outputs, so it is not a model-call count. Unselected records include nonlinguistic content for which no translation is needed. Reference-level warning counts and distinct destinations have different denominators. Legacy technical warnings and new per-claim severity counts are not comparable accuracy metrics.

The supported final assessments contain 16 holds, 264 review findings and 110 informational findings for the fixed replay; the fresh packet contains 5 holds, 340 review findings and 137 informational findings. These are findings, not unique posts or independently confirmed errors.

## Observed correctness and remaining gaps

The fixed Samsung regression previously captured only “Samsung.” The new selection rejects that fragment and preserves it alongside a fuller Kimi candidate containing `718만주` and `1.9조원`. A preliminary hold remains because the replacement's English does not clearly state the relationship between quantity and purchase cost. Separately, the live reader recheck captured “7.18 million shares” and “1.9 trillion won” with the unchanged original-text hash `sha256:82390db58687dbb27615c8f67acd5580a90da628a7a90d9179cf8a652a27b241`. The live recheck is not substituted into the fixed-input replay.

Four flagged fresh translation pairs were inspected against their source. One clear unsupported meaning change was found: the MRVL post's dollar rise was rendered as a resulting price (“to $30.62”). The final fresh packet has an explicit preliminary hold on that evidence. This is why quantity preservation and automated eligibility cannot replace source review. Other unresolved units, temporal phrases, and semantic uncertainty remain visible rather than being silently corrected.

The fixed synthetic language challenge contains 20 cases: four Korean, four Japanese, four Chinese, four mixed, and four English/nonlinguistic cases. One controlled run used 16 Kimi calls in 35.88 seconds. Agent comparison found 18 consistent with the fixture and two wording ambiguities (signed decrease; “write off shares”). These are not a population accuracy estimate or native-speaker certification. Source quantities remain in CJK notation by design; some names were rendered only in English despite the prompt instruction. See [language challenge](language_challenge.md).

All 57 fresh images have retained originals and structured transcription, observations and uncertainties. One `model_timeout` was retried once successfully using the same validated bytes. The image stage took 920.48 seconds; article HTTP capture took 79.20 seconds; final article/text preparation took 375.60 seconds. Stages overlapped with other work, so these durations are not a total task latency. Token usage and monetary cost were not captured and are not estimated.

Two images were manually spot-checked: the RMBS slide's listed amounts/percentages matched its transcription, and an oil-blockade chart's axes/legend matched. The chart observations contain approximate dates/curve values and omit unavailable units; they must not be treated as precise data. The other 55 images were not comprehensively inspected. Background book text and similar cosmetic uncertainty can be classified at claim level during review.

Four X translations still returned `translated_container_ambiguous` on direct rechecks. Both lists reported 13 page loads combined; nine explicit post checks reported another nine, plus one separate native Article export. The native oil Article was exported as a 17-page PDF, with 26,312 extracted characters and 16 retained figures. Its completeness remains unverified. Four historical MRVL posts were captured as one-hop supplemental context; their attachments/further links are unprepared and they do not count toward list quotas or eligible evidence.

The fresh HTTP results include 24 self-post links, four linked X posts, four non-articles, six missing bodies and two ambiguous bodies (40 reference dispositions). Browser-dependent or inaccessible bodies remain gaps. No new rendered-browser publisher collector or paywall workaround was used. The approved tools remain public HTTP/BeautifulSoup, the reader, and Kimi K2.6 through OpenCode Go.

## Reproduction and artifacts

Use the supported preparation/review/verify commands in [pilot runbook](pilot_runbook.md) and [reader handoff](reader_handoff.md). The v2 reassessment reused exact saved candidates and required **zero additional model calls**. Preserve an older assessment's policy and selection ID when validating it. A review annotation never releases the extraction gate.

Local ignored run root: `data/xui-reader/theme-evaluation/gap-remediation-20260910/`.

- `REVIEW_INDEX.md`: concise entry point linking the two final supported packets.
- `fixed-replay/review-v2/` and `fresh/review-v2/`: final Markdown, CSV, images, immutable assessments and follow-up manifests.
- `raw-captures/`: list/post records, native Article PDF/text and figures, and targeted follow-ups.
- `article-http/`: original saved HTTP responses and response manifest.
- `language-challenge-output.json`, `language-challenge-manual-review.json`, `image-spot-checks.json`: scoped validation evidence.
- `final-tests.log`, `frozen-baseline-verification.json`, and `final-review-summary.json` in each run: measured checks and final IDs.

Earlier `review`, `review-final` directories are retained pre-release drafts, not the current review packet. Source capture times remain in raw/baseline records; current article `retrieved_at` values during replay identify recovery/processing attempts, not fresh network downloads. No post, image, article or translation is approved for extraction by this report.

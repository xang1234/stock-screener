# Kimi translation validation — 2026-09-08

**Kimi K2.6 translation is implemented and suitable for preparing a sample for evidence review. It is not validated for unattended interpretation.** The configured route is OpenCode Go; no direct Moonshot credential was configured and no provider fallback was used.

The final policy (`translation-v3`) returned **20/20 nonempty translations**, retained **10/10 protected large quantities**, and retained both tested company names in their original scripts. Manual inspection found no material issue in 18 passages and two contextual issues requiring review. These are development-fixture observations by the assistant, not independent bilingual grading or a general accuracy estimate.

## What is implemented

- Separate opt-in `--allow-translation-calls` for the text stage; `--allow-model-calls` retains its image-only meaning. Both use `OPENCODE_GO_API_KEY` without logging it.
- Original segments and inferred/supplied language stay in the existing preparation artifacts. Provider/model and translation policy version are included in cache signatures. No theme extractions are generated.
- Image and translation adapters share bounded Go transport, safe errors, completion checks and client/session identification. Image prompt, token limit and timeout remain unchanged.
- Translation accepts at most 4,000 characters per segment, allows up to 4,096 output tokens and uses a 45-second read timeout with 5-second connection timeout. Missing, malformed, truncated or damaged quantity markers produce an explicit unavailable segment. No automatic retry or silent model switch.
- Large quantities written with Arabic digits plus CJK scale units are replaced by protected tokens before sending the text. Each must return exactly once; the adapter then restores the exact original quantity and unit. The model does not perform their magnitude conversion.

For example, `매출은 100억원이다` becomes English text containing **100억원**, not an automatically rescaled English amount. Unit reference: 억/億/亿 = 100 million; 만/万/萬 = 10 thousand; 조/兆 = one trillion. Currency and share suffixes stay as written. This deliberately favors fidelity over fully Anglicized quantities. Spelled-out numbers, unrecognized unit patterns and arbitrary quantities are not covered by this protection.

## What live testing found

Each round used the same ten Korean and ten Japanese synthetic passages, with expected meanings recorded before the first call. Subjects included revenue/profit/EPS, shares, debt, FX, signs, ranges, percentage points, forecasts, unconfirmed orders, names, cropped text and quoted instructions. No real articles, posts or production data were sent.

| Round | Outcome |
| --- | --- |
| `translation-v1`, 20 calls | Financial meanings were generally preserved, but a Korean company name became a generic phrase. Original quantity notation was inconsistently retained. The local unit-warning heuristic also misread Korean word endings as scale units. |
| `translation-v2`, 20 calls | Stronger wording did not reliably solve fidelity. Kimi rendered `100억원` as **100 billion won** instead of 10 billion, and `-320억원` as **-320 billion won** instead of -32 billion. Both were already marked for unit review, but prompt instructions alone were insufficient. |
| `translation-v3`, 20 calls | Protected-quantity restoration retained all 10 matched quantities. Company names retained. No missing translations; two contextual ambiguities remained below. |

The local unit checker now requires a numeric context, so endings such as `지만` and `XYZ만` do not trigger large-number-unit warnings. Valid protected quantities still trigger review. Other conservative number checks can warn on correct transformations such as `0.9%포인트` → `0.9 percentage points`, or `1株当たり` → `per share`; these warnings are not translation-error scores.

Final-round median latency was **2.41 seconds per short passage**, range **2.00–4.42 seconds**. There were 60 live calls across the three rounds. Token usage/quota cost was not measured. These short single-trial passages do not measure long-article throughput, stability or failure rates.

## Remaining contextual issues

| Case | Source and translation issue | Required handling |
| --- | --- | --- |
| `ko06` | Source describes 가상반도체's HBM supply negotiations without specifying direction. Translation adds “supply **to** 가상반도체.” | Do not infer a supplier/customer relationship from the English text alone. |
| `ko09` | Korean omits the speaker; English begins “**We** reduced next year's facility investment plan…” | Do not infer author/company attribution from the added pronoun. |

Both received automatic `success` status because structural and numerical checks cannot establish semantic correctness. The evidence-review gate remains necessary. Large quantities stay `needs_review` even when faithfully preserved. This run does not validate real corpus content, source reliability, every language, or ticker/entity resolution. It also tuned the policy on these same fixtures, so a fresh sample is needed to assess generalization.

## Evidence and next step

[Detailed CSV](../../data/xui-reader/theme-evaluation/kimi-translation-validation-20260908/checks.csv) contains every original passage, expected meaning, final translation, warnings and manual notes. [The local artifact folder](../../data/xui-reader/theme-evaluation/kimi-translation-validation-20260908/) retains the harness, original ground truth, all three rounds and the summary. It is ignored local data; retain it with this report when sharing. The expected meanings include normalized English quantities as reference; the final policy instead preserves their exact source notation.

Next: collect a fresh sample with the updated reader, recover articles with the approved HTTPX/BeautifulSoup route, prepare images/translations, and review the evidence packet before any theme extraction. The old 98-document pilot and reader repository were not modified by this validation.

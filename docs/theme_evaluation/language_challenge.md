# Language challenge set

The checked-in `language_challenge.json` is a fixed set of 20 synthetic cases for preparation review. It contains four Korean, four Japanese, four Chinese, four mixed-language, and four English or nonlinguistic cases. Every case is hand-authored, explicitly marked `synthetic`, and carries reviewed expected meaning plus quantity facts. None is attributed to a real post.

The set deliberately includes compound large-number quantities, signed percentages, currency notation, kana with kanji, company names, tickers, product digits, paragraph separators, market slang, and URL/emoji-only input. Expected meaning is the review reference; agreement between models is never treated as ground truth.

## Deterministic preparation checks

Run `test_multilingual_v2.py` to verify the preparation boundary before any live model exercise. The opt-in `text-language-efficient-v2` policy:

- accepts observed English metadata only when the source scripts do not contradict it;
- preserves the exact supplied metadata separately from the effective source language;
- treats Han-only text without provenance as `und`, never confidently as Chinese;
- translates mixed prose instead of dropping foreign portions;
- records URL/emoji-only input as nonlinguistic `zxx`, retains the original, and makes no translator call;
- packs adjacent paragraphs into segments of at most 4,000 characters while retaining every original character and separator;
- records failure only on the stored segment whose translator call failed.

The helper returns the existing `TextPreparation` model. This avoids new serialized payload fields and keeps legacy round trips unchanged. Nonlinguistic segments use the existing `unavailable` status with `nonlinguistic_content`; the `zxx` source language and warning state why no derivative exists.

## Optional live Kimi review

No live Kimi calls were run while creating this fixture, and this document makes no multilingual-accuracy claim. If a controller later authorizes a live validation, process each case exactly once with Kimi K2.6 through the existing OpenCode Go adapter. Save the raw preparation output separately from this fixture, then compare it manually with both `expected_meaning` and every `expected_quantity_facts` entry.

Record results by category with these columns:

| Category | Cases | Material meaning errors | Material quantity errors | Uncertain cases | Review status |
| --- | ---: | ---: | ---: | ---: | --- |
| Korean | 4 | not run | not run | not run | pending live review |
| Japanese | 4 | not run | not run | not run | pending live review |
| Chinese | 4 | not run | not run | not run | pending live review |
| Mixed | 4 | not run | not run | not run | pending live review |
| English/nonlinguistic | 4 | not run | not run | not run | pending live review |

A material meaning error changes or omits a company, action, polarity, qualification, attribution, or reported-versus-forecast relationship. A material quantity error changes or omits a number, sign, unit, currency, date/period, ticker, or product identifier. Mark a case uncertain when manual reviewers cannot establish faithful meaning from the fixture; do not resolve uncertainty by model agreement.

Optional real Japanese examples may be appended only as separately identified records with source URL and capture time. They must not be relabeled as synthetic or silently replace any fixed case.

## Pipeline integration

The pipeline opt-in should import `prepare_text_v2` and `preparation_cache_policy` from `multilingual_v2`. For a v2 `TextRequest`, pass the article or document's own observed language to both the request and `prepare_text_v2`, and set `policy_version=preparation_cache_policy(translator)`. Do not derive an article language from its linking post. Keep the current `language`, `target_language`, and `max_chars` request fields so the store cache signature covers all preparation inputs.

Requests that remain on legacy behavior must continue to call `prepare_text` and retain their existing policy version. The explicit v2 namespace ensures legacy segmentation and language decisions cannot reuse v2 cache entries.

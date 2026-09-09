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

No live Kimi calls were run while creating this fixture. The subsequent controlled run on September 10 used the approved Kimi K2.6 adapter once per required translation segment: 20 cases, 16 model calls, 35.88 seconds. Two English cases used identity and two nonlinguistic cases retained the original without a model call. Outputs and the agent's comparison against the synthetic reference are retained under `data/xui-reader/theme-evaluation/gap-remediation-20260910/`; they are not human evidence approval or a population accuracy estimate.

Record results by category with these columns:

| Category | Cases | Material meaning errors | Material quantity errors | Uncertain cases | Review status |
| --- | ---: | ---: | ---: | ---: | --- |
| Korean | 4 | 0 confirmed | 0 confirmed | 1 | Signed-decrease wording needs review |
| Japanese | 4 | 0 confirmed | 0 confirmed | 1 | “Write off shares” versus cancel/retire needs review |
| Chinese | 4 | 0 identified | 0 identified | 0 | Consistent with synthetic references |
| Mixed | 4 | 0 identified | 0 identified | 0 | Consistent; quoted bilingual quantities need careful counting |
| English/nonlinguistic | 4 | 0 identified | 0 identified | 0 | Two identities, two retained originals |

The Korean source itself combines a negative sign with a decrease; preserving that phrasing does not resolve its intended polarity. The Japanese output uses less precise corporate-action wording. CJK magnitude notation remains in the translated text by design. Some company names were rendered only in English despite the prompt's original-script instruction; source text remains visible, and prompt compliance is not guaranteed. The first adequacy-policy pass produced several false warnings on these outputs; these triggered deterministic regression fixes rather than new model calls. See the remediation validation report for final warning counts.

A material meaning error changes or omits a company, action, polarity, qualification, attribution, or reported-versus-forecast relationship. A material quantity error changes or omits a number, sign, unit, currency, date/period, ticker, or product identifier. Mark a case uncertain when manual reviewers cannot establish faithful meaning from the fixture; do not resolve uncertainty by model agreement.

Optional real Japanese examples may be appended only as separately identified records with source URL and capture time. They must not be relabeled as synthetic or silently replace any fixed case.

## Pipeline integration

The pipeline opt-in should import `prepare_text_v2` and `preparation_cache_policy` from `multilingual_v2`. For a v2 `TextRequest`, pass the article or document's own observed language to both the request and `prepare_text_v2`, and set `policy_version=preparation_cache_policy(translator)`. Do not derive an article language from its linking post. Keep the current `language`, `target_language`, and `max_chars` request fields so the store cache signature covers all preparation inputs.

Requests that remain on legacy behavior must continue to call `prepare_text` and retain their existing policy version. The explicit v2 namespace ensures legacy segmentation and language decisions cannot reuse v2 cache entries.

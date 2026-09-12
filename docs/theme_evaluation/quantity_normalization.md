# Deterministic quantity normalization

The evidence review now renders supported Korean, Japanese and Chinese large-number expressions in English without asking the translation model to perform arithmetic. For example, `718만주` becomes `7.18 million shares`, `₩320억원` becomes `KRW 32 billion`, and `1兆2,500億円` becomes `JPY 1.25 trillion`.

## Data and provenance

`quantity_display.normalize_quantities()` produces a versioned derivative containing the unchanged input, its SHA-256, the rendered text, exact decimal values as strings, explicit units, original expressions, and character spans. Spans use Python Unicode character offsets into that derivative's `original` field, not byte offsets or offsets into the rendered text. Original-language and translated segments are normalized independently; their spans are not interchangeable.

`quantity_review.normalized_segments()` is the shared entry point for saved text preparation results. Each newly rendered evidence packet writes `normalized-quantities.json`, tied to the bundle, preparation, assessment, binding and result IDs. It includes current and superseded attempts, labeled explicitly. The Markdown evidence report retains the original and raw translation, adds changed English displays, and shows normalization review flags.

Existing saved results, hashes, provider outputs, translation-quality assessments and selection decisions are unchanged. Normalization can be replayed entirely offline. It does not turn unavailable translation segments into translations or normalize translations whose target language is not English.

## Supported and conservative behavior

- Exact decimal arithmetic handles the 10,000, 100 million and trillion scales, including compounds in descending scale order.
- Explicit supported currency, share and count units are required. No currency conversion takes place.
- Signed values, repeated occurrences, and original wording such as “about”, “by”, “to” and negation are retained.
- URLs and handles are protected. Identifiers are not interpreted as financial amounts.
- Ambiguous units, contradictory currency markers, invalid number grouping and unsupported compound syntax remain unchanged with review flags.
- This is a finite supported grammar, not a general natural-language quantity parser. Unsupported expressions still require source review. Company names and other non-Latin text are not stripped.

## Downstream use

Read the canonical decimal string and explicit unit together, with its original span and result provenance. Consult the linked assessment and current/selected result before using a quantity. Do not treat all attempts in this sidecar as selected evidence. A normalization issue requires review; an absence of normalization issues is not a translation-quality verdict.

A quantity conversion cannot repair an incorrect relationship such as a price gain translated as a resulting price. Existing semantic holds remain in force. The benchmark still requires evidence approval before extraction.

The replay of the September 10 sample is in `data/xui-reader/theme-evaluation/gap-remediation-20260910/normalization-review.md`; this data directory is local and ignored by Git.

## Validation

The full theme-evaluation suite passed 387 tests, including 39 quantity-parser cases and a report/store preservation integration test. Independent review findings were reproduced and fixed. Offline replay verified all six requested display examples and preserved 1,212 prior evidence/store/review files byte-for-byte; both assessment IDs stayed unchanged. Existing dependency deprecation warnings remain.

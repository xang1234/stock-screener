# Evidence review for theme and development claims

Policy `claim-support-v2` reviews each non-empty candidate batch before returning
mentions or matching clusters. It uses the same configured model route with a
separate evidence-review system prompt. Empty extractions make no review call.
The evaluation route remains MiniMax M2.7 with the approved Kimi K2.6 fallback.
There is no theme catalog, blacklist of industries, web lookup or new provider.

## Decisions

- Supported theme: retain the theme. A faithful paraphrase or synonym is allowed;
  the name does not have to occur verbatim in the source.
- Inferred theme: retain it with `claim_support.theme = inferred`; source views
  display “Theme inferred from source context.”
- Unsupported theme: hold the whole candidate before clustering or ranking.
- Unsupported development with an accepted theme: retain the theme but set its
  development to null. Preserve the original claim and reason in the audit.
- Inferred development: prefix its display text with `Inference:`. A development
  attached to a profile-dependent inferred theme is conservatively labelled too.
- No candidate development: the reviewer must return `absent`.
- Invalid candidate: hold that candidate as `review_unavailable`, preserving valid
  siblings. A mixed outcome has audit status `partial`, visible in review exports.
- A timed-out batch is held; later batches may continue. If nothing can be reviewed,
  fail the extraction with the original error and its complete audit.
- Quota, rate-limit, authentication and other non-timeout provider failures stop
  subsequent calls and fail the extraction. Preserve the provider error classification
  and the complete audit. Unreviewed candidates never become successful mentions.

The reviewer checks source qualification, quantities, timing and product/industry
connections. It must not turn “may improve” into “will double,” a delay into
cancellation, or a codename into a specific technology without supplied evidence.

## Provenance validation

The reviewer sees the same first 10,000 primary-source characters as extraction,
plus the supplied grounding evidence and company reference fields. Candidate
excerpts are not evidence. Sources are labelled `primary`, `related:<input_id>` or
`company:<symbol>:<field>`. Supported and inferred components require citations.

Quoted text must occur in the supplied source. Whitespace differences such as OCR
line wraps and curly/straight quotation characters can normalize to the exact
original span, with an audit adjustment. The normalized span must still fit the
2,000-character limit. Words, numbers, units, negation and other punctuation cannot
change. A quote assigned to the wrong
*known* source may be reassigned only if its exact text occurs in exactly one
other supplied source. Unknown source IDs and invented/paraphrased quotes fail
validation. This does not perform fuzzy matching or invent a quote.

An ancillary company-name citation is removed when the event citation explicitly
names the candidate theme; it must still be a valid quotation. A name alone cannot
establish an unstated exposure. Short explicit theme references need no separate
industry definition, second source, or realized catalyst. Conflicting stock profiles
must not override explicit commodity evidence.

Exposure-bearing profile citations cannot be called direct event support. They are downgraded to
inference. Profile-based themes require an exposure-bearing field and an explicit
same-symbol cashtag in primary/related evidence for every cited issuer. A company
name alone, or an unrelated event quote, cannot establish exposure. A missing
cashtag citation can be added from the exact source text; the audit records the
adjustment. Raw reviewer verdicts remain alongside normalized decisions.

These checks verify citation existence and identity links, not semantic truth.
The model can still misjudge the meaning of evidence. The same-model reviewer is
an additional check, not independent corroboration or a guarantee of correctness.
Reference profiles are still subject to the grounding reader's provenance limits.

## Persistence and review

Migration `20260911_0040` adds nullable `claim_review` to the existing per-pipeline
content state and `claim_support` to ThemeMention. Per-pipeline storage prevents
one pipeline overwriting the other's audit. Success saves the audit with the
state transition; failures restore it after rollback. Held candidates therefore
remain inspectable even when they create no ThemeMention. Old records remain null.
The migration is included in the worktree but not applied to the running app.

Evaluation records store `claim_review` for successes and failures. The review
export adds `claim-review.csv` with source text, original candidates, decisions,
reasons, quotes and adjustments. Provider calls accumulate across extraction and
review, preserving both stages' actual model and token usage. Older frozen record
serialization and hashes remain unchanged.

## Validation and rollout limits

The full v1 evaluation used 91 frozen inputs and produced 182 outcomes, with
27 review failures out of 58 non-empty extractions. Its evidence, candidates,
responses and result archive remain unchanged. Version 2 first replays those
responses to isolate deterministic validation changes, then uses a separately
recorded live diagnostic for the prompt, batching, ticker cleanup and image context.
A replay cannot establish that a new prompt works or that a timeout is fixed.

Review requests contain at most three candidates (at most ten batches for the
existing thirty-candidate limit), with global indices preserved. There is no automatic
repair/retry loop. Only supported candidates enter clustering. Reports expose
partial outcomes and unavailable candidate counts instead of counting them as clean
successes. Smaller batches bound output size but can increase prompt tokens and
call counts; live throughput still needs measurement.

The guard does not guarantee ticker, sentiment or ranking quality. The accompanying
ticker safeguards address generic name aliases and stock/futures ambiguity; inherited
image company context is described in `grounded_extraction.md`. All changes remain
in the evaluation worktree, without applying a migration or deploying them.

The v2 validation artifacts are under
`data/xui-reader/theme-evaluation/gap-remediation-20260910/baseline-extraction/claim-support-v2/`.
The live check is selected review-only work, not a full fresh extraction benchmark.

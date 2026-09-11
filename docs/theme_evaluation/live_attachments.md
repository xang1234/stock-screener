# Live attachment evidence

Ordinary ingestion now preserves photo references and expanded article links from the XUI CLI and official X providers. They remain children of the original post. They are not independent posts, theme mentions or corroborating sources.

## What runs automatically

1. Ingestion saves up to ten attachment references per parent. Re-ingestion is idempotent, and fragment-only URL variants share an identity.
2. A Celery sweep checks the durable database queue each minute, preparing at most two attachments per delivery. Source permissions are checked when claiming work and again before publishing the result. Disabled sources are deferred without blocking later queue entries.
3. Images use Kimi K2.6 on OpenCode Go for transcription and factual observations. External HTML articles use the existing bounded public HTTP/BeautifulSoup recovery. Foreign-language evidence uses Kimi translation and deterministic quantity normalization. Captured text, source hashes, processing policy and warnings are retained alongside prepared text.
4. Default theme extraction receives related evidence plus verified company context. Social extraction pins evidence into new work inputs and includes it in the input hash. Attachment claims require an evidence ID and a verbatim source quotation; images and translations remain interpretations of captured evidence.
5. Successful preparation queues an update for the affected posts. Legacy mentions are replaced transactionally, preserving previous results when extraction fails. New social generations include prepared evidence for retained parents even when the provider does not return those posts again. Saved social runs and replay inputs remain unchanged.
6. Source views show attachment links and pending, ready, partial or unavailable state. Truncation, translation problems and access failures remain visible.

## Activation

This change adds Alembic revision `20260911_0041`, following the existing development/grounding/claim-review migrations. Apply the branch migrations before starting the updated application or workers. Then configure the worker environment:

```dotenv
LIVE_ATTACHMENT_PREPARATION_ENABLED=true
OPENCODE_GO_API_KEY=<approved OpenCode Go key>
```

Both variables are passed through the shared Docker worker environment. The flag defaults to false; a missing key leaves preparation blocked without consuming attempts. Restart the updated application, Celery workers and Beat after migration/configuration. The `social_ingestion` queue must have a worker. Theme extraction continues to use its configured model route; social extraction continues through its existing budget wrapper.

The implementation session did not apply migrations, change live configuration, restart services or call external model services.

## Bounds and recovery

- At most ten attachment references per post, 10 MB per image, and the existing image pixel validation.
- Article/image source text is bounded at 10,000 characters. The combined extraction grounding budget remains 6,000 characters. Longer evidence is explicitly partial/truncated. Full article chunking was intentionally excluded.
- A successful attachment record is reused on repeated ingestion. A URL on a different post is fetched again: equal URLs do not prove equal content. Identical downloaded content is deduplicated within the parent grounding snapshot.
- Preparation uses expiring leases with ownership checks; an old worker cannot overwrite a newer result. Retryable fetch/model failures and partial translations receive bounded backoff, with at most three preparation attempts. Unavailable translation segments keep their original language; partial evidence remains visible while a translation retry is pending.
- Authentication failures, unsafe addresses, inaccessible content and malformed images remain explicit failures. They are not replaced with guessed article text or chart readings.
- Pending database rows survive a missed queue delivery. A bounded recovery sweep revisits stale extraction revisions and rotates its selection; newly prepared posts take priority. Social's normal refresh also provides a recovery path if an attachment-triggered refresh dispatch is lost.

## Current limits

Public HTML recovery cannot reliably open paywalled, login-only, JavaScript-only or authenticated X Articles. These remain visible as partial/unavailable evidence and need a separate capture path. Automatic PDF body recovery is not included. Existing benchmark PDF captures remain untouched.

This connects preparation to ordinary ingestion; it does not create benchmark reference labels or validate novelty/ranking quality. Source status is preparation coverage, not a guarantee that every model interpretation is correct.

## Verification

Focused tests cover real provider attachment fields, idempotent persistence, multilingual/image preparation, late evidence, targeted dispatch, disabled sources, stale leases, bounded retries, duplicate evidence, historical social snapshots and transaction-safe mention replacement. The migration is exercised against an isolated SQLite database. Frontend source-status tests and a production build are included in verification.

Validation recorded on 2026-09-11: 807 tests passed in the related backend suite, plus the final 14-case live attachment service run including two new queue-recovery cases. The frontend attachment-status test and production build passed. Targeted lint and diff whitespace checks passed.

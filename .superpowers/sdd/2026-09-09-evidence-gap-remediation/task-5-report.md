# Task 5 report — article recovery and reference routing

## Implemented interfaces

- `parse_article(raw: bytes, final_url: str) -> ArticleRecovery` remains intact. The v2 parser prefers unique JSON-LD, then inspected Daum and Oracle body regions, then explicit generic article-body regions and a unique `article`/`main`. It removes structural controls, preserves quotations and paragraphs containing words such as “subscribe,” never marks parsed HTTP content complete, and records `body_sha256`.
- `ArticleRecovery.completeness_basis` is optional provenance for a reviewer-supplied completeness claim. Both new optional fields are omitted when absent, so loading and serializing legacy records preserves their exact shape.
- `classify_reference(url, *, parent_post_id)` returns `article_candidate`, `linked_x_post`, `same_x_post`, or `not_article` from URL structure. X Article URLs remain article candidates; X post URLs are normalized by numeric post ID.
- `route_reference`, `normalize_destination`, `destination_identity`, and `group_article_destinations` preserve each reference and source post while grouping exact destinations and conservative same-origin canonical aliases. Only known tracking parameters are removed.
- `build_linked_post_manifest(routes, *, base_post_ids, max_posts=20)` emits normalized post IDs plus a disposition for every linked-post reference. It globally deduplicates, excludes base posts, enforces one hop, and caps the pilot at 20 posts.
- `recover_references(references, *, fetcher=fetch_public, parser=parse_article)` resolves redirects, routes final URLs, reuses request and destination work, and exposes per-reference assessment codes. Distinct short links are fetched independently; references ending at one destination share one capture.

## Pipeline and CLI integration required

1. In the article stage, exclude follow-ups that already have `article_id`. This preserves imported native X Articles and their existing partial-completeness gap instead of fetching X HTML again. Also preserve source-body language when deriving any new article document.
2. Build `ReferenceInput` values from remaining reviewable follow-ups, using the handoff URL override when present, and call `recover_references`. Keep every returned route/reference binding even when several point to one capture.
3. Save each `ArticleCapture.response.body` once, create the article request with the raw asset hash and `policy_version="article-v2"`, and save one result per capture identity. Bind all matching reference IDs to that result ID. Never serve an `article-v1` cached result for an `article-v2` request; stored raw bytes may be reparsed into a new immutable result.
4. Map `ReferenceAssessment.code` directly into review output. The helper distinguishes `body_missing`, `body_ambiguous`, `access_restricted`, `http_401`, `http_403`, `render_required`, `not_article`, and `linked_x_post`; do not append a browser requirement to every partial capture.
5. Serialize the linked-post manifest and send only its `post_ids` through xui's existing POST read/import boundary. Do not parse those posts as HTML and do not include them in either list's 50-post quota.
6. For browser imports, require viewed `final_url`, explicit `retrieved_at`, exact text/hash, and `match_basis`. Verify `body_sha256` when supplied. Reject `capture_status="full"` unless a nonempty reviewer-supplied `completeness_basis` is present; the deterministic parser always leaves status partial.

## Verification

- Red run: 17 expected failures across new recovery/routing behaviors before implementation.
- Follow-up red run: native X Article classification failed before its routing branch was added.
- Focused command: `python -m pytest -q test_article_recovery.py test_reference_routing.py test_article_intake.py test_preparation_cli.py test_preparation_regressions.py` with the required SQLite environment — 48 passed.
- Python compilation and `git diff --check` passed for the owned modules and tests.
- Saved baseline replay: Daum bodies reduced to clean owned text (1,039 and 1,378 characters); Oracle announcement recovered 1,089 characters; Substack retained 9,355 readable characters with `access_restricted`; Chosun remained empty with `body_missing` and `render_required`.

## Limitations

- No live browser or renderer was used. Chosun and ZDNet stay render-required when their public response contains no body.
- Readable HTTP text remains partial until a reviewer supplies completeness evidence.
- `preparation_pipeline.py` and `preparation_cli.py` were intentionally not edited; the integration steps above are required in the shared integration task.

Commit: `fix: recover and classify article evidence by destination`

## Controller review remediation

- Canonical aliases now require positive agreement on the leaf article identity and identical retained query parameters. A canonical home page, a query-dropping alias, or another article that merely shares category/date path segments cannot redirect the destination cache.
- Owned-region text is traversed once in DOM order after explicit controls are removed. Direct text beside block elements, nested-list lead text, inline punctuation, and `figcaption` text are retained without duplicating descendant content.
- Added direct routing/parser regressions and a stage-level cross-body regression. Focused article and routing suites pass 37 tests.
- Saved baseline replay after the fix: Daum bodies contain 1,166 and 1,403 characters, Oracle contains 1,087, and Substack contains 9,425 with `access_restricted`; all remain partial and completeness-unverified.

Follow-up commit: `fix: preserve article identity and owned source text`

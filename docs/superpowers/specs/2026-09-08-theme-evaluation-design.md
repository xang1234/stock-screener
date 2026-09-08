# Theme detection evaluation: offline pilot and fresh-history capture

Date: 2026-09-08

Status: proposed design for user review

Worktree branch: `feat/theme-detection-evaluation`

Origin: idea #5 from the research recommendations kept in the main checkout at `docs/research/theme_detection_improvement_ideas_2026-09-08.md`.

## Purpose

Compare theme detector versions on the relevance, distinctness, and timeliness of the ranked themes users see. The investment horizon is several days to several weeks. Reward earlier useful discoveries while constraining irrelevant results and tracking missed opportunities.

The current application database is new and empty, as confirmed by the user. Existing historical content, scans, and rankings are not prerequisites that can be assumed. The first milestone is source selection and dataset construction, followed by reviewed labels and offline comparisons. Fresh-history capture establishes stronger future datasets.

## Accepted decisions

1. Deliver an offline benchmark first; continuous quality monitoring is outside the first release.
2. Evaluate ranked theme results primarily, using extraction and stock-membership diagnostics to explain errors.
3. Use real content plus separately reported controlled examples.
4. AI proposes reference labels with contemporaneous evidence; the user reviews them. Preserve uncertainty.
5. Judge research usefulness over days to weeks, independently of subsequent investment returns.
6. Compare speed with a noise constraint and recall reporting, not one opaque combined score.
7. Freeze article extractions initially. Replay downstream grouping, promotion, and ranking; do not claim extraction-model improvements.
8. Deliver Markdown reports with examples/evidence and CSV details.
9. With an empty database, start with a curated real-content pilot and establish capture of new history alongside it. Historical material collected now supports a retrospective simulation, not proof of actual historical system latency.
10. Both user-specified X lists are required sources: `1986290701492232693` and `1522014550211457024`. Use the local xui-reader skill to access them. When a post references an investment-related article, look up the article as well and preserve the evidence relationship.

## Approaches and choice

| Approach | Benefit | Limitation |
| --- | --- | --- |
| Curated historical material only | Immediate labeling and regression work | Collection gaps and hindsight prevent strong operational timing claims |
| Wait for fresh history only | Actual observation and output timestamps | Delays development and may initially cover few theme episodes |
| Curated pilot plus fresh-history capture | Immediate development with a route to stronger evaluation | Requires explicit separation of provenance and report claims |

Use the third approach. Implement acquisition and frozen files before a general replay runner. Do not build a dashboard, separate live crawler, or scheduled quality-monitoring service for this release.

## Initial scope and defaults

- Start with English-language content and US theme analytics, matching the reviewed ranking implementation's market scope.
- Run the existing technical and fundamental pipelines separately and report separate results. Do not combine their rankings or migrate their identities.
- Use the current ungrouped L2 theme ranking as the first evaluation surface. Grouped L1 presentation and the separate social-signal ranking engine are outside the first comparison surface.
- Review the top 10 results at predetermined daily checkpoints. Report delay at checkpoint resolution, not as minute-level latency.
- Begin with a pilot targeting approximately 100–200 real items and 10–20 reviewed theme episodes, contingent on source coverage. These are workload targets, not statistical validity claims or quotas to fill by inventing themes.
- Retain ordinary background items, repeated coverage, and no-theme examples from the selected streams. Selection must precede judging detector success.
- Include preceding content to warm up theme state. Target at least 30 calendar days of content context when available; include any longer price history required by the existing calculations. State warm-up gaps explicitly.

These defaults are part of this proposed design rather than additional user-confirmed answers.

## Milestone 1: select sources and construct the pilot

### Source intake

Required sources:

| Source ID | URL | Access |
| --- | --- | --- |
| `x-list:1986290701492232693` | https://x.com/i/lists/1986290701492232693 | Local authenticated xui-reader |
| `x-list:1522014550211457024` | https://x.com/i/lists/1522014550211457024 | Local authenticated xui-reader |

Follow the user-specified skill at `/Users/admin/Documents/Work/xui-reader/skills/xui-reader/SKILL.md`, preferably through its `scripts/xui_read.py` wrapper. Check authentication before reads. Use the `default` profile and `prompt` login policy for interactive chat collection. Never inspect session storage, expose cookies, or mutate X state. Unattended collection, if separately configured later, must fail closed on authentication/challenge errors.

Collect bounded chronological samples from both lists and record requested limits, returned counts, observation times, oldest/newest post times, source errors, and coverage limitations. A list read is not proof of a complete historical window. Both sources must have an explicit collection outcome; a failure makes required-source coverage incomplete and cannot be silently replaced by news feeds. Successful reads of the other source and document preparation may continue independently.

Deduplicate post bodies by tweet ID while preserving membership in both lists and the observation time for each membership. Record authors as distinct from list IDs: two lists carrying the same post are not two independent observations. Preserve sampled non-investment posts as background/negative examples rather than filtering them out of the dataset entirely.

Also audit the repository's existing public news/RSS candidates: Yahoo Finance, MarketWatch, Doomberg, and SentimenTrader. These are optional supplements, not replacements for the required lists. Their presence in configuration does not establish accessibility, full-text availability, or historical completeness. Record those facts before selecting a corpus.

Use openly accessible archive pages or feeds where available. An RSS feed may expose only recent entries or summaries. Preserve what is actually accessible; do not infer full archives from a lookback parameter or replace a captured summary with a later full article without recording a distinct version.

Select a small accessible source set and contiguous periods before extraction and labeling. Record selection rules, exclusions, actual enumerated items, and coverage limitations in the manifest. Do not choose periods or companies solely because their themes later succeeded. If a historical window cannot be enumerated adequately, retain it only as a case study or controlled retrospective sample; do not report it as a representative stream benchmark.

Company or regulatory original documents linked by articles may be collected as evidence. Separate documents used by the detector from supplemental reviewer evidence. Supplemental evidence cannot justify an earlier detector-eligible timestamp unless it was included in the frozen input stream by that cutoff.

An authenticated local X session is a prerequisite for complete required-source collection. A paid historical archive is not required. Social content is an input to this theme benchmark; including it does not bring the separate social-signal ranking engine into the evaluation surface.

### Article follow-up from list posts

For sampled posts that discuss an investment-relevant development and reference an article, retrieve the referenced article in addition to the post. Use the resolved outbound URL, or a targeted title/author/publication search when no usable URL is exposed. Verify that a search result is the referenced piece; record unresolved references rather than substituting a merely related article.

Use a broad relevance screen covering economic mechanisms, industry developments, company operating evidence, regulation, and investable risks. Do not require an already assigned theme or a successful stock outcome. Record the follow-up decision, reason, and classifier/prompt version so filtering can be audited and low-confidence skips sampled during review.

Capture the article as a separate content record with canonical/final URL, title, publisher, author if available, publication/update times, actual retrieval time, accessible text, and content hash. Save a `references_article` relationship from each referring post. Resolve each unique article once per captured version, preserve all referring posts, and distinguish repost attention from independent evidence.

Follow direct article references; do not recursively crawl every link within an article. Record fetch outcomes such as success, partial text, unavailable, paywalled, or unresolved. Partial text is not full text; failed follow-up remains a visible coverage gap. Use accessible content without circumventing access restrictions.

For native long-form X Articles, use the skill's supported `article-pdf` export with a deterministic local output path. Preserve the returned status, title, warnings, and PDF hash. A normal post read exposes only the post/Article flag, not necessarily the full Article body. Apply the skill's error contract instead of guessing missing content.

Freeze post and article extractions independently after collection. Admission during observed replay uses actual availability and eligibility; an article fetched after its referring post cannot affect an earlier checkpoint just because its publication date is older. Retrospective simulations retain actual retrieval dates and declare their separate simulated-availability rule, including article-follow-up assumptions.

### Immutable input records

Store local versioned records for:

- Source ID/type, URL, source eligibility, selection rule, and source configuration revision.
- X tweet ID, author handle, post URL, all observed list memberships, and post-to-article reference edges with follow-up decisions/outcomes.
- Content ID, title, exact captured text, original URL, publication time, actual retrieval time, and content hash.
- Availability mode: `observed` or `simulated`, with the simulation rule recorded explicitly.
- Extraction ID, input content hash, pipeline, full parsed extraction output including an explicit empty result, and status distinguishing failure from no themes.
- Extraction provider/model, prompt and parser versions, settings, and generation time. Unknown historic versions remain unknown.
- Supporting market/screener inputs, their effective dates, actual observation/completion times where known, and coverage.

Generate initial extractions once using the application's supported provider path, then freeze them. Save them before mutable cluster assignment. Preserve existing ticker-cleaning/identity dependencies in the extraction provenance. Routine comparisons do not call an extraction provider.

Newly generated extractions of older content are labeled accordingly. They may reflect current model knowledge and cannot establish historical extraction performance.

Use JSON/JSONL for structured local inputs and labels, Markdown for review packets and reports, and CSV for exported result rows. Keep acquired article bodies and provider outputs in ignored local data storage by default; commit schemas, manifests that are safe to share, and controlled fixtures. Do not overwrite a frozen dataset: corrections create a new version.

### Acceptance for milestone 1

Deliver a source-access/coverage inventory, a frozen pilot manifest, a small acquired sample with timestamps and hashes, fixed extractions, and review packets. If sources are insufficient, the inventory names the missing coverage; do not fabricate a complete corpus or silently substitute synthetic content.

## Milestone 2: human-reviewed reference answers

Assign reference episode IDs independently of detector cluster IDs and names. For each episode label:

- The investment mechanism and why it deserves or does not deserve research.
- Supporting content and passages available at each reviewed checkpoint.
- The earliest checkpoint at which the input stream supports research usefulness.
- Supported stock associations, unsupported associations, and uncertain associations.
- Equivalence/subset/distinctness relationships needed to judge duplicates and overmerges.
- Reviewer decision, confidence/uncertainty, and label version.

Review the underlying content, including items with empty or failed extractions, to identify missed opportunities. Uncertain and unreviewed cases are reported explicitly. Do not silently treat them as correct, incorrect, or nonexistent.

Reference stock judgments may include supplemental evidence, but that evidence retains its own availability and must not leak into detector replay. Reviewers see chronological evidence packets; later outcomes are withheld from the contemporaneous relevance question where practicable.

Freeze a development set and a later holdout before tuning. Keep syndicated event families together, or exclude boundary-spanning families from scored holdout comparisons. Do not turn user labels into runtime detector hints.

## Milestone 3: deterministic downstream replay

Use a dedicated evaluation database isolated from the running application. Default to an explicitly configured disposable PostgreSQL database for production-service parity. Local file-based scoring and pure unit tests can run without it. Never reset or seed the production database for evaluation.

Each version starts from the same frozen inputs and warm-up policy, in a separate clean state. Replay content and extraction eligibility in availability order. Current aliases, cluster assignments, stock memberships, and source activity cannot seed a historical cutoff.

Provide an evaluation clock through narrow service interfaces. Current-time calls in extraction matching, lifecycle defaults, and emerging selection need controlled behavior. Keep normal application defaults unchanged. Do not implement an independent approximate copy of the ranking algorithm just to make the benchmark run.

At each daily checkpoint:

1. Admit only content, frozen extractions, eligibility events, and supporting inputs available by the cutoff.
2. Run the selected version's grouping and candidate/lifecycle decisions.
3. Calculate ranking metrics using historical constituent state and eligible market/screener inputs.
4. Save immutable ranked results and the inputs/provenance needed to explain each row.
5. Match results to the independently reviewed reference episodes and compute metrics.

`calculate_screener_metrics` currently selects the latest completed scan. Restrict replay to scans available by the cutoff. Current scan output is not a substitute. Historical daily prices retrieved today also need declared adjustment/revision assumptions; do not present them as observed point-in-time data without evidence.

A missing market or screener input makes a full-ranking checkpoint unavailable. Report coverage and retain any valid text/grouping diagnostics separately. Do not manufacture neutral prices, silently change score weights, or use the baseline's per-theme score components for a candidate with different constituent membership.

Pin embedding models, configuration, tie-breaking, and seeds where applicable. Any downstream LLM decisions must use a versioned exact-input response cache or a separately prepared frozen run. A cache miss is explicit; never reuse a response for different inputs or silently make a live call during a claimed repeatable comparison.

## Metrics and comparison policy

Report by pipeline and dataset provenance mode. Real streams, retrospective case studies, controlled fixtures, and fresh observed histories must not share one aggregate quality score.

| Metric | Definition/constraint |
| --- | --- |
| Top-result relevance | Relevant judged rows divided by judged rows within top 10; show judged/unjudged coverage and actual list length |
| Unique useful slots | Distinct relevant reference episodes represented in the top 10; duplicates consume slots |
| Noise | Irrelevant judged rows per checkpoint, plus uncertainty bounds when judgments are incomplete |
| Episode recall | Supported reference opportunities surfaced within the scored window; denominator defined independently of detector output |
| Downstream-eligible recall | Recall restricted to opportunities represented in the frozen extractions, to isolate the chosen replay boundary |
| Discovery delay | Checkpoints from first supported research usefulness to first relevant top-10 appearance; pair with recall and censored misses |
| Duplicate/overmerge errors | Repeated representation of one episode or conflation of distinct mechanisms |
| Stock support | Supported judged stock associations divided by judged associations, with coverage and uncertainty |

Do not reward an empty result list as perfect precision. Report actual list length, unique useful slots, and recall together. Do not count missing episodes as zero delay or drop them from the report. Surface premature unsupported appearances separately from useful early discovery.

For the pilot, report paired deltas and individual examples without a universal winner. After the reviewed baseline, choose a noise ceiling and recall floor and freeze them before testing a held-out candidate. Sample counts and uncertainty accompany comparisons. No statistically supported broad improvement claim follows from a small pilot alone.

Runtime and provider usage may be logged for engineering purposes. Simulated content timing does not measure historical operational latency. The fixed-extraction mode does not evaluate extraction-model improvements.

## Fresh-history capture

Extend existing ingestion/extraction/publication points to preserve evaluation records; do not add a separate polling or monitoring system. The existing application performs collection when its normal ingestion is enabled. Capture does not bootstrap external source credentials or imply a background job has been started.

Capture exact incoming text/version, publication and observation times, pipeline eligibility history, parsed extraction output including empty results/failures, model/prompt/parser versions, source settings, ranking inputs, constituent state, and the published ranking output/revision.

Reuse existing `ContentItem`, eligibility records, scans, provider snapshots, and UI snapshots where they preserve the required facts. Add the missing extraction/provenance and evaluation-bundle records rather than treating mutable `ThemeMention` assignments as an immutable history.

Write completed evaluation bundles atomically with hashes and versioned manifests. Capture failures must leave the application operational and visibly mark the evaluation coverage gap. Ordinary feature enable/disable or source-policy changes must be included in provenance, rather than retrospectively using today's source state.

Observed bundles eventually replace simulated availability assumptions for stronger offline replay. They can measure source delay and actual first-display times independently of downstream simulated comparisons.

## Reports

Markdown report:

1. Dataset mode, source coverage, label coverage, warm-up, and limitations.
2. Baseline/candidate code, config, extraction, model/cache, and label versions.
3. Per-pipeline metrics with sample counts and paired differences.
4. Earlier useful discoveries, misses, new irrelevant results, duplicates/overmerges, and stock errors.
5. Evidence-linked examples and unresolved judgments.

CSV files contain checkpoint ranking rows and judgments, episode detection/miss records, stock-association judgments, and input/coverage failures. Include stable IDs linking rows to local evidence. Escape CSV values safely for spreadsheet consumption.

## Validation and completion criteria

- Identical frozen inputs and versions produce identical substantive results, excluding run timestamps/IDs.
- Future-dated content, aliases, eligibility, scans, and membership cannot affect earlier checkpoints.
- Empty extraction, provider failure, missing inputs, and uncertain labels remain distinct.
- Duplicate names do not create extra true discoveries; renamed clusters do not lose identity credit.
- A detector that suppresses difficult results is exposed by recall and useful-slot metrics.
- Controlled examples cover the first-week velocity problem, repeated coverage, distinct mechanisms, wrong stocks, and late/unsupported appearances.
- Report aggregates reconcile to CSV rows and explicitly identify unscored/incomplete checkpoints.
- The evaluated baseline matches the application's behavior with the same admissible inputs.
- Existing focused theme regression tests continue to pass after any clock/as-of changes.
- Capture records actual availability and fails visibly without disrupting normal ingestion or publication.
- Both required X lists appear in the manifest with explicit outcomes; failed required reads prevent a complete-coverage claim.
- The same post in both lists and multiple posts referencing the same article preserve provenance without becoming independent corroboration.
- Investment-related article references are followed or receive an explicit failure/skip reason; later article retrieval cannot leak into earlier observed checkpoints.

Completion of the initial engineering pilot requires a functioning acquisition/freeze/review/report path and reproducible comparisons on an adequately supported sample. Full chronological ranking evaluation remains explicitly unavailable for periods missing its required inputs. User-reviewed labels and observed-history accumulation are external dependencies, not results that implementation can invent.

## Implementation sequence

1. Source-access inventory, pilot manifest, and bounded real-content acquisition.
2. Frozen extraction/provenance format and user review packets.
3. Label ingestion, metric definitions, and report generation with controlled fixtures.
4. Fresh-history capture through existing application processing paths.
5. Controlled-clock downstream replay with coverage gates and baseline parity checks.
6. Reviewed pilot comparison, then held-out/prospective evaluation as sufficient history becomes available.

No theme-discovery policy fix, extraction-model upgrade, identity migration, automatic trading action, dashboard, or recurring quality-monitoring automation is part of this design.

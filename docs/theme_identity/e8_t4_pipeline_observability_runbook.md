# E8-T4 Theme Pipeline Observability Runbook

This runbook maps observability alerts to likely causes and first-response actions for
theme extraction and consolidation pipeline health.

## Parse Failure Rate

Alert key: `parse_failure_rate_high`

Primary checks:
- Verify model/provider response format stability in recent extraction runs.
- Inspect `content_item_pipeline_state.error_code` distribution for parse/schema failures.
- Confirm extraction prompt/model settings were not changed without validation.

First actions:
- Roll back to last known-good extraction model config.
- Re-run failed retryable batch with reduced limit to sample errors quickly.

## Processed-Without-Mentions Ratio

Alert key: `processed_without_mentions_ratio_high`

Primary checks:
- Inspect recent processed items for empty extraction payloads.
- Verify parse errors are being classified correctly and not masked.
- Check source mix shifts (low-signal sources can inflate no-mention outcomes).

First actions:
- Run a targeted extraction batch on recent high-confidence sources.
- Tighten failure classification to keep ambiguous parse outcomes retryable.

## New-Cluster Rate

Alert key: `new_cluster_rate_high`

Primary checks:
- Compare `match_method` mix against prior 7-day baseline.
- Review alias coverage and recent alias backfill freshness.
- Check threshold override changes for matcher policies.

First actions:
- Run alias backfill and re-evaluate match telemetry.
- Adjust matcher thresholds only after sampled mention review.

## Retry Queue Growth

Alert key: `retryable_growth_high`

Primary checks:
- Compare retryable count last 24h vs previous 24h.
- Identify top `error_code` contributors in retryable queue.
- Validate upstream provider rate limits and timeout behavior.

First actions:
- Reduce extraction concurrency and batch sizes temporarily.
- Prioritize remediation for dominant retryable error code family.

## Merge Review Backlog

Alert key: `merge_pending_backlog_high`

Primary checks:
- Inspect pending merge suggestion volume and age distribution.
- Validate reviewer throughput and queue closure in recent waves.
- Check for threshold shifts inflating suggestion creation.

First actions:
- Schedule a bounded manual-review wave.
- Tighten similarity/LLM confidence thresholds if false positives dominate.

## Merge Precision Proxy

Alert key: `merge_precision_proxy_low`

Primary checks:
- Inspect approved/rejected/auto-merged distribution in recent reviewed suggestions.
- Review LLM merge-verification outcomes for relationship-type drift.
- Verify embedding freshness campaign gates remain green.

First actions:
- Pause aggressive auto-merge waves.
- Recalibrate merge suggestion thresholds and re-run dry-run planning.

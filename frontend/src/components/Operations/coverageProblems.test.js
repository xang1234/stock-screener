import { describe, expect, it } from 'vitest';

import { marketCoverage } from './coverageProblems';

const NOW = Date.parse('2026-10-14T12:00:00Z');
const hoursAgo = (h) => new Date(NOW - h * 3600 * 1000).toISOString();

const queues = (market, consumers) => ['market_jobs', 'data_fetch', 'user_scans'].map((family) => ({
  queue: `${family}_${market.toLowerCase()}`,
  depth: 0,
  consumers: consumers[family],
}));

const pipeline = (market, run, extra = {}) => ({
  name: `daily-market-pipeline-${market.toLowerCase()}`,
  market,
  is_enabled: true,
  last_run: run,
  ...extra,
});

const ALL_SERVED = { market_jobs: 1, data_fetch: 1, user_scans: 1 };

describe('marketCoverage', () => {
  it('reads a fully wired market as OK', () => {
    const row = marketCoverage({
      market: 'US',
      enabled: true,
      queues: queues('US', ALL_SERVED),
      pipelineTask: pipeline('US', { status: 'dispatched', started_at: hoursAgo(16), triggered_by: 'auto' }),
      now: NOW,
    });

    expect(row.workers).toEqual({ market_jobs: 1, data_fetch: 1, user_scans: 1 });
    expect(row.problems).toEqual([]);
    expect(row.notes).toEqual([]);
    expect(row.pipelineNeverRan).toBe(false);
  });

  it('names each missing worker for an enabled market', () => {
    const row = marketCoverage({
      market: 'JP',
      enabled: true,
      queues: queues('JP', { market_jobs: 0, data_fetch: 0, user_scans: 0 }),
      pipelineTask: pipeline('JP', null),
      now: NOW,
    });

    expect(row.problems).toEqual([
      'No worker runs its pipeline stages (market_jobs_jp).',
      'No worker fetches its prices (data_fetch_jp).',
      'No worker runs its manual scans (user_scans_jp).',
    ]);
  });

  it('does not alarm on a pipeline that has never run: a fresh install or new market waits for its first weekday', () => {
    const row = marketCoverage({
      market: 'HK',
      enabled: true,
      queues: queues('HK', ALL_SERVED),
      pipelineTask: pipeline('HK', null),
      now: NOW,
    });

    expect(row.problems).toEqual([]);
    expect(row.pipelineNeverRan).toBe(true);
  });

  it('flags a pipeline that stopped for over four days, but not across a weekend', () => {
    const base = { market: 'HK', enabled: true, queues: queues('HK', ALL_SERVED), now: NOW };

    expect(marketCoverage({ ...base, pipelineTask: pipeline('HK', { status: 'completed', started_at: hoursAgo(72) }) }).problems)
      .toEqual([]);
    expect(marketCoverage({ ...base, pipelineTask: pipeline('HK', { status: 'completed', started_at: hoursAgo(5 * 24) }) }).problems)
      .toEqual(["No daily pipeline run has started in 5 days; check Beat and the market's pipeline worker."]);
  });

  it('does not let a recent manual run vouch for Beat', () => {
    const base = { market: 'HK', enabled: true, queues: queues('HK', ALL_SERVED), now: NOW };

    const manual = marketCoverage({
      ...base,
      pipelineTask: pipeline('HK', { status: 'dispatched', started_at: hoursAgo(2), triggered_by: 'manual' }),
    });
    expect(manual.problems).toEqual([]);
    expect(manual.pipelineScheduleUnconfirmed).toBe(true);

    // last_run is the latest of any trigger: an old manual run means no run at all since.
    const stale = marketCoverage({
      ...base,
      pipelineTask: pipeline('HK', { status: 'completed', started_at: hoursAgo(6 * 24), triggered_by: 'manual' }),
    });
    expect(stale.problems).toEqual(["No daily pipeline run has started in 6 days; check Beat and the market's pipeline worker."]);
  });

  it('notes, without alarming, that scheduled runs are turned off', () => {
    const row = marketCoverage({
      market: 'US',
      enabled: true,
      queues: queues('US', ALL_SERVED),
      pipelineTask: pipeline('US', { status: 'completed', started_at: hoursAgo(10 * 24) }, { is_enabled: false }),
      now: NOW,
    });

    expect(row.problems).toEqual([]);
    expect(row.notes).toEqual(['Scheduled pipeline runs are off (CACHE_WARMUP_ENABLED).']);
  });

  it('says when market workers serve a market the app has not enabled', () => {
    const row = marketCoverage({
      market: 'CN',
      enabled: false,
      queues: queues('CN', ALL_SERVED),
      pipelineTask: pipeline('CN', { status: 'skipped', started_at: hoursAgo(4) }),
      now: NOW,
    });

    expect(row.problems).toEqual(['Workers serve it, but it is not enabled in the app.']);
  });

  it('ignores the shared price worker, which start_celery.sh subscribes to every market', () => {
    const row = marketCoverage({
      market: 'HK',
      enabled: false,
      queues: queues('HK', { market_jobs: 0, data_fetch: 1, user_scans: 0 }),
      now: NOW,
    });

    expect(row.problems).toEqual([]);
  });

  it('does not guess when worker counts are unknown', () => {
    const row = marketCoverage({
      market: 'JP',
      enabled: true,
      queues: queues('JP', { market_jobs: null, data_fetch: null, user_scans: null }),
      pipelineTask: undefined,
      now: NOW,
    });

    expect(row.workers).toEqual({ market_jobs: null, data_fetch: null, user_scans: null });
    expect(row.problems).toEqual([]);
  });
});

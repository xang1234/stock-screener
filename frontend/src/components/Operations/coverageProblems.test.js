import { describe, expect, it } from 'vitest';

import { marketCoverage } from './coverageProblems';

const NOW = Date.parse('2026-10-14T12:00:00Z');
const hoursAgo = (h) => new Date(NOW - h * 3600 * 1000).toISOString();

const queues = (market, consumers) => ['market_jobs', 'data_fetch', 'user_scans'].map((family) => ({
  queue: `${family}_${market.toLowerCase()}`,
  depth: 0,
  consumers: consumers[family],
}));

const pipeline = (market, run) => ({ name: `daily-market-pipeline-${market.toLowerCase()}`, market, last_run: run });

describe('marketCoverage', () => {
  it('reads a fully wired market as OK', () => {
    const row = marketCoverage({
      market: 'US',
      enabled: true,
      queues: queues('US', { market_jobs: 1, data_fetch: 1, user_scans: 1 }),
      pipelineTask: pipeline('US', { status: 'dispatched', started_at: hoursAgo(16), triggered_by: 'auto' }),
      now: NOW,
    });

    expect(row.workers).toEqual({ market_jobs: 1, data_fetch: 1, user_scans: 1 });
    expect(row.problems).toEqual([]);
  });

  it('names each missing worker and a pipeline Beat never ran for an enabled market', () => {
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
      'No daily pipeline run recorded; Beat may not schedule it.',
    ]);
  });

  it('flags a pipeline with no run in over four days, but not across a weekend', () => {
    const base = { market: 'HK', enabled: true, queues: queues('HK', { market_jobs: 1, data_fetch: 1, user_scans: 1 }), now: NOW };

    expect(marketCoverage({ ...base, pipelineTask: pipeline('HK', { status: 'completed', started_at: hoursAgo(72) }) }).problems)
      .toEqual([]);
    expect(marketCoverage({ ...base, pipelineTask: pipeline('HK', { status: 'completed', started_at: hoursAgo(5 * 24) }) }).problems)
      .toEqual(['No daily pipeline run in 5 days; Beat may not schedule it.']);
  });

  it('says when workers serve a market the app has not enabled', () => {
    const row = marketCoverage({
      market: 'CN',
      enabled: false,
      queues: queues('CN', { market_jobs: 1, data_fetch: 1, user_scans: 1 }),
      pipelineTask: pipeline('CN', { status: 'skipped', started_at: hoursAgo(4) }),
      now: NOW,
    });

    expect(row.problems).toEqual(['Workers serve it, but it is not enabled in the app.']);
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

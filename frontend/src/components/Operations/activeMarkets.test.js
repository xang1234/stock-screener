import { describe, expect, it } from 'vitest';

import { findActiveMarkets, queueMarket, splitByMarket } from './activeMarkets';

const SUPPORTED = ['US', 'HK', 'IN', 'JP', 'KR', 'TW', 'CN', 'DE'];

describe('queueMarket', () => {
  it('reads the market code from the queue name suffix', () => {
    expect(queueMarket('data_fetch_jp', SUPPORTED)).toBe('JP');
    expect(queueMarket('market_jobs_cn', SUPPORTED)).toBe('CN');
  });

  it('returns null for shared and unknown queues', () => {
    expect(queueMarket('data_fetch_shared', SUPPORTED)).toBeNull();
    expect(queueMarket('celery', SUPPORTED)).toBeNull();
    expect(queueMarket('social_ingestion', SUPPORTED)).toBeNull();
  });
});

describe('findActiveMarkets', () => {
  it('counts enabled markets and markets a worker serves', () => {
    const active = findActiveMarkets({
      enabledMarkets: ['US', 'JP'],
      supportedMarkets: SUPPORTED,
      jobsPayload: { workers: [{ queues: ['data_fetch_shared', 'market_jobs_cn'] }] },
    });

    expect([...active].sort()).toEqual(['CN', 'JP', 'US']);
  });

  it('never hides a market with live state', () => {
    const active = findActiveMarkets({
      enabledMarkets: [],
      supportedMarkets: SUPPORTED,
      jobsPayload: {
        jobs: [{ market: 'KR', state: 'running' }],
        queues: [{ queue: 'data_fetch_in', depth: 1 }, { queue: 'data_fetch_de', depth: 0 }],
        leases: { market_workload: { TW: { task_id: 't' }, DE: null }, pipeline_chain: { HK: { task_id: 'c' } } },
      },
      alerts: [{ market: 'US', state: 'open' }, { market: 'DE', state: 'closed' }],
    });

    expect([...active].sort()).toEqual(['HK', 'IN', 'KR', 'TW', 'US']);
  });

  it('ignores the shared data-fetch worker, which start_celery.sh subscribes to every market', () => {
    const active = findActiveMarkets({
      enabledMarkets: ['US'],
      supportedMarkets: SUPPORTED,
      jobsPayload: {
        workers: [
          { queues: ['data_fetch_us', 'data_fetch_hk', 'data_fetch_de', 'data_fetch_in'] },
          { queues: ['market_jobs_cn'] },
          { queues: ['user_scans_kr'] },
        ],
        queues: [
          { queue: 'data_fetch_de', depth: 0, consumers: 1 },
          { queue: 'data_fetch_in', depth: 2, consumers: 1 },
        ],
      },
    });

    // IN has queued work (live state); DE only has the global subscriber.
    expect([...active].sort()).toEqual(['CN', 'IN', 'KR', 'US']);
  });

  it('counts only live jobs, not cancelled or failed ones', () => {
    const active = findActiveMarkets({
      supportedMarkets: SUPPORTED,
      jobsPayload: {
        jobs: [{ market: 'KR', state: 'cancelled' }, { market: 'TW', state: 'failed' }, { market: 'IN', state: 'stale' }],
      },
    });

    expect([...active]).toEqual(['IN']);
  });

  it('counts a market whose queue a silent worker serves', () => {
    // A busy solo-pool worker misses the inspect poll, so it is absent from
    // workers; the backend still counts it from its heartbeat (consumers).
    const active = findActiveMarkets({
      supportedMarkets: SUPPORTED,
      jobsPayload: {
        workers: [],
        queues: [{ queue: 'market_jobs_cn', depth: 0, consumers: 1 }, { queue: 'data_fetch_de', depth: 0, consumers: 0 }],
      },
    });

    expect([...active]).toEqual(['CN']);
  });

  it('copes with data that has not loaded yet', () => {
    expect([...findActiveMarkets({})]).toEqual([]);
  });
});

describe('splitByMarket', () => {
  const rows = [{ m: 'US' }, { m: null }, { m: 'IN' }, { m: 'DE' }, { m: 'DE' }];

  it('names hidden markets in the same (alphabetical) order in every section', () => {
    expect(splitByMarket(rows, (row) => row.m, new Set(['US'])).hiddenMarkets).toEqual(['DE', 'IN']);
  });

  it('puts rows of inactive markets aside and names those markets once', () => {
    const { shown, hidden, hiddenMarkets } = splitByMarket(rows, (row) => row.m, new Set(['US']));

    expect(shown).toEqual([{ m: 'US' }, { m: null }]);
    expect(hidden).toEqual([{ m: 'IN' }, { m: 'DE' }, { m: 'DE' }]);
    expect(hiddenMarkets).toEqual(['DE', 'IN']);
  });

  it('keeps a row its own predicate marks as needing attention', () => {
    const { shown } = splitByMarket(rows, (row) => row.m, new Set(['US']), (row) => row.m === 'IN');

    expect(shown).toContainEqual({ m: 'IN' });
  });

  it('shows everything when the active set is unknown', () => {
    expect(splitByMarket(rows, (row) => row.m, null).hidden).toEqual([]);
  });
});

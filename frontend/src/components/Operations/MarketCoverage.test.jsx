import { screen, within } from '@testing-library/react';
import { describe, expect, it } from 'vitest';

import { renderWithProviders } from '../../test/renderWithProviders';
import MarketCoverage from './MarketCoverage';

const hoursAgo = (h) => new Date(Date.now() - h * 3600 * 1000).toISOString();

const queues = (market, consumers) => ['market_jobs', 'data_fetch', 'user_scans'].map((family, i) => ({
  queue: `${family}_${market.toLowerCase()}`,
  depth: 0,
  consumers: consumers[i],
}));

const pipeline = (market, run) => ({ name: `daily-market-pipeline-${market.toLowerCase()}`, market, last_run: run });

function rowFor(market) {
  return screen.getByRole('rowheader', { name: market }).closest('tr');
}

describe('MarketCoverage', () => {
  it('shows one row per market, enabled ones first, with what is wrong in plain words', () => {
    renderWithProviders(
      <MarketCoverage
        markets={['US', 'HK', 'JP', 'CN']}
        enabledMarkets={['US', 'JP']}
        queues={[...queues('US', [1, 1, 1]), ...queues('JP', [0, 0, 0]), ...queues('CN', [1, 1, 1]), ...queues('HK', [1, 1, 1])]}
        tasks={[
          pipeline('US', { status: 'dispatched', started_at: hoursAgo(16), triggered_by: 'auto' }),
          pipeline('JP', null),
          pipeline('CN', { status: 'skipped', started_at: hoursAgo(5) }),
        ]}
      />,
    );

    const markets = screen.getAllByRole('rowheader').map((cell) => cell.textContent);
    expect(markets).toEqual(['US', 'JP', 'HK', 'CN']);

    expect(within(rowFor('US')).getByText('Enabled')).toBeInTheDocument();
    expect(within(rowFor('US')).getByText('OK')).toBeInTheDocument();
    expect(within(rowFor('US')).getByText(/16h ago/)).toBeInTheDocument();

    expect(within(rowFor('JP')).getByText('No worker fetches its prices (data_fetch_jp).')).toBeInTheDocument();
    expect(within(rowFor('JP')).getByText('No daily pipeline run recorded yet')).toBeInTheDocument();
    expect(within(rowFor('JP')).getByText('Never')).toBeInTheDocument();

    expect(within(rowFor('CN')).getByText('Not enabled')).toBeInTheDocument();
    expect(within(rowFor('CN')).getByText('Workers serve it, but it is not enabled in the app.')).toBeInTheDocument();
  });

  it('says so when nothing needs attention', () => {
    renderWithProviders(
      <MarketCoverage
        markets={['US']}
        enabledMarkets={['US']}
        queues={queues('US', [1, 1, 1])}
        tasks={[pipeline('US', { status: 'completed', started_at: hoursAgo(10) })]}
      />,
    );

    expect(screen.getByText('Every market shown is wired up.')).toBeInTheDocument();
  });

  it('counts the markets with problems in its summary', () => {
    renderWithProviders(
      <MarketCoverage
        markets={['US', 'JP', 'KR']}
        enabledMarkets={['US', 'JP', 'KR']}
        queues={[...queues('US', [1, 1, 1]), ...queues('JP', [0, 0, 0]), ...queues('KR', [0, 0, 0])]}
      />,
    );

    expect(screen.getByRole('alert')).toHaveTextContent('2 markets need attention: JP, KR.');
  });

  it('shows unknown worker counts as unknown, not as OK or as missing', () => {
    renderWithProviders(
      <MarketCoverage markets={['JP']} enabledMarkets={['JP']} queues={queues('JP', [null, null, null])} />,
    );

    ['Pipeline: —', 'Prices: —', 'Scans: —'].forEach((label) => {
      expect(within(rowFor('JP')).getByText(label)).toBeInTheDocument();
    });
    expect(within(rowFor('JP')).getByText('Worker counts unknown')).toBeInTheDocument();
    expect(within(rowFor('JP')).getByText('Pipeline runs unknown')).toBeInTheDocument();
    expect(within(rowFor('JP')).queryByText('OK')).not.toBeInTheDocument();
    expect(screen.queryByRole('alert')).not.toBeInTheDocument();
    // Nothing was checked, so the summary must not call it wired up.
    expect(screen.queryByText('Every market shown is wired up.')).not.toBeInTheDocument();
    expect(screen.getByText('Some checks are not available yet.')).toBeInTheDocument();
  });

  it('waits, without an alert, for a new market to record its first pipeline run', () => {
    renderWithProviders(
      <MarketCoverage markets={['HK']} enabledMarkets={['HK']} queues={queues('HK', [1, 1, 1])} tasks={[pipeline('HK', null)]} />,
    );

    expect(within(rowFor('HK')).getByText('No daily pipeline run recorded yet')).toBeInTheDocument();
    expect(screen.queryByRole('alert')).not.toBeInTheDocument();
    expect(screen.getByText('Some checks are not available yet.')).toBeInTheDocument();
  });

  it('shows a note, not a problem, when scheduled pipeline runs are turned off', () => {
    renderWithProviders(
      <MarketCoverage
        markets={['US']}
        enabledMarkets={['US']}
        queues={queues('US', [1, 1, 1])}
        tasks={[{ ...pipeline('US', { started_at: hoursAgo(20 * 24) }), is_enabled: false }]}
      />,
    );

    expect(within(rowFor('US')).getByText('Scheduled pipeline runs are off (CACHE_WARMUP_ENABLED).')).toBeInTheDocument();
    expect(screen.queryByRole('alert')).not.toBeInTheDocument();
    expect(screen.getByText('Every market shown is wired up.')).toBeInTheDocument();
  });

  it('leaves pipeline runs out when scheduled tasks are not tracked here', () => {
    renderWithProviders(
      <MarketCoverage markets={['US']} enabledMarkets={['US']} queues={queues('US', [1, 1, 1])} tasks={null} />,
    );

    expect(within(rowFor('US')).getByText('OK')).toBeInTheDocument();
    expect(screen.getByText('Every market shown is wired up.')).toBeInTheDocument();
  });

  it('claims nothing over an empty table', () => {
    renderWithProviders(<MarketCoverage markets={[]} enabledMarkets={[]} />);

    expect(screen.queryByText('Every market shown is wired up.')).not.toBeInTheDocument();
  });

  it('shows a dash, not Never, when the market has no pipeline job in the list', () => {
    // e.g. a backend older than the task list's market field.
    renderWithProviders(
      <MarketCoverage
        markets={['US']}
        enabledMarkets={['US']}
        queues={queues('US', [1, 1, 1])}
        tasks={[{ name: 'daily-market-pipeline-us', last_run: { started_at: hoursAgo(2) } }]}
      />,
    );

    expect(within(rowFor('US')).queryByText('Never')).not.toBeInTheDocument();
    expect(within(rowFor('US')).getByText('—')).toBeInTheDocument();
  });
});

import { act, fireEvent, screen, waitFor, within } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import { renderWithProviders } from '../../test/renderWithProviders';
import { fullSeRow } from '../../test/fixtures/setupEngineFixtures';
import ScanResultsSection from '../scan/components/ScanResultsSection';
import { getSocialEvidence, getSocialSymbols } from '../../api/socialSignals';

vi.mock('@tanstack/react-virtual', () => ({
  useVirtualizer: ({ count }) => ({
    getVirtualItems: () => Array.from({ length: count }, (_, index) => ({
      index, start: index * 32, end: (index + 1) * 32, size: 32, key: index,
    })),
    getTotalSize: () => count * 32,
    measure: () => {},
  }),
}));
vi.mock('../../components/Scan/RSSparkline', () => ({ default: () => null }));
vi.mock('../../components/Scan/PriceSparkline', () => ({ default: () => null }));
vi.mock('../../components/common/AddToWatchlistMenu', () => ({ default: () => null }));
vi.mock('../../api/socialSignals', () => ({ getSocialSymbols: vi.fn(), getSocialEvidence: vi.fn() }));

const row = (symbol, market) => ({ ...fullSeRow, symbol, market, company_name: `${symbol} Inc` });
const nvda = {
  candidate_key: 'US:NVDA', canonical_symbol: 'NVDA', market: 'US', state: 'actionable',
  queue_score: 72.4, social_score: 81, confirmation_score: 64, mention_count: 9,
};
const lookup = (items) => ({ supported: true, available: true, reason_code: null, window: '7d', items });

const section = (results, onOpenChart = vi.fn(), showSocialSignals = true) => (
  <ScanResultsSection
    resultsData={{ results, total: results.length, capabilities: {} }}
    page={1} perPage={25} sortBy="composite_score" sortOrder="desc"
    onPageChange={vi.fn()} onSortChange={vi.fn()} onOpenChart={onOpenChart}
    showSocialSignals={showSocialSignals}
  />
);

const headerNames = () => [...document.querySelectorAll('thead th')].map((cell) => cell.textContent);

// Plain DOM queries: getByRole over the full scan table outlasts waitFor's timeout.
function socialCell(symbol) {
  const headers = [...document.querySelectorAll('thead th')].map((cell) => cell.textContent);
  const [symbolColumn, socialColumn] = [headers.indexOf('Sym'), headers.indexOf('Social 7D')];
  const cells = [...document.querySelectorAll('tbody tr')]
    .map((tableRow) => tableRow.querySelectorAll('td'))
    .find((rowCells) => rowCells[symbolColumn]?.textContent.startsWith(symbol));
  return cells?.[socialColumn]?.textContent;
}

describe('Social column on scan results', () => {
  beforeEach(() => vi.clearAllMocks());

  it("shows each row's published state from one lookup of the covered symbols", async () => {
    getSocialSymbols.mockResolvedValue(lookup([nvda]));
    renderWithProviders(section([row('NVDA', 'US'), row('AAPL', 'US'), row('005930.KS', 'KR')]));

    await waitFor(() => expect(['NVDA', 'AAPL', '005930.KS'].map(socialCell)).toEqual([
      'Actionable 72', '-', 'n/a',
    ]));
    expect(getSocialSymbols.mock.calls).toEqual([[['AAPL', 'NVDA'], '7d']]);
  });

  it('opens the evidence drawer from the chip without opening the chart', async () => {
    getSocialSymbols.mockResolvedValue(lookup([nvda]));
    getSocialEvidence.mockReturnValue(new Promise(() => {}));
    const onOpenChart = vi.fn();
    renderWithProviders(section([row('NVDA', 'US')], onOpenChart));

    fireEvent.click(await screen.findByText('Actionable 72', {}, { timeout: 5000 }));

    const drawer = await screen.findByRole('dialog', { name: 'Social evidence' });
    expect(within(drawer).getByRole('heading', { name: 'NVDA' })).toBeInTheDocument();
    expect(within(drawer).getByText('Social 81')).toBeInTheDocument();
    expect(within(drawer).queryByRole('button', { name: 'Send visible to Scan' })).toBeNull();
    expect(getSocialEvidence).toHaveBeenCalledWith('US:NVDA', '7d');
    expect(onOpenChart).not.toHaveBeenCalled();
  });

  it('shows the column with pending cells before the first lookup answers', async () => {
    getSocialSymbols.mockReturnValue(new Promise(() => {}));
    renderWithProviders(section([row('NVDA', 'US'), row('005930.KS', 'KR')]));

    await waitFor(() => expect(['NVDA', '005930.KS'].map(socialCell)).toEqual(['…', 'n/a']));
  });

  it('marks rows as pending, not unmentioned, while a new page is looked up', async () => {
    getSocialSymbols.mockResolvedValueOnce(lookup([nvda])).mockReturnValueOnce(new Promise(() => {}));
    const view = renderWithProviders(section([row('NVDA', 'US')]));
    await waitFor(() => expect(socialCell('NVDA')).toBe('Actionable 72'));

    act(() => view.rerender(section([row('MSFT', 'US')])));

    await waitFor(() => expect(socialCell('MSFT')).toBe('…'));
    expect(getSocialSymbols).toHaveBeenLastCalledWith(['MSFT'], '7d');
  });

  it('drops the column for a page with no row in a covered market', async () => {
    getSocialSymbols.mockResolvedValue(lookup([nvda]));
    const view = renderWithProviders(section([row('NVDA', 'US')]));
    await waitFor(() => expect(socialCell('NVDA')).toBe('Actionable 72'));

    act(() => view.rerender(section([row('005930.KS', 'KR')])));

    await waitFor(() => expect(headerNames()).not.toContain('Social 7D'));
    expect(getSocialSymbols).toHaveBeenCalledTimes(1);
  });

  it('makes no lookup and shows no column when the app has Social turned off', async () => {
    getSocialSymbols.mockResolvedValue(lookup([nvda]));
    renderWithProviders(section([row('NVDA', 'US')], vi.fn(), false));

    await waitFor(() => expect(headerNames()).toContain('Comp'));
    expect(headerNames()).not.toContain('Social 7D');
    expect(getSocialSymbols).not.toHaveBeenCalled();
  });

  it('drops the column when the lookup fails', async () => {
    getSocialSymbols.mockRejectedValue(new Error('network'));
    const { queryClient } = renderWithProviders(section([row('NVDA', 'US')]));

    await waitFor(() => expect(
      queryClient.getQueryCache().find({ queryKey: ['scanSocialSymbols'], exact: false })?.state.status,
    ).toBe('error'));
    expect(headerNames()).toContain('Comp');
    expect(headerNames()).not.toContain('Social 7D');
  });

  it('keeps the column when a background refetch fails after a good lookup', async () => {
    getSocialSymbols.mockResolvedValueOnce(lookup([nvda])).mockRejectedValueOnce(new Error('blip'));
    const { queryClient } = renderWithProviders(section([row('NVDA', 'US')]));
    await waitFor(() => expect(socialCell('NVDA')).toBe('Actionable 72'));

    await act(() => queryClient.refetchQueries({ queryKey: ['scanSocialSymbols'] }));
    // Query notifications are batched on a timer; let the error render land.
    await act(() => new Promise((resolve) => { setTimeout(resolve, 50); }));

    expect(queryClient.getQueryCache().find({ queryKey: ['scanSocialSymbols'], exact: false }).state.status)
      .toBe('error');
    expect(socialCell('NVDA')).toBe('Actionable 72');
  });

  it('drops the column, not leaves it pending, when a new page lookup fails', async () => {
    getSocialSymbols.mockResolvedValueOnce(lookup([nvda])).mockRejectedValueOnce(new Error('network'));
    const view = renderWithProviders(section([row('NVDA', 'US')]));
    await waitFor(() => expect(socialCell('NVDA')).toBe('Actionable 72'));

    act(() => view.rerender(section([row('MSFT', 'US')])));

    await waitFor(() => expect(headerNames()).not.toContain('Social 7D'));
  });

  it('closes the evidence drawer when Social is turned off', async () => {
    getSocialSymbols.mockResolvedValue(lookup([nvda]));
    getSocialEvidence.mockReturnValue(new Promise(() => {}));
    const view = renderWithProviders(section([row('NVDA', 'US')]));
    fireEvent.click(await screen.findByText('Actionable 72', {}, { timeout: 5000 }));
    await screen.findByRole('dialog', { name: 'Social evidence' });

    act(() => view.rerender(section([row('NVDA', 'US')], vi.fn(), false)));

    await waitFor(() => expect(screen.queryByRole('dialog', { name: 'Social evidence' })).toBeNull());
  });

  it('makes no lookup on a phone', async () => {
    const matchMedia = window.matchMedia;
    window.matchMedia = (query) => ({
      matches: true, media: query, onchange: null,
      addListener: () => {}, removeListener: () => {},
      addEventListener: () => {}, removeEventListener: () => {}, dispatchEvent: () => false,
    });
    try {
      getSocialSymbols.mockResolvedValue(lookup([nvda]));
      renderWithProviders(section([row('NVDA', 'US')]));

      expect(await screen.findByRole('list', { name: 'Scan results' })).toBeInTheDocument();
      expect(getSocialSymbols).not.toHaveBeenCalled();
    } finally {
      window.matchMedia = matchMedia;
    }
  });

  it.each([
    ['Social is off', { supported: false, available: false, reason_code: 'social_signals_disabled', window: '7d', items: [] }],
    ['nothing is published', { supported: true, available: false, reason_code: 'no_published_run', window: '7d', items: [] }],
  ])('has no Social column when %s', async (_, response) => {
    getSocialSymbols.mockResolvedValue(response);
    const { queryClient } = renderWithProviders(section([row('NVDA', 'US')]));

    await waitFor(() => expect(
      queryClient.getQueryCache().find({ queryKey: ['scanSocialSymbols'], exact: false })?.state.status,
    ).toBe('success'));
    expect(getSocialSymbols).toHaveBeenCalledWith(['NVDA'], '7d');
    expect(headerNames()).toContain('Comp');
    expect(headerNames()).not.toContain('Social 7D');
  });
});

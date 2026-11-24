import { screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';

import { renderWithProviders } from '../../test/renderWithProviders';
import BreadthHistoryTable from './BreadthHistoryTable';
import { breadthMetricDefinitions } from './breadthMetricDefinitions';


const row = {
  date: '2026-08-21',
  stocks_up_4pct: 10,
  stocks_down_4pct: 5,
  ratio_5day: 2,
  ratio_10day: 1.5,
  stocks_up_25pct_quarter: 8,
  stocks_down_25pct_quarter: 4,
  stocks_up_25pct_month: 7,
  stocks_down_25pct_month: 3,
  stocks_up_50pct_month: 2,
  stocks_down_50pct_month: 1,
  stocks_up_13pct_34days: 9,
  stocks_down_13pct_34days: 6,
  atr_10x_extension_count: 3,
  t2108_pct: 57.89,
  above_sma20_pct: 61.25,
  above_sma50_pct: 52.75,
  above_sma200_pct: 34.5,
  down_20pct_from_high_pct: 45.25,
  broad_universe_count: 110,
};


describe('BreadthHistoryTable', () => {
  it('makes only advertised nonzero contributor cells buttons', async () => {
    const user = userEvent.setup();
    const onOpen = vi.fn();
    renderWithProviders(
      <BreadthHistoryTable
        rows={[row]}
        contributorDates={new Set([row.date])}
        onContributorCellClick={onOpen}
      />,
    );

    const button = screen.getByRole('button', {
      name: 'View 10 contributing stocks for Stocks Up 4%+',
    });
    await user.click(button);
    expect(onOpen).toHaveBeenCalledWith('stocks_up_4pct', row, button);
    expect(screen.queryByRole('button', { name: /View .*5 Day Ratio/ })).not.toBeInTheDocument();
    expect(screen.queryByRole('button', { name: /View .*Broad Universe/ })).not.toBeInTheDocument();
  });

  it('renders a date-only value as the same local calendar date', () => {
    vi.stubEnv('TZ', 'America/Los_Angeles');
    try {
      renderWithProviders(<BreadthHistoryTable rows={[row]} />);

      expect(screen.getByText('08/21/26')).toBeInTheDocument();
    } finally {
      vi.unstubAllEnvs();
    }
  });

  it('renders grouped primary, secondary, and context headers', () => {
    renderWithProviders(<BreadthHistoryTable rows={[row]} />);

    expect(screen.getByText('Primary Breadth Indicators')).toBeInTheDocument();
    expect(screen.getByText('Secondary Breadth Indicators')).toBeInTheDocument();
    expect(screen.getByText('Context')).toBeInTheDocument();
    expect(screen.getByText('10x ATR')).toBeInTheDocument();
    expect(screen.getByText('T2108')).toBeInTheDocument();
    expect(screen.getByText('Broad Universe')).toBeInTheDocument();
    expect(screen.getByText('2.00')).toBeInTheDocument();
    expect(screen.getByText('57.89%')).toBeInTheDocument();
  });

  it('renders participation columns and an em dash where a row has none', () => {
    renderWithProviders(
      <BreadthHistoryTable
        rows={[
          row,
          {
            ...row,
            date: '2026-08-20',
            above_sma200_pct: null,
            down_20pct_from_high_pct: null,
          },
        ]}
      />,
    );

    expect(screen.getByTestId('breadth-header-above_sma200_pct')).toBeInTheDocument();
    expect(screen.getByTestId('breadth-header-down_20pct_from_high_pct')).toBeInTheDocument();
    expect(screen.getByText('34.50%')).toBeInTheDocument();
    expect(screen.getByText('45.25%')).toBeInTheDocument();
    const legacyRow = screen.getByText('08/20/26').closest('tr');
    expect(within(legacyRow).getAllByText('—')).toHaveLength(2);
  });

  it('renders the 20- and 50-day participation columns', () => {
    renderWithProviders(
      <BreadthHistoryTable
        rows={[
          row,
          { ...row, date: '2026-08-20', above_sma20_pct: null, above_sma50_pct: null },
        ]}
      />,
    );

    expect(screen.getByTestId('breadth-header-above_sma20_pct')).toBeInTheDocument();
    expect(screen.getByTestId('breadth-header-above_sma50_pct')).toBeInTheDocument();
    expect(screen.getByText('61.25%')).toBeInTheDocument();
    expect(screen.getByText('52.75%')).toBeInTheDocument();
    const legacyRow = screen.getByText('08/20/26').closest('tr');
    expect(within(legacyRow).getAllByText('—')).toHaveLength(2);
  });

  it('describes each indicator family in a distinct group band', () => {
    renderWithProviders(<BreadthHistoryTable rows={[row]} />);

    expect(within(screen.getByTestId('breadth-group-primary')).getByText(
      'Daily movers & ratios',
    )).toBeInTheDocument();
    expect(within(screen.getByTestId('breadth-group-secondary')).getByText(
      'Trend windows',
    )).toBeInTheDocument();
    expect(within(screen.getByTestId('breadth-group-context')).getByText(
      'Market context',
    )).toBeInTheDocument();
  });

  it('renders compact multiline metric headings in a desktop-fit table', () => {
    renderWithProviders(<BreadthHistoryTable rows={[row]} />);

    const upHeader = screen.getByTestId('breadth-header-stocks_up_4pct');
    expect(within(upHeader).getByText('Stocks Up')).toBeInTheDocument();
    expect(within(upHeader).getByText('4%+ Today')).toBeInTheDocument();
    expect(upHeader).toHaveStyle({ whiteSpace: 'normal' });
    expect(screen.getByTestId('breadth-history-table')).toHaveStyle({
      tableLayout: 'fixed',
      width: '100%',
    });
  });

  it('marks paired cells and exposes formula tooltips accessibly', () => {
    renderWithProviders(<BreadthHistoryTable rows={[row]} />);

    expect(screen.getAllByTestId('breadth-cell-stocks_up_4pct')).toHaveLength(1);
    expect(screen.getAllByTestId('breadth-cell-stocks_down_4pct')).toHaveLength(1);
    expect(
      screen.getByRole('button', { name: /stocks up 4%\+ formula details/i }),
    ).toBeInTheDocument();
    expect(screen.getByTestId('breadth-history-scroll')).toHaveStyle({
      overflowX: 'auto',
    });
  });

  it('describes StockBee thresholds in selected-market local units', async () => {
    const user = userEvent.setup();
    renderWithProviders(<BreadthHistoryTable rows={[row]} />);

    await user.hover(
      screen.getByRole('button', { name: /stocks up 4%\+ formula details/i }),
    );

    expect(await screen.findByText(
      /local-currency traded value must meet the selected market's fixed threshold/i,
    )).toBeInTheDocument();
    expect(breadthMetricDefinitions.stocks_up_4pct.description).toMatch(
      /market-specific daily share threshold/i,
    );
    expect(breadthMetricDefinitions.stocks_up_25pct_month.requiredHistory).toMatch(
      /market-specific local reference-price threshold/i,
    );
    expect(JSON.stringify(breadthMetricDefinitions)).not.toMatch(/US\$250,000|US\$5/);
  });

  it('renders metric values as unavailable when their eligible denominator is zero', () => {
    renderWithProviders(
      <BreadthHistoryTable
        rows={[{
          ...row,
          stocks_up_25pct_month: 0,
          stocks_down_25pct_month: 0,
          stocks_up_50pct_month: 0,
          stocks_down_50pct_month: 0,
          stockbee_month_eligible_count: 0,
        }]}
      />,
    );

    expect(screen.getAllByText('—')).toHaveLength(4);
  });

  it('keeps valid rolling ratios visible when current-session eligibility is zero', () => {
    renderWithProviders(
      <BreadthHistoryTable
        rows={[{
          ...row,
          stockbee_daily_eligible_count: 0,
        }]}
      />,
    );

    expect(screen.getByTestId('breadth-cell-ratio_5day')).toHaveTextContent('2.00');
    expect(screen.getByTestId('breadth-cell-ratio_10day')).toHaveTextContent('1.50');
  });

  it('uses neutral, soft, and strong heat levels for directional count metrics', () => {
    const rows = Array.from({ length: 10 }, (_, index) => ({
      ...row,
      date: `2026-08-${String(index + 1).padStart(2, '0')}`,
      stocks_up_4pct: index + 1,
      stocks_down_4pct: index + 1,
      stockbee_daily_eligible_count: 100,
    }));

    renderWithProviders(<BreadthHistoryTable rows={rows} />);

    const upCells = screen.getAllByTestId('breadth-cell-stocks_up_4pct');
    const downCells = screen.getAllByTestId('breadth-cell-stocks_down_4pct');
    expect(upCells.map((cell) => cell.dataset.tone)).toEqual([
      'neutral', 'neutral', 'neutral', 'neutral', 'neutral',
      'neutral', 'neutral', 'up-soft', 'up-soft', 'up-strong',
    ]);
    expect(downCells.map((cell) => cell.dataset.tone)).toEqual([
      'neutral', 'neutral', 'neutral', 'neutral', 'neutral',
      'neutral', 'neutral', 'down-soft', 'down-soft', 'down-strong',
    ]);
    expect(upCells[9]).toHaveStyle({
      backgroundColor: '#0d7a3e',
      color: '#fff',
    });
    expect(downCells[9]).toHaveStyle({
      backgroundColor: '#9b1c31',
      color: '#fff',
    });
  });

  it('colors ratios around one by direction and significance', () => {
    const rows = [0.7, 0.9, 1, 1.2, 2, null].map((ratio, index) => ({
      ...row,
      date: `2026-08-${String(index + 1).padStart(2, '0')}`,
      ratio_5day: ratio,
      stockbee_daily_eligible_count: 100,
    }));

    renderWithProviders(<BreadthHistoryTable rows={rows} />);

    expect(
      screen
        .getAllByTestId('breadth-cell-ratio_5day')
        .map((cell) => cell.dataset.tone),
    ).toEqual([
      'down-strong',
      'down-soft',
      'neutral',
      'up-soft',
      'up-strong',
      'neutral',
    ]);
  });
});

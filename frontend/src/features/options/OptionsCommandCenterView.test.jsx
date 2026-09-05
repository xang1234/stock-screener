import { screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';

import { renderWithProviders } from '../../test/renderWithProviders';
import OptionsCommandCenterView from './OptionsCommandCenterView';
import { commandCenterFixture } from './__fixtures__/optionsResponses';

describe('OptionsCommandCenterView', () => {
  it('shows distinct equity and options freshness plus a compact selectable lens', async () => {
    const user = userEvent.setup();
    renderWithProviders(<OptionsCommandCenterView data={commandCenterFixture} onOpenSymbol={() => {}} />);

    expect(screen.getByText(/Equity source: Sep 4, 2026 · run 33/i)).toBeInTheDocument();
    expect(screen.getByText(/Options observed: Sep 4, 2026/i)).toBeInTheDocument();
    expect(screen.getByText(/Yahoo · options-analytics-v1 · 100% coverage/i)).toBeInTheDocument();
    expect(screen.getByRole('columnheader', { name: /Estimated Net GEX/i })).toBeInTheDocument();
    expect(screen.queryByRole('columnheader', { name: /ATM IV/i })).not.toBeInTheDocument();

    await user.click(screen.getByRole('button', { name: 'Volatility' }));
    expect(screen.getByRole('columnheader', { name: /ATM IV/i })).toBeInTheDocument();
    expect(screen.queryByRole('columnheader', { name: /Estimated Net GEX/i })).not.toBeInTheDocument();

    await user.click(screen.getByRole('button', { name: 'Activity' }));
    expect(screen.getByRole('columnheader', { name: /Call \/ Put Volume/i })).toBeInTheDocument();
    expect(screen.getByRole('columnheader', { name: /Near-Spot Open Interest Concentration/i })).toBeInTheDocument();
  });

  it('keeps every current symbol visible and ranks only available metric values', () => {
    const data = structuredClone(commandCenterFixture);
    data.items[1].metrics.net_gex = {
      available: false,
      value: null,
      label: 'Estimated Net GEX',
      reason_codes: ['provider_field_missing'],
      evidence: {},
    };
    renderWithProviders(<OptionsCommandCenterView data={data} onOpenSymbol={() => {}} />);

    const rows = screen.getAllByRole('row').slice(1);
    expect(rows).toHaveLength(2);
    expect(within(rows[0]).getByText('AAPL')).toBeInTheDocument();
    expect(within(rows[0]).getByLabelText('Metric rank 1')).toBeInTheDocument();
    expect(within(rows[1]).getByText('MSFT')).toBeInTheDocument();
    expect(within(rows[1]).getByLabelText(/Unavailable: provider field missing/i)).toBeInTheDocument();
    expect(within(rows[1]).queryByLabelText(/Metric rank/i)).not.toBeInTheDocument();
  });

  it('shows source identity and retains numeric zero', () => {
    renderWithProviders(<OptionsCommandCenterView data={commandCenterFixture} onOpenSymbol={() => {}} />);

    const aaplRow = screen.getByRole('row', { name: /AAPL/i });
    expect(within(aaplRow).getByText('Both')).toBeInTheDocument();
    expect(within(aaplRow).getByText('C1 · L2')).toBeInTheDocument();
    expect(screen.getByRole('columnheader', { name: /Estimated Net GEX/i })).toBeInTheDocument();
    expect(within(aaplRow).getByText('Max Pain: 0')).toBeInTheDocument();
  });

  it('renders a prominent stale warning', () => {
    renderWithProviders(
      <OptionsCommandCenterView
        data={{ ...commandCenterFixture, stale: true, reason_codes: ['stale_relative_to_equity'] }}
        onOpenSymbol={() => {}}
      />,
    );

    expect(screen.getByRole('alert')).toHaveTextContent(/previous published options snapshot/i);
  });

  it('opens a row with Enter', async () => {
    const user = userEvent.setup();
    const onOpenSymbol = vi.fn();
    renderWithProviders(<OptionsCommandCenterView data={commandCenterFixture} onOpenSymbol={onOpenSymbol} />);

    screen.getByRole('row', { name: /AAPL/i }).focus();
    await user.keyboard('{Enter}');
    expect(onOpenSymbol).toHaveBeenCalledWith('AAPL');
  });
});

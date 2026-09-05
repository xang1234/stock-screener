import { screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it } from 'vitest';

import { renderWithProviders } from '../../test/renderWithProviders';
import OptionsSymbolDetailView from './OptionsSymbolDetailView';
import { symbolDetailFixture } from './__fixtures__/optionsResponses';

describe('OptionsSymbolDetailView', () => {
  it('renders the compact option structure, truthful labels, and history context', () => {
    renderWithProviders(<OptionsSymbolDetailView data={symbolDetailFixture} onBack={() => {}} />);

    expect(screen.getByRole('heading', { name: /AAPL options/i })).toBeInTheDocument();
    expect(screen.getByText('Open interest and volume')).toBeInTheDocument();
    expect(screen.getByText('Estimated GEX by strike')).toBeInTheDocument();
    expect(screen.getByText('Implied volatility smile')).toBeInTheDocument();
    expect(screen.getByText('Observed history')).toBeInTheDocument();
    expect(screen.getByText(/missing sessions are not filled/i)).toBeInTheDocument();
    expect(screen.getByText(/8 lifetime observations/i)).toBeInTheDocument();
    expect(screen.getByText('Estimated Net GEX')).toBeInTheDocument();
    expect(screen.getByText('Highest Contract Activity Ratio')).toBeInTheDocument();
    expect(screen.getByText('Historical context')).toBeInTheDocument();
    expect(screen.getByText('ATM IV Percentile')).toBeInTheDocument();
    expect(screen.getByText('Max Pain')).toBeInTheDocument();
    expect(screen.getByText('0')).toBeInTheDocument();
  });

  it('places assumptions, warnings, and reason evidence behind disclosure', async () => {
    const user = userEvent.setup();
    renderWithProviders(<OptionsSymbolDetailView data={symbolDetailFixture} onBack={() => {}} />);

    const disclosure = screen.getByText(/Method and data quality/i);
    expect(disclosure).toBeInTheDocument();
    expect(screen.getByText(/dealer_proxy/i)).not.toBeVisible();
    await user.click(disclosure);
    expect(await screen.findByText(/dealer_proxy/i)).toBeVisible();
    expect(screen.getByText(/provider spot price: 201/i)).toBeVisible();
    expect(screen.getAllByText(/building history/i).some((element) => element.offsetParent !== null || element.getAttribute('aria-hidden') !== 'true')).toBe(true);
  });

  it('renders a missing spot price with a neutral marker', () => {
    const data = structuredClone(symbolDetailFixture);
    data.item.spot_price = null;

    renderWithProviders(<OptionsSymbolDetailView data={data} onBack={() => {}} />);

    const card = screen.getByText('Contract snapshot').parentElement;
    expect(within(card).getByText('—')).toBeInTheDocument();
    expect(within(card).queryByText('$')).not.toBeInTheDocument();
  });
});

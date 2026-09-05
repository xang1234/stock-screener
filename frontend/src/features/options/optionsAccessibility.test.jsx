import { screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';

import { renderWithProviders } from '../../test/renderWithProviders';
import OptionsCommandCenterView from './OptionsCommandCenterView';
import { commandCenterFixture } from './__fixtures__/optionsResponses';

describe('options accessibility', () => {
  it('supports keyboard sorting and Space row navigation', async () => {
    const user = userEvent.setup();
    const onOpenSymbol = vi.fn();
    renderWithProviders(<OptionsCommandCenterView data={commandCenterFixture} onOpenSymbol={onOpenSymbol} />);

    const sort = screen.getByRole('button', { name: /Sort by Estimated Net GEX/i });
    expect(sort).toHaveAttribute('aria-pressed', 'true');
    await user.click(sort);
    expect(sort).toHaveAttribute('aria-label', expect.stringMatching(/ascending/i));

    screen.getByRole('row', { name: /MSFT/i }).focus();
    await user.keyboard(' ');
    expect(onOpenSymbol).toHaveBeenCalledWith('MSFT');
  });

  it('gives unavailable values an accessible reason instead of directional language', async () => {
    const user = userEvent.setup();
    renderWithProviders(<OptionsCommandCenterView data={commandCenterFixture} onOpenSymbol={() => {}} />);

    await user.click(screen.getByRole('button', { name: 'Skew' }));
    expect(screen.getAllByLabelText(/Unavailable: building history/i)).not.toHaveLength(0);
    expect(screen.queryByText(/bullish|bearish|buying|selling|inflow/i)).not.toBeInTheDocument();
  });
});

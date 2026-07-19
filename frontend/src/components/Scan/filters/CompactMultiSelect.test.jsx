import { screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { vi } from 'vitest';

import { renderWithProviders } from '../../../test/renderWithProviders';
import CompactMultiSelect from './CompactMultiSelect';

describe('CompactMultiSelect', () => {
  it('explains and enforces the selection limit', async () => {
    const user = userEvent.setup();
    renderWithProviders(
      <CompactMultiSelect
        label="Industry"
        values={['A', 'B']}
        options={['A', 'B', 'C']}
        onChange={vi.fn()}
        maxValues={2}
      />,
    );

    expect(screen.getByText('Up to 2 values')).toBeInTheDocument();
    await user.click(screen.getByRole('combobox'));
    expect(screen.getByRole('option', { name: 'C' })).toHaveAttribute('aria-disabled', 'true');
  });
});

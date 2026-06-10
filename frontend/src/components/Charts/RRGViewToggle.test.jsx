import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';

import { renderWithProviders } from '../../test/renderWithProviders';
import RRGViewToggle from './RRGViewToggle';

describe('RRGViewToggle', () => {
  it('hides the RRG view option when no bundle is available', () => {
    renderWithProviders(
      <RRGViewToggle view="table" onView={vi.fn()} scope="groups" onScope={vi.fn()} rrgAvailable={false} />,
    );

    expect(screen.getByRole('button', { name: 'Table' })).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'RRG' })).not.toBeInTheDocument();
  });

  it('hides scope controls when only one scope is available', () => {
    renderWithProviders(
      <RRGViewToggle
        view="rrg"
        onView={vi.fn()}
        scope="groups"
        onScope={vi.fn()}
        availableScopes={['groups']}
      />,
    );

    expect(screen.getByRole('button', { name: 'RRG' })).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Groups' })).not.toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Sectors' })).not.toBeInTheDocument();
  });

  it('renders only available scopes and reports scope changes', async () => {
    const onScope = vi.fn();
    const user = userEvent.setup();
    render(
      <RRGViewToggle
        view="rrg"
        onView={vi.fn()}
        scope="groups"
        onScope={onScope}
        availableScopes={['groups', 'sectors']}
      />,
    );

    expect(screen.getByRole('button', { name: 'Groups' })).toBeInTheDocument();
    await user.click(screen.getByRole('button', { name: 'Sectors' }));

    expect(onScope).toHaveBeenCalledWith('sectors');
  });
});

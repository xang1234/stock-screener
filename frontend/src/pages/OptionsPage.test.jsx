import { screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import { renderWithProviders } from '../test/renderWithProviders';
import OptionsPage from './OptionsPage';
import { commandCenterFixture } from '../features/options/__fixtures__/optionsResponses';
import * as optionsApi from '../api/optionsAnalytics';

vi.mock('../api/optionsAnalytics', () => ({
  getOptionsCommandCenter: vi.fn(),
  refreshOptionsAnalytics: vi.fn(),
}));

vi.mock('../features/options/OptionsCommandCenterView', () => ({
  default: ({ data, onOpenSymbol }) => (
    <button type="button" onClick={() => onOpenSymbol(data.items[0].symbol)}>Open {data.items[0].symbol}</button>
  ),
}));

describe('OptionsPage', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    optionsApi.getOptionsCommandCenter.mockResolvedValue(commandCenterFixture);
  });

  it('loads the published run and URL-encodes row navigation', async () => {
    const data = structuredClone(commandCenterFixture);
    data.items[0].symbol = 'BRK/B';
    optionsApi.getOptionsCommandCenter.mockResolvedValue(data);
    renderWithProviders(
      <MemoryRouter initialEntries={['/options']}>
        <Routes>
          <Route path="/options" element={<OptionsPage />} />
          <Route path="/options/:symbol" element={<div>Detail route</div>} />
        </Routes>
      </MemoryRouter>,
    );

    await userEvent.click(await screen.findByRole('button', { name: 'Open BRK/B' }));
    expect(await screen.findByText('Detail route')).toBeInTheDocument();
    expect(window.location.pathname).not.toContain('BRK/B');
  });

  it('posts refresh once and waits for a different published run before settling', async () => {
    const user = userEvent.setup();
    optionsApi.refreshOptionsAnalytics.mockResolvedValue({ status: 'accepted', task_id: 'task-1' });
    renderWithProviders(
      <MemoryRouter><OptionsPage /></MemoryRouter>,
    );

    await screen.findByRole('button', { name: 'Open AAPL' });
    await user.click(screen.getByRole('button', { name: 'Refresh options analytics' }));
    expect(optionsApi.refreshOptionsAnalytics).toHaveBeenCalledTimes(1);
    expect(await screen.findByText(/Accepted as task task-1/i)).toBeInTheDocument();

    optionsApi.getOptionsCommandCenter.mockResolvedValue({ ...commandCenterFixture, run_id: 8 });
    await waitFor(() => expect(screen.queryByText(/Accepted as task task-1/i)).not.toBeInTheDocument(), {
      timeout: 7000,
    });
  });
});

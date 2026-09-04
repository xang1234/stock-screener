import { screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import { renderWithProviders } from '../test/renderWithProviders';
import OptionsPage from './OptionsPage';
import { commandCenterFixture } from '../features/options/__fixtures__/optionsResponses';
import * as optionsApi from '../api/optionsAnalytics';
import * as tasksApi from '../api/tasks';

vi.mock('../api/optionsAnalytics', () => ({
  getOptionsCommandCenter: vi.fn(),
  refreshOptionsAnalytics: vi.fn(),
}));

vi.mock('../api/tasks', () => ({
  getTaskStatus: vi.fn(),
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
    tasksApi.getTaskStatus.mockResolvedValue({ status: 'running' });
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
    expect(optionsApi.refreshOptionsAnalytics).toHaveBeenCalledWith({
      sourceRunId: null,
      force: true,
    });
    expect(await screen.findByText(/Accepted as task task-1/i)).toBeInTheDocument();

    optionsApi.getOptionsCommandCenter.mockResolvedValue({ ...commandCenterFixture, run_id: 8 });
    await waitFor(() => expect(screen.queryByText(/Accepted as task task-1/i)).not.toBeInTheDocument(), {
      timeout: 7000,
    });
  });

  it('offers refresh before the first options snapshot exists', async () => {
    const user = userEvent.setup();
    optionsApi.getOptionsCommandCenter.mockRejectedValue({ response: { status: 404 } });
    optionsApi.refreshOptionsAnalytics.mockResolvedValue({ status: 'accepted', task_id: 'task-bootstrap' });

    renderWithProviders(<MemoryRouter><OptionsPage /></MemoryRouter>);

    await screen.findByText(/No published options snapshot yet/i);
    await user.click(screen.getByRole('button', { name: 'Refresh options analytics' }));

    expect(optionsApi.refreshOptionsAnalytics).toHaveBeenCalledWith({
      sourceRunId: null,
      force: true,
    });
  });

  it.each(['failed_quality', 'cancelled', 'skipped'])(
    'clears accepted state when the refresh finishes as %s',
    async (resultStatus) => {
      const user = userEvent.setup();
      optionsApi.refreshOptionsAnalytics.mockResolvedValue({ status: 'accepted', task_id: 'task-terminal' });
      tasksApi.getTaskStatus.mockResolvedValue({
        status: 'completed',
        result: { status: resultStatus, reason_codes: ['test_reason'] },
      });
      renderWithProviders(<MemoryRouter><OptionsPage /></MemoryRouter>);

      await screen.findByRole('button', { name: 'Open AAPL' });
      await user.click(screen.getByRole('button', { name: 'Refresh options analytics' }));

      expect(await screen.findByText(new RegExp(resultStatus.replace('_', ' '), 'i'))).toBeInTheDocument();
      await waitFor(() => expect(screen.getByRole('button', { name: 'Refresh options analytics' })).toBeEnabled());
    },
  );

  it('clears accepted state when task polling fails', async () => {
    const user = userEvent.setup();
    optionsApi.refreshOptionsAnalytics.mockResolvedValue({ status: 'accepted', task_id: 'task-error' });
    tasksApi.getTaskStatus.mockRejectedValue(new Error('status unavailable'));
    renderWithProviders(<MemoryRouter><OptionsPage /></MemoryRouter>);

    await screen.findByRole('button', { name: 'Open AAPL' });
    await user.click(screen.getByRole('button', { name: 'Refresh options analytics' }));

    expect(await screen.findByText(/could not confirm refresh status/i)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Refresh options analytics' })).toBeEnabled();
  });
});

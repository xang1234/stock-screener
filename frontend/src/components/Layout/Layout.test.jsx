import { MemoryRouter } from 'react-router-dom';
import { screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';

import Layout from './Layout';
import { renderWithProviders } from '../../test/renderWithProviders';

const runtimeState = {
  auth: { required: false, authenticated: true },
  features: { chatbot: false, themes: false, tasks: false },
  isLoggingOut: false,
  logout: vi.fn(),
};

const strategyState = {
  activeProfile: 'default',
  activeProfileDetail: { label: 'Default' },
  profiles: [{ profile: 'default', label: 'Default' }],
  setActiveProfile: vi.fn(),
};

vi.mock('../../contexts/RuntimeContext', () => ({
  useRuntime: () => runtimeState,
}));

vi.mock('../../contexts/StrategyProfileContext', () => ({
  useStrategyProfile: () => strategyState,
}));

vi.mock('../../hooks/useRuntimeActivity', () => ({
  useRuntimeActivity: () => ({
    data: {
      bootstrap: { state: 'ready' },
      summary: { active_market_count: 1, status: 'active' },
      markets: [
        {
          market: 'HK',
          status: 'running',
          stage_label: 'Fundamentals Refresh',
        },
      ],
    },
  }),
}));

describe('Layout', () => {
  it('shows the runtime activity header summary and removes Digest navigation', () => {
    renderWithProviders(
      <MemoryRouter initialEntries={['/scan']}>
        <Layout>
          <div>content</div>
        </Layout>
      </MemoryRouter>
    );

    expect(screen.getByText('1 market active')).toBeInTheDocument();
    expect(screen.getByText('HK · Fundamentals Refresh')).toBeInTheDocument();
    expect(screen.queryByRole('link', { name: /digest/i })).not.toBeInTheDocument();
  });
});

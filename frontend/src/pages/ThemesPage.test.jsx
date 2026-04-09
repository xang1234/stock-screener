import { fireEvent, screen, waitFor } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import ThemesPage from './ThemesPage';
import { renderWithProviders } from '../test/renderWithProviders';

const runtimeState = {
  runtimeReady: true,
  uiSnapshots: {
    themes: false,
  },
};

vi.mock('../contexts/RuntimeContext', () => ({
  useRuntime: () => runtimeState,
}));

vi.mock('../contexts/usePipeline', () => ({
  usePipeline: () => ({
    isPipelineRunning: false,
    startPipeline: vi.fn(),
  }),
}));

vi.mock('../components/Themes/ThemeTaxonomyTable', () => ({
  default: ({ pipeline, categoryFilter }) => (
    <div data-testid="taxonomy">
      {pipeline}:{categoryFilter ?? 'none'}
    </div>
  ),
}));

vi.mock('../components/Themes/ThemeReviewDialog', () => ({
  default: ({ open }) => (open ? <h2>Theme Review</h2> : null),
}));

vi.mock('../components/Themes/ModelSettingsModal', () => ({
  default: () => null,
}));

vi.mock('../api/themes', () => {
  const getThemeRankings = vi.fn().mockResolvedValue({
    date: '2026-04-09',
    total_themes: 1,
    rankings: [
      {
        theme_cluster_id: 1,
        theme: 'AI Infrastructure',
        rank: 1,
        momentum_score: 84,
        mention_velocity: 1.7,
        mentions_7d: 18,
        basket_rs_vs_spy: 72,
        basket_return_1w: 3.8,
        pct_above_50ma: 78,
        num_constituents: 4,
        top_tickers: ['NVDA', 'AVGO'],
        status: 'trending',
      },
    ],
  });

  return {
    dismissAlert: vi.fn(),
    getAlerts: vi.fn().mockResolvedValue({ total: 0, unread: 0, alerts: [] }),
    getCandidateThemeQueue: vi.fn().mockResolvedValue({ total: 0, items: [] }),
    getEmergingThemes: vi.fn().mockResolvedValue({ count: 0, themes: [] }),
    getFailedItemsCount: vi.fn().mockResolvedValue({ failed_count: 0 }),
    getL1Categories: vi.fn().mockResolvedValue({
      categories: [{ category: 'technology', count: 1 }],
    }),
    getL1Rankings: vi.fn().mockResolvedValue({
      rankings: [{ id: 100, display_name: 'Technology', rank: 1, momentum_score: 80 }],
    }),
    getMergeSuggestions: vi.fn().mockResolvedValue({ total: 0, suggestions: [] }),
    getPipelineObservability: vi.fn().mockResolvedValue({ metrics: {}, alerts: [] }),
    getThemeRankings,
    getThemesBootstrap: vi.fn(),
    runPipelineAsync: vi.fn(),
    getThemeDetail: vi.fn(),
    getThemeHistory: vi.fn(),
    getThemeRelationshipGraph: vi.fn(),
    getThemeMentions: vi.fn(),
  };
});

function renderPage() {
  return renderWithProviders(<ThemesPage />);
}

describe('ThemesPage', () => {
  beforeEach(() => {
    localStorage.clear();
    runtimeState.runtimeReady = true;
    runtimeState.uiSnapshots = { themes: false };
  });

  it('resets grouped category filter when pipeline toggles', async () => {
    renderPage();
    await waitFor(() => expect(screen.getByTestId('taxonomy')).toBeInTheDocument());
    expect(screen.getByTestId('taxonomy')).toHaveTextContent('technical:none');

    const categoryChip = await screen.findByRole('button', { name: /technology \(1\)/i });
    fireEvent.click(categoryChip);
    expect(screen.getByTestId('taxonomy')).toHaveTextContent('technical:technology');

    fireEvent.click(screen.getByRole('button', { name: /fundamental/i }));
    expect(screen.getByTestId('taxonomy')).toHaveTextContent('fundamental:none');
  });

  it('supports flat-view switch and opens review surface', async () => {
    renderPage();
    fireEvent.click(screen.getByRole('button', { name: 'All Themes' }));

    await waitFor(() => {
      expect(screen.getByText('Theme Rankings')).toBeInTheDocument();
    });

    fireEvent.click(screen.getByRole('button', { name: 'Review' }));
    expect(screen.getByText('Theme Review')).toBeInTheDocument();
  });
});

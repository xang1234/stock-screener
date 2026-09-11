import { screen } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import { getThemeMentions } from '../../api/themes';
import { renderWithProviders } from '../../test/renderWithProviders';
import ThemeSourcesModal from './ThemeSourcesModal';

vi.mock('../../api/themes', () => ({ getThemeMentions: vi.fn() }));

describe('ThemeSourcesModal attachment evidence', () => {
  beforeEach(() => vi.clearAllMocks());

  it('shows pending image preparation and its source link without adding another source row', async () => {
    getThemeMentions.mockResolvedValue({
      theme_id: 7,
      theme_name: 'Optics',
      total_count: 1,
      mentions: [{
        mention_id: 11,
        content_title: 'CPO update',
        content_url: 'https://x.com/example/status/1',
        author: 'Analyst',
        published_at: '2026-09-11T00:00:00Z',
        excerpt: 'Attached chart gives production data.',
        sentiment: 'bullish',
        confidence: 0.8,
        tickers: [],
        source_type: 'twitter',
        source_name: 'Research list',
        attachment_status: 'pending',
        attachments: [{
          kind: 'image',
          url: 'https://pbs.twimg.com/media/chart.jpg',
          status: 'pending',
          error_code: null,
          warnings: ['low contrast'],
        }],
      }],
    });

    renderWithProviders(
      <ThemeSourcesModal open onClose={vi.fn()} themeId={7} themeName="Optics" />,
    );

    expect(await screen.findByText('Found 1 news items mentioning this theme')).toBeInTheDocument();
    expect(screen.getByText('Attachments processing')).toBeInTheDocument();
    expect(screen.getByRole('link', { name: 'Image attachment' })).toHaveAttribute(
      'href', 'https://pbs.twimg.com/media/chart.jpg',
    );
    expect(screen.getByText('Attachment note: low contrast')).toBeInTheDocument();
  });
});

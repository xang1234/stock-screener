import { describe, it, expect } from 'vitest';
import { screen } from '@testing-library/react';
import { renderWithProviders } from '../../test/renderWithProviders';
import RRGChart from './RRGChart';
import { QUADRANT_COLORS } from './rrgColors';

const sampleData = {
  date: '2024-09-29',
  market: 'US',
  scope: 'groups',
  groups: [
    {
      industry_group: 'AlphaTech',
      rank: 1,
      num_stocks: 12,
      avg_rs_rating: 64.0,
      quadrant: 'Leading',
      is_provisional: false,
      current: { date: '2024-09-29', x: 108.3, y: 106.1 },
      tail: [
        { date: '2024-08-11', x: 104.0, y: 98.0 },
        { date: '2024-09-29', x: 108.3, y: 106.1 },
      ],
    },
    {
      industry_group: 'BetaMetals',
      rank: 2,
      num_stocks: 8,
      avg_rs_rating: 36.0,
      quadrant: 'Lagging',
      is_provisional: false,
      current: { date: '2024-09-29', x: 91.7, y: 93.9 },
      tail: [
        { date: '2024-08-11', x: 95.0, y: 99.0 },
        { date: '2024-09-29', x: 91.7, y: 93.9 },
      ],
    },
  ],
};

describe('RRGChart', () => {
  it('renders the header with scope and as-of date', () => {
    renderWithProviders(<RRGChart data={sampleData} />);
    expect(screen.getByText(/Relative Rotation Graph/)).toBeInTheDocument();
    expect(screen.getByText(/Groups/)).toBeInTheDocument();
    expect(screen.getByText(/2024-09-29/)).toBeInTheDocument();
  });

  it('labels the sectors scope when provided', () => {
    renderWithProviders(<RRGChart data={{ ...sampleData, scope: 'sectors' }} />);
    expect(screen.getByText(/Sectors/)).toBeInTheDocument();
  });

  it('shows a spinner while loading', () => {
    renderWithProviders(<RRGChart data={null} isLoading />);
    expect(screen.getByRole('progressbar')).toBeInTheDocument();
  });

  it('shows an error alert on failure', () => {
    renderWithProviders(<RRGChart data={null} error={{ message: 'boom' }} />);
    expect(screen.getByText(/Failed to load RRG data/)).toBeInTheDocument();
    expect(screen.getByText(/boom/)).toBeInTheDocument();
  });

  it('shows an empty-state message when there are no groups', () => {
    renderWithProviders(<RRGChart data={{ groups: [] }} />);
    expect(screen.getByText(/No RRG data available/)).toBeInTheDocument();
  });

  it('exposes stable quadrant colors', () => {
    // Guards the live + static charts against color drift.
    expect(QUADRANT_COLORS).toMatchObject({
      Leading: '#4caf50',
      Weakening: '#ff9800',
      Lagging: '#f44336',
      Improving: '#2196f3',
    });
  });
});

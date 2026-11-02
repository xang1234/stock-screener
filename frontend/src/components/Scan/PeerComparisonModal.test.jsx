import { fireEvent, screen, within } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';

import { renderWithProviders } from '../../test/renderWithProviders';
import PeerComparisonModal from './PeerComparisonModal';

const { peers } = vi.hoisted(() => ({
  peers: [
    { symbol: 'IPO', composite_score: 72, minervini_score: null, canslim_score: null },
    { symbol: 'BEST', composite_score: 65, minervini_score: 65, canslim_score: 62 },
    { symbol: 'CANSLIM', composite_score: 60, minervini_score: null, canslim_score: 60 },
  ],
}));

vi.mock('../../api/client', () => ({
  default: { get: vi.fn(async () => ({ data: peers })) },
}));
vi.mock('../common/AddToWatchlistMenu', () => ({ default: () => null }));

const peerOrder = () =>
  screen
    .getAllByRole('row')
    .slice(1)
    .map((row) => within(row).getAllByRole('cell')[1].textContent);

describe('PeerComparisonModal', () => {
  it('ranks peers a core screener scored ahead of a higher IPO-only composite', async () => {
    renderWithProviders(
      <PeerComparisonModal open onClose={() => {}} scanId="scan-1" symbol="BEST" onOpenChart={() => {}} />
    );

    await screen.findByText('IPO');
    expect(peerOrder()).toEqual(['BEST', 'CANSLIM', 'IPO']);

    fireEvent.click(screen.getByText('Composite Score'));
    expect(peerOrder()).toEqual(['CANSLIM', 'BEST', 'IPO']);
  });
});

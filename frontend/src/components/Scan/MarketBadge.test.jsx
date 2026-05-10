import { describe, it, expect } from 'vitest';
import { screen } from '@testing-library/react';
import { renderWithProviders } from '../../test/renderWithProviders';
import MarketBadge from './MarketBadge';

describe('MarketBadge', () => {
  it('renders nothing when market is null', () => {
    const { container } = renderWithProviders(<MarketBadge market={null} />);
    expect(container).toBeEmptyDOMElement();
  });

  it('renders the market code for US/HK/IN/JP/KR/TW/CN/CA/DE', () => {
    for (const code of ['US', 'HK', 'IN', 'JP', 'KR', 'TW', 'CN', 'CA', 'DE']) {
      const { unmount } = renderWithProviders(<MarketBadge market={code} />);
      expect(screen.getByTestId(`market-badge-${code}`)).toBeInTheDocument();
      expect(screen.getByText(code)).toBeInTheDocument();
      unmount();
    }
  });

  it('falls back to the raw code for unknown markets', () => {
    renderWithProviders(<MarketBadge market="XX" />);
    // getByTestId still works and label shows the passed code
    expect(screen.getByTestId('market-badge-XX')).toBeInTheDocument();
    expect(screen.getByText('XX')).toBeInTheDocument();
  });
});

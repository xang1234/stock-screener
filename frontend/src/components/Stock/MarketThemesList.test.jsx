import { screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { renderWithProviders } from '../../test/renderWithProviders';
import MarketThemesList from './MarketThemesList';

const HK_THEMES = [
  '5G',
  'Electronic Information',
  'HK-Mainland Nexus',
  'Share Buyback',
  'Telecom Equipment',
];

describe('MarketThemesList', () => {
  describe('compact variant', () => {
    it('renders a dash for an empty array', () => {
      renderWithProviders(<MarketThemesList themes={[]} variant="compact" />);
      expect(screen.getByText('-')).toBeInTheDocument();
    });

    it('renders a single chip with no overflow indicator for one theme', () => {
      renderWithProviders(<MarketThemesList themes={['5G']} variant="compact" />);
      expect(screen.getByText('5G')).toBeInTheDocument();
      expect(screen.queryByText(/^\+\d+$/)).not.toBeInTheDocument();
    });

    it('renders first chip plus "+N" counter when there are many themes', () => {
      renderWithProviders(<MarketThemesList themes={HK_THEMES} variant="compact" />);
      expect(screen.getByText('5G')).toBeInTheDocument();
      expect(screen.getByText('+4')).toBeInTheDocument();
      // Non-inline themes should NOT be in the DOM before the popover opens.
      expect(screen.queryByText('Telecom Equipment')).not.toBeInTheDocument();
    });

    it('opens a popover with every theme when the "+N" chip is clicked', async () => {
      const user = userEvent.setup();
      renderWithProviders(<MarketThemesList themes={HK_THEMES} variant="compact" />);

      await user.click(screen.getByText('+4'));

      const dialog = await screen.findByRole('presentation');
      const scope = within(dialog);
      HK_THEMES.forEach((theme) => {
        expect(scope.getByText(theme)).toBeInTheDocument();
      });
      expect(scope.getByText('Market Themes (5)')).toBeInTheDocument();
    });

    it('does not propagate "+N" clicks to parent click handlers', async () => {
      const onRowClick = vi.fn();
      const user = userEvent.setup();
      renderWithProviders(
        <div onClick={onRowClick}>
          <MarketThemesList themes={HK_THEMES} variant="compact" />
        </div>
      );

      await user.click(screen.getByText('+4'));
      expect(onRowClick).not.toHaveBeenCalled();
    });

    it('lets first-chip clicks bubble so the row stays clickable', async () => {
      const onRowClick = vi.fn();
      const user = userEvent.setup();
      renderWithProviders(
        <div onClick={onRowClick}>
          <MarketThemesList themes={HK_THEMES} variant="compact" />
        </div>
      );

      await user.click(screen.getByText('5G'));
      expect(onRowClick).toHaveBeenCalledTimes(1);
    });

    it('does not propagate clicks from inside the popover to parent', async () => {
      const onRowClick = vi.fn();
      const user = userEvent.setup();
      renderWithProviders(
        <div onClick={onRowClick}>
          <MarketThemesList themes={HK_THEMES} variant="compact" />
        </div>
      );

      await user.click(screen.getByText('+4'));
      const dialog = await screen.findByRole('presentation');
      // Click a chip rendered inside the popover.
      await user.click(within(dialog).getByText('Telecom Equipment'));
      expect(onRowClick).not.toHaveBeenCalled();
    });
  });

  describe('wrap variant', () => {
    it('shows the default empty caption for an empty array', () => {
      renderWithProviders(<MarketThemesList themes={[]} variant="wrap" />);
      expect(
        screen.getByText(/No market taxonomy themes are available/i)
      ).toBeInTheDocument();
    });

    it('renders every theme as a chip without hiding any', () => {
      renderWithProviders(<MarketThemesList themes={HK_THEMES} variant="wrap" />);
      HK_THEMES.forEach((theme) => {
        expect(screen.getByText(theme)).toBeInTheDocument();
      });
    });

    it('allows long chip labels to wrap rather than truncate', () => {
      renderWithProviders(
        <MarketThemesList themes={['HK-Mainland Nexus']} variant="wrap" />
      );
      const label = screen.getByText('HK-Mainland Nexus');
      // The component applies `whiteSpace: normal` via the sx prop so long
      // labels grow vertically instead of being ellipsis-clipped.
      expect(getComputedStyle(label).whiteSpace).toBe('normal');
    });
  });
});

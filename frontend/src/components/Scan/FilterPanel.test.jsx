import { screen, within, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { renderWithProviders } from '../../test/renderWithProviders';
import FilterPanel from './FilterPanel';

/** Default filters object matching ScanPage.jsx initial state. */
const defaultFilters = () => ({
  symbolSearch: '',
  stage: null,
  ratings: [],
  ibdIndustries: { values: [], mode: 'include' },
  gicsSectors: { values: [], mode: 'include' },
  minVolume: null,
  minMarketCap: null,
  compositeScore: { min: null, max: null },
  minerviniScore: { min: null, max: null },
  canslimScore: { min: null, max: null },
  ipoScore: { min: null, max: null },
  customScore: { min: null, max: null },
  volBreakthroughScore: { min: null, max: null },
  seSetupScore: { min: null, max: null },
  seDistanceToPivot: { min: null, max: null },
  seBbSqueeze: { min: null, max: null },
  seVolumeVs50d: { min: null, max: null },
  seSetupReady: null,
  seRsLineNewHigh: null,
  rsRating: { min: null, max: null },
  rs1m: { min: null, max: null },
  rs3m: { min: null, max: null },
  rs12m: { min: null, max: null },
  price: { min: null, max: null },
  adrPercent: { min: null, max: null },
  epsGrowth: { min: null, max: null },
  salesGrowth: { min: null, max: null },
  vcpScore: { min: null, max: null },
  vcpPivot: { min: null, max: null },
  vcpDetected: null,
  vcpReady: null,
  maAlignment: null,
  passesTemplate: null,
  perfDay: { min: null, max: null },
  perfWeek: { min: null, max: null },
  perfMonth: { min: null, max: null },
  perf3m: { min: null, max: null },
  perf6m: { min: null, max: null },
  gapPercent: { min: null, max: null },
  volumeSurge: { min: null, max: null },
  ema10Distance: { min: null, max: null },
  ema20Distance: { min: null, max: null },
  ema50Distance: { min: null, max: null },
  week52HighDistance: { min: null, max: null },
  week52LowDistance: { min: null, max: null },
  ipoAfter: null,
  beta: { min: null, max: null },
  betaAdjRs: { min: null, max: null },
  epsRating: { min: null, max: null },
});

/** Minimal props for a renderable FilterPanel. */
const makeProps = (overrides = {}) => ({
  filters: defaultFilters(),
  onFilterChange: vi.fn(),
  onReset: vi.fn(),
  expanded: true,
  onToggle: vi.fn(),
  ...overrides,
});

describe('FilterPanel', () => {
  // ── SE filter controls render ────────────────────────────────────────
  describe('SE filter controls render', () => {
    it('renders SE Score range input', () => {
      renderWithProviders(<FilterPanel {...makeProps()} />);
      expect(screen.getByText('SE Score')).toBeInTheDocument();
    });

    it('renders Pvt Dist range input', () => {
      renderWithProviders(<FilterPanel {...makeProps()} />);
      expect(screen.getByText('Pvt Dist')).toBeInTheDocument();
    });

    it('renders Squeeze range input', () => {
      renderWithProviders(<FilterPanel {...makeProps()} />);
      expect(screen.getByText('Squeeze')).toBeInTheDocument();
    });

    it('renders Vol/50d range input', () => {
      renderWithProviders(<FilterPanel {...makeProps()} />);
      expect(screen.getByText('Vol/50d')).toBeInTheDocument();
    });

    it('renders SE Ready checkbox', () => {
      renderWithProviders(<FilterPanel {...makeProps()} />);
      expect(screen.getByText('SE Ready')).toBeInTheDocument();
    });

    it('renders RS Hi checkbox', () => {
      renderWithProviders(<FilterPanel {...makeProps()} />);
      expect(screen.getByText('RS Hi')).toBeInTheDocument();
    });
  });

  // ── SE range filter interactions ─────────────────────────────────────
  describe('SE range filter interactions', () => {
    it('fires onFilterChange with seSetupScore min after debounce', async () => {
      const onFilterChange = vi.fn();
      renderWithProviders(
        <FilterPanel {...makeProps({ onFilterChange })} />
      );

      const user = userEvent.setup();
      // SE Score is a minOnly range input — find its single spinbutton
      // by locating the label first, then the input within the same grid item
      const seScoreLabel = screen.getByText('SE Score');
      const seScoreContainer = seScoreLabel.closest('[class*="MuiGrid-item"]');
      const input = within(seScoreContainer).getByRole('spinbutton');

      await user.type(input, '70');

      await waitFor(() => {
        expect(onFilterChange).toHaveBeenCalledWith(
          expect.objectContaining({ seSetupScore: { min: 70, max: null } })
        );
      }, { timeout: 1000 });
    });

    it('fires onFilterChange with seDistanceToPivot range', async () => {
      const onFilterChange = vi.fn();
      renderWithProviders(
        <FilterPanel {...makeProps({ onFilterChange })} />
      );

      const user = userEvent.setup();
      const pvtLabel = screen.getByText('Pvt Dist');
      const pvtContainer = pvtLabel.closest('[class*="MuiGrid-item"]');
      const inputs = within(pvtContainer).getAllByRole('spinbutton');
      // Pvt Dist has both min and max inputs
      expect(inputs.length).toBe(2);

      await user.type(inputs[0], '-5');

      await waitFor(() => {
        expect(onFilterChange).toHaveBeenCalledWith(
          expect.objectContaining({ seDistanceToPivot: expect.objectContaining({ min: -5 }) })
        );
      }, { timeout: 1000 });
    });
  });

  // ── SE checkbox filter interactions ──────────────────────────────────
  describe('SE checkbox filter interactions', () => {
    it('sets seSetupReady=true when Yes is clicked', async () => {
      const onFilterChange = vi.fn();
      renderWithProviders(
        <FilterPanel {...makeProps({ onFilterChange })} />
      );

      const user = userEvent.setup();
      const seReadyLabel = screen.getByText('SE Ready');
      const seReadyContainer = seReadyLabel.closest('[class*="MuiGrid-item"]');
      const yesBtn = within(seReadyContainer).getByText('Yes');

      await user.click(yesBtn);
      expect(onFilterChange).toHaveBeenCalledWith(
        expect.objectContaining({ seSetupReady: true })
      );
    });

    it('sets seRsLineNewHigh=false when No is clicked', async () => {
      const onFilterChange = vi.fn();
      renderWithProviders(
        <FilterPanel {...makeProps({ onFilterChange })} />
      );

      const user = userEvent.setup();
      const rsHiLabel = screen.getByText('RS Hi');
      const rsHiContainer = rsHiLabel.closest('[class*="MuiGrid-item"]');
      const noBtn = within(rsHiContainer).getByText('No');

      await user.click(noBtn);
      expect(onFilterChange).toHaveBeenCalledWith(
        expect.objectContaining({ seRsLineNewHigh: false })
      );
    });

    it('resets seSetupReady to null when Yes is toggled off', async () => {
      const onFilterChange = vi.fn();
      const filters = { ...defaultFilters(), seSetupReady: true };
      renderWithProviders(
        <FilterPanel {...makeProps({ filters, onFilterChange })} />
      );

      const user = userEvent.setup();
      const seReadyLabel = screen.getByText('SE Ready');
      const seReadyContainer = seReadyLabel.closest('[class*="MuiGrid-item"]');
      const yesBtn = within(seReadyContainer).getByText('Yes');

      await user.click(yesBtn);
      // CompactCheckbox toggles: clicking already-active value -> null
      expect(onFilterChange).toHaveBeenCalledWith(
        expect.objectContaining({ seSetupReady: null })
      );
    });
  });

  // ── SE active filter chips ───────────────────────────────────────────
  describe('SE active filter chips', () => {
    it('shows "SE Score: >=70" chip when seSetupScore.min is set', () => {
      const filters = { ...defaultFilters(), seSetupScore: { min: 70, max: null } };
      renderWithProviders(
        <FilterPanel {...makeProps({ filters })} />
      );
      // The expanded area shows all active chips with the "Active:" prefix
      expect(screen.getByText(/SE Score:.*≥70/)).toBeInTheDocument();
    });

    it('shows "SE Ready: Yes" chip when seSetupReady=true', () => {
      const filters = { ...defaultFilters(), seSetupReady: true };
      renderWithProviders(
        <FilterPanel {...makeProps({ filters })} />
      );
      expect(screen.getByText('SE Ready: Yes')).toBeInTheDocument();
    });

    it('shows "RS New Hi: No" chip when seRsLineNewHigh=false', () => {
      const filters = { ...defaultFilters(), seRsLineNewHigh: false };
      renderWithProviders(
        <FilterPanel {...makeProps({ filters })} />
      );
      expect(screen.getByText('RS New Hi: No')).toBeInTheDocument();
    });

    it('removes filter when chip delete is clicked', async () => {
      const onFilterChange = vi.fn();
      const filters = { ...defaultFilters(), seSetupReady: true };
      renderWithProviders(
        <FilterPanel {...makeProps({ filters, onFilterChange })} />
      );

      const user = userEvent.setup();
      // Find the "SE Ready: Yes" chip in the expanded active area
      const chip = screen.getByText('SE Ready: Yes').closest('.MuiChip-root');
      const deleteBtn = within(chip).getByTestId('CancelIcon');

      await user.click(deleteBtn);
      expect(onFilterChange).toHaveBeenCalledWith(
        expect.objectContaining({ seSetupReady: null })
      );
    });
  });

  // ── structural ───────────────────────────────────────────────────────
  describe('structural', () => {
    it('renders all 3 section headers', () => {
      renderWithProviders(<FilterPanel {...makeProps()} />);
      expect(screen.getByText('Fundamental')).toBeInTheDocument();
      expect(screen.getByText('Technical')).toBeInTheDocument();
      expect(screen.getByText('Rating / Score')).toBeInTheDocument();
    });

    it('calls onReset when Reset button is clicked', async () => {
      const onReset = vi.fn();
      renderWithProviders(
        <FilterPanel {...makeProps({ onReset })} />
      );

      const user = userEvent.setup();
      await user.click(screen.getByText('Reset'));
      expect(onReset).toHaveBeenCalledTimes(1);
    });

    it('shows active count badge on Rating section when SE filter is set', () => {
      const filters = { ...defaultFilters(), seSetupScore: { min: 70, max: null } };
      renderWithProviders(
        <FilterPanel {...makeProps({ filters })} />
      );
      // FilterSection shows "{N} active" chip when activeCount > 0
      expect(screen.getByText('1 active')).toBeInTheDocument();
    });
  });
});

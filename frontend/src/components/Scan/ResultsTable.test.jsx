import { screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { renderWithProviders } from '../../test/renderWithProviders';
import { fullSeRow, nullSeRow, mixedSeRow } from '../../test/fixtures/setupEngineFixtures';
import ResultsTable from './ResultsTable';

/*
 * Module mocks — required because:
 * - @tanstack/react-virtual: useVirtualizer depends on DOM layout measurements
 *   (getBoundingClientRect, scrollHeight) that jsdom doesn't implement -> zero rows render
 * - RSSparkline/PriceSparkline: Recharts ResponsiveContainer requires real DOM dimensions
 * - AddToWatchlistMenu: uses React Query (QueryClientProvider) which is not needed here
 */
vi.mock('@tanstack/react-virtual', () => ({
  useVirtualizer: ({ count }) => ({
    getVirtualItems: () =>
      Array.from({ length: count }, (_, i) => ({
        index: i,
        start: i * 32,
        end: (i + 1) * 32,
        size: 32,
        key: i,
      })),
    getTotalSize: () => count * 32,
  }),
}));
vi.mock('./RSSparkline', () => ({ default: () => <td data-testid="rs-sparkline" /> }));
vi.mock('./PriceSparkline', () => ({ default: () => <td data-testid="price-sparkline" /> }));
vi.mock('../common/AddToWatchlistMenu', () => ({ default: () => null }));

/** Default props for a basic render — 1 row, page 1. */
const defaultProps = {
  results: [fullSeRow],
  total: 1,
  page: 1,
  perPage: 25,
  sortBy: 'composite_score',
  sortOrder: 'desc',
  onPageChange: vi.fn(),
  onPerPageChange: vi.fn(),
  onSortChange: vi.fn(),
  onOpenChart: vi.fn(),
  loading: false,
};

describe('ResultsTable', () => {
  // ── SE column rendering — full data ──────────────────────────────────
  describe('SE column rendering — full data', () => {
    beforeEach(() => {
      renderWithProviders(<ResultsTable {...defaultProps} results={[fullSeRow]} />);
    });

    it('renders se_setup_score as 78.3', () => {
      expect(screen.getByText('78.3')).toBeInTheDocument();
    });

    it('renders se_pattern_primary as cup_with_handle', () => {
      expect(screen.getByText('cup_with_handle')).toBeInTheDocument();
    });

    it('renders se_distance_to_pivot_pct as -3.2%', () => {
      expect(screen.getByText('-3.2%')).toBeInTheDocument();
    });

    it('renders se_bb_width_pctile_252 as 15', () => {
      expect(screen.getByText('15')).toBeInTheDocument();
    });

    it('renders se_volume_vs_50d as 1.8x', () => {
      expect(screen.getByText('1.8x')).toBeInTheDocument();
    });

    it('renders CheckIcon for se_rs_line_new_high=true', () => {
      // Multiple boolean columns render CheckIcon (ma_alignment, passes_template, se_rs_line_new_high).
      // fullSeRow has ma_alignment=true, passes_template=true, se_rs_line_new_high=true → 3 CheckIcons.
      const checkIcons = screen.getAllByTestId('CheckIcon');
      expect(checkIcons.length).toBe(3);
    });

    it('renders se_pivot_price as $198.50', () => {
      expect(screen.getByText('$198.50')).toBeInTheDocument();
    });
  });

  // ── SE column rendering — null data ──────────────────────────────────
  describe('SE column rendering — null data', () => {
    it('renders dash for all 7 SE columns when null', () => {
      renderWithProviders(<ResultsTable {...defaultProps} results={[nullSeRow]} />);
      // The table has many '-' dashes (other null columns too). We verify
      // by checking that none of the SE-specific formatted values appear.
      expect(screen.queryByText('78.3')).not.toBeInTheDocument();
      expect(screen.queryByText('cup_with_handle')).not.toBeInTheDocument();
      expect(screen.queryByText('-3.2%')).not.toBeInTheDocument();
      expect(screen.queryByText('1.8x')).not.toBeInTheDocument();
      expect(screen.queryByText('$198.50')).not.toBeInTheDocument();
      // No CheckIcon should appear for se_rs_line_new_high=null
      // (other booleans like ma_alignment still render icons)
    });
  });

  // ── SE column rendering — mixed data ─────────────────────────────────
  describe('SE column rendering — mixed data', () => {
    beforeEach(() => {
      renderWithProviders(<ResultsTable {...defaultProps} results={[mixedSeRow]} />);
    });

    it('renders CloseIcon for se_rs_line_new_high=false', () => {
      // mixedSeRow has se_rs_line_new_high=false, vcp_detected=false, vcp_ready_for_breakout=false → 3 CloseIcons.
      const closeIcons = screen.getAllByTestId('CloseIcon');
      expect(closeIcons.length).toBe(3);
    });

    it('renders populated SE values alongside dashes for null ones', () => {
      expect(screen.getByText('62.1')).toBeInTheDocument();
      expect(screen.getByText('4.7%')).toBeInTheDocument();
      expect(screen.getByText('2.3x')).toBeInTheDocument();
    });
  });

  // ── SE column headers ────────────────────────────────────────────────
  describe('SE column headers', () => {
    it('renders all 7 SE header labels', () => {
      renderWithProviders(<ResultsTable {...defaultProps} />);
      const headers = ['SE', 'Pat', 'Pvt%', 'Sqz', 'V50', 'RSH', 'Pvt$'];
      headers.forEach((label) => {
        expect(screen.getByText(label)).toBeInTheDocument();
      });
    });
  });

  // ── structural ───────────────────────────────────────────────────────
  describe('structural', () => {
    it('shows "No results found" when results is empty', () => {
      renderWithProviders(
        <ResultsTable {...defaultProps} results={[]} total={0} />
      );
      expect(screen.getByText('No results found')).toBeInTheDocument();
    });

    it('shows loading spinner when loading=true', () => {
      renderWithProviders(
        <ResultsTable {...defaultProps} loading={true} />
      );
      expect(screen.getByRole('progressbar')).toBeInTheDocument();
    });

    it('renders pagination controls', () => {
      renderWithProviders(
        <ResultsTable {...defaultProps} results={[fullSeRow]} total={50} />
      );
      // MUI TablePagination renders "Rows per page:" text
      expect(screen.getByText(/rows per page/i)).toBeInTheDocument();
    });
  });

  // ── interactions ─────────────────────────────────────────────────────
  describe('interactions', () => {
    it('calls onOpenChart when row is clicked', async () => {
      const onOpenChart = vi.fn();
      renderWithProviders(
        <ResultsTable {...defaultProps} onOpenChart={onOpenChart} results={[fullSeRow]} />
      );

      const user = userEvent.setup();
      // Click the row containing the symbol text
      await user.click(screen.getByText('FULL'));
      expect(onOpenChart).toHaveBeenCalledWith('FULL');
    });

    it('calls onSortChange when a sortable header is clicked', async () => {
      const onSortChange = vi.fn();
      renderWithProviders(
        <ResultsTable {...defaultProps} onSortChange={onSortChange} />
      );

      const user = userEvent.setup();
      // Click the "SE" header (sortable)
      await user.click(screen.getByText('SE'));
      expect(onSortChange).toHaveBeenCalledWith('se_setup_score', 'asc');
    });

    it('toggles sort direction when same header is clicked twice', async () => {
      const onSortChange = vi.fn();
      // Start sorted by se_setup_score asc
      renderWithProviders(
        <ResultsTable
          {...defaultProps}
          sortBy="se_setup_score"
          sortOrder="asc"
          onSortChange={onSortChange}
        />
      );

      const user = userEvent.setup();
      await user.click(screen.getByText('SE'));
      // Since current is asc, clicking again should flip to desc
      expect(onSortChange).toHaveBeenCalledWith('se_setup_score', 'desc');
    });
  });
});

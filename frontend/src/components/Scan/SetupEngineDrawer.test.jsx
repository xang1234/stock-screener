import { screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { renderWithProviders } from '../../test/renderWithProviders';
import SetupEngineDrawer from './SetupEngineDrawer';
import {
  fullPayloadStock,
  partialPayloadStock,
  insufficientDataStock,
  noExplainStock,
  malformedExplainStock,
} from '../../test/fixtures/setupEngineFixtures';

const defaultProps = { open: true, onClose: vi.fn() };

describe('SetupEngineDrawer — full payload', () => {
  beforeEach(() => {
    renderWithProviders(
      <SetupEngineDrawer {...defaultProps} stockData={fullPayloadStock} />
    );
  });

  it('shows the stock symbol in the header', () => {
    expect(screen.getByText(/NVDA/)).toBeInTheDocument();
  });

  it('renders setup score with correct value', () => {
    // Score appears in both main grid and candidate section
    const matches = screen.getAllByText('82.5');
    expect(matches.length).toBeGreaterThanOrEqual(1);
  });

  it('renders quality score', () => {
    const matches = screen.getAllByText('75.3');
    expect(matches.length).toBeGreaterThanOrEqual(1);
  });

  it('renders readiness score', () => {
    const matches = screen.getAllByText('68.9');
    expect(matches.length).toBeGreaterThanOrEqual(1);
  });

  it('formats pattern name from snake_case to Title Case', () => {
    // "Three Weeks Tight" appears in main view and candidate section
    const matches = screen.getAllByText('Three Weeks Tight');
    expect(matches.length).toBeGreaterThanOrEqual(1);
  });

  it('renders confidence chip with percentage', () => {
    // 85% appears for both main confidence and first candidate
    const matches = screen.getAllByText('85%');
    expect(matches.length).toBeGreaterThanOrEqual(1);
  });

  it('renders READY chip', () => {
    expect(screen.getByText('READY')).toBeInTheDocument();
  });

  it('renders timeframe chip', () => {
    // "weekly" appears for main stock and first candidate
    const matches = screen.getAllByText('weekly');
    expect(matches.length).toBeGreaterThanOrEqual(1);
  });

  it('renders passed checks with human-readable names', () => {
    // Check names may appear in both passed_checks and candidate.checks sections
    const setupChecks = screen.getAllByText('Setup score meets threshold');
    expect(setupChecks.length).toBeGreaterThanOrEqual(1);
    const volumeChecks = screen.getAllByText('Volume above minimum');
    expect(volumeChecks.length).toBeGreaterThanOrEqual(1);
    expect(screen.getByText('RS leadership confirmed')).toBeInTheDocument();
  });

  it('renders failed checks', () => {
    expect(screen.getByText('ATR14 exceeds volatility limit')).toBeInTheDocument();
  });

  it('renders key levels with formatted prices', () => {
    // $142.50 appears in pivot info, key_levels, and candidate pivot
    const pivotMatches = screen.getAllByText('$142.50');
    expect(pivotMatches.length).toBeGreaterThanOrEqual(1);
    expect(screen.getByText('$135.20')).toBeInTheDocument();
  });

  it('does not render invalidation flags section when empty', () => {
    expect(screen.queryByText('INVALIDATION FLAGS')).not.toBeInTheDocument();
  });

  it('renders pattern candidates with scores', () => {
    // Second candidate: flat_base (unique name)
    expect(screen.getByText('Flat Base')).toBeInTheDocument();
  });

  it('renders candidate confidence chips', () => {
    // 62% for flat_base candidate (unique)
    expect(screen.getByText('62%')).toBeInTheDocument();
  });

  it('renders candidate notes', () => {
    expect(screen.getByText('Tight weekly closes within 1.5% range')).toBeInTheDocument();
  });
});

describe('SetupEngineDrawer — partial/degraded payload', () => {
  beforeEach(() => {
    renderWithProviders(
      <SetupEngineDrawer {...defaultProps} stockData={partialPayloadStock} />
    );
  });

  it('shows degraded data info banner', () => {
    expect(screen.getByText(/Some data sources are degraded/)).toBeInTheDocument();
  });

  it('renders present scores', () => {
    // Scores appear in main grid and candidate section
    const setupMatches = screen.getAllByText('55.0');
    expect(setupMatches.length).toBeGreaterThanOrEqual(1);
    const readinessMatches = screen.getAllByText('48.2');
    expect(readinessMatches.length).toBeGreaterThanOrEqual(1);
  });

  it('shows "-" for null quality score', () => {
    // The quality score is null, so ScoreItem renders '-'
    const dashes = screen.getAllByText('-');
    expect(dashes.length).toBeGreaterThanOrEqual(1);
  });

  it('does not render FAILED CHECKS header when list is empty', () => {
    expect(screen.queryByText('FAILED CHECKS')).not.toBeInTheDocument();
  });

  it('renders NOT READY chip', () => {
    expect(screen.getByText('NOT READY')).toBeInTheDocument();
  });
});

describe('SetupEngineDrawer — insufficient data', () => {
  beforeEach(() => {
    renderWithProviders(
      <SetupEngineDrawer {...defaultProps} stockData={insufficientDataStock} />
    );
  });

  it('shows insufficient data warning banner', () => {
    expect(screen.getByText(/Insufficient historical data/)).toBeInTheDocument();
  });

  it('hides scores grid when all scores are null', () => {
    // No scores should be rendered — the grid condition checks for at least one non-null
    expect(screen.queryByText('Setup')).not.toBeInTheDocument();
  });

  it('shows "-" for null pattern name', () => {
    const dashes = screen.getAllByText('-');
    expect(dashes.length).toBeGreaterThanOrEqual(1);
  });

  it('renders invalidation flags', () => {
    expect(screen.getByText('INVALIDATION FLAGS')).toBeInTheDocument();
    expect(screen.getByText('Insufficient data')).toBeInTheDocument();
  });
});

describe('SetupEngineDrawer — no explain / malformed explain', () => {
  it('shows "not available" when se_explain is null', () => {
    renderWithProviders(
      <SetupEngineDrawer {...defaultProps} stockData={noExplainStock} />
    );
    expect(screen.getByText(/Setup details not available/)).toBeInTheDocument();
  });

  it('treats array explain as null (SE-F4 type guard)', () => {
    renderWithProviders(
      <SetupEngineDrawer {...defaultProps} stockData={malformedExplainStock} />
    );
    expect(screen.getByText(/Setup details not available/)).toBeInTheDocument();
  });

  it('renders nothing when stockData is null', () => {
    const { container } = renderWithProviders(
      <SetupEngineDrawer {...defaultProps} stockData={null} />
    );
    expect(container.innerHTML).toBe('');
  });

  it('shows loading state while setup payload is being fetched', () => {
    renderWithProviders(
      <SetupEngineDrawer
        {...defaultProps}
        stockData={noExplainStock}
        isLoading={true}
      />
    );
    expect(screen.getByRole('progressbar')).toBeInTheDocument();
    expect(screen.queryByText(/Setup details not available/)).not.toBeInTheDocument();
  });
});

describe('SetupEngineDrawer — interactions', () => {
  it('calls onClose when close button is clicked', async () => {
    const onClose = vi.fn();
    renderWithProviders(
      <SetupEngineDrawer open={true} onClose={onClose} stockData={fullPayloadStock} />
    );

    const user = userEvent.setup();
    const closeButton = screen.getByRole('button');
    await user.click(closeButton);
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it('renders nothing visible when open={false}', () => {
    // MUI Drawer with keepMounted={false} (default) unmounts children when closed
    renderWithProviders(
      <SetupEngineDrawer open={false} onClose={vi.fn()} stockData={fullPayloadStock} />
    );
    expect(screen.queryByText('NVDA')).not.toBeInTheDocument();
  });
});

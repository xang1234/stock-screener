import { render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';

import { createEmptyExpression } from '../filterExpression';
import ScanResultsSection from './ScanResultsSection';

vi.mock('../../../components/Scan/ResultsTable', () => ({
  default: () => <div>previous-results-table</div>,
}));

const baseProps = {
  resultsLoading: false,
  resultsData: { results: [{ symbol: 'NVDA' }], total: 1, unfiltered_total: 2 },
  expression: createEmptyExpression([
    { kind: 'range', field: 'price', min: 10, max: null },
  ]),
  onExport: vi.fn(),
  page: 1,
  perPage: 50,
  sortBy: 'composite_score',
  sortOrder: 'desc',
  onPageChange: vi.fn(),
  onPerPageChange: vi.fn(),
  onSortChange: vi.fn(),
  onOpenChart: vi.fn(),
  onRowHover: vi.fn(),
  onRetry: vi.fn(),
};

describe('ScanResultsSection', () => {
  it('keeps prior rows visible and shows a retryable request error', () => {
    render(
      <ScanResultsSection
        {...baseProps}
        resultsError={new Error('Request timed out')}
      />,
    );

    expect(screen.getByText('previous-results-table')).toBeInTheDocument();
    expect(screen.getByText('Request timed out')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Retry' })).toBeInTheDocument();
  });

  it('does not report an errored initial request as an empty result set', () => {
    render(
      <ScanResultsSection
        {...baseProps}
        resultsData={undefined}
        resultsError={new Error('Server unavailable')}
      />,
    );

    expect(screen.getByText('Could not update scan results')).toBeInTheDocument();
    expect(screen.queryByText('No stocks match the applied logic')).not.toBeInTheDocument();
  });
});

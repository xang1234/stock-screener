import { screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';

import { renderWithProviders } from '../../test/renderWithProviders';
import ExposureResearchPanel from './ExposureResearchPanel';
import { reviewRequiredJob, shadowCompletedJob, shadowPreview } from './fixtures';

describe('ExposureResearchPanel', () => {
  it('shows a usable shadow preview without claiming live membership', () => {
    renderWithProviders(<ExposureResearchPanel job={shadowCompletedJob} preview={shadowPreview} />);
    expect(screen.getByText(/shadow preview/i)).toBeVisible();
    expect(screen.getByText(/not separately disclosed/i)).toBeVisible();
    expect(screen.queryByText(/added to live basket/i)).not.toBeInTheDocument();
    expect(screen.getByText('Research progress — not accepted')).toBeVisible();
    expect(screen.getByText('Primary source — explicit')).toBeVisible();
    expect(screen.getByText(/http_status_404/)).toBeVisible();
  });

  it('does not relabel unknown materiality or research as a verified company', () => {
    renderWithProviders(<ExposureResearchPanel job={shadowCompletedJob} preview={shadowPreview} />);
    expect(screen.queryByText(/\d+% exposure/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/^verified$/i)).not.toBeInTheDocument();
  });

  it('shows the failed review condition while paused', () => {
    renderWithProviders(<ExposureResearchPanel job={reviewRequiredJob} />);
    expect(screen.getByText('review_required')).toBeVisible();
    expect(screen.getByText(/more than one sec cik/i)).toBeVisible();
    expect(screen.queryByText(/shadow preview/i)).not.toBeInTheDocument();
  });

  it('renders original passages as text, never as markup', () => {
    const hostile = {
      ...shadowPreview,
      claims: [{
        ...shadowPreview.claims[0],
        evidence: [{ ...shadowPreview.claims[0].evidence[0], quote: '<img src=x onerror=alert(1)>HBM' }],
      }],
    };
    const { container } = renderWithProviders(
      <ExposureResearchPanel job={shadowCompletedJob} preview={hostile} />,
    );
    expect(container.querySelector('img')).toBeNull();
    expect(screen.getByText('<img src=x onerror=alert(1)>HBM')).toBeVisible();
  });

  it('marks held or stale claims as unusable for new automated use', () => {
    const held = {
      ...shadowPreview,
      claims: [{ ...shadowPreview.claims[0], active_holds: ['disputed'], freshness_state: 'stale' }],
    };
    renderWithProviders(<ExposureResearchPanel job={shadowCompletedJob} preview={held} />);
    expect(screen.getByText('Hold: disputed')).toBeVisible();
    expect(screen.getByText(/held from new automated use/i)).toBeVisible();
  });
});

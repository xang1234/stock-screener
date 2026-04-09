import { screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';

import ThemeInsightsCards, { VelocityIndicator } from './ThemeInsightsCards';
import { renderWithProviders } from '../../../test/renderWithProviders';

describe('ThemeInsightsCards', () => {
  it('renders a placeholder when velocity is not numeric', () => {
    renderWithProviders(<VelocityIndicator velocity="fast" />);

    expect(screen.getByText('-')).toBeInTheDocument();
  });

  it('renders safely when trending props are omitted during bootstrap', () => {
    renderWithProviders(
      <ThemeInsightsCards
        emerging={{
          count: 1,
          themes: [{ theme: 'AI Infrastructure', velocity: 'fast', mentions_7d: 7 }],
        }}
        isLoadingEmerging={false}
        observability={null}
        isLoadingObservability={false}
        alerts={[]}
        isLoadingAlerts={false}
        dismissingAlertId={null}
      />
    );

    expect(screen.getByText('Top Trending')).toBeInTheDocument();
    expect(screen.getByText('Emerging Themes')).toBeInTheDocument();
    expect(screen.getByText('-')).toBeInTheDocument();
  });
});

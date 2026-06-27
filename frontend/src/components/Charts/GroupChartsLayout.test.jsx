import { screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';

import { renderWithProviders } from '../../test/renderWithProviders';
import GroupChartsLayout, { GroupChartCell } from './GroupChartsLayout';

const useMediaQuerySpy = vi.hoisted(() => vi.fn(() => true));

vi.mock('@mui/material', async () => {
  const actual = await vi.importActual('@mui/material');
  return {
    ...actual,
    useMediaQuery: (...args) => useMediaQuerySpy(...args),
  };
});

describe('GroupChartsLayout', () => {
  it('provides the shared two-column desktop chart grid contract', () => {
    renderWithProviders(
      <GroupChartsLayout data-testid="shared-group-charts-layout" gap={2}>
        <GroupChartCell data-testid="shared-group-chart-cell">AAA</GroupChartCell>
        <GroupChartCell>BBB</GroupChartCell>
      </GroupChartsLayout>,
    );

    const layout = screen.getByTestId('shared-group-charts-layout');
    expect(layout).toHaveStyle({
      display: 'grid',
    });
    const generatedCss = document.head.textContent.replace(/\s/g, '');
    expect(generatedCss).toContain('grid-template-columns:1fr');
    expect(generatedCss).toContain('grid-template-columns:repeat(2,minmax(0,1fr))');
    expect(screen.getByTestId('shared-group-chart-cell')).toHaveStyle({ minWidth: '0' });
  });

  it('keeps the responsive contract in CSS instead of JS media-query state', () => {
    renderWithProviders(
      <GroupChartsLayout data-testid="shared-group-charts-layout-css-only">
        <GroupChartCell>AAA</GroupChartCell>
      </GroupChartsLayout>,
    );

    expect(screen.getByTestId('shared-group-charts-layout-css-only')).toBeInTheDocument();
    expect(useMediaQuerySpy).not.toHaveBeenCalled();
  });
});

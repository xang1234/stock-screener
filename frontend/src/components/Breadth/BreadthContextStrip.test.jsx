import { screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';

import { renderWithProviders } from '../../test/renderWithProviders';
import BreadthContextStrip from './BreadthContextStrip';
import { breadthMetricDefinitions } from './breadthMetricDefinitions';


const row = {
  advancing_count: 60,
  declining_count: 35,
  unchanged_count: 5,
  advance_decline_eligible_count: 100,
  new_high_52week_count: 8,
  new_low_52week_count: 2,
  high_low_52week_eligible_count: 80,
  t2108_count: 55,
  t2108_pct: 57.89,
  t2108_eligible_count: 95,
  atr_10x_extension_count: 3,
  atr_extension_eligible_count: 92,
  broad_universe_count: 110,
};


describe('BreadthContextStrip', () => {
  it('shows paired health bars with widths derived from the correct denominators', () => {
    renderWithProviders(<BreadthContextStrip row={row} />);

    expect(screen.getByText('60 (60.0%)')).toBeInTheDocument();
    expect(screen.getByText('35 (35.0%)')).toBeInTheDocument();
    expect(screen.getByText('8 (80.0%)')).toBeInTheDocument();
    expect(screen.getByText('2 (20.0%)')).toBeInTheDocument();
    expect(screen.getByText('57.89% (55 / 95)')).toBeInTheDocument();
    expect(screen.getByText('3 / 92')).toBeInTheDocument();
    expect(screen.getByText('110')).toBeInTheDocument();

    expect(screen.getByTestId('breadth-health-advance-decline')).toHaveAttribute(
      'aria-label',
      'Advancing 60.0%, Declining 35.0%, Unchanged 5.0%',
    );
    expect(
      screen.getByTestId('breadth-health-advance-decline-advancing'),
    ).toHaveStyle({ width: '60%' });
    expect(
      screen.getByTestId('breadth-health-advance-decline-declining'),
    ).toHaveStyle({ width: '35%' });
    expect(screen.getByTestId('breadth-health-high-low')).toHaveAttribute(
      'aria-label',
      'New High 80.0%, New Low 20.0%',
    );
    expect(screen.getByTestId('breadth-health-high-low-high')).toHaveStyle({
      width: '80%',
    });
    expect(screen.getByTestId('breadth-health-high-low-low')).toHaveStyle({
      width: '20%',
    });
  });

  it('uses an em dash when a metric has no eligible stocks', () => {
    renderWithProviders(
      <BreadthContextStrip
        row={{ ...row, t2108_eligible_count: 0, t2108_pct: null }}
      />,
    );

    expect(screen.getByTestId('breadth-context-t2108_pct')).toHaveTextContent('—');
  });

  it('shows valid zero high and low counts when eligible stocks exist', () => {
    renderWithProviders(
      <BreadthContextStrip
        row={{
          ...row,
          new_high_52week_count: 0,
          new_low_52week_count: 0,
          high_low_52week_eligible_count: 80,
        }}
      />,
    );

    expect(screen.getByTestId('breadth-health-high-low')).toHaveAttribute(
      'aria-label',
      'New High 0.0%, New Low 0.0%',
    );
    expect(screen.getAllByText('0 (0.0%)')).toHaveLength(2);
  });

  it('shows participation as a percent of its own eligible universe', () => {
    renderWithProviders(
      <BreadthContextStrip
        row={{
          ...row,
          above_sma200_count: 34,
          above_sma200_pct: 34,
          sma200_eligible_count: 100,
          down_20pct_from_high_count: 45,
          down_20pct_from_high_pct: 45,
          off_high_eligible_count: 100,
        }}
      />,
    );

    expect(screen.getByTestId('breadth-context-above_sma200_pct')).toHaveTextContent(
      '34.00% (34 / 100)',
    );
    expect(
      screen.getByTestId('breadth-context-down_20pct_from_high_pct'),
    ).toHaveTextContent('45.00% (45 / 100)');
  });

  it('shows an em dash for participation on rows stored before it existed', () => {
    renderWithProviders(<BreadthContextStrip row={row} />);

    expect(screen.getByTestId('breadth-context-above_sma200_pct')).toHaveTextContent('—');
    expect(
      screen.getByTestId('breadth-context-down_20pct_from_high_pct'),
    ).toHaveTextContent('—');
  });

  it('labels StockBee and screenshot-derived formulas explicitly', () => {
    expect(breadthMetricDefinitions.t2108_pct.formulaOrigin).toBe('StockBee');
    expect(breadthMetricDefinitions.atr_10x_extension_count.formulaOrigin).toBe(
      'Screenshot-derived',
    );
    expect(breadthMetricDefinitions.stocks_up_4pct.groupOrigin).toBe(
      'Screenshot grouping',
    );
    expect(breadthMetricDefinitions.stocks_up_4pct.formulaOrigin).toBe('StockBee');
  });
});

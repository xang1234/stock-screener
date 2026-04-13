import { describe, it, expect } from 'vitest';
import { screen, fireEvent } from '@testing-library/react';
import { renderWithProviders } from '../../test/renderWithProviders';
import FieldAvailabilityChip from './FieldAvailabilityChip';

describe('FieldAvailabilityChip', () => {
  it('renders nothing when field_availability is null and no growth warning', () => {
    const { container } = renderWithProviders(
      <FieldAvailabilityChip fieldAvailability={null} growthMetricBasis={null} />
    );
    expect(container).toBeEmptyDOMElement();
  });

  it('renders nothing when field_availability is empty and growth basis is quarterly_qoq', () => {
    const { container } = renderWithProviders(
      <FieldAvailabilityChip fieldAvailability={{}} growthMetricBasis="quarterly_qoq" />
    );
    expect(container).toBeEmptyDOMElement();
  });

  it('renders count chip for unsupported ownership entries and opens dialog with reasons', () => {
    const fa = {
      institutional_ownership: {
        status: 'unsupported',
        reason_code: 'unsupported_market_policy_excludes_canonical_provider',
      },
      short_interest: {
        status: 'unsupported',
        reason_code: 'unsupported_market_policy_excludes_canonical_provider',
      },
    };
    renderWithProviders(
      <FieldAvailabilityChip fieldAvailability={fa} growthMetricBasis={null} />
    );

    const chip = screen.getByTestId('field-availability-chip');
    expect(chip.textContent).toContain('2');

    fireEvent.click(chip);
    expect(screen.getByText('Data Availability')).toBeInTheDocument();
    expect(screen.getByText('institutional_ownership')).toBeInTheDocument();
    expect(screen.getByText('short_interest')).toBeInTheDocument();
    // Secondary shows status + reason_code
    expect(
      screen.getAllByText(/unsupported — unsupported_market_policy/).length
    ).toBeGreaterThanOrEqual(1);
  });

  it('renders computed entries (not just unsupported)', () => {
    const fa = {
      eps_growth_qq: {
        status: 'computed',
        reason_code: 'comparable_period_yoy_fallback',
      },
    };
    renderWithProviders(
      <FieldAvailabilityChip fieldAvailability={fa} growthMetricBasis="comparable_period_yoy" />
    );
    const chip = screen.getByTestId('field-availability-chip');
    expect(chip.textContent).toContain('1');
  });

  it('ignores entries with status="available" (server sends them on full-ownership US rows)', () => {
    // Safety net: even if a non-filtered dict leaks through the API, the
    // chip should only count non-available entries.
    const fa = {
      institutional_ownership: { status: 'available', reason_code: null },
      insider_ownership: { status: 'missing', reason_code: 'missing_supported_field_value' },
    };
    renderWithProviders(
      <FieldAvailabilityChip fieldAvailability={fa} growthMetricBasis={null} />
    );
    const chip = screen.getByTestId('field-availability-chip');
    expect(chip.textContent).toContain('1');
  });

  it('renders growth-only cadence warning when field_availability is empty but basis is unavailable', () => {
    renderWithProviders(
      <FieldAvailabilityChip fieldAvailability={null} growthMetricBasis="unavailable" />
    );
    const chip = screen.getByTestId('field-availability-chip');
    expect(chip).toBeInTheDocument();
    fireEvent.click(chip);
    expect(screen.getByText(/Growth metrics are unavailable/)).toBeInTheDocument();
  });

  it('click on chip does not bubble up to parent handlers', () => {
    let parentClicks = 0;
    const fa = { short_interest: { status: 'unsupported', reason_code: 'x' } };
    renderWithProviders(
      <div onClick={() => { parentClicks += 1; }}>
        <FieldAvailabilityChip fieldAvailability={fa} growthMetricBasis={null} />
      </div>
    );
    fireEvent.click(screen.getByTestId('field-availability-chip'));
    expect(parentClicks).toBe(0);
  });
});

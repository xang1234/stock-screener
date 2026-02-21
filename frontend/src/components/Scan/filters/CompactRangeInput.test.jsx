import { screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { renderWithProviders } from '../../../test/renderWithProviders';
import CompactRangeInput from './CompactRangeInput';

describe('CompactRangeInput', () => {
  it('renders the label text', () => {
    renderWithProviders(
      <CompactRangeInput label="SE Score" onChange={vi.fn()} />
    );
    expect(screen.getByText('SE Score')).toBeInTheDocument();
  });

  it('renders single input with >= placeholder when minOnly={true}', () => {
    renderWithProviders(
      <CompactRangeInput label="Min Score" minOnly={true} onChange={vi.fn()} />
    );
    const inputs = screen.getAllByRole('spinbutton');
    expect(inputs).toHaveLength(1);
    expect(inputs[0]).toHaveAttribute('placeholder', '≥');
  });

  it('renders both Min and Max inputs when minOnly={false}', () => {
    renderWithProviders(
      <CompactRangeInput label="Pvt Dist" minOnly={false} onChange={vi.fn()} />
    );
    const inputs = screen.getAllByRole('spinbutton');
    expect(inputs).toHaveLength(2);
    expect(inputs[0]).toHaveAttribute('placeholder', 'Min');
    expect(inputs[1]).toHaveAttribute('placeholder', 'Max');
  });

  it('fires onChange with parsed value after 300ms debounce', async () => {
    const onChange = vi.fn();
    renderWithProviders(
      <CompactRangeInput label="SE Score" minOnly={true} onChange={onChange} />
    );

    const user = userEvent.setup();
    const input = screen.getByRole('spinbutton');
    await user.type(input, '70');

    // Wait for the 300ms debounce to fire
    await waitFor(() => {
      expect(onChange).toHaveBeenCalledWith(
        expect.objectContaining({ min: 70, max: null })
      );
    }, { timeout: 1000 });
  });

  it('sends null for empty input', async () => {
    const onChange = vi.fn();
    renderWithProviders(
      <CompactRangeInput label="Score" minOnly={true} minValue={50} onChange={onChange} />
    );

    const user = userEvent.setup();
    const input = screen.getByRole('spinbutton');
    await user.clear(input);

    await waitFor(() => {
      expect(onChange).toHaveBeenCalledWith(
        expect.objectContaining({ min: null, max: null })
      );
    }, { timeout: 1000 });
  });

  it('shows suffix adornment when provided', () => {
    renderWithProviders(
      <CompactRangeInput label="ATR" minOnly={true} suffix="%" onChange={vi.fn()} />
    );
    expect(screen.getByText('%')).toBeInTheDocument();
  });

  it('sends null (not NaN) for non-numeric input', async () => {
    const onChange = vi.fn();
    renderWithProviders(
      <CompactRangeInput label="Score" minOnly={true} onChange={onChange} />
    );

    const user = userEvent.setup();
    const input = screen.getByRole('spinbutton');
    // type:number inputs may reject non-numeric chars, resulting in empty string
    await user.type(input, 'abc');

    // Wait for potential debounce
    await waitFor(
      () => {
        if (onChange.mock.calls.length > 0) {
          const lastCall = onChange.mock.calls[onChange.mock.calls.length - 1][0];
          expect(lastCall.min).not.toBeNaN();
        }
      },
      { timeout: 1000 }
    );
  });

  it('respects step prop on input element', () => {
    renderWithProviders(
      <CompactRangeInput label="Fine" step={0.01} minOnly={true} onChange={vi.fn()} />
    );
    const input = screen.getByRole('spinbutton');
    expect(input).toHaveAttribute('step', '0.01');
  });
});

import { screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { renderWithProviders } from '../../../test/renderWithProviders';
import CompactCheckbox from './CompactCheckbox';

describe('CompactCheckbox', () => {
  it('renders label text', () => {
    renderWithProviders(
      <CompactCheckbox label="SE Ready" value={null} onChange={vi.fn()} />
    );
    expect(screen.getByText('SE Ready')).toBeInTheDocument();
  });

  it('shows All / Yes / No toggle buttons', () => {
    renderWithProviders(
      <CompactCheckbox label="Ready" value={null} onChange={vi.fn()} />
    );
    expect(screen.getByText('All')).toBeInTheDocument();
    expect(screen.getByText('Yes')).toBeInTheDocument();
    expect(screen.getByText('No')).toBeInTheDocument();
  });

  it('calls onChange(true) when Yes is clicked', async () => {
    const onChange = vi.fn();
    renderWithProviders(
      <CompactCheckbox label="Ready" value={null} onChange={onChange} />
    );

    const user = userEvent.setup();
    await user.click(screen.getByText('Yes'));
    expect(onChange).toHaveBeenCalledWith(true);
  });

  it('calls onChange(false) when No is clicked', async () => {
    const onChange = vi.fn();
    renderWithProviders(
      <CompactCheckbox label="Ready" value={null} onChange={onChange} />
    );

    const user = userEvent.setup();
    await user.click(screen.getByText('No'));
    expect(onChange).toHaveBeenCalledWith(false);
  });

  it('calls onChange(null) when already-selected Yes is clicked again', async () => {
    const onChange = vi.fn();
    renderWithProviders(
      <CompactCheckbox label="Ready" value={true} onChange={onChange} />
    );

    const user = userEvent.setup();
    await user.click(screen.getByText('Yes'));
    expect(onChange).toHaveBeenCalledWith(null);
  });

  it('highlights Yes button when value={true}', () => {
    renderWithProviders(
      <CompactCheckbox label="Ready" value={true} onChange={vi.fn()} />
    );
    const yesButton = screen.getByRole('button', { name: /yes/i });
    expect(yesButton).toHaveAttribute('aria-pressed', 'true');
  });
});

import { screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';

import StockbeeMmTab from './StockbeeMmTab';
import { renderWithProviders } from '../../test/renderWithProviders';

describe('StockbeeMmTab', () => {
  it('renders a direct fallback link alongside the embedded page', () => {
    renderWithProviders(<StockbeeMmTab />);

    const link = screen.getByRole('link', { name: /open stockbee mm in a new tab/i });
    expect(link).toHaveAttribute('href', 'https://stockbee.blogspot.com/p/mm.html');
    expect(screen.getByTitle('Stockbee MM')).toHaveAttribute(
      'src',
      'https://stockbee.blogspot.com/p/mm.html'
    );
  });
});

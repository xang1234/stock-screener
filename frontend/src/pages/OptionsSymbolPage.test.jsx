import { screen } from '@testing-library/react';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import { renderWithProviders } from '../test/renderWithProviders';
import OptionsSymbolPage from './OptionsSymbolPage';
import * as optionsApi from '../api/optionsAnalytics';

vi.mock('../api/optionsAnalytics', () => ({ getOptionsSymbolDetail: vi.fn() }));

describe('OptionsSymbolPage', () => {
  beforeEach(() => vi.clearAllMocks());

  it('shows a clear missing-symbol state and route back', async () => {
    optionsApi.getOptionsSymbolDetail.mockRejectedValue({ response: { status: 404 } });
    renderWithProviders(
      <MemoryRouter initialEntries={['/options/MISSING']}>
        <Routes><Route path="/options/:symbol" element={<OptionsSymbolPage />} /></Routes>
      </MemoryRouter>,
    );

    expect(await screen.findByRole('heading', { name: /MISSING is not in the published options cohort/i })).toBeInTheDocument();
    expect(screen.getByRole('link', { name: /Back to Command Center/i })).toHaveAttribute('href', '/options');
  });
});

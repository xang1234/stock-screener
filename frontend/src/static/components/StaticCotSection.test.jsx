import { fireEvent, screen, waitFor } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import { renderWithProviders } from '../../test/renderWithProviders';
import { fetchStaticJson } from '../dataClient';
import StaticCotSection from './StaticCotSection';
import {
  makeCotHistory,
  staticCotIndexFixture,
} from '../../features/cot/__fixtures__/cotResponses';

vi.mock('../dataClient', async () => {
  const actual = await vi.importActual('../dataClient');
  return { ...actual, fetchStaticJson: vi.fn(), useStaticGeneration: () => ({ generation: 'flat', dataRoot: '' }), };
});

describe('StaticCotSection', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    fetchStaticJson.mockImplementation(async (path) => {
      if (path === 'cot/index.json') return staticCotIndexFixture;
      if (path === 'cot/sp-500.json') return makeCotHistory({ weekCount: 260, range: '5y' });
      if (path === 'cot/nasdaq-100.json') return makeCotHistory({ weekCount: 260, range: '5y', slug: 'nasdaq-100' });
      throw new Error(`Unexpected static path: ${path}`);
    });
  });

  it('is absent when the root manifest does not advertise COT', () => {
    renderWithProviders(<StaticCotSection manifest={{ assets: {} }} />);
    expect(screen.queryByTestId('static-cot-section')).not.toBeInTheDocument();
    expect(fetchStaticJson).not.toHaveBeenCalled();
  });

  it('loads only the selected history, slices ranges locally, and has no full table', async () => {
    renderWithProviders(<StaticCotSection manifest={{ assets: { cot: { path: 'cot/index.json' } } }} />);

    expect(await screen.findByTestId('static-cot-section')).toBeInTheDocument();
    await screen.findByRole('img', { name: /S&P 500 net positioning.*52 report weeks/i });
    expect(fetchStaticJson.mock.calls).toEqual([
      ['cot/index.json', ''],
      ['cot/sp-500.json', ''],
    ]);
    fireEvent.click(screen.getByRole('button', { name: '3Y' }));
    expect(await screen.findByRole('img', { name: /156 report weeks/i })).toBeInTheDocument();
    expect(fetchStaticJson).toHaveBeenCalledTimes(2);
    fireEvent.change(screen.getByLabelText('COT instrument'), { target: { value: 'nasdaq-100' } });
    await waitFor(() => expect(fetchStaticJson).toHaveBeenCalledWith('cot/nasdaq-100.json', ''));
    expect(screen.queryByRole('table', { name: /COT positioning snapshot/i })).not.toBeInTheDocument();
  });

  it('contains artifact errors without breaking the surrounding page', async () => {
    fetchStaticJson.mockRejectedValue(new Error('corrupt artifact'));
    renderWithProviders(<StaticCotSection manifest={{ assets: { cot: { path: 'cot/index.json' } } }} />);
    expect(await screen.findByText(/Static COT data is unavailable/i)).toBeInTheDocument();
  });
});

import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import ThemeEquivalencePanel from './ThemeEquivalencePanel';
import ThemeDevelopmentTimeline from './ThemeDevelopmentTimeline';
import * as api from '../../api/themes';
vi.mock('../../api/themes', () => ({ searchThemeEquivalence: vi.fn(), previewThemeEquivalence: vi.fn(), applyThemeEquivalence: vi.fn(), getThemeEquivalenceHistory: vi.fn(), undoThemeEquivalence: vi.fn(), getThemeDevelopments: vi.fn() }));
function show(component) { return render(<QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>{component}</QueryClientProvider>); }
beforeEach(() => {
  vi.clearAllMocks();
  api.searchThemeEquivalence.mockResolvedValue({ themes: [{ id: 1, name: 'CPO' }, { id: 2, name: 'Co-Packaged Optics' }] });
  api.getThemeEquivalenceHistory.mockResolvedValue({ operations: [] });
});
it('requires preview and attribution before grouping', async () => {
  api.previewThemeEquivalence.mockResolvedValue({ version: 'a'.repeat(64), parent_posts: 4, aliases: [{ name: 'CPO' }, { name: 'Co-Packaged Optics' }] });
  api.applyThemeEquivalence.mockResolvedValue({ refresh_status: 'pending' });
  show(<ThemeEquivalencePanel />);
  expect(screen.getByRole('button', { name: 'Apply reviewed grouping' })).toBeDisabled();
  fireEvent.change(screen.getByLabelText('Theme to group'), { target: { value: 'CPO' } });
  fireEvent.click(await screen.findByRole('option', { name: 'CPO' }));
  fireEvent.change(screen.getByLabelText('Display under theme'), { target: { value: 'Co-' } });
  fireEvent.click(await screen.findByRole('option', { name: 'Co-Packaged Optics' }));
  fireEvent.click(screen.getByRole('button', { name: 'Preview grouping' }));
  await screen.findByText(/4 distinct source posts/);
  expect(screen.getByRole('button', { name: 'Apply reviewed grouping' })).toBeDisabled();
  fireEvent.change(screen.getByLabelText('Reviewer'), { target: { value: 'Reviewer' } });
  fireEvent.change(screen.getByLabelText('Reason for grouping or undo'), { target: { value: 'Same exposure' } });
  fireEvent.click(screen.getByRole('button', { name: 'Apply reviewed grouping' }));
  expect(await screen.findByText('Grouping saved. Current results are awaiting a refresh.')).toBeInTheDocument();
  await waitFor(() => expect(api.applyThemeEquivalence).toHaveBeenCalledWith(expect.objectContaining({ source_id: 1, target_id: 2, expected_version: 'a'.repeat(64), reason: 'Same exposure' })));
});
it('shows repeated and superseded evidence and failures without unsafe links', async () => {
  api.getThemeDevelopments.mockResolvedValue({ tracking_enabled: true, event_count: 1, material_update_count: 0, work_counts: { failed: 1 }, observations: [{ id: 1, event_id: 8, facts: { summary: 'An attributed report', status: 'rumored' }, classification: 'repeated_coverage', superseded: true, available_at: '2026-09-12T00:00:00Z', url: 'javascript:alert(1)', citations: [{ source_id: 'primary', quote: 'Exact evidence' }] }] });
  show(<ThemeDevelopmentTimeline themeId={1} />);
  expect(await screen.findByText('Repeated coverage')).toBeInTheDocument();
  expect(screen.getByText('Superseded revision')).toBeInTheDocument();
  expect(screen.getByText(/could not be prepared/)).toBeInTheDocument();
  expect(screen.getByText(/Exact evidence/)).toBeInTheDocument();
  expect(screen.queryByRole('link')).not.toBeInTheDocument();
});

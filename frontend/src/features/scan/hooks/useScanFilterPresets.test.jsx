import { act, renderHook } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import { buildDefaultScanFilters } from '../defaultFilters';
import { useScanFilterPresets } from './useScanFilterPresets';

function setup(overrides = {}) {
  const setFilters = vi.fn();
  const setSortBy = vi.fn();
  const setSortOrder = vi.fn();
  const setPage = vi.fn();
  const createPresetAsync = vi.fn().mockResolvedValue({ id: 'preset-2' });
  const updatePresetAsync = vi.fn().mockResolvedValue({});
  const deletePreset = vi.fn();

  const props = {
    presets: [
      {
        id: 'preset-1',
        name: 'Momentum',
        filters: { ...buildDefaultScanFilters(), symbolSearch: 'NVDA' },
        sort_by: 'composite_score',
        sort_order: 'desc',
      },
    ],
    createPresetAsync,
    updatePresetAsync,
    deletePreset,
    filters: buildDefaultScanFilters(),
    sortBy: 'composite_score',
    sortOrder: 'desc',
    setFilters,
    setSortBy,
    setSortOrder,
    setPage,
    ...overrides,
  };

  const hook = renderHook((currentProps) => useScanFilterPresets(currentProps), {
    initialProps: props,
  });

  return {
    hook,
    props,
    setFilters,
    setSortBy,
    setSortOrder,
    setPage,
    createPresetAsync,
    updatePresetAsync,
    deletePreset,
  };
}

describe('useScanFilterPresets', () => {
  it('loads a preset and updates filter + sort state', () => {
    const { hook, setFilters, setSortBy, setSortOrder, setPage } = setup();

    act(() => {
      hook.result.current.handleLoadPreset('preset-1');
    });

    expect(setFilters).toHaveBeenCalledWith(expect.objectContaining({ symbolSearch: 'NVDA' }));
    expect(setSortBy).toHaveBeenCalledWith('composite_score');
    expect(setSortOrder).toHaveBeenCalledWith('desc');
    expect(setPage).toHaveBeenCalledWith(1);
  });

  it('tracks unsaved changes after preset load', () => {
    const { hook, props } = setup();

    act(() => {
      hook.result.current.handleLoadPreset('preset-1');
    });
    hook.rerender({
      ...props,
      filters: { ...buildDefaultScanFilters(), symbolSearch: 'NVDA' },
    });
    expect(hook.result.current.hasUnsavedChanges()).toBe(false);

    hook.rerender({
      ...props,
      filters: { ...props.filters, symbolSearch: 'AAPL' },
    });

    expect(hook.result.current.hasUnsavedChanges()).toBe(true);
  });

  it('creates a new preset from save dialog', async () => {
    const { hook, createPresetAsync } = setup();

    act(() => {
      hook.result.current.handleOpenSaveDialog();
    });

    await act(async () => {
      await hook.result.current.handleSaveDialogSave('My preset', 'desc');
    });

    expect(createPresetAsync).toHaveBeenCalledWith(
      expect.objectContaining({
        name: 'My preset',
        description: 'desc',
      })
    );
  });
});

import { act, renderHook } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import { buildDefaultScanFilters } from '../defaultFilters';
import { legacyFiltersToExpression } from '../legacyFilterExpression';
import { useScanFilterPresets } from './useScanFilterPresets';

function setup(overrides = {}) {
  const applyQuery = vi.fn();
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
      {
        id: 'preset-2',
        name: 'Growth',
        description: 'growth profile',
        filters: { ...buildDefaultScanFilters(), symbolSearch: 'AAPL' },
        sort_by: 'rs_rating',
        sort_order: 'asc',
      },
    ],
    createPresetAsync,
    updatePresetAsync,
    deletePreset,
    filters: buildDefaultScanFilters(),
    sortBy: 'composite_score',
    sortOrder: 'desc',
    applyQuery,
    ...overrides,
  };

  const hook = renderHook((currentProps) => useScanFilterPresets(currentProps), {
    initialProps: props,
  });

  return {
    hook,
    props,
    applyQuery,
    createPresetAsync,
    updatePresetAsync,
    deletePreset,
  };
}

describe('useScanFilterPresets', () => {
  it('loads a preset as one canonical filter + sort transition', () => {
    const { hook, applyQuery } = setup();

    act(() => {
      hook.result.current.handleLoadPreset('preset-1');
    });

    expect(applyQuery).toHaveBeenCalledWith(expect.objectContaining({
      expression: expect.objectContaining({
        expression_version: 1,
        required: expect.objectContaining({
          conditions: expect.arrayContaining([
            expect.objectContaining({ kind: 'text', pattern: 'NVDA' }),
          ]),
        }),
      }),
      sortBy: 'composite_score',
      sortOrder: 'desc',
    }));
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

  it('stores one canonical V2 expression using the current quick-filter draft', async () => {
    const originalFilters = { ...buildDefaultScanFilters(), symbolSearch: 'NVDA' };
    const currentFilters = { ...buildDefaultScanFilters(), symbolSearch: 'AAPL' };
    const { hook, createPresetAsync } = setup({
      expression: legacyFiltersToExpression(originalFilters),
      filters: currentFilters,
    });

    act(() => {
      hook.result.current.handleOpenSaveDialog();
    });
    await act(async () => {
      await hook.result.current.handleSaveDialogSave('Current draft', '');
    });

    const payload = createPresetAsync.mock.calls[0][0].filters;
    expect(payload).toEqual({
      schema_version: 2,
      expression: expect.objectContaining({
        required: expect.objectContaining({
          conditions: expect.arrayContaining([
            expect.objectContaining({ kind: 'text', pattern: 'AAPL' }),
          ]),
        }),
      }),
    });
    expect(payload).not.toHaveProperty('legacy_filters');
  });

  it('loads V2 presets from the expression even when stale legacy filters exist', () => {
    const canonicalFilters = { ...buildDefaultScanFilters(), symbolSearch: 'GOOGL' };
    const staleFilters = { ...buildDefaultScanFilters(), symbolSearch: 'STALE' };
    const { hook, applyQuery } = setup({
      expression: legacyFiltersToExpression(buildDefaultScanFilters()),
      presets: [
        {
          id: 'preset-v2',
          name: 'Canonical',
          filters: {
            schema_version: 2,
            expression: legacyFiltersToExpression(canonicalFilters),
            legacy_filters: staleFilters,
          },
          sort_by: 'rs_rating',
          sort_order: 'desc',
        },
      ],
    });

    act(() => {
      hook.result.current.handleLoadPreset('preset-v2');
    });

    expect(applyQuery).toHaveBeenCalledWith(expect.objectContaining({
      expression: expect.objectContaining({
        required: expect.objectContaining({
          conditions: expect.arrayContaining([
            expect.objectContaining({ kind: 'text', pattern: 'GOOGL' }),
          ]),
        }),
      }),
    }));
  });

  it('renames the preset selected in the rename dialog, not the active preset', async () => {
    const { hook, updatePresetAsync } = setup();

    act(() => {
      hook.result.current.handleLoadPreset('preset-1');
      hook.result.current.handleRenamePreset('preset-2');
    });

    await act(async () => {
      await hook.result.current.handleSaveDialogSave('Renamed Growth', 'updated');
    });

    expect(updatePresetAsync).toHaveBeenCalledWith({
      presetId: 'preset-2',
      updates: { name: 'Renamed Growth', description: 'updated' },
    });
  });
});

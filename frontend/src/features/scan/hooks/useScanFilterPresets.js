import { useCallback, useMemo, useState } from 'react';
import { buildDefaultScanFilters } from '../defaultFilters';
import {
  expressionToLegacyFilters,
  legacyFiltersToExpression,
} from '../filterExpression';

export function useScanFilterPresets({
  presets,
  createPresetAsync,
  updatePresetAsync,
  deletePreset,
  filters,
  sortBy,
  sortOrder,
  setFilters,
  setSortBy,
  setSortOrder,
  setPage,
  expression = null,
  setExpression = null,
}) {
  const [activePresetId, setActivePresetId] = useState(null);
  const [presetFiltersSnapshot, setPresetFiltersSnapshot] = useState(null);
  const [presetSortSnapshot, setPresetSortSnapshot] = useState(null);
  const [saveDialogOpen, setSaveDialogOpen] = useState(false);
  const [saveDialogMode, setSaveDialogMode] = useState('save');
  const [saveDialogPresetId, setSaveDialogPresetId] = useState(null);
  const [saveDialogInitialName, setSaveDialogInitialName] = useState('');
  const [saveDialogInitialDescription, setSaveDialogInitialDescription] = useState('');
  const [saveDialogError, setSaveDialogError] = useState(null);

  const currentPresetFilters = useMemo(
    () => (expression
      ? { schema_version: 2, expression, legacy_filters: filters }
      : filters),
    [expression, filters],
  );

  const clearActivePreset = useCallback(() => {
    setActivePresetId(null);
    setPresetFiltersSnapshot(null);
    setPresetSortSnapshot(null);
  }, []);

  const hasUnsavedChanges = useCallback(() => {
    if (!activePresetId || !presetFiltersSnapshot) {
      return false;
    }
    const filtersChanged = JSON.stringify(currentPresetFilters) !== JSON.stringify(presetFiltersSnapshot);
    const sortChanged =
      presetSortSnapshot &&
      (sortBy !== presetSortSnapshot.sortBy || sortOrder !== presetSortSnapshot.sortOrder);
    return filtersChanged || sortChanged;
  }, [activePresetId, currentPresetFilters, presetFiltersSnapshot, presetSortSnapshot, sortBy, sortOrder]);

  const handleLoadPreset = useCallback(
    (presetId) => {
      if (!presetId) {
        clearActivePreset();
        return;
      }

      const preset = presets.find((item) => item.id === presetId);
      if (!preset) {
        return;
      }

      const isExpressionPreset = preset.filters?.schema_version === 2 && preset.filters?.expression;
      const nextFilters = isExpressionPreset
        ? (preset.filters.legacy_filters
            || expressionToLegacyFilters(preset.filters.expression, buildDefaultScanFilters()))
        : preset.filters;
      setFilters(nextFilters);
      if (setExpression) {
        setExpression(
          isExpressionPreset
            ? preset.filters.expression
            : legacyFiltersToExpression(nextFilters),
        );
      }
      setSortBy(preset.sort_by);
      setSortOrder(preset.sort_order);
      setActivePresetId(presetId);
      setPresetFiltersSnapshot(
        isExpressionPreset
          ? preset.filters
          : (setExpression
              ? {
                  schema_version: 2,
                  expression: legacyFiltersToExpression(nextFilters),
                  legacy_filters: nextFilters,
                }
              : preset.filters),
      );
      setPresetSortSnapshot({ sortBy: preset.sort_by, sortOrder: preset.sort_order });
      setPage(1);
    },
    [clearActivePreset, presets, setExpression, setFilters, setPage, setSortBy, setSortOrder]
  );

  const handleOpenSaveDialog = useCallback(() => {
    setSaveDialogMode('save');
    setSaveDialogPresetId(null);
    setSaveDialogInitialName('');
    setSaveDialogInitialDescription('');
    setSaveDialogError(null);
    setSaveDialogOpen(true);
  }, []);

  const handleUpdatePreset = useCallback(async () => {
    if (!activePresetId) {
      return;
    }
    try {
      await updatePresetAsync({
        presetId: activePresetId,
        updates: {
          filters: currentPresetFilters,
          sort_by: sortBy,
          sort_order: sortOrder,
        },
      });
      setPresetFiltersSnapshot(currentPresetFilters);
      setPresetSortSnapshot({ sortBy, sortOrder });
    } catch (error) {
      console.error('Failed to update preset:', error);
      alert('Failed to update preset. Please try again.');
    }
  }, [activePresetId, currentPresetFilters, sortBy, sortOrder, updatePresetAsync]);

  const handleRenamePreset = useCallback(
    (presetId) => {
      const preset = presets.find((item) => item.id === presetId);
      if (!preset) {
        return;
      }
      setSaveDialogMode('rename');
      setSaveDialogPresetId(presetId);
      setSaveDialogInitialName(preset.name);
      setSaveDialogInitialDescription(preset.description || '');
      setSaveDialogError(null);
      setSaveDialogOpen(true);
    },
    [presets]
  );

  const handleDeletePreset = useCallback(
    (presetId) => {
      deletePreset(presetId);
      if (activePresetId === presetId) {
        clearActivePreset();
      }
    },
    [activePresetId, clearActivePreset, deletePreset]
  );

  const handleSaveDialogClose = useCallback(() => {
    setSaveDialogOpen(false);
    setSaveDialogError(null);
    setSaveDialogPresetId(null);
  }, []);

  const handleSaveDialogSave = useCallback(
    async (name, description) => {
      setSaveDialogError(null);

      try {
        if (saveDialogMode === 'save') {
          const newPreset = await createPresetAsync({
            name,
            description: description || null,
            filters: currentPresetFilters,
            sort_by: sortBy,
            sort_order: sortOrder,
          });
          setActivePresetId(newPreset.id);
          setPresetFiltersSnapshot(currentPresetFilters);
          setPresetSortSnapshot({ sortBy, sortOrder });
        } else {
          const targetPresetId = saveDialogPresetId ?? activePresetId;
          if (!targetPresetId) {
            setSaveDialogError('No preset selected for rename');
            return;
          }
          await updatePresetAsync({
            presetId: targetPresetId,
            updates: { name, description: description || null },
          });
        }

        setSaveDialogOpen(false);
      } catch (error) {
        console.error('Failed to save preset:', error);
        const errorMessage = error.response?.data?.detail || 'Failed to save preset';
        setSaveDialogError(errorMessage);
      }
    },
    [
      activePresetId,
      createPresetAsync,
      currentPresetFilters,
      saveDialogMode,
      saveDialogPresetId,
      sortBy,
      sortOrder,
      updatePresetAsync,
    ]
  );

  return {
    activePresetId,
    hasUnsavedChanges,
    clearActivePreset,
    handleLoadPreset,
    handleOpenSaveDialog,
    handleUpdatePreset,
    handleRenamePreset,
    handleDeletePreset,
    saveDialogOpen,
    saveDialogMode,
    saveDialogInitialName,
    saveDialogInitialDescription,
    saveDialogError,
    handleSaveDialogClose,
    handleSaveDialogSave,
  };
}

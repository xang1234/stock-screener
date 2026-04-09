import { useCallback, useState } from 'react';

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
}) {
  const [activePresetId, setActivePresetId] = useState(null);
  const [presetFiltersSnapshot, setPresetFiltersSnapshot] = useState(null);
  const [presetSortSnapshot, setPresetSortSnapshot] = useState(null);
  const [saveDialogOpen, setSaveDialogOpen] = useState(false);
  const [saveDialogMode, setSaveDialogMode] = useState('save');
  const [saveDialogInitialName, setSaveDialogInitialName] = useState('');
  const [saveDialogInitialDescription, setSaveDialogInitialDescription] = useState('');
  const [saveDialogError, setSaveDialogError] = useState(null);

  const clearActivePreset = useCallback(() => {
    setActivePresetId(null);
    setPresetFiltersSnapshot(null);
    setPresetSortSnapshot(null);
  }, []);

  const hasUnsavedChanges = useCallback(() => {
    if (!activePresetId || !presetFiltersSnapshot) {
      return false;
    }
    const filtersChanged = JSON.stringify(filters) !== JSON.stringify(presetFiltersSnapshot);
    const sortChanged =
      presetSortSnapshot &&
      (sortBy !== presetSortSnapshot.sortBy || sortOrder !== presetSortSnapshot.sortOrder);
    return filtersChanged || sortChanged;
  }, [activePresetId, filters, presetFiltersSnapshot, presetSortSnapshot, sortBy, sortOrder]);

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

      setFilters(preset.filters);
      setSortBy(preset.sort_by);
      setSortOrder(preset.sort_order);
      setActivePresetId(presetId);
      setPresetFiltersSnapshot(preset.filters);
      setPresetSortSnapshot({ sortBy: preset.sort_by, sortOrder: preset.sort_order });
      setPage(1);
    },
    [clearActivePreset, presets, setFilters, setPage, setSortBy, setSortOrder]
  );

  const handleOpenSaveDialog = useCallback(() => {
    setSaveDialogMode('save');
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
          filters,
          sort_by: sortBy,
          sort_order: sortOrder,
        },
      });
      setPresetFiltersSnapshot(filters);
      setPresetSortSnapshot({ sortBy, sortOrder });
    } catch (error) {
      console.error('Failed to update preset:', error);
      alert('Failed to update preset. Please try again.');
    }
  }, [activePresetId, filters, sortBy, sortOrder, updatePresetAsync]);

  const handleRenamePreset = useCallback(
    (presetId) => {
      const preset = presets.find((item) => item.id === presetId);
      if (!preset) {
        return;
      }
      setSaveDialogMode('rename');
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
  }, []);

  const handleSaveDialogSave = useCallback(
    async (name, description) => {
      setSaveDialogError(null);

      try {
        if (saveDialogMode === 'save') {
          const newPreset = await createPresetAsync({
            name,
            description: description || null,
            filters,
            sort_by: sortBy,
            sort_order: sortOrder,
          });
          setActivePresetId(newPreset.id);
          setPresetFiltersSnapshot(filters);
          setPresetSortSnapshot({ sortBy, sortOrder });
        } else {
          await updatePresetAsync({
            presetId: activePresetId,
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
    [activePresetId, createPresetAsync, filters, saveDialogMode, sortBy, sortOrder, updatePresetAsync]
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

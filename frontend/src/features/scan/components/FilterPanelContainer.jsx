import { memo } from 'react';
import {
  Paper,
  Typography,
  Box,
  Button,
  Chip,
  Collapse,
  IconButton,
} from '@mui/material';
import FilterListIcon from '@mui/icons-material/FilterList';
import ClearIcon from '@mui/icons-material/Clear';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import ExpandLessIcon from '@mui/icons-material/ExpandLess';
import {
  FilterPresets,
  SavePresetDialog,
} from '../../../components/Scan/filters';
import {
  FUNDAMENTAL_KEYS,
  TECHNICAL_KEYS,
  RATING_KEYS,
} from './filterPanel/constants';
import {
  buildActiveFilters,
  countActiveInCategory,
  resetFilterValue,
} from './filterPanel/utils';
import FundamentalFiltersSection from './filterPanel/FundamentalFiltersSection';
import TechnicalFiltersSection from './filterPanel/TechnicalFiltersSection';
import RatingFiltersSection from './filterPanel/RatingFiltersSection';

function FilterPanel({
  filters,
  onFilterChange,
  onReset,
  filterOptions = {},
  expanded = true,
  onToggle,
  presetsEnabled = true,
  sectionDefaultExpanded = {
    fundamental: true,
    technical: true,
    rating: true,
  },
  presets = [],
  activePresetId = null,
  hasUnsavedChanges = false,
  presetsLoading = false,
  presetsSaving = false,
  onLoadPreset,
  onSavePreset,
  onUpdatePreset,
  onRenamePreset,
  onDeletePreset,
  saveDialogOpen = false,
  saveDialogMode = 'save',
  saveDialogInitialName = '',
  saveDialogInitialDescription = '',
  saveDialogError = null,
  onSaveDialogClose,
  onSaveDialogSave,
}) {
  const updateFilter = (key, value) => {
    onFilterChange({ ...filters, [key]: value });
  };

  const updateRangeFilter = (key, range) => {
    onFilterChange({ ...filters, [key]: range });
  };

  const handleDeleteFilter = (key) => {
    onFilterChange({ ...filters, [key]: resetFilterValue(key) });
  };

  const activeFilters = buildActiveFilters(filters);
  const fundamentalCount = countActiveInCategory(filters, FUNDAMENTAL_KEYS);
  const technicalCount = countActiveInCategory(filters, TECHNICAL_KEYS);
  const ratingCount = countActiveInCategory(filters, RATING_KEYS);

  return (
    <Paper elevation={1} sx={{ p: 1.5, mb: 2 }}>
      <Box
        sx={{
          display: 'flex',
          alignItems: 'center',
          cursor: 'pointer',
          userSelect: 'none',
        }}
        onClick={onToggle}
      >
        <FilterListIcon sx={{ mr: 1, fontSize: 18, color: 'text.secondary' }} />
        <Typography variant="subtitle2" sx={{ mr: 2 }}>
          Filters
          {activeFilters.length > 0 && (
            <Chip
              label={activeFilters.length}
              size="small"
              color="primary"
              sx={{ ml: 1, height: 18, fontSize: '10px', '& .MuiChip-label': { px: 0.75 } }}
            />
          )}
        </Typography>

        {presetsEnabled && (
          <Box
            onClick={(event) => event.stopPropagation()}
            onMouseDown={(event) => event.stopPropagation()}
          >
            <FilterPresets
              presets={presets}
              activePresetId={activePresetId}
              hasUnsavedChanges={hasUnsavedChanges}
              isLoading={presetsLoading}
              isSaving={presetsSaving}
              onLoadPreset={onLoadPreset}
              onSavePreset={onSavePreset}
              onUpdatePreset={onUpdatePreset}
              onRenamePreset={onRenamePreset}
              onDeletePreset={onDeletePreset}
            />
          </Box>
        )}

        <Box sx={{ flexGrow: 1 }} />

        {!expanded && activeFilters.length > 0 && (
          <Box sx={{ display: 'flex', gap: 0.5, mr: 1, flexWrap: 'wrap', maxWidth: '60%' }}>
            {activeFilters.slice(0, 5).map(({ key, label }) => (
              <Chip
                key={key}
                label={label}
                size="small"
                onDelete={(event) => {
                  event.stopPropagation();
                  handleDeleteFilter(key);
                }}
                sx={{
                  height: 20,
                  fontSize: '10px',
                  '& .MuiChip-label': { px: 0.75 },
                  '& .MuiChip-deleteIcon': { fontSize: '14px' },
                }}
              />
            ))}
            {activeFilters.length > 5 && (
              <Chip
                label={`+${activeFilters.length - 5}`}
                size="small"
                sx={{ height: 20, fontSize: '10px', '& .MuiChip-label': { px: 0.75 } }}
              />
            )}
          </Box>
        )}

        <Button
          startIcon={<ClearIcon sx={{ fontSize: 14 }} />}
          onClick={(event) => {
            event.stopPropagation();
            onReset();
          }}
          size="small"
          sx={{ fontSize: '0.7rem', py: 0.25, px: 0.75, minWidth: 0 }}
        >
          Reset
        </Button>
        <IconButton size="small" sx={{ ml: 0.5 }}>
          {expanded ? <ExpandLessIcon fontSize="small" /> : <ExpandMoreIcon fontSize="small" />}
        </IconButton>
      </Box>

      <Collapse in={expanded}>
        <Box sx={{ mt: 1.5 }}>
          <FundamentalFiltersSection
            filters={filters}
            filterOptions={filterOptions}
            updateFilter={updateFilter}
            updateRangeFilter={updateRangeFilter}
            activeCount={fundamentalCount}
            defaultExpanded={sectionDefaultExpanded.fundamental ?? true}
          />

          <TechnicalFiltersSection
            filters={filters}
            updateFilter={updateFilter}
            updateRangeFilter={updateRangeFilter}
            activeCount={technicalCount}
            defaultExpanded={sectionDefaultExpanded.technical ?? true}
          />

          <RatingFiltersSection
            filters={filters}
            updateFilter={updateFilter}
            updateRangeFilter={updateRangeFilter}
            activeCount={ratingCount}
            defaultExpanded={sectionDefaultExpanded.rating ?? true}
          />

          {activeFilters.length > 0 && (
            <Box sx={{ mt: 1, pt: 1, borderTop: '1px solid', borderColor: 'divider' }}>
              <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                <Typography variant="caption" color="text.secondary" sx={{ mr: 0.5, alignSelf: 'center' }}>
                  Active:
                </Typography>
                {activeFilters.map(({ key, label }) => (
                  <Chip
                    key={key}
                    label={label}
                    size="small"
                    onDelete={() => handleDeleteFilter(key)}
                    sx={{
                      height: 22,
                      fontSize: '0.7rem',
                      '& .MuiChip-label': { px: 1 },
                      '& .MuiChip-deleteIcon': { fontSize: '0.9rem' },
                    }}
                  />
                ))}
              </Box>
            </Box>
          )}
        </Box>
      </Collapse>

      {presetsEnabled && (
        <SavePresetDialog
          open={saveDialogOpen}
          onClose={onSaveDialogClose}
          onSave={onSaveDialogSave}
          mode={saveDialogMode}
          initialName={saveDialogInitialName}
          initialDescription={saveDialogInitialDescription}
          error={saveDialogError}
          isLoading={presetsSaving}
        />
      )}
    </Paper>
  );
}

export default memo(FilterPanel, (prevProps, nextProps) => (
  prevProps.filters === nextProps.filters &&
  prevProps.expanded === nextProps.expanded &&
  prevProps.filterOptions === nextProps.filterOptions &&
  prevProps.presetsEnabled === nextProps.presetsEnabled &&
  prevProps.sectionDefaultExpanded === nextProps.sectionDefaultExpanded &&
  prevProps.presets === nextProps.presets &&
  prevProps.activePresetId === nextProps.activePresetId &&
  prevProps.hasUnsavedChanges === nextProps.hasUnsavedChanges &&
  prevProps.presetsLoading === nextProps.presetsLoading &&
  prevProps.presetsSaving === nextProps.presetsSaving &&
  prevProps.saveDialogOpen === nextProps.saveDialogOpen &&
  prevProps.saveDialogMode === nextProps.saveDialogMode &&
  prevProps.saveDialogInitialName === nextProps.saveDialogInitialName &&
  prevProps.saveDialogInitialDescription === nextProps.saveDialogInitialDescription &&
  prevProps.saveDialogError === nextProps.saveDialogError
));

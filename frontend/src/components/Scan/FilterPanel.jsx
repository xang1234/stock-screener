import { memo, useCallback, useMemo } from 'react';
import {
  Paper,
  Typography,
  Box,
  Button,
  TextField,
  Chip,
  Grid,
  Collapse,
  IconButton,
} from '@mui/material';
import FilterListIcon from '@mui/icons-material/FilterList';
import ClearIcon from '@mui/icons-material/Clear';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import ExpandLessIcon from '@mui/icons-material/ExpandLess';
import {
  CompactRangeInput,
  CompactSelect,
  CompactMultiSelect,
  CompactCheckbox,
  FilterSection,
  FilterPresets,
  SavePresetDialog,
  IpoDateFilter,
} from './filters';

const STAGE_OPTIONS = [
  { value: 1, label: 'S1 - Basing' },
  { value: 2, label: 'S2 - Advancing' },
  { value: 3, label: 'S3 - Topping' },
  { value: 4, label: 'S4 - Declining' },
];

const VOLUME_OPTIONS = [
  { value: 10000000, label: '>$10M' },
  { value: 50000000, label: '>$50M' },
  { value: 100000000, label: '>$100M' },
  { value: 500000000, label: '>$500M' },
  { value: 1000000000, label: '>$1B' },
  { value: 5000000000, label: '>$5B' },
  { value: 10000000000, label: '>$10B' },
];

const MARKET_CAP_OPTIONS = [
  { value: 100000000, label: '>$100M' },
  { value: 200000000, label: '>$200M' },
  { value: 500000000, label: '>$500M' },
  { value: 1000000000, label: '>$1B' },
  { value: 2000000000, label: '>$2B' },
  { value: 5000000000, label: '>$5B' },
  { value: 10000000000, label: '>$10B' },
];

// Filter keys grouped by category
const FUNDAMENTAL_KEYS = [
  'symbolSearch', 'minMarketCap', 'minVolume', 'price',
  'epsGrowth', 'salesGrowth', 'epsRating', 'ibdIndustries', 'gicsSectors', 'ipoAfter'
];

const TECHNICAL_KEYS = [
  'stage', 'rsRating', 'rs1m', 'rs3m', 'rs12m', 'maAlignment',
  'adrPercent', 'perfDay', 'perfWeek', 'perfMonth',
  'perf3m', 'perf6m', 'gapPercent', 'volumeSurge',
  'ema10Distance', 'ema20Distance', 'ema50Distance',
  'week52HighDistance', 'week52LowDistance',
  'beta', 'betaAdjRs'
];

const RATING_KEYS = [
  'compositeScore', 'minerviniScore', 'canslimScore', 'ipoScore',
  'customScore', 'volBreakthroughScore',
  'seSetupScore', 'seDistanceToPivot', 'seBbSqueeze', 'seVolumeVs50d',
  'seSetupReady', 'seRsLineNewHigh',
  'vcpScore', 'vcpDetected', 'vcpReady', 'passesTemplate'
];

/**
 * Compact filter panel for scan results with all column filters
 * Organized into 3 collapsible categories: Fundamental, Technical, Rating/Score
 */
function FilterPanel({
  filters,
  onFilterChange,
  onReset,
  filterOptions = {},
  expanded = true,
  onToggle,
  // Preset props
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
  // Dialog props
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

  // Check if a filter has an active value
  const isFilterActive = (key) => {
    const value = filters[key];
    if (value === null || value === undefined) return false;
    if (typeof value === 'string') return value.length > 0;
    if (Array.isArray(value)) return value.length > 0;
    if (typeof value === 'boolean') return true;
    if (typeof value === 'object') {
      // Range filters
      if ('min' in value || 'max' in value) {
        return value.min != null || value.max != null;
      }
      // Multi-select with mode
      if ('values' in value) {
        return value.values?.length > 0;
      }
    }
    return true;
  };

  // Count active filters in a category
  const countActiveInCategory = (keys) => {
    return keys.filter(key => isFilterActive(key)).length;
  };

  // Build active filters list for chips
  const getActiveFilters = () => {
    const active = [];

    if (filters.symbolSearch) {
      active.push({ key: 'symbolSearch', label: `Symbol: ${filters.symbolSearch}` });
    }
    if (filters.stage != null) {
      active.push({ key: 'stage', label: `Stage: ${filters.stage}` });
    }
    if (filters.ratings?.length) {
      active.push({ key: 'ratings', label: `Rating: ${filters.ratings.join(', ')}` });
    }
    if (filters.ibdIndustries?.values?.length) {
      const modeLabel = filters.ibdIndustries.mode === 'exclude' ? ' (Exclude)' : '';
      active.push({ key: 'ibdIndustries', label: `Industry${modeLabel}: ${filters.ibdIndustries.values.length} selected` });
    }
    if (filters.gicsSectors?.values?.length) {
      const modeLabel = filters.gicsSectors.mode === 'exclude' ? ' (Exclude)' : '';
      active.push({ key: 'gicsSectors', label: `Sector${modeLabel}: ${filters.gicsSectors.values.length} selected` });
    }
    if (filters.minVolume != null) {
      const volLabel = VOLUME_OPTIONS.find(o => o.value === filters.minVolume)?.label || `>${filters.minVolume}`;
      active.push({ key: 'minVolume', label: `Volume: ${volLabel}` });
    }
    if (filters.minMarketCap != null) {
      const capLabel = MARKET_CAP_OPTIONS.find(o => o.value === filters.minMarketCap)?.label || `>${filters.minMarketCap}`;
      active.push({ key: 'minMarketCap', label: `Mkt Cap: ${capLabel}` });
    }
    if (filters.ipoAfter) {
      const ipoLabel = filters.ipoAfter.toUpperCase();
      active.push({ key: 'ipoAfter', label: `IPO: >${ipoLabel}` });
    }

    // Score filters
    const scoreFilters = [
      { key: 'compositeScore', label: 'Composite' },
      { key: 'minerviniScore', label: 'Minervini' },
      { key: 'canslimScore', label: 'CANSLIM' },
      { key: 'ipoScore', label: 'IPO' },
      { key: 'customScore', label: 'Custom' },
      { key: 'volBreakthroughScore', label: 'Vol BT' },
      { key: 'seSetupScore', label: 'SE Score' },
    ];

    scoreFilters.forEach(({ key, label }) => {
      const range = filters[key];
      if (range?.min != null || range?.max != null) {
        const minStr = range.min != null ? `≥${range.min}` : '';
        const maxStr = range.max != null ? `≤${range.max}` : '';
        active.push({ key, label: `${label}: ${minStr}${minStr && maxStr ? ', ' : ''}${maxStr}` });
      }
    });

    // RS filters
    const rsFilters = [
      { key: 'rsRating', label: 'RS' },
      { key: 'rs1m', label: 'RS 1M' },
      { key: 'rs3m', label: 'RS 3M' },
      { key: 'rs12m', label: 'RS 12M' },
      { key: 'epsRating', label: 'EPS Rtg' },
    ];

    rsFilters.forEach(({ key, label }) => {
      const range = filters[key];
      if (range?.min != null || range?.max != null) {
        const minStr = range.min != null ? `≥${range.min}` : '';
        const maxStr = range.max != null ? `≤${range.max}` : '';
        active.push({ key, label: `${label}: ${minStr}${minStr && maxStr ? ', ' : ''}${maxStr}` });
      }
    });

    // Price & Growth
    if (filters.price?.min != null || filters.price?.max != null) {
      const { min, max } = filters.price;
      active.push({ key: 'price', label: `Price: ${min != null ? `≥$${min}` : ''}${max != null ? ` ≤$${max}` : ''}` });
    }
    if (filters.adrPercent?.min != null || filters.adrPercent?.max != null) {
      const { min, max } = filters.adrPercent;
      active.push({ key: 'adrPercent', label: `ADR: ${min != null ? `≥${min}%` : ''}${max != null ? ` ≤${max}%` : ''}` });
    }
    if (filters.epsGrowth?.min != null || filters.epsGrowth?.max != null) {
      const { min, max } = filters.epsGrowth;
      active.push({ key: 'epsGrowth', label: `EPS: ${min != null ? `≥${min}%` : ''}${max != null ? ` ≤${max}%` : ''}` });
    }
    if (filters.salesGrowth?.min != null || filters.salesGrowth?.max != null) {
      const { min, max } = filters.salesGrowth;
      active.push({ key: 'salesGrowth', label: `Sales: ${min != null ? `≥${min}%` : ''}${max != null ? ` ≤${max}%` : ''}` });
    }

    // VCP
    if (filters.vcpScore?.min != null || filters.vcpScore?.max != null) {
      const { min, max } = filters.vcpScore;
      active.push({ key: 'vcpScore', label: `VCP Score: ${min != null ? `≥${min}` : ''}${max != null ? ` ≤${max}` : ''}` });
    }
    if (filters.vcpPivot?.min != null || filters.vcpPivot?.max != null) {
      const { min, max } = filters.vcpPivot;
      active.push({ key: 'vcpPivot', label: `VCP Pivot: ${min != null ? `≥$${min}` : ''}${max != null ? ` ≤$${max}` : ''}` });
    }

    // Booleans
    if (filters.vcpDetected != null) {
      active.push({ key: 'vcpDetected', label: `VCP: ${filters.vcpDetected ? 'Yes' : 'No'}` });
    }
    if (filters.vcpReady != null) {
      active.push({ key: 'vcpReady', label: `VCP Ready: ${filters.vcpReady ? 'Yes' : 'No'}` });
    }
    if (filters.maAlignment != null) {
      active.push({ key: 'maAlignment', label: `MA Align: ${filters.maAlignment ? 'Yes' : 'No'}` });
    }
    if (filters.passesTemplate != null) {
      active.push({ key: 'passesTemplate', label: `Passes: ${filters.passesTemplate ? 'Yes' : 'No'}` });
    }
    if (filters.seSetupReady != null) {
      active.push({ key: 'seSetupReady', label: `SE Ready: ${filters.seSetupReady ? 'Yes' : 'No'}` });
    }
    if (filters.seRsLineNewHigh != null) {
      active.push({ key: 'seRsLineNewHigh', label: `RS New Hi: ${filters.seRsLineNewHigh ? 'Yes' : 'No'}` });
    }

    // Technical Filters
    const techFilters = [
      { key: 'perfDay', label: '1D Chg' },
      { key: 'perfWeek', label: '1W Chg' },
      { key: 'perfMonth', label: '1M Chg' },
      { key: 'perf3m', label: '3M Chg' },
      { key: 'perf6m', label: '6M Chg' },
      { key: 'gapPercent', label: 'Gap' },
      { key: 'volumeSurge', label: 'Vol Surge' },
      { key: 'ema10Distance', label: 'vs EMA10' },
      { key: 'ema20Distance', label: 'vs EMA20' },
      { key: 'ema50Distance', label: 'vs EMA50' },
      { key: 'week52HighDistance', label: '52W Hi' },
      { key: 'week52LowDistance', label: '52W Lo' },
      { key: 'beta', label: 'Beta' },
      { key: 'betaAdjRs', label: 'β-adj RS' },
      { key: 'seDistanceToPivot', label: 'Pvt Dist' },
      { key: 'seBbSqueeze', label: 'Squeeze' },
      { key: 'seVolumeVs50d', label: 'Vol/50d' },
    ];

    techFilters.forEach(({ key, label }) => {
      const range = filters[key];
      if (range?.min != null || range?.max != null) {
        const minStr = range.min != null ? `≥${range.min}%` : '';
        const maxStr = range.max != null ? `≤${range.max}%` : '';
        active.push({ key, label: `${label}: ${minStr}${minStr && maxStr ? ', ' : ''}${maxStr}` });
      }
    });

    return active;
  };

  const handleDeleteFilter = (key) => {
    const newFilters = { ...filters };

    // Reset to default based on type
    if (['symbolSearch'].includes(key)) {
      newFilters[key] = '';
    } else if (['stage', 'minVolume', 'minMarketCap', 'ipoAfter'].includes(key)) {
      newFilters[key] = null;
    } else if (['ratings'].includes(key)) {
      newFilters[key] = [];
    } else if (['ibdIndustries', 'gicsSectors'].includes(key)) {
      newFilters[key] = { values: [], mode: 'include' };
    } else if (['vcpDetected', 'vcpReady', 'maAlignment', 'passesTemplate', 'seSetupReady', 'seRsLineNewHigh'].includes(key)) {
      newFilters[key] = null;
    } else {
      // Range filters
      newFilters[key] = { min: null, max: null };
    }

    onFilterChange(newFilters);
  };

  const activeFilters = getActiveFilters();

  // Memoize category counts to prevent recalculation on expand/collapse
  const { fundamentalCount, technicalCount, ratingCount } = useMemo(() => ({
    fundamentalCount: countActiveInCategory(FUNDAMENTAL_KEYS),
    technicalCount: countActiveInCategory(TECHNICAL_KEYS),
    ratingCount: countActiveInCategory(RATING_KEYS),
  }), [filters]);

  return (
    <Paper elevation={1} sx={{ p: 1.5, mb: 2 }}>
      {/* Collapsible Header */}
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

        {/* Filter Presets */}
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

        <Box sx={{ flexGrow: 1 }} />

        {/* Active filter chips shown inline when collapsed */}
        {!expanded && activeFilters.length > 0 && (
          <Box sx={{ display: 'flex', gap: 0.5, mr: 1, flexWrap: 'wrap', maxWidth: '60%' }}>
            {activeFilters.slice(0, 5).map(({ key, label }) => (
              <Chip
                key={key}
                label={label}
                size="small"
                onDelete={(e) => { e.stopPropagation(); handleDeleteFilter(key); }}
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
          onClick={(e) => { e.stopPropagation(); onReset(); }}
          size="small"
          sx={{ fontSize: '0.7rem', py: 0.25, px: 0.75, minWidth: 0 }}
        >
          Reset
        </Button>
        <IconButton size="small" sx={{ ml: 0.5 }}>
          {expanded ? <ExpandLessIcon fontSize="small" /> : <ExpandMoreIcon fontSize="small" />}
        </IconButton>
      </Box>

      {/* Collapsible Content */}
      <Collapse in={expanded}>
        <Box sx={{ mt: 1.5 }}>
          {/* FUNDAMENTAL Section */}
          <FilterSection
            title="Fundamental"
            category="fundamental"
            activeCount={fundamentalCount}
            defaultExpanded={true}
          >
            <Grid container spacing={3}>
              <Grid item xs={12} sm={6} md={1.5}>
                <Box>
                  <Typography
                    variant="caption"
                    color="text.secondary"
                    sx={{ display: 'block', mb: 0.5, fontSize: '0.7rem' }}
                  >
                    Symbol
                  </Typography>
                  <TextField
                    size="small"
                    value={filters.symbolSearch || ''}
                    onChange={(e) => updateFilter('symbolSearch', e.target.value)}
                    placeholder="Search..."
                    sx={{
                      width: '100%',
                      '& .MuiOutlinedInput-root': { height: 28 },
                      '& .MuiOutlinedInput-input': { padding: '4px 8px', fontSize: '0.75rem' },
                    }}
                  />
                </Box>
              </Grid>
              <Grid item xs={6} sm={3} md={1.5}>
                <CompactSelect
                  label="Mkt Cap"
                  value={filters.minMarketCap}
                  options={MARKET_CAP_OPTIONS}
                  onChange={(value) => updateFilter('minMarketCap', value)}
                />
              </Grid>
              <Grid item xs={6} sm={3} md={1.5}>
                <CompactSelect
                  label="Volume"
                  value={filters.minVolume}
                  options={VOLUME_OPTIONS}
                  onChange={(value) => updateFilter('minVolume', value)}
                />
              </Grid>
              <Grid item xs={6} sm={4} md={1.5}>
                <CompactRangeInput
                  label="Price"
                  minValue={filters.price?.min}
                  maxValue={filters.price?.max}
                  onChange={(range) => updateRangeFilter('price', range)}
                  step={1}
                  minLimit={0}
                  prefix="$"
                />
              </Grid>
              <Grid item xs={6} sm={4} md={1.5}>
                <CompactRangeInput
                  label="EPS Growth"
                  minValue={filters.epsGrowth?.min}
                  maxValue={filters.epsGrowth?.max}
                  onChange={(range) => updateRangeFilter('epsGrowth', range)}
                  step={5}
                  suffix="%"
                  minOnly
                />
              </Grid>
              <Grid item xs={6} sm={4} md={1.5}>
                <CompactRangeInput
                  label="Sales Growth"
                  minValue={filters.salesGrowth?.min}
                  maxValue={filters.salesGrowth?.max}
                  onChange={(range) => updateRangeFilter('salesGrowth', range)}
                  step={5}
                  suffix="%"
                  minOnly
                />
              </Grid>
              <Grid item xs={6} sm={4} md={1.5}>
                <CompactRangeInput
                  label="EPS Rating"
                  minValue={filters.epsRating?.min}
                  maxValue={filters.epsRating?.max}
                  onChange={(range) => updateRangeFilter('epsRating', range)}
                  step={5}
                  minLimit={0}
                  maxLimit={99}
                  minOnly
                />
              </Grid>
              <Grid item xs={12} sm={6} md={2.5}>
                <CompactMultiSelect
                  label="IBD Industry"
                  values={filters.ibdIndustries?.values || []}
                  options={filterOptions.ibdIndustries || []}
                  onChange={(values) => updateFilter('ibdIndustries', { ...filters.ibdIndustries, values })}
                  mode={filters.ibdIndustries?.mode || 'include'}
                  onModeChange={(mode) => updateFilter('ibdIndustries', { ...filters.ibdIndustries, mode })}
                  showModeToggle
                />
              </Grid>
              <Grid item xs={12} sm={6} md={2.5}>
                <CompactMultiSelect
                  label="GICS Sector"
                  values={filters.gicsSectors?.values || []}
                  options={filterOptions.gicsSectors || []}
                  onChange={(values) => updateFilter('gicsSectors', { ...filters.gicsSectors, values })}
                  mode={filters.gicsSectors?.mode || 'include'}
                  onModeChange={(mode) => updateFilter('gicsSectors', { ...filters.gicsSectors, mode })}
                  showModeToggle
                />
              </Grid>
              <Grid item xs={6} sm={4} md={2}>
                <IpoDateFilter
                  value={filters.ipoAfter}
                  onChange={(value) => updateFilter('ipoAfter', value)}
                />
              </Grid>
            </Grid>
          </FilterSection>

          {/* TECHNICAL Section */}
          <FilterSection
            title="Technical"
            category="technical"
            activeCount={technicalCount}
            defaultExpanded={true}
          >
            <Grid container spacing={1.5}>
              <Grid item xs={6} sm={4} md={1.5}>
                <CompactRangeInput
                  label="ADR %"
                  minValue={filters.adrPercent?.min}
                  maxValue={filters.adrPercent?.max}
                  onChange={(range) => updateRangeFilter('adrPercent', range)}
                  step={0.5}
                  minLimit={0}
                  suffix="%"
                />
              </Grid>
              <Grid item xs={6} sm={3} md={1.5}>
                <CompactSelect
                  label="Stage"
                  value={filters.stage}
                  options={STAGE_OPTIONS}
                  onChange={(value) => updateFilter('stage', value)}
                />
              </Grid>
              <Grid item xs={6} sm={4} md={1.5}>
                <CompactRangeInput
                  label="RS Rating"
                  minValue={filters.rsRating?.min}
                  maxValue={filters.rsRating?.max}
                  onChange={(range) => updateRangeFilter('rsRating', range)}
                  step={5}
                  minLimit={0}
                  maxLimit={100}
                  minOnly
                />
              </Grid>
              <Grid item xs={6} sm={4} md={1.5}>
                <CompactRangeInput
                  label="RS 1M"
                  minValue={filters.rs1m?.min}
                  maxValue={filters.rs1m?.max}
                  onChange={(range) => updateRangeFilter('rs1m', range)}
                  step={5}
                  minLimit={0}
                  maxLimit={100}
                  minOnly
                />
              </Grid>
              <Grid item xs={6} sm={4} md={1.5}>
                <CompactRangeInput
                  label="RS 3M"
                  minValue={filters.rs3m?.min}
                  maxValue={filters.rs3m?.max}
                  onChange={(range) => updateRangeFilter('rs3m', range)}
                  step={5}
                  minLimit={0}
                  maxLimit={100}
                  minOnly
                />
              </Grid>
              <Grid item xs={6} sm={4} md={1.5}>
                <CompactRangeInput
                  label="RS 12M"
                  minValue={filters.rs12m?.min}
                  maxValue={filters.rs12m?.max}
                  onChange={(range) => updateRangeFilter('rs12m', range)}
                  step={5}
                  minLimit={0}
                  maxLimit={100}
                  minOnly
                />
              </Grid>
              <Grid item xs={6} sm={4} md={1.5}>
                <CompactRangeInput
                  label="Beta"
                  minValue={filters.beta?.min}
                  maxValue={filters.beta?.max}
                  onChange={(range) => updateRangeFilter('beta', range)}
                  step={0.1}
                  minLimit={0}
                  maxLimit={5}
                />
              </Grid>
              <Grid item xs={6} sm={4} md={1.5}>
                <CompactRangeInput
                  label="β-adj RS"
                  minValue={filters.betaAdjRs?.min}
                  maxValue={filters.betaAdjRs?.max}
                  onChange={(range) => updateRangeFilter('betaAdjRs', range)}
                  step={5}
                  minLimit={0}
                  maxLimit={100}
                  minOnly
                />
              </Grid>
              <Grid item xs={6} sm={3} md={1}>
                <CompactCheckbox
                  label="MA Align"
                  value={filters.maAlignment}
                  onChange={(value) => updateFilter('maAlignment', value)}
                />
              </Grid>
              <Grid item xs={6} sm={4} md={1.5}>
                <CompactRangeInput
                  label="1D Chg %"
                  minValue={filters.perfDay?.min}
                  maxValue={filters.perfDay?.max}
                  onChange={(range) => updateRangeFilter('perfDay', range)}
                  step={1}
                  suffix="%"
                  minOnly
                />
              </Grid>
              <Grid item xs={6} sm={4} md={1.5}>
                <CompactRangeInput
                  label="1W Chg %"
                  minValue={filters.perfWeek?.min}
                  maxValue={filters.perfWeek?.max}
                  onChange={(range) => updateRangeFilter('perfWeek', range)}
                  step={1}
                  suffix="%"
                  minOnly
                />
              </Grid>
              <Grid item xs={6} sm={4} md={1.5}>
                <CompactRangeInput
                  label="1M Chg %"
                  minValue={filters.perfMonth?.min}
                  maxValue={filters.perfMonth?.max}
                  onChange={(range) => updateRangeFilter('perfMonth', range)}
                  step={1}
                  suffix="%"
                  minOnly
                />
              </Grid>
              <Grid item xs={6} sm={4} md={1.5}>
                <CompactRangeInput
                  label="3M Chg %"
                  minValue={filters.perf3m?.min}
                  maxValue={filters.perf3m?.max}
                  onChange={(range) => updateRangeFilter('perf3m', range)}
                  step={5}
                  suffix="%"
                  minOnly
                />
              </Grid>
              <Grid item xs={6} sm={4} md={1.5}>
                <CompactRangeInput
                  label="6M Chg %"
                  minValue={filters.perf6m?.min}
                  maxValue={filters.perf6m?.max}
                  onChange={(range) => updateRangeFilter('perf6m', range)}
                  step={10}
                  suffix="%"
                  minOnly
                />
              </Grid>
              <Grid item xs={6} sm={4} md={1.5}>
                <CompactRangeInput
                  label="Gap %"
                  minValue={filters.gapPercent?.min}
                  maxValue={filters.gapPercent?.max}
                  onChange={(range) => updateRangeFilter('gapPercent', range)}
                  step={1}
                  suffix="%"
                />
              </Grid>
              <Grid item xs={6} sm={4} md={1.5}>
                <CompactRangeInput
                  label="Vol Surge"
                  minValue={filters.volumeSurge?.min}
                  maxValue={filters.volumeSurge?.max}
                  onChange={(range) => updateRangeFilter('volumeSurge', range)}
                  step={0.5}
                  minLimit={0}
                  suffix="x"
                />
              </Grid>
              <Grid item xs={6} sm={4} md={1.5}>
                <CompactRangeInput
                  label="% vs EMA10"
                  minValue={filters.ema10Distance?.min}
                  maxValue={filters.ema10Distance?.max}
                  onChange={(range) => updateRangeFilter('ema10Distance', range)}
                  step={1}
                  suffix="%"
                  minOnly
                />
              </Grid>
              <Grid item xs={6} sm={4} md={1.5}>
                <CompactRangeInput
                  label="% vs EMA20"
                  minValue={filters.ema20Distance?.min}
                  maxValue={filters.ema20Distance?.max}
                  onChange={(range) => updateRangeFilter('ema20Distance', range)}
                  step={1}
                  suffix="%"
                  minOnly
                />
              </Grid>
              <Grid item xs={6} sm={4} md={1.5}>
                <CompactRangeInput
                  label="% vs EMA50"
                  minValue={filters.ema50Distance?.min}
                  maxValue={filters.ema50Distance?.max}
                  onChange={(range) => updateRangeFilter('ema50Distance', range)}
                  step={1}
                  suffix="%"
                  minOnly
                />
              </Grid>
              <Grid item xs={6} sm={4} md={1.5}>
                <CompactRangeInput
                  label="% from 52W Hi"
                  minValue={filters.week52HighDistance?.min}
                  maxValue={filters.week52HighDistance?.max}
                  onChange={(range) => updateRangeFilter('week52HighDistance', range)}
                  step={1}
                  suffix="%"
                />
              </Grid>
              <Grid item xs={6} sm={4} md={1.5}>
                <CompactRangeInput
                  label="% from 52W Lo"
                  minValue={filters.week52LowDistance?.min}
                  maxValue={filters.week52LowDistance?.max}
                  onChange={(range) => updateRangeFilter('week52LowDistance', range)}
                  step={1}
                  suffix="%"
                />
              </Grid>
            </Grid>
          </FilterSection>

          {/* RATING / SCORE Section */}
          <FilterSection
            title="Rating / Score"
            category="rating"
            activeCount={ratingCount}
            defaultExpanded={true}
          >
            <Grid container spacing={1.5}>
              <Grid item xs={6} sm={4} md={1.2}>
                <CompactRangeInput
                  label="Composite"
                  minValue={filters.compositeScore?.min}
                  maxValue={filters.compositeScore?.max}
                  onChange={(range) => updateRangeFilter('compositeScore', range)}
                  step={5}
                  minLimit={0}
                  maxLimit={100}
                  minOnly
                />
              </Grid>
              <Grid item xs={6} sm={4} md={1.2}>
                <CompactRangeInput
                  label="Minervini"
                  minValue={filters.minerviniScore?.min}
                  maxValue={filters.minerviniScore?.max}
                  onChange={(range) => updateRangeFilter('minerviniScore', range)}
                  step={5}
                  minLimit={0}
                  maxLimit={100}
                  minOnly
                />
              </Grid>
              <Grid item xs={6} sm={4} md={1.2}>
                <CompactRangeInput
                  label="CANSLIM"
                  minValue={filters.canslimScore?.min}
                  maxValue={filters.canslimScore?.max}
                  onChange={(range) => updateRangeFilter('canslimScore', range)}
                  step={5}
                  minLimit={0}
                  maxLimit={100}
                  minOnly
                />
              </Grid>
              <Grid item xs={6} sm={4} md={1.2}>
                <CompactRangeInput
                  label="IPO"
                  minValue={filters.ipoScore?.min}
                  maxValue={filters.ipoScore?.max}
                  onChange={(range) => updateRangeFilter('ipoScore', range)}
                  step={5}
                  minLimit={0}
                  maxLimit={100}
                  minOnly
                />
              </Grid>
              <Grid item xs={6} sm={4} md={1.2}>
                <CompactRangeInput
                  label="Custom"
                  minValue={filters.customScore?.min}
                  maxValue={filters.customScore?.max}
                  onChange={(range) => updateRangeFilter('customScore', range)}
                  step={5}
                  minLimit={0}
                  maxLimit={100}
                  minOnly
                />
              </Grid>
              <Grid item xs={6} sm={4} md={1.2}>
                <CompactRangeInput
                  label="Vol BT"
                  minValue={filters.volBreakthroughScore?.min}
                  maxValue={filters.volBreakthroughScore?.max}
                  onChange={(range) => updateRangeFilter('volBreakthroughScore', range)}
                  step={5}
                  minLimit={0}
                  maxLimit={100}
                  minOnly
                />
              </Grid>
              <Grid item xs={6} sm={4} md={1.2}>
                <CompactRangeInput
                  label="SE Score"
                  minValue={filters.seSetupScore?.min}
                  maxValue={filters.seSetupScore?.max}
                  onChange={(range) => updateRangeFilter('seSetupScore', range)}
                  step={5}
                  minLimit={0}
                  maxLimit={100}
                  minOnly
                />
              </Grid>
              <Grid item xs={6} sm={4} md={1.2}>
                <CompactRangeInput
                  label="Pvt Dist"
                  minValue={filters.seDistanceToPivot?.min}
                  maxValue={filters.seDistanceToPivot?.max}
                  onChange={(range) => updateRangeFilter('seDistanceToPivot', range)}
                  step={1}
                  suffix="%"
                />
              </Grid>
              <Grid item xs={6} sm={4} md={1.2}>
                <CompactRangeInput
                  label="Squeeze"
                  minValue={filters.seBbSqueeze?.min}
                  maxValue={filters.seBbSqueeze?.max}
                  onChange={(range) => updateRangeFilter('seBbSqueeze', range)}
                  step={5}
                  minLimit={0}
                  maxLimit={100}
                />
              </Grid>
              <Grid item xs={6} sm={4} md={1.2}>
                <CompactRangeInput
                  label="Vol/50d"
                  minValue={filters.seVolumeVs50d?.min}
                  maxValue={filters.seVolumeVs50d?.max}
                  onChange={(range) => updateRangeFilter('seVolumeVs50d', range)}
                  step={0.5}
                  minLimit={0}
                  suffix="x"
                />
              </Grid>
              <Grid item xs={6} sm={3} md={1}>
                <CompactCheckbox
                  label="SE Ready"
                  value={filters.seSetupReady}
                  onChange={(value) => updateFilter('seSetupReady', value)}
                />
              </Grid>
              <Grid item xs={6} sm={3} md={1}>
                <CompactCheckbox
                  label="RS Hi"
                  value={filters.seRsLineNewHigh}
                  onChange={(value) => updateFilter('seRsLineNewHigh', value)}
                />
              </Grid>
              <Grid item xs={6} sm={4} md={1.2}>
                <CompactRangeInput
                  label="VCP Score"
                  minValue={filters.vcpScore?.min}
                  maxValue={filters.vcpScore?.max}
                  onChange={(range) => updateRangeFilter('vcpScore', range)}
                  step={5}
                  minLimit={0}
                  maxLimit={100}
                  minOnly
                />
              </Grid>
              <Grid item xs={6} sm={3} md={1}>
                <CompactCheckbox
                  label="VCP"
                  value={filters.vcpDetected}
                  onChange={(value) => updateFilter('vcpDetected', value)}
                />
              </Grid>
              <Grid item xs={6} sm={3} md={1}>
                <CompactCheckbox
                  label="VCP Ready"
                  value={filters.vcpReady}
                  onChange={(value) => updateFilter('vcpReady', value)}
                />
              </Grid>
              <Grid item xs={6} sm={3} md={1}>
                <CompactCheckbox
                  label="Passes"
                  value={filters.passesTemplate}
                  onChange={(value) => updateFilter('passesTemplate', value)}
                />
              </Grid>
            </Grid>
          </FilterSection>

          {/* Active Filters Chips */}
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

      {/* Save Preset Dialog */}
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
    </Paper>
  );
}

// Memoize component to prevent re-renders when parent state changes
export default memo(FilterPanel, (prevProps, nextProps) => {
  // Only re-render if these props change
  return (
    prevProps.filters === nextProps.filters &&
    prevProps.expanded === nextProps.expanded &&
    prevProps.filterOptions === nextProps.filterOptions &&
    prevProps.presets === nextProps.presets &&
    prevProps.activePresetId === nextProps.activePresetId &&
    prevProps.hasUnsavedChanges === nextProps.hasUnsavedChanges &&
    prevProps.presetsLoading === nextProps.presetsLoading &&
    prevProps.presetsSaving === nextProps.presetsSaving &&
    prevProps.saveDialogOpen === nextProps.saveDialogOpen
  );
});

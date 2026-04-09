import { Box, Chip, Paper, Tab, Tabs, ToggleButton, ToggleButtonGroup, Typography } from '@mui/material';
import CategoryIcon from '@mui/icons-material/Category';
import ViewListIcon from '@mui/icons-material/ViewList';
import { SOURCE_TYPES } from '../constants';

export default function ThemesFiltersPanel({
  themeView,
  onViewChange,
  l1Categories,
  categoryFilter,
  onCategoryFilterChange,
  selectedTab,
  onTabChange,
  selectedSourceTypes,
  onSourceTypeToggle,
}) {
  return (
    <Paper sx={{ mb: 3 }}>
      <Box
        sx={{
          px: 2,
          py: 1,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          borderBottom: 1,
          borderColor: 'divider',
        }}
      >
        <ToggleButtonGroup value={themeView} exclusive onChange={onViewChange} size="small" sx={{ height: 28 }}>
          <ToggleButton value="grouped" sx={{ px: 1.5, fontSize: '11px' }}>
            <CategoryIcon sx={{ mr: 0.5, fontSize: 16 }} />
            L1 Grouped
          </ToggleButton>
          <ToggleButton value="flat" sx={{ px: 1.5, fontSize: '11px' }}>
            <ViewListIcon sx={{ mr: 0.5, fontSize: 16 }} />
            All Themes
          </ToggleButton>
        </ToggleButtonGroup>

        {themeView === 'grouped' && l1Categories?.categories?.length > 0 && (
          <Box display="flex" gap={0.5} alignItems="center">
            <Typography variant="caption" color="text.secondary" sx={{ mr: 0.5 }}>
              Sector:
            </Typography>
            <Chip
              label="All"
              size="small"
              variant={categoryFilter === null ? 'filled' : 'outlined'}
              color={categoryFilter === null ? 'primary' : 'default'}
              onClick={() => onCategoryFilterChange(null)}
              sx={{ cursor: 'pointer', height: 22, fontSize: '10px' }}
            />
            {l1Categories.categories.map((category) => (
              <Chip
                key={category.category}
                label={`${category.category} (${category.count})`}
                size="small"
                variant={categoryFilter === category.category ? 'filled' : 'outlined'}
                color={categoryFilter === category.category ? 'primary' : 'default'}
                onClick={() => onCategoryFilterChange(categoryFilter === category.category ? null : category.category)}
                sx={{ cursor: 'pointer', height: 22, fontSize: '10px', textTransform: 'capitalize' }}
              />
            ))}
          </Box>
        )}
      </Box>

      {themeView === 'flat' && (
        <>
          <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
            <Tabs value={selectedTab} onChange={onTabChange}>
              <Tab label="All Themes" value="all" />
              <Tab
                label={
                  <Box display="flex" alignItems="center">
                    Trending
                    <Chip label="Hot" size="small" color="success" sx={{ ml: 0.5, height: 18 }} />
                  </Box>
                }
                value="trending"
              />
              <Tab label="Emerging" value="emerging" />
              <Tab label="Active" value="active" />
              <Tab label="Fading" value="fading" />
            </Tabs>
          </Box>

          <Box
            sx={{
              px: 2,
              py: 1.5,
              display: 'flex',
              alignItems: 'center',
              gap: 1,
              borderTop: 1,
              borderColor: 'divider',
            }}
          >
            <Typography variant="body2" color="text.secondary" sx={{ mr: 1 }}>
              Sources:
            </Typography>
            {SOURCE_TYPES.map((source) => (
              <Chip
                key={source.value}
                label={source.label}
                size="small"
                variant={selectedSourceTypes.includes(source.value) ? 'filled' : 'outlined'}
                color={selectedSourceTypes.includes(source.value) ? 'primary' : 'default'}
                onClick={() => onSourceTypeToggle(source.value)}
                sx={{ cursor: 'pointer' }}
              />
            ))}
          </Box>
        </>
      )}
    </Paper>
  );
}

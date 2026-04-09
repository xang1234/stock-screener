/**
 * Themes Tab Component
 *
 * Main tab for user-defined themes display.
 * Features:
 * - Theme toggle buttons to switch between themes
 * - Settings icon to open ThemeManager modal
 * - Renders ThemeTable for selected theme
 */
import { useState, useEffect } from 'react';
import { useQuery } from '@tanstack/react-query';
import {
  Box,
  Typography,
  IconButton,
  CircularProgress,
  ToggleButtonGroup,
  ToggleButton,
  Tooltip,
} from '@mui/material';
import SettingsIcon from '@mui/icons-material/Settings';
import RefreshIcon from '@mui/icons-material/Refresh';
import AddIcon from '@mui/icons-material/Add';
import { getThemes, getThemeData } from '../../api/userThemes';
import ThemeTable from './ThemeTable';
import ThemeManager from './ThemeManager';

const EMPTY_THEMES = [];

function ThemesTab() {
  const [selectedThemeId, setSelectedThemeId] = useState(null);
  const [managerOpen, setManagerOpen] = useState(false);

  // Fetch list of themes for toggle
  const {
    data: themesData,
    isLoading: themesLoading,
    refetch: refetchThemes,
  } = useQuery({
    queryKey: ['userThemes'],
    queryFn: getThemes,
  });

  const themes = themesData?.themes ?? EMPTY_THEMES;

  // Auto-select first theme if none selected
  useEffect(() => {
    if (!selectedThemeId && themes.length > 0) {
      setSelectedThemeId(themes[0].id);
    }
  }, [themes, selectedThemeId]);

  // Fetch selected theme data with sparklines
  const {
    data: themeData,
    isLoading: dataLoading,
    refetch: refetchData,
  } = useQuery({
    queryKey: ['userThemeData', selectedThemeId],
    queryFn: () => getThemeData(selectedThemeId),
    enabled: !!selectedThemeId,
  });

  const handleRefresh = () => {
    refetchThemes();
    if (selectedThemeId) {
      refetchData();
    }
  };

  const handleThemeChange = (event, newThemeId) => {
    if (newThemeId !== null) {
      setSelectedThemeId(newThemeId);
    }
  };

  if (themesLoading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" height="100%">
        <CircularProgress />
      </Box>
    );
  }

  return (
    <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      {/* Header with theme toggle and settings */}
      <Box
        sx={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          mb: 1,
          pb: 1,
          borderBottom: 1,
          borderColor: 'divider',
        }}
      >
        {/* Theme Toggle */}
        <Box display="flex" alignItems="center" gap={2}>
          <Typography variant="subtitle2" color="text.secondary">
            Theme:
          </Typography>
          {themes.length > 0 ? (
            <ToggleButtonGroup
              value={selectedThemeId}
              exclusive
              onChange={handleThemeChange}
              size="small"
            >
              {themes.map((theme) => (
                <ToggleButton
                  key={theme.id}
                  value={theme.id}
                  sx={{ px: 2, py: 0.5, textTransform: 'none' }}
                >
                  {theme.name}
                </ToggleButton>
              ))}
            </ToggleButtonGroup>
          ) : (
            <Typography variant="body2" color="text.secondary">
              No themes created yet
            </Typography>
          )}
        </Box>

        {/* Actions */}
        <Box display="flex" alignItems="center" gap={0.5}>
          <Tooltip title="Refresh data">
            <IconButton onClick={handleRefresh} size="small">
              <RefreshIcon fontSize="small" />
            </IconButton>
          </Tooltip>
          <Tooltip title="Manage themes">
            <IconButton onClick={() => setManagerOpen(true)} size="small">
              <SettingsIcon fontSize="small" />
            </IconButton>
          </Tooltip>
        </Box>
      </Box>

      {/* Theme Table */}
      <Box sx={{ flex: 1, overflow: 'auto' }}>
        {dataLoading ? (
          <Box display="flex" justifyContent="center" py={4}>
            <CircularProgress />
          </Box>
        ) : themeData ? (
          <ThemeTable themeData={themeData} onRefresh={handleRefresh} />
        ) : themes.length === 0 ? (
          <Box textAlign="center" py={4}>
            <Typography color="text.secondary" gutterBottom>
              No themes yet. Create your first theme to get started.
            </Typography>
            <Tooltip title="Manage themes">
              <IconButton color="primary" onClick={() => setManagerOpen(true)} size="large">
                <AddIcon />
              </IconButton>
            </Tooltip>
          </Box>
        ) : null}
      </Box>

      {/* Theme Manager Modal */}
      <ThemeManager open={managerOpen} onClose={() => setManagerOpen(false)} onUpdate={handleRefresh} />
    </Box>
  );
}

export default ThemesTab;

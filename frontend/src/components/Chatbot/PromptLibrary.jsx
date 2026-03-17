/**
 * PromptLibrary - Popover for selecting and managing saved prompts.
 */
import { useState } from 'react';
import {
  Box,
  IconButton,
  Popover,
  Typography,
  Button,
  List,
  ListItem,
  ListItemButton,
  ListItemText,
  Chip,
  CircularProgress,
  Divider,
} from '@mui/material';
import BookmarkIcon from '@mui/icons-material/Bookmark';
import AddIcon from '@mui/icons-material/Add';
import EditIcon from '@mui/icons-material/Edit';
import DeleteIcon from '@mui/icons-material/Delete';

import { usePromptPresets } from '../../hooks/usePromptPresets';
import SavePromptDialog from './SavePromptDialog';
import TickerInputDialog from './TickerInputDialog';

function PromptLibrary({ onInsertPrompt, disabled }) {
  const [anchorEl, setAnchorEl] = useState(null);
  const [saveDialogOpen, setSaveDialogOpen] = useState(false);
  const [tickerDialogOpen, setTickerDialogOpen] = useState(false);
  const [editingPreset, setEditingPreset] = useState(null);
  const [selectedPreset, setSelectedPreset] = useState(null);

  const {
    presets,
    isLoading,
    createPresetAsync,
    updatePresetAsync,
    deletePreset,
    isCreating,
    isUpdating,
    isDeleting,
  } = usePromptPresets();

  const handleClick = (event) => {
    setAnchorEl(event.currentTarget);
  };

  const handleClose = () => {
    setAnchorEl(null);
  };

  const handleAddNew = () => {
    setEditingPreset(null);
    setSaveDialogOpen(true);
  };

  const handleEdit = (preset, e) => {
    e.stopPropagation();
    setEditingPreset(preset);
    setSaveDialogOpen(true);
  };

  const handleDelete = (preset, e) => {
    e.stopPropagation();
    if (window.confirm(`Delete prompt "${preset.name}"?`)) {
      deletePreset(preset.id);
    }
  };

  const handleSelectPreset = (preset) => {
    handleClose();

    if (preset.has_ticker_placeholder) {
      // Show ticker input dialog
      setSelectedPreset(preset);
      setTickerDialogOpen(true);
    } else {
      // Insert directly
      onInsertPrompt(preset.content);
    }
  };

  const handleSavePreset = async (data) => {
    try {
      if (editingPreset) {
        await updatePresetAsync({
          presetId: editingPreset.id,
          updates: data,
        });
      } else {
        await createPresetAsync(data);
      }
      setSaveDialogOpen(false);
      setEditingPreset(null);
    } catch (error) {
      console.error('Failed to save preset:', error);
    }
  };

  const handleTickerInsert = (processedContent) => {
    onInsertPrompt(processedContent);
    setSelectedPreset(null);
  };

  const open = Boolean(anchorEl);

  return (
    <>
      <IconButton
        onClick={handleClick}
        size="small"
        disabled={disabled}
        sx={{
          color: open ? 'primary.main' : 'text.secondary',
          backgroundColor: open ? 'action.selected' : 'transparent',
          '&:hover': {
            backgroundColor: 'action.hover',
          },
        }}
        title="Prompt Library"
      >
        <BookmarkIcon />
      </IconButton>

      <Popover
        open={open}
        anchorEl={anchorEl}
        onClose={handleClose}
        anchorOrigin={{
          vertical: 'top',
          horizontal: 'left',
        }}
        transformOrigin={{
          vertical: 'bottom',
          horizontal: 'left',
        }}
        slotProps={{
          paper: {
            sx: {
              width: 320,
              maxHeight: 400,
            },
          },
        }}
      >
        {/* Header */}
        <Box
          sx={{
            px: 2,
            py: 1.5,
            borderBottom: 1,
            borderColor: 'divider',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
          }}
        >
          <Typography variant="subtitle2" fontWeight={600}>
            Prompt Library
          </Typography>
          <Button
            size="small"
            startIcon={<AddIcon />}
            onClick={handleAddNew}
            sx={{ textTransform: 'none', fontSize: '0.75rem' }}
          >
            Add New
          </Button>
        </Box>

        {/* Content */}
        {isLoading ? (
          <Box sx={{ p: 3, display: 'flex', justifyContent: 'center' }}>
            <CircularProgress size={24} />
          </Box>
        ) : presets.length === 0 ? (
          <Box sx={{ p: 3, textAlign: 'center' }}>
            <Typography variant="body2" color="text.secondary">
              No saved prompts yet.
            </Typography>
            <Typography variant="caption" color="text.secondary">
              Click &quot;Add New&quot; to create your first prompt template.
            </Typography>
          </Box>
        ) : (
          <List dense sx={{ py: 0, maxHeight: 300, overflowY: 'auto' }}>
            {presets.map((preset, index) => (
              <Box key={preset.id}>
                {index > 0 && <Divider />}
                <ListItem
                  disablePadding
                  secondaryAction={
                    <Box sx={{ display: 'flex', gap: 0.5 }}>
                      <IconButton
                        edge="end"
                        size="small"
                        onClick={(e) => handleEdit(preset, e)}
                        sx={{ opacity: 0.6, '&:hover': { opacity: 1 } }}
                      >
                        <EditIcon fontSize="small" />
                      </IconButton>
                      <IconButton
                        edge="end"
                        size="small"
                        onClick={(e) => handleDelete(preset, e)}
                        disabled={isDeleting}
                        sx={{ opacity: 0.6, '&:hover': { opacity: 1, color: 'error.main' } }}
                      >
                        <DeleteIcon fontSize="small" />
                      </IconButton>
                    </Box>
                  }
                >
                  <ListItemButton onClick={() => handleSelectPreset(preset)} sx={{ pr: 10 }}>
                    <ListItemText
                      primary={
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                          <Typography variant="body2" fontWeight={500} noWrap>
                            {preset.name}
                          </Typography>
                          {preset.has_ticker_placeholder && (
                            <Chip
                              label="{ticker}"
                              size="small"
                              variant="outlined"
                              sx={{
                                height: 18,
                                fontSize: '0.65rem',
                                '& .MuiChip-label': { px: 0.75 },
                              }}
                            />
                          )}
                        </Box>
                      }
                      secondary={
                        preset.description ? (
                          <Typography
                            variant="caption"
                            color="text.secondary"
                            sx={{
                              display: '-webkit-box',
                              WebkitLineClamp: 1,
                              WebkitBoxOrient: 'vertical',
                              overflow: 'hidden',
                            }}
                          >
                            {preset.description}
                          </Typography>
                        ) : null
                      }
                    />
                  </ListItemButton>
                </ListItem>
              </Box>
            ))}
          </List>
        )}
      </Popover>

      {/* Save/Edit Dialog */}
      <SavePromptDialog
        open={saveDialogOpen}
        onClose={() => {
          setSaveDialogOpen(false);
          setEditingPreset(null);
        }}
        onSave={handleSavePreset}
        preset={editingPreset}
        isSaving={isCreating || isUpdating}
      />

      {/* Ticker Input Dialog */}
      <TickerInputDialog
        open={tickerDialogOpen}
        onClose={() => {
          setTickerDialogOpen(false);
          setSelectedPreset(null);
        }}
        onInsert={handleTickerInsert}
        promptContent={selectedPreset?.content || ''}
        promptName={selectedPreset?.name || ''}
      />
    </>
  );
}

export default PromptLibrary;

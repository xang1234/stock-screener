/**
 * SavePromptDialog - Dialog for creating/editing prompt presets.
 */
import { useState, useEffect } from 'react';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  Button,
  Typography,
  Box,
  Alert,
} from '@mui/material';

function SavePromptDialog({ open, onClose, onSave, preset, isSaving }) {
  const [name, setName] = useState('');
  const [description, setDescription] = useState('');
  const [content, setContent] = useState('');
  const [error, setError] = useState('');

  const isEditing = Boolean(preset);

  // Reset form when dialog opens/closes or preset changes
  useEffect(() => {
    if (open) {
      if (preset) {
        setName(preset.name || '');
        setDescription(preset.description || '');
        setContent(preset.content || '');
      } else {
        setName('');
        setDescription('');
        setContent('');
      }
      setError('');
    }
  }, [open, preset]);

  const handleSave = () => {
    // Validation
    if (!name.trim()) {
      setError('Name is required');
      return;
    }
    if (!content.trim()) {
      setError('Prompt content is required');
      return;
    }

    onSave({
      name: name.trim(),
      description: description.trim() || null,
      content: content.trim(),
    });
  };

  const handleKeyDown = (e) => {
    // Submit on Ctrl/Cmd + Enter
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
      e.preventDefault();
      handleSave();
    }
  };

  return (
    <Dialog
      open={open}
      onClose={onClose}
      maxWidth="sm"
      fullWidth
      PaperProps={{
        sx: { maxHeight: '80vh' },
      }}
    >
      <DialogTitle>
        {isEditing ? 'Edit Prompt' : 'Create New Prompt'}
      </DialogTitle>
      <DialogContent>
        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2, mt: 1 }}>
          {error && (
            <Alert severity="error" onClose={() => setError('')}>
              {error}
            </Alert>
          )}

          <TextField
            label="Name"
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder="e.g., Minervini Analysis"
            required
            fullWidth
            size="small"
            autoFocus
          />

          <TextField
            label="Description"
            value={description}
            onChange={(e) => setDescription(e.target.value)}
            placeholder="Optional description of what this prompt does"
            fullWidth
            size="small"
          />

          <Box>
            <TextField
              label="Prompt Content"
              value={content}
              onChange={(e) => setContent(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Enter your prompt template here..."
              required
              fullWidth
              multiline
              rows={6}
              size="small"
            />
            <Typography
              variant="caption"
              color="text.secondary"
              sx={{ mt: 0.5, display: 'block' }}
            >
              Use {'{ticker}'} as a placeholder for stock symbol. When using this prompt, you&apos;ll be asked to enter a ticker.
            </Typography>
          </Box>
        </Box>
      </DialogContent>
      <DialogActions sx={{ px: 3, pb: 2 }}>
        <Button onClick={onClose} disabled={isSaving}>
          Cancel
        </Button>
        <Button
          onClick={handleSave}
          variant="contained"
          disabled={isSaving || !name.trim() || !content.trim()}
        >
          {isSaving ? 'Saving...' : isEditing ? 'Save Changes' : 'Create Prompt'}
        </Button>
      </DialogActions>
    </Dialog>
  );
}

export default SavePromptDialog;

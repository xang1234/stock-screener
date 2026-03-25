import { useEffect, useState } from 'react';
import {
  Box,
  Dialog,
  DialogContent,
  DialogTitle,
  IconButton,
  Tab,
  Tabs,
  Typography,
} from '@mui/material';
import CloseIcon from '@mui/icons-material/Close';
import { ManageSourcesContent } from './ManageSourcesModal';
import { ThemePolicySettingsContent } from './ThemePolicySettingsModal';

function ThemeSettingsDialog({ open, onClose, pipeline, initialTab = 0 }) {
  const [tab, setTab] = useState(initialTab);

  useEffect(() => {
    if (open) setTab(initialTab);
  }, [open, initialTab]);

  return (
    <Dialog open={open} onClose={onClose} maxWidth="xl" fullWidth>
      <DialogTitle sx={{ pb: 0 }}>
        <Box display="flex" justifyContent="space-between" alignItems="center">
          <Typography variant="h6">Theme Settings</Typography>
          <IconButton onClick={onClose} size="small">
            <CloseIcon />
          </IconButton>
        </Box>
        <Tabs value={tab} onChange={(_, v) => setTab(v)} sx={{ mt: 1 }}>
          <Tab label="Content Sources" />
          <Tab label="Policy Controls" />
        </Tabs>
      </DialogTitle>
      <DialogContent dividers>
        {tab === 0 && <ManageSourcesContent />}
        {tab === 1 && <ThemePolicySettingsContent pipeline={pipeline} />}
      </DialogContent>
    </Dialog>
  );
}

export default ThemeSettingsDialog;

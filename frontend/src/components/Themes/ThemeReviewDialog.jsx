import { useEffect, useState } from 'react';
import {
  Badge,
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
import CompareArrowsIcon from '@mui/icons-material/CompareArrows';
import AutoAwesomeIcon from '@mui/icons-material/AutoAwesome';
import { ThemeMergeReviewContent } from './ThemeMergeReviewModal';
import { ThemeCandidateReviewContent } from './ThemeCandidateReviewModal';

function ThemeReviewDialog({ open, onClose, pipeline, initialTab = 0, mergeCount = 0, candidateCount = 0 }) {
  const [tab, setTab] = useState(initialTab);

  useEffect(() => {
    if (open) setTab(initialTab);
  }, [open, initialTab]);

  return (
    <Dialog open={open} onClose={onClose} maxWidth="lg" fullWidth>
      <DialogTitle sx={{ pb: 0 }}>
        <Box display="flex" justifyContent="space-between" alignItems="center">
          <Typography variant="h6">Theme Review</Typography>
          <IconButton onClick={onClose} size="small">
            <CloseIcon />
          </IconButton>
        </Box>
        <Tabs value={tab} onChange={(_, v) => setTab(v)} sx={{ mt: 1 }}>
          <Tab
            icon={
              <Badge badgeContent={mergeCount} color="warning" max={99}>
                <CompareArrowsIcon sx={{ fontSize: 18 }} />
              </Badge>
            }
            iconPosition="start"
            label="Merges"
            sx={{ minHeight: 48 }}
          />
          <Tab
            icon={
              <Badge badgeContent={candidateCount} color="info" max={99}>
                <AutoAwesomeIcon sx={{ fontSize: 18 }} />
              </Badge>
            }
            iconPosition="start"
            label="Candidates"
            sx={{ minHeight: 48 }}
          />
        </Tabs>
      </DialogTitle>
      <DialogContent dividers sx={{ p: 0 }}>
        {tab === 0 && <ThemeMergeReviewContent onClose={onClose} />}
        {tab === 1 && (
          <Box sx={{ p: 0 }}>
            <ThemeCandidateReviewContent pipeline={pipeline} />
          </Box>
        )}
      </DialogContent>
    </Dialog>
  );
}

export default ThemeReviewDialog;

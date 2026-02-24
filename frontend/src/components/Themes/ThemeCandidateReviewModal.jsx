import { useEffect, useMemo, useState } from 'react';
import {
  Alert,
  Box,
  Button,
  Chip,
  Checkbox,
  CircularProgress,
  Dialog,
  DialogContent,
  DialogTitle,
  IconButton,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Tooltip,
  Typography,
} from '@mui/material';
import CloseIcon from '@mui/icons-material/Close';
import CheckCircleOutlineIcon from '@mui/icons-material/CheckCircleOutline';
import CancelOutlinedIcon from '@mui/icons-material/CancelOutlined';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { getCandidateThemeQueue, reviewCandidateThemes } from '../../api/themes';

const confidenceBandColor = (band) => {
  if (band === '0.85-1.00') return 'success';
  if (band === '0.70-0.84') return 'info';
  if (band === '0.40-0.69') return 'warning';
  return 'default';
};

function ThemeCandidateReviewModal({ open, onClose, pipeline = 'technical' }) {
  const queryClient = useQueryClient();
  const [selectedIds, setSelectedIds] = useState([]);
  const [errorText, setErrorText] = useState(null);

  const { data: queue, isLoading, error } = useQuery({
    queryKey: ['candidateThemeQueue', pipeline],
    queryFn: () => getCandidateThemeQueue({ limit: 200, offset: 0, pipeline }),
    enabled: open,
    staleTime: 30_000,
  });

  const selectedSet = useMemo(() => new Set(selectedIds), [selectedIds]);
  const items = queue?.items || [];
  const visibleItemIds = useMemo(() => items.map((row) => row.theme_cluster_id), [items]);
  const selectedVisibleIds = useMemo(
    () => visibleItemIds.filter((id) => selectedSet.has(id)),
    [visibleItemIds, selectedSet],
  );
  const allSelected = items.length > 0 && selectedVisibleIds.length === items.length;

  useEffect(() => {
    if (!open) {
      setSelectedIds([]);
      setErrorText(null);
    }
  }, [open]);

  useEffect(() => {
    setSelectedIds((current) => current.filter((id) => visibleItemIds.includes(id)));
  }, [visibleItemIds]);

  const reviewMutation = useMutation({
    mutationFn: ({ action, ids }) =>
      reviewCandidateThemes(
        {
          theme_cluster_ids: ids,
          action,
          actor: 'analyst_ui',
        },
        pipeline,
      ),
    onSuccess: () => {
      setSelectedIds([]);
      queryClient.invalidateQueries({ queryKey: ['candidateThemeQueue', pipeline] });
      queryClient.invalidateQueries({ queryKey: ['themeRankings'] });
      queryClient.invalidateQueries({ queryKey: ['themeDetail'] });
    },
    onError: (mutationError) => {
      setErrorText(mutationError?.message || 'Review action failed');
    },
  });

  const toggleTheme = (themeId) => {
    setSelectedIds((current) => {
      if (current.includes(themeId)) return current.filter((id) => id !== themeId);
      return [...current, themeId];
    });
  };

  const toggleSelectAll = () => {
    if (allSelected) {
      setSelectedIds([]);
      return;
    }
    setSelectedIds(items.map((row) => row.theme_cluster_id));
  };

  const runAction = (action) => {
    setErrorText(null);
    if (!selectedVisibleIds.length) {
      setErrorText('Select at least one candidate theme.');
      return;
    }
    reviewMutation.mutate({ action, ids: selectedVisibleIds });
  };

  return (
    <Dialog open={open} onClose={onClose} maxWidth="lg" fullWidth>
      <DialogTitle>
        <Box display="flex" justifyContent="space-between" alignItems="center">
          <Box>
            <Typography variant="h6">Candidate Theme Queue Review</Typography>
            <Typography variant="body2" color="text.secondary">
              Promote high-quality candidates to active or reject noise in bulk.
            </Typography>
          </Box>
          <IconButton onClick={onClose} size="small">
            <CloseIcon />
          </IconButton>
        </Box>
      </DialogTitle>
      <DialogContent>
        {(errorText || error) && (
          <Alert severity="error" sx={{ mb: 2 }}>
            {errorText || error?.message || 'Failed to load candidate queue'}
          </Alert>
        )}

        {queue?.confidence_bands?.length > 0 && (
          <Box sx={{ display: 'flex', gap: 0.75, flexWrap: 'wrap', mb: 1.5 }}>
            {queue.confidence_bands.map((band) => (
              <Chip
                key={band.band}
                size="small"
                label={`${band.band}: ${band.count}`}
                color={confidenceBandColor(band.band)}
                variant="outlined"
              />
            ))}
          </Box>
        )}

        <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 1.5 }}>
          <Box sx={{ fontSize: '12px', color: 'text.secondary' }}>
            {queue?.total ?? 0} candidates in queue
          </Box>
          <Box sx={{ display: 'flex', gap: 1 }}>
            <Button
              size="small"
              color="success"
              variant="contained"
              startIcon={<CheckCircleOutlineIcon />}
              onClick={() => runAction('promote')}
              disabled={reviewMutation.isPending || selectedVisibleIds.length === 0}
            >
              Promote Selected
            </Button>
            <Button
              size="small"
              color="error"
              variant="outlined"
              startIcon={<CancelOutlinedIcon />}
              onClick={() => runAction('reject')}
              disabled={reviewMutation.isPending || selectedVisibleIds.length === 0}
            >
              Reject Selected
            </Button>
          </Box>
        </Box>

        {isLoading ? (
          <Box display="flex" justifyContent="center" p={4}>
            <CircularProgress />
          </Box>
        ) : (
          <TableContainer sx={{ maxHeight: 460 }}>
            <Table size="small" stickyHeader>
              <TableHead>
                <TableRow>
                  <TableCell padding="checkbox">
                    <Checkbox
                      size="small"
                      indeterminate={selectedVisibleIds.length > 0 && !allSelected}
                      checked={allSelected}
                      onChange={toggleSelectAll}
                    />
                  </TableCell>
                  <TableCell>Theme</TableCell>
                  <TableCell align="right">Mentions (7D)</TableCell>
                  <TableCell align="right">Sources (7D)</TableCell>
                  <TableCell align="right">Persistence</TableCell>
                  <TableCell align="right">Avg Conf (30D)</TableCell>
                  <TableCell>Band</TableCell>
                  <TableCell>Reason</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {items.map((row) => (
                  <TableRow
                    key={row.theme_cluster_id}
                    hover
                    selected={selectedSet.has(row.theme_cluster_id)}
                    onClick={() => toggleTheme(row.theme_cluster_id)}
                    sx={{ cursor: 'pointer' }}
                  >
                    <TableCell padding="checkbox">
                      <Checkbox
                        size="small"
                        checked={selectedSet.has(row.theme_cluster_id)}
                        onClick={(e) => e.stopPropagation()}
                        onChange={(e) => {
                          e.stopPropagation();
                          toggleTheme(row.theme_cluster_id);
                        }}
                      />
                    </TableCell>
                    <TableCell sx={{ fontWeight: 600 }}>
                      {row.theme_display_name || row.theme_name}
                    </TableCell>
                    <TableCell align="right" sx={{ fontFamily: 'monospace' }}>
                      {row.mentions_7d}
                    </TableCell>
                    <TableCell align="right" sx={{ fontFamily: 'monospace' }}>
                      {row.source_diversity_7d}
                    </TableCell>
                    <TableCell align="right" sx={{ fontFamily: 'monospace' }}>
                      {row.persistence_days_7d}d
                    </TableCell>
                    <TableCell align="right" sx={{ fontFamily: 'monospace' }}>
                      {(row.avg_confidence_30d * 100).toFixed(0)}%
                    </TableCell>
                    <TableCell>
                      <Chip
                        label={row.confidence_band}
                        size="small"
                        color={confidenceBandColor(row.confidence_band)}
                        variant="outlined"
                      />
                    </TableCell>
                    <TableCell sx={{ maxWidth: 240 }}>
                      <Tooltip title={row.queue_reason || '-'}>
                        <Box sx={{ whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                          {row.queue_reason || '-'}
                        </Box>
                      </Tooltip>
                    </TableCell>
                  </TableRow>
                ))}
                {!items.length && (
                  <TableRow>
                    <TableCell colSpan={8} align="center" sx={{ py: 4, color: 'text.secondary' }}>
                      Candidate queue is empty.
                    </TableCell>
                  </TableRow>
                )}
              </TableBody>
            </Table>
          </TableContainer>
        )}
      </DialogContent>
    </Dialog>
  );
}

export default ThemeCandidateReviewModal;

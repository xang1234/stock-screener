import { useEffect, useMemo, useState } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import {
  Alert,
  Box,
  Button,
  CircularProgress,
  Dialog,
  DialogActions,
  DialogContent,
  DialogTitle,
  FormControl,
  InputLabel,
  MenuItem,
  Select,
  Stack,
  Typography,
} from '@mui/material';
import { previewWatchlistAdd } from '../../api/assistant';
import { bulkAddItems, getWatchlists } from '../../api/userWatchlists';

function AssistantWatchlistDialog({ open, symbols, onClose }) {
  const queryClient = useQueryClient();
  const [selectedWatchlistId, setSelectedWatchlistId] = useState('');

  const watchlistsQuery = useQuery({
    queryKey: ['assistant-watchlists'],
    queryFn: getWatchlists,
    enabled: open,
  });

  const watchlists = useMemo(
    () => watchlistsQuery.data?.watchlists || [],
    [watchlistsQuery.data?.watchlists],
  );
  const selectedWatchlist = useMemo(
    () => watchlists.find((watchlist) => String(watchlist.id) === String(selectedWatchlistId)) || null,
    [selectedWatchlistId, watchlists],
  );

  useEffect(() => {
    if (!open) {
      setSelectedWatchlistId('');
      return;
    }
    if (watchlists.length === 0) {
      return;
    }
    const hasSelected = watchlists.some(
      (watchlist) => String(watchlist.id) === String(selectedWatchlistId),
    );
    if (!hasSelected) {
      setSelectedWatchlistId(String(watchlists[0].id));
    }
  }, [open, selectedWatchlistId, watchlists]);

  const previewQuery = useQuery({
    queryKey: ['assistant-watchlist-preview', selectedWatchlistId, symbols],
    queryFn: () => previewWatchlistAdd({ watchlist: selectedWatchlist.name, symbols }),
    enabled: open && Boolean(selectedWatchlist),
  });

  const addMutation = useMutation({
    mutationFn: () => bulkAddItems(Number(selectedWatchlistId), previewQuery.data?.addable_symbols || []),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['userWatchlists'] });
      queryClient.invalidateQueries({ queryKey: ['userWatchlistData', Number(selectedWatchlistId)] });
      onClose();
    },
  });

  const noWatchlists = open && !watchlistsQuery.isLoading && watchlists.length === 0;
  const preview = previewQuery.data;
  const canConfirm = Boolean(preview?.addable_symbols?.length)
    && !watchlistsQuery.isError
    && !previewQuery.isError
    && !addMutation.isPending;

  return (
    <Dialog open={open} onClose={onClose} fullWidth maxWidth="sm">
      <DialogTitle>Add assistant tickers to watchlist</DialogTitle>
      <DialogContent dividers>
        <Stack spacing={2}>
          <Typography variant="body2" color="text.secondary">
            Symbols extracted from the latest assistant response: {symbols.join(', ')}
          </Typography>

          {watchlistsQuery.isLoading ? (
            <Box sx={{ display: 'flex', justifyContent: 'center', py: 2 }}>
              <CircularProgress size={24} />
            </Box>
          ) : watchlistsQuery.isError ? (
            <Alert severity="error">
              {watchlistsQuery.error?.response?.data?.detail
                || watchlistsQuery.error?.message
                || 'Failed to load watchlists.'}
            </Alert>
          ) : noWatchlists ? (
            <Alert severity="info">Create a watchlist first to add assistant suggestions.</Alert>
          ) : (
            <FormControl fullWidth size="small">
              <InputLabel id="assistant-watchlist-select-label">Watchlist</InputLabel>
              <Select
                labelId="assistant-watchlist-select-label"
                value={selectedWatchlistId}
                label="Watchlist"
                onChange={(event) => setSelectedWatchlistId(event.target.value)}
              >
                {watchlists.map((watchlist) => (
                  <MenuItem key={watchlist.id} value={String(watchlist.id)}>
                    {watchlist.name}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
          )}

          {selectedWatchlist && previewQuery.isLoading && (
            <Box sx={{ display: 'flex', justifyContent: 'center', py: 1 }}>
              <CircularProgress size={24} />
            </Box>
          )}

          {selectedWatchlist && previewQuery.isError && (
            <Alert severity="error">
              {previewQuery.error?.response?.data?.detail
                || previewQuery.error?.message
                || 'Failed to preview symbols for this watchlist.'}
            </Alert>
          )}

          {preview?.summary && <Alert severity="info">{preview.summary}</Alert>}

          {preview && (
            <Stack spacing={1}>
              <Typography variant="body2">
                Addable: {preview.addable_symbols.join(', ') || 'None'}
              </Typography>
              <Typography variant="body2">
                Already present: {preview.existing_symbols.join(', ') || 'None'}
              </Typography>
              <Typography variant="body2">
                Invalid: {preview.invalid_symbols.join(', ') || 'None'}
              </Typography>
            </Stack>
          )}

          {addMutation.isError && (
            <Alert severity="error">
              {addMutation.error?.response?.data?.detail || addMutation.error?.message || 'Failed to add symbols.'}
            </Alert>
          )}
        </Stack>
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose}>Cancel</Button>
        <Button
          variant="contained"
          onClick={() => addMutation.mutate()}
          disabled={!canConfirm}
        >
          {addMutation.isPending ? 'Adding...' : 'Confirm add'}
        </Button>
      </DialogActions>
    </Dialog>
  );
}

export default AssistantWatchlistDialog;

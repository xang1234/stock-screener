/**
 * Add to Watchlist Menu Component
 *
 * A reusable dropdown menu for adding stock(s) to a watchlist.
 * Shows existing watchlists and allows creating a new one.
 *
 * Props:
 * - symbols: string | string[] - Symbol(s) to add
 * - trigger: ReactNode - Optional custom trigger element (default: PlaylistAddIcon button)
 * - onSuccess: () => void - Callback after successful add
 * - size: 'small' | 'medium' - Button size
 */
import { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import {
  IconButton,
  Menu,
  MenuItem,
  ListItemIcon,
  ListItemText,
  Divider,
  TextField,
  Box,
  Typography,
  CircularProgress,
  Tooltip,
  Alert,
} from '@mui/material';
import PlaylistAddIcon from '@mui/icons-material/PlaylistAdd';
import AddIcon from '@mui/icons-material/Add';
import CheckIcon from '@mui/icons-material/Check';
import {
  getWatchlists,
  getWatchlistMemberships,
  createWatchlist,
  addItem,
  bulkAddItems,
} from '../../api/userWatchlists';
import { invalidateWatchlistItems } from '../../api/userWatchlistQueries';
import { getErrorMessage } from '../../utils/errorMessage';

function AddToWatchlistMenu({ symbols, trigger, onSuccess, size = 'small' }) {
  const [anchorEl, setAnchorEl] = useState(null);
  const [newWatchlistName, setNewWatchlistName] = useState('');
  const [showNewInput, setShowNewInput] = useState(false);
  const queryClient = useQueryClient();

  const open = Boolean(anchorEl);

  // Normalize symbols to array
  const symbolList = Array.isArray(symbols) ? symbols : [symbols];
  const normalizedSymbols = [...new Set(symbolList.map(
    (symbol) => symbol.trim().replace(/^\$+/, '').toUpperCase(),
  ))];
  const membershipQueryKey = ['userWatchlistMemberships', normalizedSymbols];

  // Fetch watchlists
  const {
    data: watchlistsData,
    isLoading: isWatchlistsLoading,
    error: watchlistsError,
  } = useQuery({
    queryKey: ['userWatchlists'],
    queryFn: getWatchlists,
    enabled: open,
  });

  const {
    data: membershipData,
    isFetching: isMembershipFetching,
    error: membershipError,
  } = useQuery({
    queryKey: membershipQueryKey,
    queryFn: () => getWatchlistMemberships(normalizedSymbols),
    enabled: open && normalizedSymbols.length > 0,
  });

  const watchlists = watchlistsData?.watchlists || [];
  const memberships = membershipData?.memberships || {};

  const recordMembership = (watchlistId, querySymbols, symbolsToRecord = querySymbols) => {
    queryClient.setQueryData(['userWatchlistMemberships', querySymbols], (current) => {
      const nextMemberships = { ...(current?.memberships || {}) };
      symbolsToRecord.forEach((symbol) => {
        nextMemberships[symbol] = [
          ...new Set([...(nextMemberships[symbol] || []), watchlistId]),
        ];
      });
      return { memberships: nextMemberships };
    });
  };

  // Create watchlist mutation
  const createMutation = useMutation({
    mutationFn: createWatchlist,
    onSuccess: async (newWatchlist) => {
      queryClient.invalidateQueries({ queryKey: ['userWatchlists'] });
      setNewWatchlistName('');
      setShowNewInput(false);
      // Add symbols to the new watchlist
      await handleAddToWatchlist(newWatchlist.id);
    },
  });

  // Add item mutation (single)
  const addItemMutation = useMutation({
    mutationFn: ({ watchlistId, data }) => addItem(watchlistId, data),
    onSuccess: (_, variables) => {
      invalidateWatchlistItems(queryClient, variables.watchlistId);
      recordMembership(variables.watchlistId, [variables.data.symbol]);
      onSuccess?.();
    },
  });

  // Bulk add mutation
  const bulkAddMutation = useMutation({
    mutationFn: ({ watchlistId, symbols }) => bulkAddItems(watchlistId, symbols),
    onSuccess: (addedItems, variables) => {
      invalidateWatchlistItems(queryClient, variables.watchlistId);
      recordMembership(
        variables.watchlistId,
        variables.symbols,
        addedItems.map((item) => item.symbol),
      );
      onSuccess?.();
    },
  });

  const handleClick = (event) => {
    event.stopPropagation();
    queryClient.invalidateQueries({
      queryKey: membershipQueryKey,
      refetchType: 'none',
    });
    setAnchorEl(event.currentTarget);
  };

  const handleClose = () => {
    setAnchorEl(null);
    setShowNewInput(false);
    setNewWatchlistName('');
  };

  const handleAddToWatchlist = async (watchlistId) => {
    if (normalizedSymbols.length === 1) {
      addItemMutation.mutate({
        watchlistId,
        data: { symbol: normalizedSymbols[0] },
      });
    } else {
      bulkAddMutation.mutate({
        watchlistId,
        symbols: normalizedSymbols,
      });
    }
  };

  const handleCreateAndAdd = () => {
    if (newWatchlistName.trim()) {
      createMutation.mutate({ name: newWatchlistName.trim() });
    }
  };

  const isPending = addItemMutation.isPending || bulkAddMutation.isPending || createMutation.isPending;
  const isLoading = isWatchlistsLoading || isMembershipFetching;
  const visibleError = watchlistsError
    || membershipError
    || addItemMutation.error
    || bulkAddMutation.error
    || createMutation.error;

  return (
    <>
      {trigger ? (
        <Box onClick={handleClick}>{trigger}</Box>
      ) : (
        <Tooltip title="Add to watchlist">
          <IconButton
            size={size}
            onClick={handleClick}
            sx={{ p: size === 'small' ? 0.25 : undefined }}
          >
            <PlaylistAddIcon fontSize={size} />
          </IconButton>
        </Tooltip>
      )}

      <Menu
        anchorEl={anchorEl}
        open={open}
        onClose={handleClose}
        onClick={(e) => e.stopPropagation()}
        PaperProps={{
          sx: { minWidth: 200, maxHeight: 400 },
        }}
      >
        <Box sx={{ px: 2, py: 1 }}>
          <Typography variant="caption" color="text.secondary">
            Add {symbolList.length > 1 ? `${symbolList.length} stocks` : symbolList[0]} to:
          </Typography>
        </Box>

        <Divider />

        {visibleError && (
          <Alert severity="error" sx={{ mx: 1, my: 1 }}>
            {getErrorMessage(visibleError, 'Unable to update watchlist')}
          </Alert>
        )}

        {isLoading ? (
          <Box display="flex" justifyContent="center" py={2}>
            <CircularProgress size={24} />
          </Box>
        ) : (
          <>
            {watchlists.map((watchlist) => {
              const alreadyAdded = normalizedSymbols.every(
                (symbol) => memberships[symbol]?.includes(watchlist.id),
              );
              return (
                <MenuItem
                  key={watchlist.id}
                  onClick={() => {
                    if (!alreadyAdded) handleAddToWatchlist(watchlist.id);
                  }}
                  disabled={isPending || Boolean(membershipError) || alreadyAdded}
                >
                  <ListItemText
                    primary={watchlist.name}
                    secondary={alreadyAdded ? 'Already added' : undefined}
                  />
                  {alreadyAdded && (
                    <ListItemIcon sx={{ minWidth: 'auto', ml: 1 }}>
                      <CheckIcon fontSize="small" color="success" />
                    </ListItemIcon>
                  )}
                </MenuItem>
              );
            })}

            {watchlists.length === 0 && (
              <Box sx={{ px: 2, py: 1 }}>
                <Typography variant="body2" color="text.secondary">
                  No watchlists yet
                </Typography>
              </Box>
            )}
          </>
        )}

        <Divider />

        {showNewInput ? (
          <Box sx={{ px: 1.5, py: 1, display: 'flex', gap: 0.5 }}>
            <TextField
              size="small"
              placeholder="Watchlist name"
              value={newWatchlistName}
              onChange={(e) => setNewWatchlistName(e.target.value)}
              onKeyPress={(e) => e.key === 'Enter' && handleCreateAndAdd()}
              autoFocus
              sx={{ flex: 1, '& .MuiInputBase-input': { fontSize: '0.875rem' } }}
            />
            <IconButton
              size="small"
              onClick={handleCreateAndAdd}
              disabled={!newWatchlistName.trim() || isPending}
              color="primary"
            >
              <AddIcon fontSize="small" />
            </IconButton>
          </Box>
        ) : (
          <MenuItem onClick={() => setShowNewInput(true)}>
            <ListItemIcon>
              <AddIcon fontSize="small" />
            </ListItemIcon>
            <ListItemText primary="Create new watchlist" />
          </MenuItem>
        )}
      </Menu>
    </>
  );
}

export default AddToWatchlistMenu;

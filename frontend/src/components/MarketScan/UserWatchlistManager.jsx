/**
 * User Watchlist Manager Modal
 *
 * Allows users to:
 * - Create/rename/delete watchlists
 * - Add/remove stocks from watchlists
 * - Reorder watchlists and stocks via drag-and-drop
 */
import { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  TextField,
  List,
  ListItem,
  ListItemText,
  ListItemSecondaryAction,
  ListItemButton,
  IconButton,
  Box,
  Typography,
  Alert,
  Stack,
} from '@mui/material';
import DeleteIcon from '@mui/icons-material/Delete';
import AddIcon from '@mui/icons-material/Add';
import DragIndicatorIcon from '@mui/icons-material/DragIndicator';
import EditIcon from '@mui/icons-material/Edit';
import UploadFileIcon from '@mui/icons-material/UploadFile';
import { DragDropContext, Droppable, Draggable } from '@hello-pangea/dnd';
import {
  getWatchlists,
  createWatchlist,
  updateWatchlist,
  deleteWatchlist,
  getWatchlistData,
  addItem,
  removeItem,
  reorderWatchlists,
  reorderItems,
  importItems,
} from '../../api/userWatchlists';

function UserWatchlistManager({ open, onClose, onUpdate }) {
  const queryClient = useQueryClient();
  const [selectedWatchlistId, setSelectedWatchlistId] = useState(null);
  const [newWatchlistName, setNewWatchlistName] = useState('');
  const [editingName, setEditingName] = useState(false);
  const [editedName, setEditedName] = useState('');
  const [newSymbol, setNewSymbol] = useState('');
  const [feedback, setFeedback] = useState(null);
  const [importDialogOpen, setImportDialogOpen] = useState(false);
  const [importContent, setImportContent] = useState('');

  // Fetch watchlists
  const { data: watchlistsData, isLoading: watchlistsLoading } = useQuery({
    queryKey: ['userWatchlists'],
    queryFn: getWatchlists,
    enabled: open,
  });

  const watchlists = watchlistsData?.watchlists || [];

  // Fetch watchlist data for editing
  const { data: watchlistData } = useQuery({
    queryKey: ['userWatchlistData', selectedWatchlistId],
    queryFn: () => getWatchlistData(selectedWatchlistId),
    enabled: !!selectedWatchlistId && open,
  });

  // Mutations
  const createMutation = useMutation({
    mutationFn: (data) => createWatchlist(data),
    onSuccess: (newWatchlist) => {
      queryClient.invalidateQueries({ queryKey: ['userWatchlists'] });
      setNewWatchlistName('');
      setFeedback(null);
      setSelectedWatchlistId(newWatchlist.id);
      onUpdate?.();
    },
    onError: (err) => setFeedback({ severity: 'error', message: err.response?.data?.detail || 'Failed to create watchlist' }),
  });

  const updateMutation = useMutation({
    mutationFn: ({ id, updates }) => updateWatchlist(id, updates),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['userWatchlists'] });
      setEditingName(false);
      onUpdate?.();
    },
    onError: (err) => setFeedback({ severity: 'error', message: err.response?.data?.detail || 'Failed to update watchlist' }),
  });

  const deleteMutation = useMutation({
    mutationFn: (id) => deleteWatchlist(id),
    onSuccess: (_, deletedId) => {
      queryClient.invalidateQueries({ queryKey: ['userWatchlists'] });
      if (selectedWatchlistId === deletedId) {
        setSelectedWatchlistId(null);
      }
      onUpdate?.();
    },
    onError: (err) => setFeedback({ severity: 'error', message: err.response?.data?.detail || 'Failed to delete watchlist' }),
  });

  const addItemMutation = useMutation({
    mutationFn: ({ watchlistId, data }) => addItem(watchlistId, data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['userWatchlistData', selectedWatchlistId] });
      setNewSymbol('');
      onUpdate?.();
    },
    onError: (err) => setFeedback({ severity: 'error', message: err.response?.data?.detail || 'Failed to add stock' }),
  });

  const removeItemMutation = useMutation({
    mutationFn: (id) => removeItem(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['userWatchlistData', selectedWatchlistId] });
      onUpdate?.();
    },
    onError: (err) => setFeedback({ severity: 'error', message: err.response?.data?.detail || 'Failed to remove stock' }),
  });

  const reorderWatchlistsMutation = useMutation({
    mutationFn: (ids) => reorderWatchlists(ids),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['userWatchlists'] });
      onUpdate?.();
    },
  });

  const reorderItemsMutation = useMutation({
    mutationFn: ({ watchlistId, itemIds }) => reorderItems(watchlistId, itemIds),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['userWatchlistData', selectedWatchlistId] });
      onUpdate?.();
    },
  });

  const importItemsMutation = useMutation({
    mutationFn: ({ watchlistId, payload }) => importItems(watchlistId, payload),
    onSuccess: (result) => {
      queryClient.invalidateQueries({ queryKey: ['userWatchlistData', selectedWatchlistId] });
      setImportContent('');
      setImportDialogOpen(false);
      const parts = [
        `Imported ${result.added.length} symbol${result.added.length === 1 ? '' : 's'}`,
      ];
      if (result.skipped_existing.length > 0) {
        parts.push(`${result.skipped_existing.length} already existed`);
      }
      if (result.invalid_symbols.length > 0) {
        parts.push(`${result.invalid_symbols.length} invalid`);
      }
      setFeedback({ severity: 'success', message: parts.join(', ') });
      onUpdate?.();
    },
    onError: (err) => setFeedback({ severity: 'error', message: err.response?.data?.detail || 'Failed to import symbols' }),
  });

  const handleCreate = () => {
    if (newWatchlistName.trim()) {
      createMutation.mutate({ name: newWatchlistName.trim() });
    }
  };

  const handleDelete = (id) => {
    if (window.confirm('Delete this watchlist and all its stocks?')) {
      deleteMutation.mutate(id);
    }
  };

  const handleStartRename = () => {
    const selected = watchlists.find((w) => w.id === selectedWatchlistId);
    if (selected) {
      setEditedName(selected.name);
      setEditingName(true);
    }
  };

  const handleSaveRename = () => {
    if (editedName.trim() && selectedWatchlistId) {
      updateMutation.mutate({ id: selectedWatchlistId, updates: { name: editedName.trim() } });
    }
  };

  const handleAddSymbol = () => {
    if (newSymbol.trim() && selectedWatchlistId) {
      addItemMutation.mutate({
        watchlistId: selectedWatchlistId,
        data: { symbol: newSymbol.trim().toUpperCase() },
      });
    }
  };

  const handleRemoveItem = (itemId) => {
    removeItemMutation.mutate(itemId);
  };

  // Drag-drop handlers
  const handleWatchlistDragEnd = (result) => {
    if (!result.destination || result.destination.index === result.source.index) return;

    const items = Array.from(watchlists);
    const [reorderedItem] = items.splice(result.source.index, 1);
    items.splice(result.destination.index, 0, reorderedItem);

    queryClient.setQueryData(['userWatchlists'], {
      ...watchlistsData,
      watchlists: items,
    });

    reorderWatchlistsMutation.mutate(items.map((item) => item.id));
  };

  const handleItemDragEnd = (result) => {
    if (!result.destination || result.destination.index === result.source.index) return;
    if (!watchlistData?.items) return;

    const items = Array.from(watchlistData.items);
    const [reorderedItem] = items.splice(result.source.index, 1);
    items.splice(result.destination.index, 0, reorderedItem);

    queryClient.setQueryData(['userWatchlistData', selectedWatchlistId], {
      ...watchlistData,
      items,
    });

    reorderItemsMutation.mutate({
      watchlistId: selectedWatchlistId,
      itemIds: items.map((item) => item.id),
    });
  };

  const selectedWatchlist = watchlists.find((w) => w.id === selectedWatchlistId);

  return (
    <Dialog open={open} onClose={onClose} maxWidth="md" fullWidth>
      <DialogTitle>Manage Watchlists</DialogTitle>
      <DialogContent sx={{ minHeight: 450 }}>
        {feedback && (
          <Alert severity={feedback.severity} sx={{ mb: 2 }} onClose={() => setFeedback(null)}>
            {feedback.message}
          </Alert>
        )}

        <Box sx={{ display: 'flex', gap: 2, height: '100%' }}>
          {/* Left Panel: Watchlist List */}
          <Box
            sx={{
              width: 220,
              borderRight: 1,
              borderColor: 'divider',
              pr: 2,
              flexShrink: 0,
            }}
          >
            <Typography variant="subtitle2" gutterBottom>
              Watchlists
            </Typography>

            {/* Create new watchlist */}
            <Box display="flex" gap={0.5} mb={2}>
              <TextField
                size="small"
                placeholder="New watchlist"
                value={newWatchlistName}
                onChange={(e) => setNewWatchlistName(e.target.value)}
                onKeyPress={(e) => e.key === 'Enter' && handleCreate()}
                fullWidth
                sx={{ '& .MuiInputBase-input': { fontSize: '0.875rem' } }}
              />
              <IconButton
                color="primary"
                onClick={handleCreate}
                disabled={!newWatchlistName.trim() || createMutation.isPending}
                size="small"
              >
                <AddIcon />
              </IconButton>
            </Box>

            {/* Watchlist list with drag-drop */}
            <DragDropContext onDragEnd={handleWatchlistDragEnd}>
              <Droppable droppableId="watchlists">
                {(provided) => (
                  <List dense sx={{ mx: -1 }} ref={provided.innerRef} {...provided.droppableProps}>
                    {watchlists.map((watchlist, index) => (
                      <Draggable key={watchlist.id} draggableId={`wl-${watchlist.id}`} index={index}>
                        {(provided, snapshot) => (
                          <ListItemButton
                            ref={provided.innerRef}
                            {...provided.draggableProps}
                            selected={watchlist.id === selectedWatchlistId}
                            onClick={() => setSelectedWatchlistId(watchlist.id)}
                            sx={{
                              borderRadius: 1,
                              mb: 0.5,
                              bgcolor: snapshot.isDragging ? 'action.hover' : undefined,
                            }}
                          >
                            <Box
                              {...provided.dragHandleProps}
                              sx={{ mr: 0.5, display: 'flex', alignItems: 'center', cursor: 'grab' }}
                            >
                              <DragIndicatorIcon fontSize="small" color="action" />
                            </Box>
                            <ListItemText
                              primary={watchlist.name}
                              primaryTypographyProps={{ fontSize: '0.875rem' }}
                            />
                            <IconButton
                              size="small"
                              onClick={(e) => {
                                e.stopPropagation();
                                handleDelete(watchlist.id);
                              }}
                              sx={{ opacity: 0.5, '&:hover': { opacity: 1 } }}
                            >
                              <DeleteIcon fontSize="small" />
                            </IconButton>
                          </ListItemButton>
                        )}
                      </Draggable>
                    ))}
                    {provided.placeholder}
                    {watchlists.length === 0 && !watchlistsLoading && (
                      <Typography variant="caption" color="text.secondary" sx={{ px: 1 }}>
                        No watchlists yet
                      </Typography>
                    )}
                  </List>
                )}
              </Droppable>
            </DragDropContext>
          </Box>

          {/* Right Panel: Watchlist Details */}
          <Box sx={{ flex: 1, overflow: 'auto' }}>
            {selectedWatchlist ? (
              <>
                {/* Watchlist name with rename */}
                <Box display="flex" alignItems="center" gap={1} mb={2}>
                  {editingName ? (
                    <>
                      <TextField
                        size="small"
                        value={editedName}
                        onChange={(e) => setEditedName(e.target.value)}
                        onKeyPress={(e) => e.key === 'Enter' && handleSaveRename()}
                        autoFocus
                        sx={{ flex: 1 }}
                      />
                      <Button size="small" onClick={handleSaveRename}>
                        Save
                      </Button>
                      <Button size="small" onClick={() => setEditingName(false)}>
                        Cancel
                      </Button>
                    </>
                  ) : (
                    <>
                      <Typography variant="h6">{selectedWatchlist.name}</Typography>
                      <IconButton size="small" onClick={handleStartRename}>
                        <EditIcon fontSize="small" />
                      </IconButton>
                    </>
                  )}
                </Box>

                {/* Add symbol */}
                <Stack direction={{ xs: 'column', sm: 'row' }} spacing={1} mb={2}>
                  <TextField
                    size="small"
                    placeholder="Add symbol (e.g., NVDA)"
                    value={newSymbol}
                    onChange={(e) => setNewSymbol(e.target.value)}
                    onKeyPress={(e) => e.key === 'Enter' && handleAddSymbol()}
                    sx={{ width: 200 }}
                  />
                  <Button
                    variant="contained"
                    size="small"
                    onClick={handleAddSymbol}
                    disabled={!newSymbol.trim() || addItemMutation.isPending}
                    startIcon={<AddIcon />}
                  >
                    Add
                  </Button>
                  <Button
                    variant="outlined"
                    size="small"
                    onClick={() => setImportDialogOpen(true)}
                    startIcon={<UploadFileIcon />}
                  >
                    Import
                  </Button>
                </Stack>

                {/* Items list with drag-drop */}
                {watchlistData?.items && watchlistData.items.length > 0 ? (
                  <DragDropContext onDragEnd={handleItemDragEnd}>
                    <Droppable droppableId="items">
                      {(provided) => (
                        <List dense ref={provided.innerRef} {...provided.droppableProps}>
                          {watchlistData.items.map((item, index) => (
                            <Draggable key={item.id} draggableId={`item-${item.id}`} index={index}>
                              {(provided, snapshot) => (
                                <ListItem
                                  ref={provided.innerRef}
                                  {...provided.draggableProps}
                                  sx={{
                                    bgcolor: snapshot.isDragging ? 'action.hover' : undefined,
                                    borderRadius: 1,
                                  }}
                                >
                                  <Box
                                    {...provided.dragHandleProps}
                                    sx={{
                                      mr: 1,
                                      display: 'flex',
                                      alignItems: 'center',
                                      cursor: 'grab',
                                    }}
                                  >
                                    <DragIndicatorIcon fontSize="small" color="action" />
                                  </Box>
                                  <ListItemText
                                    primary={item.symbol}
                                    secondary={item.company_name}
                                    primaryTypographyProps={{ fontWeight: 500 }}
                                  />
                                  <ListItemSecondaryAction>
                                    <IconButton
                                      size="small"
                                      onClick={() => handleRemoveItem(item.id)}
                                      sx={{ opacity: 0.5, '&:hover': { opacity: 1 } }}
                                    >
                                      <DeleteIcon fontSize="small" />
                                    </IconButton>
                                  </ListItemSecondaryAction>
                                </ListItem>
                              )}
                            </Draggable>
                          ))}
                          {provided.placeholder}
                        </List>
                      )}
                    </Droppable>
                  </DragDropContext>
                ) : (
                  <Typography variant="body2" color="text.secondary">
                    No stocks in this watchlist yet.
                  </Typography>
                )}
              </>
            ) : (
              <Box
                display="flex"
                justifyContent="center"
                alignItems="center"
                height="100%"
                minHeight={300}
              >
                <Typography color="text.secondary" textAlign="center">
                  {watchlists.length > 0
                    ? 'Select a watchlist to edit'
                    : 'Create a watchlist to get started'}
                </Typography>
              </Box>
            )}
          </Box>
        </Box>
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose}>Close</Button>
      </DialogActions>

      <Dialog open={importDialogOpen} onClose={() => setImportDialogOpen(false)} maxWidth="sm" fullWidth>
        <DialogTitle>Import Symbols</DialogTitle>
        <DialogContent>
          <Typography variant="body2" color="text.secondary" sx={{ mb: 1.5 }}>
            Paste newline, comma, tab, or CSV-formatted symbols. The importer will dedupe entries and report invalid symbols.
          </Typography>
          <TextField
            autoFocus
            multiline
            minRows={8}
            fullWidth
            placeholder={'NVDA\nMSFT\nAAPL'}
            value={importContent}
            onChange={(event) => setImportContent(event.target.value)}
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setImportDialogOpen(false)}>Cancel</Button>
          <Button
            variant="contained"
            onClick={() => importItemsMutation.mutate({
              watchlistId: selectedWatchlistId,
              payload: { content: importContent, format: 'auto' },
            })}
            disabled={!selectedWatchlistId || !importContent.trim() || importItemsMutation.isPending}
          >
            Import
          </Button>
        </DialogActions>
      </Dialog>
    </Dialog>
  );
}

export default UserWatchlistManager;

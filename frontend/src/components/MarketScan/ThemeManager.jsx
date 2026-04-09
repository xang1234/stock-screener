/**
 * Theme Manager Modal
 *
 * Allows users to:
 * - Create/edit/delete themes
 * - Create/edit/delete subgroups within themes
 * - Add/remove stocks from subgroups
 * - Reorder themes, subgroups, and stocks via drag-and-drop
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
  Chip,
  Alert,
  Accordion,
  AccordionSummary,
  AccordionDetails,
} from '@mui/material';
import DeleteIcon from '@mui/icons-material/Delete';
import AddIcon from '@mui/icons-material/Add';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import DragIndicatorIcon from '@mui/icons-material/DragIndicator';
import EditIcon from '@mui/icons-material/Edit';
import CheckIcon from '@mui/icons-material/Check';
import CloseIcon from '@mui/icons-material/Close';
import CircularProgress from '@mui/material/CircularProgress';
import { DragDropContext, Droppable, Draggable } from '@hello-pangea/dnd';
import {
  getThemes,
  createTheme,
  updateTheme,
  deleteTheme,
  getThemeData,
  createSubgroup,
  updateSubgroup,
  deleteSubgroup,
  addStock,
  removeStock,
  reorderThemes,
  reorderSubgroups,
  reorderStocks,
} from '../../api/userThemes';

function ThemeManager({ open, onClose, onUpdate }) {
  const queryClient = useQueryClient();
  const [selectedThemeId, setSelectedThemeId] = useState(null);
  const [newThemeName, setNewThemeName] = useState('');
  const [newSubgroupName, setNewSubgroupName] = useState('');
  const [newStockSymbol, setNewStockSymbol] = useState('');
  const [expandedSubgroup, setExpandedSubgroup] = useState(null);
  const [error, setError] = useState('');

  // Theme rename state
  const [editingThemeId, setEditingThemeId] = useState(null);
  const [editingThemeName, setEditingThemeName] = useState('');

  // Subgroup rename state
  const [editingSubgroupId, setEditingSubgroupId] = useState(null);
  const [editingSubgroupName, setEditingSubgroupName] = useState('');

  // Fetch themes
  const { data: themesData, isLoading: themesLoading } = useQuery({
    queryKey: ['userThemes'],
    queryFn: getThemes,
    enabled: open,
  });

  const themes = themesData?.themes || [];

  // Fetch theme data for editing
  const { data: themeData } = useQuery({
    queryKey: ['userThemeData', selectedThemeId],
    queryFn: () => getThemeData(selectedThemeId),
    enabled: !!selectedThemeId && open,
  });

  // Mutations
  const createThemeMutation = useMutation({
    mutationFn: (data) => createTheme(data),
    onSuccess: (newTheme) => {
      queryClient.invalidateQueries({ queryKey: ['userThemes'] });
      setNewThemeName('');
      setError('');
      setSelectedThemeId(newTheme.id);
      onUpdate?.();
    },
    onError: (err) => setError(err.response?.data?.detail || 'Failed to create theme'),
  });

  const deleteThemeMutation = useMutation({
    mutationFn: (id) => deleteTheme(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['userThemes'] });
      if (selectedThemeId === arguments[0]) {
        setSelectedThemeId(null);
      }
      onUpdate?.();
    },
    onError: (err) => setError(err.response?.data?.detail || 'Failed to delete theme'),
  });

  const createSubgroupMutation = useMutation({
    mutationFn: ({ themeId, data }) => createSubgroup(themeId, data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['userThemeData', selectedThemeId] });
      setNewSubgroupName('');
      onUpdate?.();
    },
    onError: (err) => setError(err.response?.data?.detail || 'Failed to create subgroup'),
  });

  const deleteSubgroupMutation = useMutation({
    mutationFn: (id) => deleteSubgroup(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['userThemeData', selectedThemeId] });
      onUpdate?.();
    },
    onError: (err) => setError(err.response?.data?.detail || 'Failed to delete subgroup'),
  });

  const addStockMutation = useMutation({
    mutationFn: ({ subgroupId, data }) => addStock(subgroupId, data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['userThemeData', selectedThemeId] });
      setNewStockSymbol('');
      onUpdate?.();
    },
    onError: (err) => setError(err.response?.data?.detail || 'Failed to add stock'),
  });

  const removeStockMutation = useMutation({
    mutationFn: (id) => removeStock(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['userThemeData', selectedThemeId] });
      onUpdate?.();
    },
    onError: (err) => setError(err.response?.data?.detail || 'Failed to remove stock'),
  });

  // Reorder mutations
  const reorderThemesMutation = useMutation({
    mutationFn: (themeIds) => reorderThemes(themeIds),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['userThemes'] });
      onUpdate?.();
    },
  });

  const reorderSubgroupsMutation = useMutation({
    mutationFn: ({ themeId, subgroupIds }) => reorderSubgroups(themeId, subgroupIds),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['userThemeData', selectedThemeId] });
      onUpdate?.();
    },
  });

  const reorderStocksMutation = useMutation({
    mutationFn: ({ subgroupId, stockIds }) => reorderStocks(subgroupId, stockIds),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['userThemeData', selectedThemeId] });
      onUpdate?.();
    },
  });

  // Update mutations for renaming
  const updateThemeMutation = useMutation({
    mutationFn: ({ themeId, name }) => updateTheme(themeId, { name }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['userThemes'] });
      setEditingThemeId(null);
      setEditingThemeName('');
      onUpdate?.();
    },
    onError: (err) => setError(err.response?.data?.detail || 'Failed to rename theme'),
  });

  const updateSubgroupMutation = useMutation({
    mutationFn: ({ subgroupId, name }) => updateSubgroup(subgroupId, { name }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['userThemeData', selectedThemeId] });
      setEditingSubgroupId(null);
      setEditingSubgroupName('');
      onUpdate?.();
    },
    onError: (err) => setError(err.response?.data?.detail || 'Failed to rename subgroup'),
  });

  const handleCreateTheme = () => {
    if (newThemeName.trim()) {
      createThemeMutation.mutate({ name: newThemeName.trim() });
    }
  };

  const handleDeleteTheme = (themeId) => {
    if (window.confirm('Delete this theme and all its subgroups and stocks?')) {
      deleteThemeMutation.mutate(themeId);
      if (selectedThemeId === themeId) {
        setSelectedThemeId(null);
      }
    }
  };

  const handleCreateSubgroup = () => {
    if (newSubgroupName.trim() && selectedThemeId) {
      createSubgroupMutation.mutate({
        themeId: selectedThemeId,
        data: { name: newSubgroupName.trim() },
      });
    }
  };

  const handleDeleteSubgroup = (subgroupId, e) => {
    e.stopPropagation();
    if (window.confirm('Delete this subgroup and all its stocks?')) {
      deleteSubgroupMutation.mutate(subgroupId);
    }
  };

  const handleAddStock = (subgroupId) => {
    if (newStockSymbol.trim()) {
      addStockMutation.mutate({
        subgroupId,
        data: { symbol: newStockSymbol.trim().toUpperCase() },
      });
    }
  };

  const handleRemoveStock = (stockId) => {
    removeStockMutation.mutate(stockId);
  };

  // Theme rename handlers
  const handleStartThemeRename = (theme, e) => {
    e.stopPropagation();
    setEditingThemeId(theme.id);
    setEditingThemeName(theme.name);
  };

  const handleSaveThemeRename = () => {
    const trimmedName = editingThemeName.trim();
    if (!trimmedName) {
      setError('Theme name cannot be empty');
      return;
    }
    // Check for duplicate names (case-insensitive)
    const duplicate = themes.find(
      (t) => t.id !== editingThemeId && t.name.toLowerCase() === trimmedName.toLowerCase()
    );
    if (duplicate) {
      setError('A theme with this name already exists');
      return;
    }
    updateThemeMutation.mutate({ themeId: editingThemeId, name: trimmedName });
  };

  const handleCancelThemeRename = () => {
    setEditingThemeId(null);
    setEditingThemeName('');
  };

  // Subgroup rename handlers
  const handleStartSubgroupRename = (subgroup, e) => {
    e.stopPropagation();
    setEditingSubgroupId(subgroup.id);
    setEditingSubgroupName(subgroup.name);
  };

  const handleSaveSubgroupRename = () => {
    const trimmedName = editingSubgroupName.trim();
    if (!trimmedName) {
      setError('Subgroup name cannot be empty');
      return;
    }
    // Check for duplicate names within the same theme (case-insensitive)
    const duplicate = themeData?.subgroups?.find(
      (sg) => sg.id !== editingSubgroupId && sg.name.toLowerCase() === trimmedName.toLowerCase()
    );
    if (duplicate) {
      setError('A subgroup with this name already exists in this theme');
      return;
    }
    updateSubgroupMutation.mutate({ subgroupId: editingSubgroupId, name: trimmedName });
  };

  const handleCancelSubgroupRename = () => {
    setEditingSubgroupId(null);
    setEditingSubgroupName('');
  };

  // Keyboard handler for rename fields
  const handleRenameKeyDown = (e, saveHandler, cancelHandler) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      saveHandler();
    } else if (e.key === 'Escape') {
      e.preventDefault();
      cancelHandler();
    }
  };

  // Drag-drop handlers
  const handleThemeDragEnd = (result) => {
    if (!result.destination || result.destination.index === result.source.index) return;

    const items = Array.from(themes);
    const [reorderedItem] = items.splice(result.source.index, 1);
    items.splice(result.destination.index, 0, reorderedItem);

    // Optimistically update the cache
    queryClient.setQueryData(['userThemes'], {
      ...themesData,
      themes: items,
    });

    // Send new order to backend
    const newOrder = items.map((item) => item.id);
    reorderThemesMutation.mutate(newOrder);
  };

  const handleSubgroupDragEnd = (result) => {
    if (!result.destination || result.destination.index === result.source.index) return;
    if (!themeData?.subgroups) return;

    const items = Array.from(themeData.subgroups);
    const [reorderedItem] = items.splice(result.source.index, 1);
    items.splice(result.destination.index, 0, reorderedItem);

    // Optimistically update the cache
    queryClient.setQueryData(['userThemeData', selectedThemeId], {
      ...themeData,
      subgroups: items,
    });

    // Send new order to backend
    const newOrder = items.map((item) => item.id);
    reorderSubgroupsMutation.mutate({ themeId: selectedThemeId, subgroupIds: newOrder });
  };

  const handleStockDragEnd = (subgroupId) => (result) => {
    if (!result.destination || result.destination.index === result.source.index) return;
    if (!themeData?.subgroups) return;

    const subgroupIndex = themeData.subgroups.findIndex((sg) => sg.id === subgroupId);
    if (subgroupIndex === -1) return;

    const subgroup = themeData.subgroups[subgroupIndex];
    const items = Array.from(subgroup.stocks);
    const [reorderedItem] = items.splice(result.source.index, 1);
    items.splice(result.destination.index, 0, reorderedItem);

    // Optimistically update the cache
    const updatedSubgroups = [...themeData.subgroups];
    updatedSubgroups[subgroupIndex] = { ...subgroup, stocks: items };
    queryClient.setQueryData(['userThemeData', selectedThemeId], {
      ...themeData,
      subgroups: updatedSubgroups,
    });

    // Send new order to backend
    const newOrder = items.map((item) => item.id);
    reorderStocksMutation.mutate({ subgroupId, stockIds: newOrder });
  };

  const selectedTheme = themes.find((t) => t.id === selectedThemeId);

  return (
    <Dialog open={open} onClose={onClose} maxWidth="md" fullWidth>
      <DialogTitle>Manage Themes</DialogTitle>
      <DialogContent sx={{ minHeight: 450 }}>
        {error && (
          <Alert severity="error" sx={{ mb: 2 }} onClose={() => setError('')}>
            {error}
          </Alert>
        )}

        <Box sx={{ display: 'flex', gap: 2, height: '100%' }}>
          {/* Left Panel: Theme List */}
          <Box
            sx={{
              width: 200,
              borderRight: 1,
              borderColor: 'divider',
              pr: 2,
              flexShrink: 0,
            }}
          >
            <Typography variant="subtitle2" gutterBottom>
              Themes
            </Typography>

            {/* Create new theme */}
            <Box display="flex" gap={0.5} mb={2}>
              <TextField
                size="small"
                placeholder="New theme"
                value={newThemeName}
                onChange={(e) => setNewThemeName(e.target.value)}
                onKeyPress={(e) => e.key === 'Enter' && handleCreateTheme()}
                fullWidth
                sx={{ '& .MuiInputBase-input': { fontSize: '0.875rem' } }}
              />
              <IconButton
                color="primary"
                onClick={handleCreateTheme}
                disabled={!newThemeName.trim() || createThemeMutation.isPending}
                size="small"
              >
                <AddIcon />
              </IconButton>
            </Box>

            {/* Theme list with drag-drop */}
            <DragDropContext onDragEnd={handleThemeDragEnd}>
              <Droppable droppableId="themes">
                {(provided) => (
                  <List
                    dense
                    sx={{ mx: -1 }}
                    ref={provided.innerRef}
                    {...provided.droppableProps}
                  >
                    {themes.map((theme, index) => (
                      <Draggable
                        key={theme.id}
                        draggableId={`theme-${theme.id}`}
                        index={index}
                      >
                        {(provided, snapshot) => (
                          <ListItemButton
                            ref={provided.innerRef}
                            {...provided.draggableProps}
                            selected={theme.id === selectedThemeId}
                            onClick={() => setSelectedThemeId(theme.id)}
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
                            {editingThemeId === theme.id ? (
                              <>
                                <TextField
                                  size="small"
                                  value={editingThemeName}
                                  onChange={(e) => setEditingThemeName(e.target.value)}
                                  onKeyDown={(e) =>
                                    handleRenameKeyDown(e, handleSaveThemeRename, handleCancelThemeRename)
                                  }
                                  onClick={(e) => e.stopPropagation()}
                                  autoFocus
                                  sx={{
                                    flex: 1,
                                    '& .MuiInputBase-input': { fontSize: '0.875rem', py: 0.5 },
                                  }}
                                />
                                <IconButton
                                  size="small"
                                  onClick={(e) => {
                                    e.stopPropagation();
                                    handleSaveThemeRename();
                                  }}
                                  disabled={updateThemeMutation.isPending}
                                  color="primary"
                                >
                                  {updateThemeMutation.isPending ? (
                                    <CircularProgress size={16} />
                                  ) : (
                                    <CheckIcon fontSize="small" />
                                  )}
                                </IconButton>
                                <IconButton
                                  size="small"
                                  onClick={(e) => {
                                    e.stopPropagation();
                                    handleCancelThemeRename();
                                  }}
                                  disabled={updateThemeMutation.isPending}
                                >
                                  <CloseIcon fontSize="small" />
                                </IconButton>
                              </>
                            ) : (
                              <>
                                <ListItemText
                                  primary={theme.name}
                                  primaryTypographyProps={{ fontSize: '0.875rem' }}
                                />
                                <IconButton
                                  size="small"
                                  onClick={(e) => handleStartThemeRename(theme, e)}
                                  sx={{ opacity: 0.5, '&:hover': { opacity: 1 } }}
                                >
                                  <EditIcon fontSize="small" />
                                </IconButton>
                                <IconButton
                                  size="small"
                                  onClick={(e) => {
                                    e.stopPropagation();
                                    handleDeleteTheme(theme.id);
                                  }}
                                  sx={{ opacity: 0.5, '&:hover': { opacity: 1 } }}
                                >
                                  <DeleteIcon fontSize="small" />
                                </IconButton>
                              </>
                            )}
                          </ListItemButton>
                        )}
                      </Draggable>
                    ))}
                    {provided.placeholder}
                    {themes.length === 0 && !themesLoading && (
                      <Typography variant="caption" color="text.secondary" sx={{ px: 1 }}>
                        No themes yet
                      </Typography>
                    )}
                  </List>
                )}
              </Droppable>
            </DragDropContext>
          </Box>

          {/* Right Panel: Theme Details */}
          <Box sx={{ flex: 1, overflow: 'auto' }}>
            {selectedTheme ? (
              <>
                <Typography variant="h6" gutterBottom>
                  {selectedTheme.name}
                </Typography>

                {/* Create new subgroup */}
                <Box display="flex" gap={1} mb={2}>
                  <TextField
                    size="small"
                    placeholder="New subgroup name"
                    value={newSubgroupName}
                    onChange={(e) => setNewSubgroupName(e.target.value)}
                    onKeyPress={(e) => e.key === 'Enter' && handleCreateSubgroup()}
                    sx={{ flex: 1 }}
                  />
                  <Button
                    variant="outlined"
                    onClick={handleCreateSubgroup}
                    disabled={!newSubgroupName.trim() || createSubgroupMutation.isPending}
                    startIcon={<AddIcon />}
                    size="small"
                  >
                    Add Subgroup
                  </Button>
                </Box>

                {/* Subgroups with drag-drop */}
                <DragDropContext onDragEnd={handleSubgroupDragEnd}>
                  <Droppable droppableId="subgroups">
                    {(provided) => (
                      <Box ref={provided.innerRef} {...provided.droppableProps}>
                        {themeData?.subgroups?.map((subgroup, index) => (
                          <Draggable
                            key={subgroup.id}
                            draggableId={`subgroup-${subgroup.id}`}
                            index={index}
                          >
                            {(provided, snapshot) => (
                              <Accordion
                                ref={provided.innerRef}
                                {...provided.draggableProps}
                                expanded={expandedSubgroup === subgroup.id}
                                onChange={() =>
                                  setExpandedSubgroup(
                                    expandedSubgroup === subgroup.id ? null : subgroup.id
                                  )
                                }
                                sx={{
                                  mb: 1,
                                  bgcolor: snapshot.isDragging ? 'action.hover' : undefined,
                                }}
                              >
                                <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                                  <Box display="flex" alignItems="center" gap={1} flex={1}>
                                    <Box
                                      {...provided.dragHandleProps}
                                      sx={{ display: 'flex', alignItems: 'center', cursor: 'grab' }}
                                      onClick={(e) => e.stopPropagation()}
                                    >
                                      <DragIndicatorIcon fontSize="small" color="action" />
                                    </Box>
                                    {editingSubgroupId === subgroup.id ? (
                                      <>
                                        <TextField
                                          size="small"
                                          value={editingSubgroupName}
                                          onChange={(e) => setEditingSubgroupName(e.target.value)}
                                          onKeyDown={(e) =>
                                            handleRenameKeyDown(
                                              e,
                                              handleSaveSubgroupRename,
                                              handleCancelSubgroupRename
                                            )
                                          }
                                          onClick={(e) => e.stopPropagation()}
                                          autoFocus
                                          sx={{
                                            flex: 1,
                                            '& .MuiInputBase-input': { fontSize: '0.875rem' },
                                          }}
                                        />
                                        <IconButton
                                          size="small"
                                          onClick={(e) => {
                                            e.stopPropagation();
                                            handleSaveSubgroupRename();
                                          }}
                                          disabled={updateSubgroupMutation.isPending}
                                          color="primary"
                                        >
                                          {updateSubgroupMutation.isPending ? (
                                            <CircularProgress size={16} />
                                          ) : (
                                            <CheckIcon fontSize="small" />
                                          )}
                                        </IconButton>
                                        <IconButton
                                          size="small"
                                          onClick={(e) => {
                                            e.stopPropagation();
                                            handleCancelSubgroupRename();
                                          }}
                                          disabled={updateSubgroupMutation.isPending}
                                        >
                                          <CloseIcon fontSize="small" />
                                        </IconButton>
                                      </>
                                    ) : (
                                      <>
                                        <Typography variant="subtitle2">{subgroup.name}</Typography>
                                        <Chip size="small" label={`${subgroup.stocks.length} stocks`} />
                                        <Box flex={1} />
                                        <IconButton
                                          size="small"
                                          onClick={(e) => handleStartSubgroupRename(subgroup, e)}
                                          sx={{ opacity: 0.5, '&:hover': { opacity: 1 } }}
                                        >
                                          <EditIcon fontSize="small" />
                                        </IconButton>
                                        <IconButton
                                          size="small"
                                          onClick={(e) => handleDeleteSubgroup(subgroup.id, e)}
                                          sx={{ opacity: 0.5, '&:hover': { opacity: 1 } }}
                                        >
                                          <DeleteIcon fontSize="small" />
                                        </IconButton>
                                      </>
                                    )}
                                  </Box>
                                </AccordionSummary>
                                <AccordionDetails>
                                  {/* Add stock */}
                                  <Box display="flex" gap={1} mb={2}>
                                    <TextField
                                      size="small"
                                      placeholder="Add symbol (e.g., NVDA)"
                                      value={expandedSubgroup === subgroup.id ? newStockSymbol : ''}
                                      onChange={(e) => setNewStockSymbol(e.target.value)}
                                      onKeyPress={(e) =>
                                        e.key === 'Enter' && handleAddStock(subgroup.id)
                                      }
                                      sx={{ width: 200 }}
                                    />
                                    <Button
                                      size="small"
                                      variant="contained"
                                      onClick={() => handleAddStock(subgroup.id)}
                                      disabled={!newStockSymbol.trim() || addStockMutation.isPending}
                                    >
                                      Add
                                    </Button>
                                  </Box>

                                  {/* Stock list with drag-drop */}
                                  {subgroup.stocks.length > 0 ? (
                                    <DragDropContext onDragEnd={handleStockDragEnd(subgroup.id)}>
                                      <Droppable droppableId={`stocks-${subgroup.id}`}>
                                        {(stocksProvided) => (
                                          <List
                                            dense
                                            sx={{ mx: -2 }}
                                            ref={stocksProvided.innerRef}
                                            {...stocksProvided.droppableProps}
                                          >
                                            {subgroup.stocks.map((stock, stockIndex) => (
                                              <Draggable
                                                key={stock.id}
                                                draggableId={`stock-${stock.id}`}
                                                index={stockIndex}
                                              >
                                                {(stockProvided, stockSnapshot) => (
                                                  <ListItem
                                                    ref={stockProvided.innerRef}
                                                    {...stockProvided.draggableProps}
                                                    sx={{
                                                      bgcolor: stockSnapshot.isDragging
                                                        ? 'action.hover'
                                                        : undefined,
                                                      borderRadius: 1,
                                                    }}
                                                  >
                                                    <Box
                                                      {...stockProvided.dragHandleProps}
                                                      sx={{
                                                        mr: 1,
                                                        display: 'flex',
                                                        alignItems: 'center',
                                                        cursor: 'grab',
                                                      }}
                                                    >
                                                      <DragIndicatorIcon
                                                        fontSize="small"
                                                        color="action"
                                                      />
                                                    </Box>
                                                    <ListItemText
                                                      primary={stock.symbol}
                                                      secondary={stock.display_name}
                                                      primaryTypographyProps={{ fontWeight: 500 }}
                                                    />
                                                    <ListItemSecondaryAction>
                                                      <IconButton
                                                        size="small"
                                                        onClick={() => handleRemoveStock(stock.id)}
                                                        sx={{
                                                          opacity: 0.5,
                                                          '&:hover': { opacity: 1 },
                                                        }}
                                                      >
                                                        <DeleteIcon fontSize="small" />
                                                      </IconButton>
                                                    </ListItemSecondaryAction>
                                                  </ListItem>
                                                )}
                                              </Draggable>
                                            ))}
                                            {stocksProvided.placeholder}
                                          </List>
                                        )}
                                      </Droppable>
                                    </DragDropContext>
                                  ) : (
                                    <Typography variant="caption" color="text.secondary">
                                      No stocks in this subgroup
                                    </Typography>
                                  )}
                                </AccordionDetails>
                              </Accordion>
                            )}
                          </Draggable>
                        ))}
                        {provided.placeholder}
                      </Box>
                    )}
                  </Droppable>
                </DragDropContext>

                {(!themeData?.subgroups || themeData.subgroups.length === 0) && (
                  <Typography variant="body2" color="text.secondary" sx={{ mt: 2 }}>
                    No subgroups yet. Add a subgroup to start organizing stocks.
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
                  {themes.length > 0
                    ? 'Select a theme to edit'
                    : 'Create a theme to get started'}
                </Typography>
              </Box>
            )}
          </Box>
        </Box>
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose}>Close</Button>
      </DialogActions>
    </Dialog>
  );
}

export default ThemeManager;

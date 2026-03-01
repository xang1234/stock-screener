import { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Typography,
  Box,
  CircularProgress,
  IconButton,
  Tooltip,
  TextField,
  Select,
  MenuItem,
  Switch,
  Chip,
  Alert,
  Checkbox,
  FormControlLabel,
  FormGroup,
} from '@mui/material';
import CloseIcon from '@mui/icons-material/Close';
import EditIcon from '@mui/icons-material/Edit';
import SaveIcon from '@mui/icons-material/Save';
import CancelIcon from '@mui/icons-material/Cancel';
import AddIcon from '@mui/icons-material/Add';
import DeleteIcon from '@mui/icons-material/Delete';
import {
  getContentSources,
  addContentSource,
  updateContentSource,
  deleteContentSource,
  getTwitterSessionStatus,
  createTwitterSessionChallenge,
} from '../../api/themes';
import apiClient from '../../api/client';
import { captureXCookies, isBridgeAvailable } from '../../services/xuiBridge';

const SOURCE_TYPES = [
  { value: 'substack', label: 'Substack' },
  { value: 'twitter', label: 'Twitter' },
  { value: 'news', label: 'News' },
  { value: 'reddit', label: 'Reddit' },
];

const PIPELINES = [
  { value: 'technical', label: 'Technical' },
  { value: 'fundamental', label: 'Fundamental' },
];

const EMPTY_SOURCE = {
  name: '',
  source_type: 'substack',
  url: '',
  priority: 50,
  fetch_interval_minutes: 60,
  is_active: true,
  pipelines: ['technical', 'fundamental'],
};

function ManageSourcesModal({ open, onClose }) {
  const queryClient = useQueryClient();
  const [editingId, setEditingId] = useState(null);
  const [editFormData, setEditFormData] = useState({});
  const [isAddingNew, setIsAddingNew] = useState(false);
  const [newSourceData, setNewSourceData] = useState(EMPTY_SOURCE);
  const [error, setError] = useState(null);
  const [sessionError, setSessionError] = useState(null);
  const [isConnectingSession, setIsConnectingSession] = useState(false);

  const { data: sources, isLoading } = useQuery({
    queryKey: ['contentSources'],
    queryFn: () => getContentSources(false),
    enabled: open,
  });
  const {
    data: twitterSessionStatus,
    isLoading: isLoadingSessionStatus,
    isFetching: isFetchingSessionStatus,
    refetch: refetchSessionStatus,
  } = useQuery({
    queryKey: ['twitterSessionStatus'],
    queryFn: getTwitterSessionStatus,
    enabled: open,
    retry: false,
  });

  const addMutation = useMutation({
    mutationFn: addContentSource,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['contentSources'] });
      setIsAddingNew(false);
      setNewSourceData(EMPTY_SOURCE);
      setError(null);
    },
    onError: (err) => {
      setError(err.response?.data?.detail || 'Failed to add source');
    },
  });

  const updateMutation = useMutation({
    mutationFn: ({ id, data }) => updateContentSource(id, data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['contentSources'] });
      setEditingId(null);
      setEditFormData({});
      setError(null);
    },
    onError: (err) => {
      setError(err.response?.data?.detail || 'Failed to update source');
    },
  });

  const deleteMutation = useMutation({
    mutationFn: deleteContentSource,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['contentSources'] });
      queryClient.refetchQueries({ queryKey: ['contentSources'] });
      setError(null);
    },
    onError: (err) => {
      setError(err.response?.data?.detail || 'Failed to deactivate source');
    },
  });

  const handleStartEdit = (source) => {
    setEditingId(source.id);
    setEditFormData({
      name: source.name,
      source_type: source.source_type,
      url: source.url,
      priority: source.priority,
      fetch_interval_minutes: source.fetch_interval_minutes,
      is_active: source.is_active,
      pipelines: source.pipelines || ['technical', 'fundamental'],
    });
  };

  // Pipeline checkbox handler for editing
  const handleEditPipelineToggle = (pipeline) => {
    const currentPipelines = editFormData.pipelines || [];
    if (currentPipelines.includes(pipeline)) {
      // Don't allow removing all pipelines
      if (currentPipelines.length <= 1) return;
      setEditFormData({
        ...editFormData,
        pipelines: currentPipelines.filter((p) => p !== pipeline),
      });
    } else {
      setEditFormData({
        ...editFormData,
        pipelines: [...currentPipelines, pipeline],
      });
    }
  };

  // Pipeline checkbox handler for new source
  const handleNewPipelineToggle = (pipeline) => {
    const currentPipelines = newSourceData.pipelines || [];
    if (currentPipelines.includes(pipeline)) {
      // Don't allow removing all pipelines
      if (currentPipelines.length <= 1) return;
      setNewSourceData({
        ...newSourceData,
        pipelines: currentPipelines.filter((p) => p !== pipeline),
      });
    } else {
      setNewSourceData({
        ...newSourceData,
        pipelines: [...currentPipelines, pipeline],
      });
    }
  };

  const handleCancelEdit = () => {
    setEditingId(null);
    setEditFormData({});
  };

  const handleSaveEdit = () => {
    updateMutation.mutate({ id: editingId, data: editFormData });
  };

  const handleAddNew = () => {
    addMutation.mutate(newSourceData);
  };

  const handleCancelAdd = () => {
    setIsAddingNew(false);
    setNewSourceData(EMPTY_SOURCE);
  };

  const handleDelete = (sourceId) => {
    if (window.confirm('Are you sure you want to deactivate this source?')) {
      deleteMutation.mutate(sourceId);
    }
  };

  const handleConnectFromBrowser = async () => {
    setSessionError(null);
    setIsConnectingSession(true);
    try {
      const bridgeReady = await isBridgeAvailable();
      if (!bridgeReady) {
        throw new Error(
          'XUI bridge extension was not detected. Install/load the extension and refresh this page.',
        );
      }

      const challenge = await createTwitterSessionChallenge();
      const rawBaseUrl = String(apiClient.defaults.baseURL || '').trim();
      const normalizedBaseUrl = rawBaseUrl.startsWith('http')
        ? rawBaseUrl
        : `${window.location.origin}${rawBaseUrl.startsWith('/') ? '' : '/'}${rawBaseUrl}`;
      const baseUrl = normalizedBaseUrl.replace(/\/+$/, '');
      const importUrl = `${baseUrl}/v1/themes/twitter/session/import`;
      await captureXCookies({
        challengeId: challenge.challenge_id,
        challengeToken: challenge.challenge_token,
        importUrl,
        appOrigin: window.location.origin,
      });
      await queryClient.invalidateQueries({ queryKey: ['twitterSessionStatus'] });
      await refetchSessionStatus();
    } catch (err) {
      setSessionError(err?.response?.data?.detail || err?.message || 'Failed to import browser session.');
    } finally {
      setIsConnectingSession(false);
    }
  };

  const formatDate = (dateString) => {
    if (!dateString) return 'Never';
    return new Date(dateString).toLocaleString();
  };

  const sessionStatusColor = twitterSessionStatus?.authenticated
    ? 'success'
    : twitterSessionStatus?.status_code?.startsWith('blocked_')
      ? 'warning'
      : 'default';
  const sessionStatusLabel = isLoadingSessionStatus || isFetchingSessionStatus
    ? 'Checking...'
    : twitterSessionStatus?.authenticated
      ? 'Authenticated'
      : twitterSessionStatus?.status_code || 'Unavailable';

  return (
    <Dialog open={open} onClose={onClose} maxWidth="xl" fullWidth>
      <DialogTitle>
        <Box display="flex" justifyContent="space-between" alignItems="center">
          <Typography variant="h6">Manage Content Sources</Typography>
          <IconButton onClick={onClose} size="small">
            <CloseIcon />
          </IconButton>
        </Box>
      </DialogTitle>

      <DialogContent dividers>
        {error && (
          <Alert severity="error" sx={{ mb: 2 }} onClose={() => setError(null)}>
            {error}
          </Alert>
        )}

        <Paper variant="outlined" sx={{ p: 2, mb: 2 }}>
          <Box display="flex" justifyContent="space-between" alignItems="center" flexWrap="wrap" gap={1}>
            <Box>
              <Typography variant="subtitle2">Twitter/X Session</Typography>
              <Typography variant="caption" color="text.secondary">
                {twitterSessionStatus?.message || 'Connect the current browser session used for x.com.'}
              </Typography>
            </Box>
            <Chip label={sessionStatusLabel} color={sessionStatusColor} size="small" />
          </Box>
          <Box display="flex" gap={1} flexWrap="wrap" sx={{ mt: 1.5 }}>
            <Button
              variant="contained"
              size="small"
              onClick={handleConnectFromBrowser}
              disabled={isConnectingSession || isLoadingSessionStatus}
            >
              {isConnectingSession ? 'Connecting...' : 'Connect From Current Browser'}
            </Button>
            <Button
              variant="outlined"
              size="small"
              onClick={() => refetchSessionStatus()}
              disabled={isLoadingSessionStatus || isFetchingSessionStatus}
            >
              Check Status
            </Button>
            <Button
              variant="text"
              size="small"
              onClick={() =>
                window.open(
                  'https://developer.chrome.com/docs/extensions/get-started/tutorial/hello-world#load-unpacked',
                  '_blank',
                  'noopener,noreferrer',
                )
              }
            >
              Install Extension
            </Button>
          </Box>
          <Typography variant="caption" color="text.secondary" display="block" sx={{ mt: 1 }}>
            Load unpacked extension from: <code>browser-extension/xui-session-bridge</code>
          </Typography>
          {sessionError && (
            <Alert severity="error" sx={{ mt: 1 }} onClose={() => setSessionError(null)}>
              {sessionError}
            </Alert>
          )}
        </Paper>

        {isLoading ? (
          <Box display="flex" justifyContent="center" p={4}>
            <CircularProgress />
          </Box>
        ) : (
          <>
            {!isAddingNew && (
              <Box mb={2}>
                <Button
                  variant="contained"
                  startIcon={<AddIcon />}
                  onClick={() => setIsAddingNew(true)}
                >
                  Add New Source
                </Button>
              </Box>
            )}

            <TableContainer component={Paper} variant="outlined">
              <Table size="small" stickyHeader>
                <TableHead>
                  <TableRow>
                    <TableCell sx={{ minWidth: 100 }}>Actions</TableCell>
                    <TableCell sx={{ minWidth: 150 }}>Name</TableCell>
                    <TableCell sx={{ minWidth: 100 }}>Type</TableCell>
                    <TableCell sx={{ minWidth: 200 }}>URL</TableCell>
                    <TableCell sx={{ minWidth: 150 }}>Pipelines</TableCell>
                    <TableCell align="center" sx={{ minWidth: 80 }}>Priority</TableCell>
                    <TableCell align="center" sx={{ minWidth: 100 }}>Interval (min)</TableCell>
                    <TableCell align="center" sx={{ minWidth: 80 }}>Active</TableCell>
                    <TableCell sx={{ minWidth: 160 }}>Last Fetched</TableCell>
                    <TableCell align="right" sx={{ minWidth: 80 }}>Items</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {isAddingNew && (
                    <TableRow sx={{ backgroundColor: 'action.hover' }}>
                      <TableCell>
                        <Box display="flex" gap={0.5}>
                          <IconButton
                            size="small"
                            color="primary"
                            onClick={handleAddNew}
                            disabled={addMutation.isPending}
                          >
                            <SaveIcon fontSize="small" />
                          </IconButton>
                          <IconButton size="small" onClick={handleCancelAdd}>
                            <CancelIcon fontSize="small" />
                          </IconButton>
                        </Box>
                      </TableCell>
                      <TableCell>
                        <TextField
                          size="small"
                          value={newSourceData.name}
                          onChange={(e) => setNewSourceData({ ...newSourceData, name: e.target.value })}
                          placeholder="Source name"
                          fullWidth
                        />
                      </TableCell>
                      <TableCell>
                        <Select
                          size="small"
                          value={newSourceData.source_type}
                          onChange={(e) => setNewSourceData({ ...newSourceData, source_type: e.target.value })}
                          fullWidth
                        >
                          {SOURCE_TYPES.map((t) => (
                            <MenuItem key={t.value} value={t.value}>{t.label}</MenuItem>
                          ))}
                        </Select>
                      </TableCell>
                      <TableCell>
                        <TextField
                          size="small"
                          value={newSourceData.url}
                          onChange={(e) => setNewSourceData({ ...newSourceData, url: e.target.value })}
                          placeholder="URL"
                          fullWidth
                        />
                      </TableCell>
                      <TableCell>
                        <FormGroup row>
                          {PIPELINES.map((p) => (
                            <FormControlLabel
                              key={p.value}
                              control={
                                <Checkbox
                                  size="small"
                                  checked={newSourceData.pipelines?.includes(p.value)}
                                  onChange={() => handleNewPipelineToggle(p.value)}
                                />
                              }
                              label={<Typography variant="caption">{p.label}</Typography>}
                            />
                          ))}
                        </FormGroup>
                      </TableCell>
                      <TableCell>
                        <TextField
                          size="small"
                          type="number"
                          value={newSourceData.priority}
                          onChange={(e) => setNewSourceData({ ...newSourceData, priority: parseInt(e.target.value) || 50 })}
                          inputProps={{ min: 1, max: 100, style: { width: 60, textAlign: 'center' } }}
                        />
                      </TableCell>
                      <TableCell>
                        <TextField
                          size="small"
                          type="number"
                          value={newSourceData.fetch_interval_minutes}
                          onChange={(e) => setNewSourceData({ ...newSourceData, fetch_interval_minutes: parseInt(e.target.value) || 60 })}
                          inputProps={{ min: 1, style: { width: 60, textAlign: 'center' } }}
                        />
                      </TableCell>
                      <TableCell align="center">
                        <Switch
                          checked={newSourceData.is_active}
                          onChange={(e) => setNewSourceData({ ...newSourceData, is_active: e.target.checked })}
                          size="small"
                        />
                      </TableCell>
                      <TableCell>-</TableCell>
                      <TableCell align="right">-</TableCell>
                    </TableRow>
                  )}

                  {sources?.map((source) => (
                    <TableRow
                      key={source.id}
                      sx={{
                        opacity: source.is_active ? 1 : 0.6,
                        backgroundColor: editingId === source.id ? 'action.selected' : 'inherit',
                      }}
                    >
                      <TableCell>
                        {editingId === source.id ? (
                          <Box display="flex" gap={0.5}>
                            <IconButton
                              size="small"
                              color="primary"
                              onClick={handleSaveEdit}
                              disabled={updateMutation.isPending}
                            >
                              <SaveIcon fontSize="small" />
                            </IconButton>
                            <IconButton size="small" onClick={handleCancelEdit}>
                              <CancelIcon fontSize="small" />
                            </IconButton>
                          </Box>
                        ) : (
                          <Box display="flex" gap={0.5}>
                            <Tooltip title="Edit">
                              <IconButton size="small" onClick={() => handleStartEdit(source)}>
                                <EditIcon fontSize="small" />
                              </IconButton>
                            </Tooltip>
                            {source.is_active && (
                              <Tooltip title="Deactivate">
                                <IconButton
                                  size="small"
                                  color="error"
                                  onClick={() => handleDelete(source.id)}
                                >
                                  <DeleteIcon fontSize="small" />
                                </IconButton>
                              </Tooltip>
                            )}
                          </Box>
                        )}
                      </TableCell>

                      <TableCell>
                        {editingId === source.id ? (
                          <TextField
                            size="small"
                            value={editFormData.name}
                            onChange={(e) => setEditFormData({ ...editFormData, name: e.target.value })}
                            fullWidth
                          />
                        ) : (
                          <Typography variant="body2" fontWeight="medium">
                            {source.name}
                          </Typography>
                        )}
                      </TableCell>

                      <TableCell>
                        {editingId === source.id ? (
                          <Select
                            size="small"
                            value={editFormData.source_type}
                            onChange={(e) => setEditFormData({ ...editFormData, source_type: e.target.value })}
                            fullWidth
                          >
                            {SOURCE_TYPES.map((t) => (
                              <MenuItem key={t.value} value={t.value}>{t.label}</MenuItem>
                            ))}
                          </Select>
                        ) : (
                          <Chip
                            label={source.source_type}
                            size="small"
                            variant="outlined"
                          />
                        )}
                      </TableCell>

                      <TableCell>
                        {editingId === source.id ? (
                          <TextField
                            size="small"
                            value={editFormData.url}
                            onChange={(e) => setEditFormData({ ...editFormData, url: e.target.value })}
                            fullWidth
                          />
                        ) : (
                          <Typography
                            variant="body2"
                            sx={{
                              maxWidth: 200,
                              overflow: 'hidden',
                              textOverflow: 'ellipsis',
                              whiteSpace: 'nowrap',
                            }}
                            title={source.url}
                          >
                            {source.url}
                          </Typography>
                        )}
                      </TableCell>

                      <TableCell>
                        {editingId === source.id ? (
                          <FormGroup row>
                            {PIPELINES.map((p) => (
                              <FormControlLabel
                                key={p.value}
                                control={
                                  <Checkbox
                                    size="small"
                                    checked={editFormData.pipelines?.includes(p.value)}
                                    onChange={() => handleEditPipelineToggle(p.value)}
                                  />
                                }
                                label={<Typography variant="caption">{p.label}</Typography>}
                              />
                            ))}
                          </FormGroup>
                        ) : (
                          <Box display="flex" gap={0.5}>
                            {(source.pipelines || ['technical', 'fundamental']).map((p) => (
                              <Chip
                                key={p}
                                label={p === 'technical' ? 'Tech' : 'Fund'}
                                size="small"
                                color={p === 'technical' ? 'primary' : 'secondary'}
                                variant="outlined"
                                sx={{ fontSize: '10px', height: 20 }}
                              />
                            ))}
                          </Box>
                        )}
                      </TableCell>

                      <TableCell align="center">
                        {editingId === source.id ? (
                          <TextField
                            size="small"
                            type="number"
                            value={editFormData.priority}
                            onChange={(e) => setEditFormData({ ...editFormData, priority: parseInt(e.target.value) || 50 })}
                            inputProps={{ min: 1, max: 100, style: { width: 60, textAlign: 'center' } }}
                          />
                        ) : (
                          source.priority
                        )}
                      </TableCell>

                      <TableCell align="center">
                        {editingId === source.id ? (
                          <TextField
                            size="small"
                            type="number"
                            value={editFormData.fetch_interval_minutes}
                            onChange={(e) => setEditFormData({ ...editFormData, fetch_interval_minutes: parseInt(e.target.value) || 60 })}
                            inputProps={{ min: 1, style: { width: 60, textAlign: 'center' } }}
                          />
                        ) : (
                          `${source.fetch_interval_minutes}`
                        )}
                      </TableCell>

                      <TableCell align="center">
                        {editingId === source.id ? (
                          <Switch
                            checked={editFormData.is_active}
                            onChange={(e) => setEditFormData({ ...editFormData, is_active: e.target.checked })}
                            size="small"
                          />
                        ) : (
                          <Chip
                            label={source.is_active ? 'Active' : 'Inactive'}
                            size="small"
                            color={source.is_active ? 'success' : 'default'}
                          />
                        )}
                      </TableCell>

                      <TableCell>
                        <Typography variant="caption" color="text.secondary">
                          {formatDate(source.last_fetched_at)}
                        </Typography>
                      </TableCell>

                      <TableCell align="right">
                        {source.total_items_fetched}
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          </>
        )}
      </DialogContent>

      <DialogActions>
        <Button onClick={onClose}>Close</Button>
      </DialogActions>
    </Dialog>
  );
}

export default ManageSourcesModal;

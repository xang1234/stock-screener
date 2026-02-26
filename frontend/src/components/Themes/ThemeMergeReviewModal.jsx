import { useState, useEffect } from 'react';
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
  Tabs,
  Tab,
  Chip,
  Alert,
  Collapse,
  Tooltip,
  Checkbox,
} from '@mui/material';
import CloseIcon from '@mui/icons-material/Close';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import BlockIcon from '@mui/icons-material/Block';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import ExpandLessIcon from '@mui/icons-material/ExpandLess';
import CompareArrowsIcon from '@mui/icons-material/CompareArrows';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import {
  getMergeSuggestions,
  approveMergeSuggestion,
  rejectMergeSuggestion,
  getMergeHistory,
  runThemeConsolidation,
} from '../../api/themes';

// Color helpers based on score thresholds
const getSimilarityColor = (score) => {
  if (score >= 95) return 'success';
  if (score >= 85) return 'warning';
  return 'default';
};

const getConfidenceColor = (score) => {
  if (score >= 90) return 'success';
  if (score >= 70) return 'warning';
  return 'error';
};

const getSimilarityTooltip = (score) => {
  if (!Number.isFinite(score)) return 'Similarity unavailable.';
  if (score >= 95) return 'Very high lexical/embedding overlap. Candidate duplicate.';
  if (score >= 85) return 'High overlap. Requires confidence + relationship validation.';
  return 'Lower overlap. Merge only with strong supporting reasoning.';
};

const getConfidenceTooltip = (score) => {
  if (!Number.isFinite(score)) return 'Model confidence unavailable.';
  if (score >= 90) return 'High confidence. Safe to fast-track when relationship is identical.';
  if (score >= 70) return 'Medium confidence. Review reasoning and aliases before action.';
  return 'Low confidence. Prefer reject or defer for additional evidence.';
};

const getRelationshipColor = (relationship) => {
  switch (relationship) {
    case 'identical':
      return 'success';
    case 'subset':
      return 'warning';
    case 'related':
      return 'info';
    case 'distinct':
      return 'error';
    default:
      return 'default';
  }
};

const getRelationshipTooltip = (relationship) => {
  switch (relationship) {
    case 'identical':
      return 'Themes are essentially the same - recommended to merge';
    case 'subset':
      return 'One theme is a subset of the other - review carefully';
    case 'related':
      return 'Themes are related but distinct - probably should not merge';
    case 'distinct':
      return 'Themes are clearly different - do not merge';
    default:
      return '';
  }
};

// Expandable row component for showing LLM reasoning
function SuggestionRow({ suggestion, onReject, isRejecting, checked, onToggle, disabled }) {
  const [expanded, setExpanded] = useState(false);

  const hasSimilarity = Number.isFinite(suggestion.similarity_score);
  const hasConfidence = Number.isFinite(suggestion.llm_confidence);
  const similarityPercent = hasSimilarity ? (suggestion.similarity_score * 100).toFixed(0) : null;
  const confidencePercent = hasConfidence ? (suggestion.llm_confidence * 100).toFixed(0) : null;
  const relationshipType = suggestion.relationship_type || 'unknown';

  return (
    <>
      <TableRow
        hover
        sx={{ cursor: 'pointer', '& > *': { borderBottom: expanded ? 0 : undefined } }}
        onClick={() => setExpanded(!expanded)}
      >
        <TableCell sx={{ width: 40 }} onClick={(e) => e.stopPropagation()}>
          <Checkbox
            checked={checked}
            onChange={() => onToggle(suggestion.id)}
            disabled={disabled}
            size="small"
          />
        </TableCell>
        <TableCell sx={{ width: 40 }}>
          <IconButton size="small">
            {expanded ? <ExpandLessIcon /> : <ExpandMoreIcon />}
          </IconButton>
        </TableCell>
        <TableCell>
          <Box>
            <Typography variant="body2" fontWeight="medium">
              {suggestion.source_theme_name}
            </Typography>
            {suggestion.source_aliases?.length > 0 && (
              <Typography variant="caption" color="text.secondary" sx={{ display: 'block' }}>
                {suggestion.source_aliases.slice(0, 2).join(', ')}
                {suggestion.source_aliases.length > 2 && ` +${suggestion.source_aliases.length - 2}`}
              </Typography>
            )}
          </Box>
        </TableCell>
        <TableCell>
          <Box>
            <Typography variant="body2" fontWeight="medium">
              {suggestion.target_theme_name}
            </Typography>
            {suggestion.target_aliases?.length > 0 && (
              <Typography variant="caption" color="text.secondary" sx={{ display: 'block' }}>
                {suggestion.target_aliases.slice(0, 2).join(', ')}
                {suggestion.target_aliases.length > 2 && ` +${suggestion.target_aliases.length - 2}`}
              </Typography>
            )}
          </Box>
        </TableCell>
        <TableCell align="center">
          <Tooltip title={getSimilarityTooltip(parseFloat(similarityPercent))}>
            <Chip
              label={hasSimilarity ? `${similarityPercent}%` : 'N/A'}
              size="small"
              color={hasSimilarity ? getSimilarityColor(parseFloat(similarityPercent)) : 'default'}
              variant="filled"
              sx={{ minWidth: 55, fontFamily: 'monospace', fontWeight: 600 }}
            />
          </Tooltip>
        </TableCell>
        <TableCell align="center">
          <Tooltip title={getConfidenceTooltip(parseFloat(confidencePercent))}>
            <Chip
              label={hasConfidence ? `${confidencePercent}%` : 'N/A'}
              size="small"
              color={hasConfidence ? getConfidenceColor(parseFloat(confidencePercent)) : 'default'}
              variant="filled"
              sx={{ minWidth: 55, fontFamily: 'monospace', fontWeight: 600 }}
            />
          </Tooltip>
        </TableCell>
        <TableCell align="center">
          <Tooltip title={getRelationshipTooltip(relationshipType)}>
            <Chip
              label={relationshipType}
              size="small"
              color={getRelationshipColor(relationshipType)}
              variant="outlined"
              sx={{ textTransform: 'capitalize' }}
            />
          </Tooltip>
        </TableCell>
        <TableCell align="right" onClick={(e) => e.stopPropagation()}>
          <Tooltip title="Reject merge">
            <span>
              <IconButton
                size="small"
                color="error"
                onClick={() => onReject(suggestion.id)}
                disabled={isRejecting || disabled}
              >
                {isRejecting ? <CircularProgress size={18} /> : <BlockIcon />}
              </IconButton>
            </span>
          </Tooltip>
        </TableCell>
      </TableRow>
      <TableRow>
        <TableCell colSpan={8} sx={{ py: 0 }}>
          <Collapse in={expanded} timeout="auto" unmountOnExit>
            <Box sx={{ py: 2, px: 2, bgcolor: 'action.hover', borderRadius: 1, my: 1 }}>
              <Box display="flex" gap={4} mb={2}>
                <Box>
                  <Typography variant="caption" color="text.secondary">
                    Relationship
                  </Typography>
                  <Typography variant="body2" fontWeight="medium" sx={{ textTransform: 'capitalize' }}>
                    {relationshipType}
                  </Typography>
                </Box>
                <Box>
                  <Typography variant="caption" color="text.secondary">
                    Suggested Name
                  </Typography>
                  <Typography variant="body2" fontWeight="medium">
                    {suggestion.suggested_name || suggestion.target_theme_name}
                  </Typography>
                </Box>
                <Box>
                  <Typography variant="caption" color="text.secondary">
                    Created
                  </Typography>
                  <Typography variant="body2">
                    {new Date(suggestion.created_at).toLocaleDateString()}
                  </Typography>
                </Box>
              </Box>
              <Box>
                <Typography variant="caption" color="text.secondary">
                  LLM Reasoning
                </Typography>
                <Paper variant="outlined" sx={{ p: 1.5, mt: 0.5, bgcolor: 'background.paper' }}>
                  <Typography variant="body2" sx={{ whiteSpace: 'pre-wrap' }}>
                    {suggestion.reasoning || 'No reasoning provided'}
                  </Typography>
                </Paper>
              </Box>
              <Box display="flex" gap={1} mt={2} justifyContent="flex-end">
                <Button
                  variant="outlined"
                  color="error"
                  size="small"
                  startIcon={isRejecting ? <CircularProgress size={16} /> : <BlockIcon />}
                  onClick={() => onReject(suggestion.id)}
                  disabled={isRejecting || disabled}
                >
                  Reject
                </Button>
              </Box>
            </Box>
          </Collapse>
        </TableCell>
      </TableRow>
    </>
  );
}

// History row component
function HistoryRow({ entry }) {
  return (
    <TableRow hover>
      <TableCell>
        <Typography variant="body2">
          {entry.source_name}
        </Typography>
      </TableCell>
      <TableCell>
        <Typography variant="body2">
          {entry.target_name}
        </Typography>
      </TableCell>
      <TableCell>
        <Chip
          label={entry.merge_type}
          size="small"
          color={entry.merge_type === 'merged' ? 'success' : 'default'}
          variant="outlined"
          sx={{ textTransform: 'capitalize' }}
        />
      </TableCell>
      <TableCell>
        <Typography variant="body2" color="text.secondary">
          {new Date(entry.merged_at || entry.created_at).toLocaleString()}
        </Typography>
      </TableCell>
    </TableRow>
  );
}

function ThemeMergeReviewModal({ open, onClose }) {
  const queryClient = useQueryClient();
  const [activeTab, setActiveTab] = useState(0);
  const [error, setError] = useState(null);
  const [successMessage, setSuccessMessage] = useState(null);
  const [actioningId, setActioningId] = useState(null);
  const [actionType, setActionType] = useState(null);
  const [isConsolidating, setIsConsolidating] = useState(false);
  const [selectedIds, setSelectedIds] = useState(new Set());
  const [isApprovingBatch, setIsApprovingBatch] = useState(false);
  const [batchProgress, setBatchProgress] = useState({ current: 0, total: 0 });

  // Consolidation mutation
  const consolidationMutation = useMutation({
    mutationFn: () => runThemeConsolidation(false),
    onMutate: () => {
      setIsConsolidating(true);
      setError(null);
      setSuccessMessage(null);
    },
    onSuccess: (data) => {
      setSuccessMessage(`Consolidation started (Task ID: ${data.task_id}). New suggestions will appear shortly.`);
      // Refetch suggestions after a delay to allow task to complete
      setTimeout(() => {
        queryClient.invalidateQueries({ queryKey: ['mergeSuggestions'] });
      }, 5000);
    },
    onError: (err) => {
      setError(err.response?.data?.detail || 'Failed to start consolidation');
    },
    onSettled: () => {
      setIsConsolidating(false);
    },
  });

  // Fetch pending suggestions
  const {
    data: pendingSuggestions,
    isLoading: isLoadingPending,
    refetch: refetchPending,
  } = useQuery({
    queryKey: ['mergeSuggestions', 'pending'],
    queryFn: () => getMergeSuggestions('pending', 100),
    enabled: open && activeTab === 0,
  });

  // Fetch merge history
  const {
    data: history,
    isLoading: isLoadingHistory,
  } = useQuery({
    queryKey: ['mergeHistory'],
    queryFn: () => getMergeHistory(50),
    enabled: open && activeTab === 1,
  });

  // Auto-select all pending suggestions when they load
  const suggestionsList = pendingSuggestions?.suggestions || pendingSuggestions || [];
  useEffect(() => {
    if (suggestionsList.length > 0) {
      setSelectedIds(new Set(suggestionsList.map((s) => s.id)));
    }
  }, [suggestionsList.length]);

  // Toggle individual selection
  const handleToggleSelection = (id) => {
    setSelectedIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) {
        next.delete(id);
      } else {
        next.add(id);
      }
      return next;
    });
  };

  // Toggle all selections
  const handleToggleAll = () => {
    if (selectedIds.size === suggestionsList.length) {
      setSelectedIds(new Set());
    } else {
      setSelectedIds(new Set(suggestionsList.map((s) => s.id)));
    }
  };

  // Batch approve selected suggestions
  const handleBatchApprove = async () => {
    const idsToApprove = Array.from(selectedIds);
    if (idsToApprove.length === 0) return;

    setIsApprovingBatch(true);
    setBatchProgress({ current: 0, total: idsToApprove.length });
    setError(null);

    let successCount = 0;
    let errorCount = 0;

    for (let i = 0; i < idsToApprove.length; i++) {
      const id = idsToApprove[i];
      setBatchProgress({ current: i + 1, total: idsToApprove.length });

      try {
        const result = await approveMergeSuggestion(id);
        if (result.success) {
          successCount++;
          // Only remove from selected on actual success
          setSelectedIds((prev) => {
            const next = new Set(prev);
            next.delete(id);
            return next;
          });
        } else {
          errorCount++;
          console.error(`Failed to approve suggestion ${id}:`, result.error);
        }
      } catch (err) {
        errorCount++;
        console.error(`Network error approving suggestion ${id}:`, err);
      }
    }

    setIsApprovingBatch(false);
    setBatchProgress({ current: 0, total: 0 });

    // Refresh data
    queryClient.invalidateQueries({ queryKey: ['mergeSuggestions'] });
    queryClient.invalidateQueries({ queryKey: ['mergeHistory'] });
    queryClient.refetchQueries({ queryKey: ['themeRankings'] });
    queryClient.refetchQueries({ queryKey: ['emergingThemes'] });

    if (errorCount > 0) {
      setError(`Approved ${successCount} merges, but ${errorCount} failed`);
    } else if (successCount > 0) {
      setSuccessMessage(`Successfully approved ${successCount} merge${successCount > 1 ? 's' : ''}`);
    }
  };

  // Reject mutation
  const rejectMutation = useMutation({
    mutationFn: rejectMergeSuggestion,
    onMutate: (id) => {
      setActioningId(id);
      setActionType('reject');
    },
    onSuccess: (result) => {
      if (!result?.success) {
        setError(result?.error || 'Failed to reject suggestion');
        return;
      }
      queryClient.invalidateQueries({ queryKey: ['mergeSuggestions'] });
      queryClient.invalidateQueries({ queryKey: ['mergeHistory'] });
      setError(null);
    },
    onError: (err) => {
      setError(err.response?.data?.detail || err.message || 'Failed to reject suggestion');
    },
    onSettled: () => {
      setActioningId(null);
      setActionType(null);
    },
  });

  const handleReject = (suggestionId) => {
    rejectMutation.mutate(suggestionId);
  };

  const pendingCount = suggestionsList.length;
  const selectedCount = selectedIds.size;

  return (
    <Dialog open={open} onClose={onClose} maxWidth="lg" fullWidth>
      <DialogTitle>
        <Box display="flex" justifyContent="space-between" alignItems="center">
          <Box display="flex" alignItems="center" gap={1}>
            <CompareArrowsIcon color="primary" />
            <Typography variant="h6">Review Theme Merge Suggestions</Typography>
          </Box>
          <Box display="flex" alignItems="center" gap={1}>
            <Tooltip title="Run theme consolidation to find new duplicate themes">
              <span>
                <Button
                  variant="outlined"
                  size="small"
                  startIcon={isConsolidating ? <CircularProgress size={16} /> : <PlayArrowIcon />}
                  onClick={() => consolidationMutation.mutate()}
                  disabled={isConsolidating}
                >
                  {isConsolidating ? 'Running...' : 'Run Consolidation'}
                </Button>
              </span>
            </Tooltip>
            <IconButton onClick={onClose} size="small">
              <CloseIcon />
            </IconButton>
          </Box>
        </Box>
      </DialogTitle>

      <DialogContent dividers sx={{ p: 0 }}>
        {error && (
          <Alert severity="error" sx={{ m: 2, mb: 0 }} onClose={() => setError(null)}>
            {error}
          </Alert>
        )}
        {successMessage && (
          <Alert severity="success" sx={{ m: 2, mb: 0 }} onClose={() => setSuccessMessage(null)}>
            {successMessage}
          </Alert>
        )}
        <Alert severity="info" sx={{ m: 2, mb: 0 }}>
          Decision guide: prioritize merges when <strong>Relationship=identical</strong> and <strong>Confidence is high</strong>.
          For <strong>subset/related</strong>, verify intent before approving.
        </Alert>

        <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
          <Tabs value={activeTab} onChange={(e, v) => setActiveTab(v)}>
            <Tab
              label={
                <Box display="flex" alignItems="center" gap={1}>
                  Pending Review
                  {pendingCount > 0 && (
                    <Chip label={pendingCount} size="small" color="warning" />
                  )}
                </Box>
              }
            />
            <Tab label="History" />
          </Tabs>
        </Box>

        {/* Pending Tab */}
        {activeTab === 0 && (
          <Box sx={{ minHeight: 400 }}>
            {isLoadingPending ? (
              <Box display="flex" justifyContent="center" alignItems="center" p={4}>
                <CircularProgress />
              </Box>
            ) : pendingCount === 0 ? (
              <Box display="flex" flexDirection="column" alignItems="center" justifyContent="center" p={4}>
                <CheckCircleIcon sx={{ fontSize: 48, color: 'success.main', mb: 2 }} />
                <Typography variant="h6" color="text.secondary">
                  No pending merge suggestions
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  All theme merge suggestions have been reviewed
                </Typography>
              </Box>
            ) : (
              <TableContainer sx={{ maxHeight: 500 }}>
                <Table size="small" stickyHeader>
                  <TableHead>
                    <TableRow>
                      <TableCell sx={{ width: 40 }}>
                        <Checkbox
                          checked={selectedIds.size === suggestionsList.length && suggestionsList.length > 0}
                          indeterminate={selectedIds.size > 0 && selectedIds.size < suggestionsList.length}
                          onChange={handleToggleAll}
                          disabled={isApprovingBatch}
                          size="small"
                        />
                      </TableCell>
                      <TableCell sx={{ width: 40 }} />
                      <TableCell>Source Theme</TableCell>
                      <TableCell>Target Theme</TableCell>
                      <TableCell align="center" sx={{ width: 80 }}>Similarity</TableCell>
                      <TableCell align="center" sx={{ width: 80 }}>Confidence</TableCell>
                      <TableCell align="center" sx={{ width: 100 }}>Relationship</TableCell>
                      <TableCell align="right" sx={{ width: 80 }}>Reject</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {suggestionsList.map((suggestion) => (
                      <SuggestionRow
                        key={suggestion.id}
                        suggestion={suggestion}
                        onReject={handleReject}
                        isRejecting={actioningId === suggestion.id && actionType === 'reject'}
                        checked={selectedIds.has(suggestion.id)}
                        onToggle={handleToggleSelection}
                        disabled={isApprovingBatch}
                      />
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>
            )}
          </Box>
        )}

        {/* History Tab */}
        {activeTab === 1 && (
          <Box sx={{ minHeight: 400 }}>
            {isLoadingHistory ? (
              <Box display="flex" justifyContent="center" alignItems="center" p={4}>
                <CircularProgress />
              </Box>
            ) : (history?.history?.length || history?.length || 0) === 0 ? (
              <Box display="flex" flexDirection="column" alignItems="center" justifyContent="center" p={4}>
                <Typography variant="h6" color="text.secondary">
                  No merge history yet
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Approved merges will appear here
                </Typography>
              </Box>
            ) : (
              <TableContainer sx={{ maxHeight: 500 }}>
                <Table size="small" stickyHeader>
                  <TableHead>
                    <TableRow>
                      <TableCell>Source Theme</TableCell>
                      <TableCell>Target Theme</TableCell>
                      <TableCell sx={{ width: 100 }}>Action</TableCell>
                      <TableCell sx={{ width: 180 }}>Date</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {(history?.history || history || []).map((entry, index) => (
                      <HistoryRow key={entry.id || index} entry={entry} />
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>
            )}
          </Box>
        )}
      </DialogContent>

      <DialogActions sx={{ justifyContent: 'space-between' }}>
        <Box>
          {activeTab === 0 && pendingCount > 0 && (
            <Button
              variant="contained"
              color="success"
              onClick={handleBatchApprove}
              disabled={selectedCount === 0 || isApprovingBatch}
              startIcon={isApprovingBatch ? <CircularProgress size={16} color="inherit" /> : <CheckCircleIcon />}
            >
              {isApprovingBatch
                ? `Approving ${batchProgress.current}/${batchProgress.total}...`
                : `Approve Selected (${selectedCount})`}
            </Button>
          )}
        </Box>
        <Button onClick={onClose} disabled={isApprovingBatch}>Close</Button>
      </DialogActions>
    </Dialog>
  );
}

export default ThemeMergeReviewModal;

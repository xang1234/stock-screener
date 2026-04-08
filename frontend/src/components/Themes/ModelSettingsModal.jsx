import { useState, useEffect } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  Typography,
  Box,
  CircularProgress,
  IconButton,
  TextField,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Chip,
  Alert,
  Divider,
  Card,
  CardContent,
  Tooltip,
} from '@mui/material';
import CloseIcon from '@mui/icons-material/Close';
import CloudIcon from '@mui/icons-material/Cloud';
import ComputerIcon from '@mui/icons-material/Computer';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import ErrorIcon from '@mui/icons-material/Error';
import RefreshIcon from '@mui/icons-material/Refresh';
import SettingsIcon from '@mui/icons-material/Settings';
import {
  getLLMConfig,
  updateLLMModel,
  updateOllamaSettings,
  getOllamaModels,
} from '../../api/themes';

const STATUS_COLORS = {
  connected: 'success',
  disconnected: 'error',
  timeout: 'warning',
  error: 'error',
  not_configured: 'default',
};

const STATUS_LABELS = {
  connected: 'Connected',
  disconnected: 'Disconnected',
  timeout: 'Timeout',
  error: 'Error',
  not_configured: 'Not Configured',
};

function ModelSettingsModal({ open, onClose }) {
  const queryClient = useQueryClient();
  const [selectedExtractionModel, setSelectedExtractionModel] = useState('');
  const [ollamaApiBase, setOllamaApiBase] = useState('');
  const [error, setError] = useState(null);
  const [successMessage, setSuccessMessage] = useState(null);

  // Fetch current config
  const { data: config, isLoading, refetch: refetchConfig } = useQuery({
    queryKey: ['llmConfig'],
    queryFn: getLLMConfig,
    enabled: open,
  });

  // Fetch Ollama models (only when connected)
  const { data: ollamaModelsData, refetch: refetchOllamaModels } = useQuery({
    queryKey: ['ollamaModels'],
    queryFn: getOllamaModels,
    enabled: open && config?.ollama_status === 'connected',
  });

  // Update local state when config loads
  useEffect(() => {
    if (config) {
      setSelectedExtractionModel(config.extraction?.current_model || '');
      setOllamaApiBase(config.ollama_api_base || 'http://localhost:11434');
    }
  }, [config]);

  // Mutation for updating model
  const updateModelMutation = useMutation({
    mutationFn: ({ modelId, useCase }) => updateLLMModel(modelId, useCase),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['llmConfig'] });
      setSuccessMessage('Model updated successfully');
      setError(null);
      setTimeout(() => setSuccessMessage(null), 3000);
    },
    onError: (err) => {
      setError(err.response?.data?.detail || 'Failed to update model');
      setSuccessMessage(null);
    },
  });

  // Mutation for updating Ollama settings
  const updateOllamaMutation = useMutation({
    mutationFn: (apiBase) => updateOllamaSettings(apiBase),
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: ['llmConfig'] });
      queryClient.invalidateQueries({ queryKey: ['ollamaModels'] });
      if (data.ollama_status === 'connected') {
        setSuccessMessage('Ollama connected successfully');
      } else {
        setError(`Ollama status: ${data.ollama_status}`);
      }
    },
    onError: (err) => {
      setError(err.response?.data?.detail || 'Failed to update Ollama settings');
    },
  });

  const handleModelChange = (modelId) => {
    setSelectedExtractionModel(modelId);
    updateModelMutation.mutate({ modelId, useCase: 'extraction' });
  };

  const handleOllamaUpdate = () => {
    updateOllamaMutation.mutate(ollamaApiBase);
  };

  const handleRefresh = () => {
    refetchConfig();
    if (config?.ollama_status === 'connected') {
      refetchOllamaModels();
    }
  };

  // Group models by provider/category
  const cloudModels = config?.available_models?.filter(m => m.category === 'cloud') || [];
  const localModels = config?.available_models?.filter(m => m.category === 'local') || [];
  const hasLocalModels = localModels.length > 0;

  // Get current model info
  const currentModelInfo = config?.extraction?.model_info;
  const isLocalModel = currentModelInfo?.category === 'local';

  return (
    <Dialog open={open} onClose={onClose} maxWidth="sm" fullWidth>
      <DialogTitle>
        <Box display="flex" justifyContent="space-between" alignItems="center">
          <Box display="flex" alignItems="center" gap={1}>
            <SettingsIcon color="primary" />
            <Typography variant="h6">LLM Settings</Typography>
          </Box>
          <Box display="flex" alignItems="center" gap={1}>
            <Tooltip title="Refresh status">
              <IconButton onClick={handleRefresh} size="small">
                <RefreshIcon />
              </IconButton>
            </Tooltip>
            <IconButton onClick={onClose} size="small">
              <CloseIcon />
            </IconButton>
          </Box>
        </Box>
      </DialogTitle>

      <DialogContent dividers>
        {error && (
          <Alert severity="error" sx={{ mb: 2 }} onClose={() => setError(null)}>
            {error}
          </Alert>
        )}

        {successMessage && (
          <Alert severity="success" sx={{ mb: 2 }} onClose={() => setSuccessMessage(null)}>
            {successMessage}
          </Alert>
        )}

        {isLoading ? (
          <Box display="flex" justifyContent="center" p={4}>
            <CircularProgress />
          </Box>
        ) : (
          <>
            {/* Current Model Status */}
            <Card variant="outlined" sx={{ mb: 3 }}>
              <CardContent>
                <Typography variant="subtitle2" color="text.secondary" gutterBottom>
                  Current Extraction Model
                </Typography>
                <Box display="flex" alignItems="center" gap={1}>
                  {isLocalModel ? (
                    <ComputerIcon color="primary" />
                  ) : (
                    <CloudIcon color="info" />
                  )}
                  <Typography variant="h6">
                    {currentModelInfo?.name || selectedExtractionModel}
                  </Typography>
                  <Chip
                    label={isLocalModel ? 'Local' : 'Cloud'}
                    size="small"
                    color={isLocalModel ? 'primary' : 'info'}
                    variant="outlined"
                  />
                </Box>
                <Typography variant="caption" color="text.secondary">
                  Provider: {currentModelInfo?.provider || 'unknown'}
                </Typography>
              </CardContent>
            </Card>

            {/* Model Selection */}
            <FormControl fullWidth sx={{ mb: 3 }}>
              <InputLabel>Extraction Model</InputLabel>
              <Select
                value={selectedExtractionModel}
                label="Extraction Model"
                onChange={(e) => handleModelChange(e.target.value)}
                disabled={updateModelMutation.isPending}
              >
                {/* Cloud Models */}
                <MenuItem disabled>
                  <Box display="flex" alignItems="center" gap={1}>
                    <CloudIcon fontSize="small" color="info" />
                    <Typography variant="caption" fontWeight="bold">CLOUD MODELS</Typography>
                  </Box>
                </MenuItem>
                {cloudModels.map((model) => (
                  <MenuItem key={model.id} value={model.id}>
                    <Box display="flex" alignItems="center" justifyContent="space-between" width="100%">
                      <Typography>{model.name}</Typography>
                      <Chip label={model.provider} size="small" variant="outlined" />
                    </Box>
                  </MenuItem>
                ))}

                {hasLocalModels && (
                  <MenuItem disabled>
                    <Box display="flex" alignItems="center" gap={1}>
                      <ComputerIcon fontSize="small" color="primary" />
                      <Typography variant="caption" fontWeight="bold">LOCAL MODELS (OLLAMA)</Typography>
                    </Box>
                  </MenuItem>
                )}
                {localModels.map((model) => (
                  <MenuItem key={model.id} value={model.id}>
                    <Box display="flex" alignItems="center" justifyContent="space-between" width="100%">
                      <Typography>{model.name}</Typography>
                      <Chip label="local" size="small" color="primary" variant="outlined" />
                    </Box>
                  </MenuItem>
                ))}
              </Select>
            </FormControl>

            {hasLocalModels && (
              <>
                <Divider sx={{ my: 3 }} />

                {/* Ollama Configuration */}
                <Typography variant="subtitle1" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                  <ComputerIcon color="primary" />
                  Ollama Configuration
                </Typography>

                <Box display="flex" alignItems="center" gap={1} mb={2}>
                  <Typography variant="body2" color="text.secondary">Status:</Typography>
                  <Chip
                    icon={config?.ollama_status === 'connected' ? <CheckCircleIcon /> : <ErrorIcon />}
                    label={STATUS_LABELS[config?.ollama_status] || 'Unknown'}
                    size="small"
                    color={STATUS_COLORS[config?.ollama_status] || 'default'}
                  />
                </Box>

                <Box display="flex" gap={1} mb={2}>
                  <TextField
                    fullWidth
                    size="small"
                    label="Ollama API URL"
                    value={ollamaApiBase}
                    onChange={(e) => setOllamaApiBase(e.target.value)}
                    placeholder="http://localhost:11434"
                    helperText="Default: http://localhost:11434"
                  />
                  <Button
                    variant="outlined"
                    onClick={handleOllamaUpdate}
                    disabled={updateOllamaMutation.isPending}
                  >
                    {updateOllamaMutation.isPending ? <CircularProgress size={20} /> : 'Test'}
                  </Button>
                </Box>

                {/* Installed Ollama Models */}
                {config?.ollama_status === 'connected' && ollamaModelsData?.models?.length > 0 && (
                  <Box>
                    <Typography variant="body2" color="text.secondary" gutterBottom>
                      Installed Models ({ollamaModelsData.models.length})
                    </Typography>
                    <Box display="flex" flexWrap="wrap" gap={0.5}>
                      {ollamaModelsData.models.map((model) => (
                        <Chip
                          key={model.id}
                          label={model.name}
                          size="small"
                          variant="outlined"
                          onClick={() => handleModelChange(model.id)}
                          sx={{ cursor: 'pointer' }}
                        />
                      ))}
                    </Box>
                  </Box>
                )}

                {config?.ollama_status === 'disconnected' && (
                  <Alert severity="info" sx={{ mt: 2 }}>
                    <Typography variant="body2">
                      To use local models, install and run Ollama:
                    </Typography>
                    <Typography variant="caption" component="div" sx={{ mt: 1, fontFamily: 'monospace' }}>
                      1. Install: curl -fsSL https://ollama.com/install.sh | sh<br />
                      2. Pull a model: ollama pull llama3.1:8b<br />
                      3. Start server: ollama serve
                    </Typography>
                  </Alert>
                )}
              </>
            )}
          </>
        )}
      </DialogContent>

      <DialogActions>
        <Button onClick={onClose}>Close</Button>
      </DialogActions>
    </Dialog>
  );
}

export default ModelSettingsModal;

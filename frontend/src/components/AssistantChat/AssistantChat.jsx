import { useEffect, useMemo, useState } from 'react';
import {
  Alert,
  Box,
  Button,
  Chip,
  Divider,
  IconButton,
  Paper,
  Stack,
  TextField,
  Tooltip,
  Typography,
} from '@mui/material';
import AddIcon from '@mui/icons-material/Add';
import CloseIcon from '@mui/icons-material/Close';
import PlaylistAddIcon from '@mui/icons-material/PlaylistAdd';
import SendIcon from '@mui/icons-material/Send';

import { useAssistantChat } from '../../contexts/AssistantChatContext';
import AssistantMessageList from './AssistantMessageList';
import AssistantWatchlistDialog from './AssistantWatchlistDialog';

const STOPWORDS = new Set([
  'THE', 'AND', 'FOR', 'WITH', 'THIS', 'THAT', 'FROM', 'YOUR', 'THEME', 'THEMES',
  'BREADTH', 'WATCH', 'WATCHLIST', 'BUY', 'SELL', 'HOLD', 'RISK', 'RISKS', 'JSON',
  'HTTP', 'MARKET', 'STAGE', 'SCORE', 'SCORES', 'USES', 'RS', 'EPS', 'AI',
]);

function extractTickerCandidates(content) {
  const matches = content?.match(/\b[A-Z]{2,5}\b/g) || [];
  return [...new Set(matches.filter((symbol) => !STOPWORDS.has(symbol)))].slice(0, 12);
}

function AssistantChat({ mode = 'page', onClose = null }) {
  const {
    assistantHealth,
    conversationTitle,
    displayedMessages,
    ensureConversation,
    isLoadingConversation,
    isStreaming,
    sendMessage,
    startNewConversation,
  } = useAssistantChat();
  const [input, setInput] = useState('');
  const [watchlistDialogOpen, setWatchlistDialogOpen] = useState(false);

  useEffect(() => {
    ensureConversation().catch(() => {});
  }, [ensureConversation]);

  const latestAssistantSymbols = useMemo(() => {
    const latestAssistantMessage = [...displayedMessages]
      .reverse()
      .find((message) => message.role === 'assistant' && !message.isStreaming && !message.isError);
    return extractTickerCandidates(latestAssistantMessage?.content || '');
  }, [displayedMessages]);

  const handleSubmit = async () => {
    if (!input.trim() || isStreaming) return;
    const content = input;
    setInput('');
    await sendMessage(content);
  };

  const handleKeyDown = async (event) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      await handleSubmit();
    }
  };

  return (
    <Paper
      elevation={mode === 'page' ? 1 : 0}
      sx={{
        display: 'flex',
        flexDirection: 'column',
        height: '100%',
        minHeight: mode === 'page' ? '70vh' : '100%',
        borderRadius: mode === 'page' ? 2 : 0,
        overflow: 'hidden',
      }}
    >
      <Box sx={{ px: 2, py: 1.5, display: 'flex', alignItems: 'center', gap: 1 }}>
        <Box sx={{ flex: 1 }}>
          <Typography variant="h6">
            Assistant
          </Typography>
          <Typography variant="caption" color="text.secondary">
            {conversationTitle || 'Assistant'}
          </Typography>
        </Box>

        <Chip
          size="small"
          color={assistantHealth.available ? 'success' : 'default'}
          label={assistantHealth.available ? 'Hermes online' : 'Hermes offline'}
        />
        <Tooltip title="Start new conversation">
          <IconButton size="small" onClick={() => startNewConversation()}>
            <AddIcon fontSize="small" />
          </IconButton>
        </Tooltip>
        {onClose && (
          <Tooltip title="Close assistant">
            <IconButton size="small" onClick={onClose}>
              <CloseIcon fontSize="small" />
            </IconButton>
          </Tooltip>
        )}
      </Box>
      <Divider />

      {!assistantHealth.available && (
        <Box sx={{ p: 2, pb: 0 }}>
          <Alert severity="warning">
            {assistantHealth.detail || 'The assistant runtime is not currently available.'}
          </Alert>
        </Box>
      )}

      <AssistantMessageList
        messages={displayedMessages}
        isLoading={isLoadingConversation || isStreaming}
      />

      <Divider />

      <Box sx={{ p: 2 }}>
        <Stack direction="row" spacing={1} sx={{ mb: latestAssistantSymbols.length ? 1 : 0 }}>
          {latestAssistantSymbols.length > 0 && (
            <Button
              size="small"
              startIcon={<PlaylistAddIcon />}
              onClick={() => setWatchlistDialogOpen(true)}
            >
              Add tickers to watchlist
            </Button>
          )}
          {latestAssistantSymbols.length > 0 && (
            <Typography variant="caption" color="text.secondary" sx={{ alignSelf: 'center' }}>
              {latestAssistantSymbols.join(', ')}
            </Typography>
          )}
        </Stack>

        <Box sx={{ display: 'flex', gap: 1, alignItems: 'flex-end' }}>
          <TextField
            fullWidth
            multiline
            minRows={2}
            maxRows={6}
            value={input}
            onChange={(event) => setInput(event.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask about scans, themes, breadth, symbols, or compare internal signals with current web context."
            disabled={!assistantHealth.available || isStreaming}
          />
          <Button
            variant="contained"
            endIcon={<SendIcon />}
            onClick={handleSubmit}
            disabled={!input.trim() || !assistantHealth.available || isStreaming}
            sx={{ minWidth: 110, height: 40 }}
          >
            Send
          </Button>
        </Box>
      </Box>

      <AssistantWatchlistDialog
        open={watchlistDialogOpen}
        symbols={latestAssistantSymbols}
        onClose={() => setWatchlistDialogOpen(false)}
      />
    </Paper>
  );
}

export default AssistantChat;

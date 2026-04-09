import { useEffect, useRef } from 'react';
import { Box, CircularProgress, Typography } from '@mui/material';
import AssistantMessageBubble from './AssistantMessageBubble';

const SUGGESTIONS = [
  'What are the strongest current scan candidates?',
  'Which themes are accelerating right now?',
  'How does market breadth look?',
  'Compare StockScreenClaude signals with current web context for NVDA.',
];

function AssistantMessageList({ messages, isLoading }) {
  const bottomRef = useRef(null);

  useEffect(() => {
    if (bottomRef.current) {
      bottomRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [messages, isLoading]);

  if (messages.length === 0 && !isLoading) {
    return (
      <Box
        sx={{
          flex: 1,
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          p: 3,
        }}
      >
        <Typography variant="body1" color="text.secondary" gutterBottom>
          Start a conversation with the assistant
        </Typography>
        <Typography variant="body2" color="text.secondary" sx={{ maxWidth: 440, textAlign: 'center' }}>
          It can use StockScreenClaude scan outputs, themes, breadth, watchlists, and wider web context in one response.
        </Typography>
        <Box sx={{ mt: 2, display: 'flex', flexDirection: 'column', gap: 1 }}>
          {SUGGESTIONS.map((suggestion) => (
            <Typography
              key={suggestion}
              variant="body2"
              sx={{
                p: 1,
                px: 2,
                borderRadius: 2,
                backgroundColor: 'action.hover',
                cursor: 'default',
                fontSize: '13px',
              }}
            >
              &quot;{suggestion}&quot;
            </Typography>
          ))}
        </Box>
      </Box>
    );
  }

  return (
    <Box
      sx={{
        flex: 1,
        overflow: 'auto',
        p: 2,
        display: 'flex',
        flexDirection: 'column',
        gap: 2,
      }}
    >
      {messages.map((message) => (
        <AssistantMessageBubble key={message.id} message={message} />
      ))}

      {isLoading && messages.length > 0 && !messages[messages.length - 1]?.isStreaming && (
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, pl: 1 }}>
          <CircularProgress size={16} />
          <Typography variant="caption" color="text.secondary">
            Thinking...
          </Typography>
        </Box>
      )}

      <div ref={bottomRef} />
    </Box>
  );
}

export default AssistantMessageList;

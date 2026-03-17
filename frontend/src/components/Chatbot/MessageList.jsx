import { useRef, useEffect } from 'react';
import { Box, Typography, CircularProgress } from '@mui/material';
import MessageBubble from './MessageBubble';

function MessageList({ messages, isLoading }) {
  const bottomRef = useRef(null);

  // Auto-scroll to bottom on new messages
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
          Start a conversation
        </Typography>
        <Typography variant="body2" color="text.secondary" sx={{ maxWidth: 400, textAlign: 'center' }}>
          Try asking:
        </Typography>
        <Box sx={{ mt: 2, display: 'flex', flexDirection: 'column', gap: 1 }}>
          {[
            "How is NVDA performing?",
            "What are the trending themes?",
            "Show me stocks with high RS ratings",
            "Compare AAPL, MSFT, and GOOGL",
          ].map((suggestion, i) => (
            <Typography
              key={i}
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
        <MessageBubble key={message.id} message={message} />
      ))}

      {/* Loading indicator when waiting for initial response */}
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

export default MessageList;

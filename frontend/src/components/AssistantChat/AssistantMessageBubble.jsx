import { useMemo, useState } from 'react';
import {
  Box,
  CircularProgress,
  Collapse,
  IconButton,
  Paper,
  Typography,
  useTheme,
} from '@mui/material';
import PersonIcon from '@mui/icons-material/Person';
import SmartToyIcon from '@mui/icons-material/SmartToy';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import ExpandLessIcon from '@mui/icons-material/ExpandLess';
import LinkIcon from '@mui/icons-material/Link';

import ReactMarkdown from 'react-markdown';

const EMPTY_ARRAY = [];

function processContentWithCitations(content, references) {
  if (!content || !references?.length) {
    return content || '';
  }

  const referenceMap = new Map();
  references.forEach((reference, index) => {
    referenceMap.set(reference.reference_number || index + 1, reference);
  });

  return content.replace(/\[(\d+)\]/g, (match, number) => {
    const reference = referenceMap.get(Number(number));
    if (!reference?.url) {
      return match;
    }
    const title = (reference.title || `Source ${number}`).replace(/"/g, "'");
    return `[[${number}]](${reference.url} "${title}")`;
  });
}

function AssistantMessageBubble({ message }) {
  const theme = useTheme();
  const [showSources, setShowSources] = useState(false);
  const [showTools, setShowTools] = useState(false);

  const isUser = message.role === 'user';
  const isStreaming = Boolean(message.isStreaming);
  const references = message.source_references ?? message.sourceReferences ?? EMPTY_ARRAY;
  const toolCalls = message.tool_calls ?? message.toolCalls ?? EMPTY_ARRAY;
  const processedContent = useMemo(
    () => processContentWithCitations(message.content, references),
    [message.content, references],
  );

  return (
    <Box
      sx={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: isUser ? 'flex-end' : 'flex-start',
        maxWidth: '88%',
        alignSelf: isUser ? 'flex-end' : 'flex-start',
      }}
    >
      <Box
        sx={{
          display: 'flex',
          alignItems: 'center',
          gap: 1,
          mb: 0.5,
          flexDirection: isUser ? 'row-reverse' : 'row',
        }}
      >
        <Box
          sx={{
            width: 28,
            height: 28,
            borderRadius: '50%',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            backgroundColor: isUser ? 'primary.main' : 'secondary.main',
            color: 'white',
          }}
        >
          {isUser ? <PersonIcon fontSize="small" /> : <SmartToyIcon fontSize="small" />}
        </Box>
        <Typography variant="caption" color="text.secondary">
          {isUser ? 'You' : 'Assistant'}
        </Typography>
      </Box>

      <Paper
        elevation={1}
        sx={{
          p: 2,
          borderRadius: 2,
          backgroundColor: isUser
            ? 'primary.main'
            : message.isError
              ? 'error.dark'
              : theme.palette.mode === 'dark'
                ? 'grey.800'
                : 'grey.100',
          color: isUser || message.isError ? 'white' : 'text.primary',
          maxWidth: '100%',
          wordBreak: 'break-word',
        }}
      >
        {isStreaming && !message.content ? (
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <CircularProgress size={16} color="inherit" />
            <Typography variant="body2">Thinking...</Typography>
          </Box>
        ) : (
          <Box
            sx={{
              fontSize: '14px',
              lineHeight: 1.6,
              '& p': { m: 0, mb: 1, '&:last-child': { mb: 0 } },
              '& ul, & ol': { m: 0, pl: 2.5, mb: 1 },
              '& li': { mb: 0.5 },
              '& code': {
                backgroundColor: theme.palette.mode === 'dark' ? 'grey.900' : 'grey.200',
                p: 0.5,
                borderRadius: 0.5,
                fontSize: '0.85em',
              },
              '& pre': {
                backgroundColor: theme.palette.mode === 'dark' ? 'grey.900' : 'grey.200',
                p: 1.5,
                borderRadius: 1,
                overflow: 'auto',
                fontSize: '12px',
                '& code': { p: 0, backgroundColor: 'transparent' },
              },
              '& strong': { fontWeight: 600 },
              '& a[href]': {
                color: isUser ? 'inherit' : 'primary.main',
                textDecoration: 'none',
                fontWeight: 500,
                '&:hover': {
                  textDecoration: 'underline',
                },
              },
            }}
          >
            <ReactMarkdown>{processedContent}</ReactMarkdown>
          </Box>
        )}

        {isStreaming && message.content && (
          <Box
            component="span"
            sx={{
              display: 'inline-block',
              width: 8,
              height: 16,
              backgroundColor: 'primary.main',
              ml: 0.5,
              animation: 'blink 1s infinite',
              '@keyframes blink': {
                '0%, 50%': { opacity: 1 },
                '51%, 100%': { opacity: 0 },
              },
            }}
          />
        )}
      </Paper>

      {!isUser && toolCalls.length > 0 && (
        <Box sx={{ mt: 1, width: '100%' }}>
          <Box
            sx={{ display: 'flex', alignItems: 'center', gap: 0.5, cursor: 'pointer' }}
            onClick={() => setShowTools((previous) => !previous)}
          >
            <Typography variant="caption" color="text.secondary">
              Tool activity ({toolCalls.length})
            </Typography>
            <IconButton size="small">
              {showTools ? <ExpandLessIcon fontSize="small" /> : <ExpandMoreIcon fontSize="small" />}
            </IconButton>
          </Box>

          <Collapse in={showTools}>
            <Box sx={{ mt: 1, display: 'flex', flexDirection: 'column', gap: 0.75 }}>
              {toolCalls.map((toolCall, index) => (
                <Paper key={`${toolCall.tool}-${index}`} variant="outlined" sx={{ p: 1 }}>
                  <Typography variant="caption" sx={{ fontWeight: 600 }}>
                    {toolCall.tool}
                  </Typography>
                  {toolCall.args && (
                    <Typography
                      variant="caption"
                      component="pre"
                      sx={{
                        m: 0,
                        mt: 0.5,
                        p: 0.5,
                        backgroundColor: 'action.hover',
                        borderRadius: 0.5,
                        overflow: 'auto',
                        maxHeight: 120,
                      }}
                    >
                      {JSON.stringify(toolCall.args, null, 2)}
                    </Typography>
                  )}
                  {toolCall.result && (
                    <Typography
                      variant="caption"
                      component="pre"
                      sx={{
                        m: 0,
                        mt: 0.5,
                        p: 0.5,
                        backgroundColor: 'action.hover',
                        borderRadius: 0.5,
                        overflow: 'auto',
                        maxHeight: 150,
                      }}
                    >
                      {JSON.stringify(toolCall.result, null, 2)}
                    </Typography>
                  )}
                </Paper>
              ))}
            </Box>
          </Collapse>
        </Box>
      )}

      {!isUser && references.length > 0 && (
        <Box sx={{ mt: 1, width: '100%' }}>
          <Box
            sx={{ display: 'flex', alignItems: 'center', gap: 0.5, cursor: 'pointer' }}
            onClick={() => setShowSources((previous) => !previous)}
          >
            <LinkIcon fontSize="small" color="action" />
            <Typography variant="caption" color="text.secondary">
              Sources ({references.length})
            </Typography>
            <IconButton size="small">
              {showSources ? <ExpandLessIcon fontSize="small" /> : <ExpandMoreIcon fontSize="small" />}
            </IconButton>
          </Box>

          <Collapse in={showSources}>
            <Box sx={{ mt: 1, display: 'flex', flexDirection: 'column', gap: 0.75 }}>
              {references.map((reference, index) => (
                <Paper key={`${reference.url}-${index}`} variant="outlined" sx={{ p: 1 }}>
                  <Box sx={{ display: 'flex', gap: 1 }}>
                    <Typography variant="caption" sx={{ fontWeight: 600, flexShrink: 0 }}>
                      [{reference.reference_number || index + 1}]
                    </Typography>
                    <Box sx={{ minWidth: 0 }}>
                      <Typography
                        component="a"
                        href={reference.url}
                        target="_blank"
                        rel="noopener noreferrer"
                        sx={{
                          color: 'primary.main',
                          fontSize: '12px',
                          textDecoration: 'none',
                          fontWeight: 600,
                          '&:hover': { textDecoration: 'underline' },
                        }}
                      >
                        {reference.title}
                      </Typography>
                      {(reference.section || reference.snippet) && (
                        <Typography
                          variant="caption"
                          color="text.secondary"
                          sx={{ display: 'block', mt: 0.25, whiteSpace: 'pre-wrap' }}
                        >
                          {[reference.section, reference.snippet].filter(Boolean).join(' • ')}
                        </Typography>
                      )}
                    </Box>
                  </Box>
                </Paper>
              ))}
            </Box>
          </Collapse>
        </Box>
      )}
    </Box>
  );
}

export default AssistantMessageBubble;

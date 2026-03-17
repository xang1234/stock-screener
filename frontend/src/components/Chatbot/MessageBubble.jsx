import { useState, useMemo } from 'react';
import {
  Box,
  Paper,
  Typography,
  Collapse,
  IconButton,
  Chip,
  useTheme,
  CircularProgress,
} from '@mui/material';
import PersonIcon from '@mui/icons-material/Person';
import SmartToyIcon from '@mui/icons-material/SmartToy';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import ExpandLessIcon from '@mui/icons-material/ExpandLess';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import ErrorIcon from '@mui/icons-material/Error';
import PsychologyIcon from '@mui/icons-material/Psychology';
import LinkIcon from '@mui/icons-material/Link';

import ReactMarkdown from 'react-markdown';

/**
 * Process content to make inline citations [N] clickable links.
 * Converts [1], [2], etc. to markdown links pointing to the reference URL.
 * Matches citations by reference_number field, not array index.
 */
function processContentWithCitations(content, references) {
  if (!content || !references || references.length === 0) {
    return content;
  }

  // Build a map of reference_number -> reference for O(1) lookup
  const refMap = new Map();
  references.forEach((ref, idx) => {
    // Use reference_number if present, otherwise fall back to idx + 1
    // parseInt ensures consistent integer keys (handles string "1" from old messages)
    const refNum = parseInt(ref.reference_number, 10) || (idx + 1);
    refMap.set(refNum, ref);
  });

  // Replace [N] with markdown links, looking up by reference_number
  return content.replace(/\[(\d+)\]/g, (match, num) => {
    const refNum = parseInt(num, 10);
    const ref = refMap.get(refNum);
    if (ref && ref.url) {
      const title = (ref.title || `Source ${num}`).replace(/"/g, "'");
      // Create a markdown link - the brackets stay visible as link text
      return `[[${num}]](${ref.url} "${title}")`;
    }
    return match;
  });
}

function MessageBubble({ message }) {
  const theme = useTheme();
  const [showTools, setShowTools] = useState(false);
  const [showThinking, setShowThinking] = useState(false);
  const [showSources, setShowSources] = useState(false);

  const isUser = message.role === 'user';
  const isStreaming = message.isStreaming;
  const isError = message.isError;

  // Handle both camelCase (streaming) and snake_case (loaded from backend)
  const toolCalls = message.toolCalls || message.tool_calls || [];
  const thinkingTraces = message.thinkingTraces || message.thinking_traces || [];
  const references = message.sourceReferences || message.source_references || [];

  const hasToolCalls = toolCalls.length > 0;
  const hasThinking = thinkingTraces.length > 0;
  const hasReferences = references.length > 0;

  // Process content to make inline citations clickable
  const processedContent = useMemo(() => {
    return processContentWithCitations(message.content, references);
  }, [message.content, references]);

  return (
    <Box
      sx={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: isUser ? 'flex-end' : 'flex-start',
        maxWidth: '85%',
        alignSelf: isUser ? 'flex-end' : 'flex-start',
      }}
    >
      {/* Avatar and name */}
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
          {message.agent && ` (${message.agent})`}
        </Typography>
      </Box>

      {/* Message content */}
      <Paper
        elevation={1}
        sx={{
          p: 2,
          borderRadius: 2,
          backgroundColor: isUser
            ? 'primary.main'
            : isError
            ? 'error.dark'
            : theme.palette.mode === 'dark'
            ? 'grey.800'
            : 'grey.100',
          color: isUser || isError ? 'white' : 'text.primary',
          maxWidth: '100%',
          wordBreak: 'break-word',
        }}
      >
        {isStreaming && !message.content ? (
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <CircularProgress size={16} color="inherit" />
            <Typography variant="body2">
              {message.agent === 'planning' && 'Creating research plan...'}
              {message.agent === 'action' && 'Gathering data...'}
              {message.agent === 'validation' && 'Validating results...'}
              {message.agent === 'answer' && 'Generating response...'}
              {!message.agent && 'Thinking...'}
            </Typography>
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
              '& h1, & h2, & h3, & h4': { mt: 1, mb: 0.5 },
              '& table': {
                borderCollapse: 'collapse',
                width: '100%',
                mb: 1,
              },
              '& th, & td': {
                border: `1px solid ${theme.palette.divider}`,
                p: 0.5,
                fontSize: '12px',
              },
              // Style for inline citation links
              '& a[href]': {
                color: 'primary.main',
                textDecoration: 'none',
                fontWeight: 500,
                '&:hover': {
                  textDecoration: 'underline',
                },
              },
            }}
          >
            <ReactMarkdown>{processedContent || ''}</ReactMarkdown>
          </Box>
        )}

        {/* Streaming cursor */}
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

      {/* Sources section (collapsible) */}
      {!isUser && hasReferences && !isStreaming && (
        <Box sx={{ mt: 1, width: '100%' }}>
          <Box
            sx={{
              display: 'flex',
              alignItems: 'center',
              gap: 0.5,
              cursor: 'pointer',
            }}
            onClick={() => setShowSources(!showSources)}
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
            <Box sx={{ mt: 1, display: 'flex', flexDirection: 'column', gap: 0.5 }}>
              {references.map((ref, idx) => {
                // Use reference_number if present, otherwise fall back to idx + 1
                const displayNum = ref.reference_number || (idx + 1);
                return (
                  <Paper
                    key={idx}
                    variant="outlined"
                    sx={{
                      p: 1,
                      fontSize: '12px',
                      backgroundColor: 'background.default',
                      borderLeft: '3px solid',
                      borderLeftColor: 'primary.main',
                    }}
                  >
                    <Box
                      sx={{
                        display: 'flex',
                        alignItems: 'flex-start',
                        gap: 0.75,
                      }}
                    >
                      <Typography
                        component="span"
                        sx={{
                          color: 'text.secondary',
                          fontSize: '12px',
                          fontWeight: 600,
                          flexShrink: 0,
                        }}
                      >
                        [{displayNum}]
                      </Typography>
                      <Typography
                        component="a"
                        href={ref.url}
                        target="_blank"
                        rel="noopener noreferrer"
                        sx={{
                          color: 'primary.main',
                          fontSize: '12px',
                          textDecoration: 'none',
                          '&:hover': {
                            textDecoration: 'underline',
                          },
                          overflow: 'hidden',
                          textOverflow: 'ellipsis',
                          whiteSpace: 'nowrap',
                        }}
                      >
                        {ref.title}{ref.section ? ` - ${ref.section}` : ''}
                      </Typography>
                    </Box>
                  </Paper>
                );
              })}
            </Box>
          </Collapse>
        </Box>
      )}

      {/* Thinking traces (collapsible) */}
      {hasThinking && (
        <Box sx={{ mt: 1, width: '100%' }}>
          <Box
            sx={{
              display: 'flex',
              alignItems: 'center',
              gap: 0.5,
              cursor: 'pointer',
            }}
            onClick={() => setShowThinking(!showThinking)}
          >
            <PsychologyIcon fontSize="small" color="action" />
            <Typography variant="caption" color="text.secondary">
              Agent Thinking ({thinkingTraces.length})
            </Typography>
            <IconButton size="small">
              {showThinking ? <ExpandLessIcon fontSize="small" /> : <ExpandMoreIcon fontSize="small" />}
            </IconButton>
          </Box>

          <Collapse in={showThinking}>
            <Box sx={{ mt: 1, display: 'flex', flexDirection: 'column', gap: 0.5 }}>
              {thinkingTraces.map((trace, idx) => (
                <Paper
                  key={idx}
                  variant="outlined"
                  sx={{
                    p: 1,
                    fontSize: '12px',
                    backgroundColor: 'background.default',
                    borderLeft: '3px solid',
                    borderLeftColor: 'info.main',
                  }}
                >
                  {trace.agent && (
                    <Chip
                      size="small"
                      label={trace.agent}
                      color="info"
                      variant="outlined"
                      sx={{ mb: 0.5 }}
                    />
                  )}
                  <Typography
                    variant="body2"
                    sx={{
                      fontStyle: 'italic',
                      color: 'text.secondary',
                      whiteSpace: 'pre-wrap',  // Preserve line breaks in reasoning
                    }}
                  >
                    {trace.content}
                  </Typography>
                </Paper>
              ))}
            </Box>
          </Collapse>
        </Box>
      )}

      {/* Tool calls (collapsible) */}
      {hasToolCalls && (
        <Box sx={{ mt: 1, width: '100%' }}>
          <Box
            sx={{
              display: 'flex',
              alignItems: 'center',
              gap: 0.5,
              cursor: 'pointer',
            }}
            onClick={() => setShowTools(!showTools)}
          >
            <Typography variant="caption" color="text.secondary">
              {toolCalls.length} tool calls
            </Typography>
            <IconButton size="small">
              {showTools ? <ExpandLessIcon fontSize="small" /> : <ExpandMoreIcon fontSize="small" />}
            </IconButton>
          </Box>

          <Collapse in={showTools}>
            <Box sx={{ mt: 1, display: 'flex', flexDirection: 'column', gap: 1 }}>
              {toolCalls.map((tool, idx) => {
                // Detect format: streaming has 'type' field, database format has 'args' without 'type'
                const isStreamingFormat = 'type' in tool;
                const isCallOnly = isStreamingFormat && tool.type === 'call';
                const isResultOnly = isStreamingFormat && tool.type === 'result';
                const isCombined = !isStreamingFormat && 'args' in tool; // Database format

                // Determine status for chip color
                const hasResult = isResultOnly || (isCombined && tool.result);
                const isSuccess = isResultOnly ? tool.status === 'success' : (isCombined && tool.result && !tool.result.error);
                const chipColor = isCallOnly ? 'primary' : (hasResult && isSuccess) ? 'success' : hasResult ? 'error' : 'primary';

                // Get params/args and result based on format
                const params = tool.params || tool.args;
                const result = tool.result;

                return (
                  <Paper
                    key={idx}
                    variant="outlined"
                    sx={{
                      p: 1,
                      fontSize: '12px',
                      backgroundColor: 'background.default',
                    }}
                  >
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 0.5 }}>
                      <Chip
                        size="small"
                        label={tool.tool}
                        color={chipColor}
                        variant="outlined"
                      />
                      {hasResult && (
                        isSuccess ? (
                          <CheckCircleIcon fontSize="small" color="success" />
                        ) : (
                          <ErrorIcon fontSize="small" color="error" />
                        )
                      )}
                    </Box>

                    {/* Show params for call-only or combined format */}
                    {(isCallOnly || isCombined) && params && (
                      <Typography
                        variant="caption"
                        component="pre"
                        sx={{
                          m: 0,
                          p: 0.5,
                          backgroundColor: 'action.hover',
                          borderRadius: 0.5,
                          overflow: 'auto',
                          maxHeight: 100,
                        }}
                      >
                        {JSON.stringify(params, null, 2)}
                      </Typography>
                    )}

                    {/* Show result for result-only or combined format */}
                    {(isResultOnly || isCombined) && result && (
                      <Box sx={{ mt: 0.5 }}>
                        {/* Special rendering for news/search results */}
                        {(tool.tool === 'search_news' || tool.tool === 'web_search' || tool.tool === 'search_finance') && result.results ? (
                          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 0.5 }}>
                            <Typography variant="caption" color="text.secondary">
                              {result.total_results} results for &quot;{result.query}&quot;
                            </Typography>
                            {result.results.map((item, i) => (
                              <Box
                                key={i}
                                sx={{
                                  p: 0.75,
                                  backgroundColor: 'action.hover',
                                  borderRadius: 0.5,
                                  borderLeft: '2px solid',
                                  borderLeftColor: 'primary.main',
                                }}
                              >
                                <Typography
                                  variant="caption"
                                  component="a"
                                  href={item.url}
                                  target="_blank"
                                  rel="noopener noreferrer"
                                  sx={{
                                    fontWeight: 600,
                                    color: 'primary.main',
                                    textDecoration: 'none',
                                    display: 'block',
                                    '&:hover': { textDecoration: 'underline' },
                                  }}
                                >
                                  {item.title}
                                </Typography>
                                {item.snippet && (
                                  <Typography
                                    variant="caption"
                                    sx={{
                                      color: 'text.secondary',
                                      display: 'block',
                                      mt: 0.25,
                                      fontSize: '11px',
                                    }}
                                  >
                                    {item.snippet}
                                  </Typography>
                                )}
                              </Box>
                            ))}
                          </Box>
                        ) : (
                          <Typography
                            variant="caption"
                            component="pre"
                            sx={{
                              m: 0,
                              p: 0.5,
                              backgroundColor: 'action.hover',
                              borderRadius: 0.5,
                              overflow: 'auto',
                              maxHeight: 150,
                            }}
                          >
                            {JSON.stringify(result, null, 2)}
                          </Typography>
                        )}
                      </Box>
                    )}
                  </Paper>
                );
              })}
            </Box>
          </Collapse>
        </Box>
      )}

      {/* Timestamp */}
      <Typography
        variant="caption"
        color="text.secondary"
        sx={{ mt: 0.5, fontSize: '10px' }}
      >
        {new Date(message.created_at).toLocaleTimeString()}
      </Typography>
    </Box>
  );
}

export default MessageBubble;

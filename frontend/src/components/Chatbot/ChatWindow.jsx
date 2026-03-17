import { useState, useRef, useEffect } from 'react';
import {
  Box,
  Paper,
  TextField,
  IconButton,
  Typography,
  CircularProgress,
} from '@mui/material';
import SendIcon from '@mui/icons-material/Send';

import MessageList from './MessageList';
import ToolSelector from './ToolSelector';
import ResearchModeToggle from './ResearchModeToggle';
import ResearchProgressIndicator from './ResearchProgressIndicator';
import PromptLibrary from './PromptLibrary';
import { sendMessageStream } from '../../api/chatbot';
import { useToolSelection } from '../../hooks/useToolSelection';

function ChatWindow({ conversationId, messages, onMessagesUpdate }) {
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [streamingContent, setStreamingContent] = useState('');
  const [currentAgent, setCurrentAgent] = useState(null);
  const [toolCalls, setToolCalls] = useState([]);
  const [thinkingTraces, setThinkingTraces] = useState([]);
  const abortRef = useRef(null);
  const inputRef = useRef(null);
  const contentRef = useRef('');  // Track content synchronously for 'done' handler
  const toolCallsRef = useRef([]);  // Track tool calls synchronously
  const thinkingRef = useRef([]);  // Track thinking traces synchronously

  // Research mode state
  const [researchMode, setResearchMode] = useState(false);
  const [researchPhase, setResearchPhase] = useState(null);
  const [researchProgress, setResearchProgress] = useState({
    totalUnits: 0,
    completedUnits: 0,
    sourcesFound: 0,
    subQuestions: [],
  });

  // Tool selection state
  const {
    enabledTools,
    toggleTool,
    toggleCategory,
    enableAll,
    disableAll,
    getEnabledToolsArray,
    enabledCount,
    totalCount,
    allEnabled,
  } = useToolSelection();

  // Focus input on mount
  useEffect(() => {
    if (inputRef.current) {
      inputRef.current.focus();
    }
  }, [conversationId]);

  const handleSend = async () => {
    if (!input.trim() || isLoading) return;

    const userMessage = {
      id: Date.now(),
      role: 'user',
      content: input.trim(),
      created_at: new Date().toISOString(),
    };

    // Add user message immediately
    const updatedMessages = [...messages, userMessage];
    onMessagesUpdate(updatedMessages);

    setInput('');
    setIsLoading(true);
    setStreamingContent('');
    setCurrentAgent(null);
    setToolCalls([]);
    setThinkingTraces([]);
    contentRef.current = '';  // Reset ref
    toolCallsRef.current = [];  // Reset ref
    thinkingRef.current = [];  // Reset ref

    // Reset research state if in research mode
    if (researchMode) {
      setResearchPhase('planning');
      setResearchProgress({
        totalUnits: 0,
        completedUnits: 0,
        sourcesFound: 0,
        subQuestions: [],
      });
    }

    // Start streaming
    abortRef.current = sendMessageStream(
      conversationId,
      userMessage.content,
      getEnabledToolsArray(),  // Pass enabled tools (null if all enabled)
      researchMode,  // Pass research mode flag
      // onChunk
      (chunk) => {
        switch (chunk.type) {
          // Research-specific chunk types
          case 'research_phase':
            setResearchPhase(chunk.phase);
            break;

          case 'research_plan':
            setResearchProgress(prev => ({
              ...prev,
              subQuestions: chunk.sub_questions || [],
            }));
            break;

          case 'research_progress':
            setResearchProgress(prev => ({
              ...prev,
              totalUnits: chunk.total_units ?? prev.totalUnits,
              completedUnits: chunk.completed ?? prev.completedUnits,
              sourcesFound: chunk.total_notes ?? prev.sourcesFound,
            }));
            break;

          case 'research_complete':
            setResearchProgress(prev => ({
              ...prev,
              completedUnits: chunk.completed_units ?? prev.completedUnits,
              sourcesFound: chunk.total_notes ?? prev.sourcesFound,
            }));
            break;

          case 'thinking':
            setCurrentAgent(chunk.agent);
            if (chunk.content) {
              const trace = { content: chunk.content, agent: chunk.agent, timestamp: Date.now() };
              thinkingRef.current = [...thinkingRef.current, trace];
              setThinkingTraces(prev => [...prev, trace]);
            }
            break;

          case 'tool_call': {
            const toolCall = {
              type: 'call',
              tool: chunk.tool,
              params: chunk.params,
            };
            toolCallsRef.current = [...toolCallsRef.current, toolCall];
            setToolCalls((prev) => [...prev, toolCall]);
            break;
          }

          case 'tool_result': {
            const toolResult = {
              type: 'result',
              tool: chunk.tool,
              status: chunk.status,
              result: chunk.result,
            };
            toolCallsRef.current = [...toolCallsRef.current, toolResult];
            setToolCalls((prev) => [...prev, toolResult]);
            break;
          }

          case 'content':
            contentRef.current += chunk.content;  // Update ref synchronously
            setStreamingContent((prev) => prev + chunk.content);
            break;

          case 'done': {
            // Finalize the assistant message using refs (synchronous values)
            const assistantMessage = {
              id: chunk.message_id || Date.now() + 1,
              role: 'assistant',
              content: contentRef.current || 'Response complete.',
              created_at: new Date().toISOString(),
              tool_calls: toolCallsRef.current,
              thinkingTraces: thinkingRef.current,
              sourceReferences: chunk.references || [],
              isResearchReport: researchMode,
              researchStats: chunk.stats || null,
            };
            onMessagesUpdate([...updatedMessages, assistantMessage]);
            setIsLoading(false);
            setStreamingContent('');
            setToolCalls([]);
            setThinkingTraces([]);
            setCurrentAgent(null);
            setResearchPhase(null);
            contentRef.current = '';
            toolCallsRef.current = [];
            thinkingRef.current = [];
            break;
          }

          case 'error': {
            const errorMessage = {
              id: Date.now() + 1,
              role: 'assistant',
              content: `Error: ${chunk.error}`,
              created_at: new Date().toISOString(),
              isError: true,
            };
            onMessagesUpdate([...updatedMessages, errorMessage]);
            setIsLoading(false);
            setStreamingContent('');
            break;
          }

          default:
            break;
        }
      },
      // onError
      (error) => {
        console.error('Stream error:', error);
        const errorMessage = {
          id: Date.now() + 1,
          role: 'assistant',
          content: `Connection error: ${error.message}`,
          created_at: new Date().toISOString(),
          isError: true,
        };
        onMessagesUpdate([...updatedMessages, errorMessage]);
        setIsLoading(false);
        setStreamingContent('');
      },
      // onDone
      () => {
        // Stream ended - handled in 'done' chunk
      }
    );
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  // Build display messages including streaming content
  const displayMessages = [...messages];
  if (isLoading && (streamingContent || currentAgent || thinkingTraces.length > 0 || researchPhase)) {
    displayMessages.push({
      id: 'streaming',
      role: 'assistant',
      content: streamingContent,
      created_at: new Date().toISOString(),
      isStreaming: true,
      agent: currentAgent,
      toolCalls: toolCalls,
      thinkingTraces: thinkingTraces,
      isResearchReport: researchMode,
      researchPhase: researchPhase,
      researchProgress: researchProgress,
    });
  }

  return (
    <Box
      sx={{
        flex: 1,
        display: 'flex',
        flexDirection: 'column',
        height: '100%',
        overflow: 'hidden',
      }}
    >
      {/* Research Progress Indicator */}
      {isLoading && researchMode && researchPhase && (
        <Box sx={{ px: 2, pt: 2 }}>
          <ResearchProgressIndicator
            phase={researchPhase}
            currentUnit={researchProgress.completedUnits}
            totalUnits={researchProgress.totalUnits}
            completedUnits={researchProgress.completedUnits}
            sourcesFound={researchProgress.sourcesFound}
            subQuestions={researchProgress.subQuestions}
          />
        </Box>
      )}

      {/* Messages Area */}
      <MessageList messages={displayMessages} isLoading={isLoading} />

      {/* Input Area */}
      <Paper
        elevation={2}
        sx={{
          p: 2,
          borderTop: 1,
          borderColor: 'divider',
          backgroundColor: 'background.paper',
        }}
      >
        <Box sx={{ display: 'flex', gap: 1, alignItems: 'flex-end' }}>
          <ToolSelector
            enabledTools={enabledTools}
            toggleTool={toggleTool}
            toggleCategory={toggleCategory}
            enableAll={enableAll}
            disableAll={disableAll}
            enabledCount={enabledCount}
            totalCount={totalCount}
            allEnabled={allEnabled}
          />
          <ResearchModeToggle
            researchMode={researchMode}
            onToggle={setResearchMode}
            disabled={isLoading}
          />
          <PromptLibrary
            onInsertPrompt={(text) => setInput(text)}
            disabled={isLoading}
          />
          <TextField
            inputRef={inputRef}
            fullWidth
            multiline
            maxRows={4}
            placeholder="Ask about stocks, themes, or market analysis..."
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={handleKeyPress}
            disabled={isLoading}
            variant="outlined"
            size="small"
            sx={{
              '& .MuiOutlinedInput-root': {
                backgroundColor: 'background.default',
              },
            }}
          />
          <IconButton
            color="primary"
            onClick={handleSend}
            disabled={!input.trim() || isLoading}
            sx={{
              backgroundColor: 'primary.main',
              color: 'white',
              '&:hover': {
                backgroundColor: 'primary.dark',
              },
              '&:disabled': {
                backgroundColor: 'action.disabledBackground',
              },
            }}
          >
            {isLoading ? <CircularProgress size={24} color="inherit" /> : <SendIcon />}
          </IconButton>
        </Box>

        {/* Status indicator */}
        {isLoading && (currentAgent || researchPhase) && (
          <Typography
            variant="caption"
            color="text.secondary"
            sx={{ mt: 1, display: 'block' }}
          >
            {researchPhase === 'planning' && 'Planning research strategy...'}
            {researchPhase === 'researching' && 'Gathering data from multiple sources...'}
            {researchPhase === 'compressing' && 'Analyzing and consolidating findings...'}
            {researchPhase === 'writing' && 'Writing research report...'}
            {!researchPhase && currentAgent === 'planning' && 'Planning research strategy...'}
            {!researchPhase && currentAgent === 'action' && 'Gathering data...'}
            {!researchPhase && currentAgent === 'validation' && 'Validating results...'}
            {!researchPhase && currentAgent === 'answer' && 'Generating response...'}
            {!researchPhase && currentAgent === 'tool_agent' && 'Processing...'}
          </Typography>
        )}
      </Paper>
    </Box>
  );
}

export default ChatWindow;

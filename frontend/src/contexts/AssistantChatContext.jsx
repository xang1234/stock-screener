/* eslint-disable react-refresh/only-export-components */

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
} from 'react';
import { useQuery } from '@tanstack/react-query';
import {
  checkHealth,
  createConversation,
  getConversation,
  sendMessageStream,
} from '../api/assistant';
import { useRuntime } from './RuntimeContext';

const STORAGE_KEY = 'assistant_conversation_id';
const AssistantChatContext = createContext(null);

const summarizeTitle = (content) => {
  const normalized = content.trim().replace(/\s+/g, ' ');
  if (normalized.length <= 60) return normalized;
  return `${normalized.slice(0, 57).trim()}...`;
};

const buildStreamingDraft = () => ({
  id: 'assistant-streaming',
  role: 'assistant',
  content: '',
  created_at: new Date().toISOString(),
  isStreaming: true,
  tool_calls: [],
  source_references: [],
});

export function AssistantChatProvider({ children }) {
  const { auth, features } = useRuntime();
  const [conversationId, setConversationId] = useState(() => {
    if (typeof window === 'undefined') return null;
    return window.localStorage.getItem(STORAGE_KEY);
  });
  const [conversationTitle, setConversationTitle] = useState('Assistant');
  const [messages, setMessages] = useState([]);
  const [draftMessage, setDraftMessage] = useState(null);
  const [isLoadingConversation, setIsLoadingConversation] = useState(false);
  const [isStreaming, setIsStreaming] = useState(false);
  const abortRef = useRef(null);
  const draftRef = useRef(null);
  const initialConversationIdRef = useRef(conversationId);
  const assistantEnabled = Boolean(features?.chatbot);
  const authSatisfied = !auth?.required || Boolean(auth?.authenticated);

  const healthQuery = useQuery({
    queryKey: ['assistant-health', assistantEnabled, authSatisfied],
    queryFn: checkHealth,
    enabled: assistantEnabled && authSatisfied,
    retry: 1,
    staleTime: 30_000,
    refetchInterval: 30_000,
  });

  useEffect(() => {
    if (typeof window === 'undefined') return;
    if (conversationId) {
      window.localStorage.setItem(STORAGE_KEY, conversationId);
      return;
    }
    window.localStorage.removeItem(STORAGE_KEY);
  }, [conversationId]);

  const resetDraft = useCallback(() => {
    draftRef.current = null;
    setDraftMessage(null);
    setIsStreaming(false);
    abortRef.current = null;
  }, []);

  const updateDraft = useCallback((updater) => {
    const current = draftRef.current || buildStreamingDraft();
    const nextValue = typeof updater === 'function' ? updater(current) : updater;
    draftRef.current = nextValue;
    setDraftMessage(nextValue);
  }, []);

  const loadConversation = useCallback(async (targetConversationId) => {
    if (!targetConversationId) {
      setMessages([]);
      setConversationTitle('Assistant');
      return null;
    }

    setIsLoadingConversation(true);
    try {
      const conversation = await getConversation(targetConversationId);
      setConversationId(conversation.conversation_id);
      setConversationTitle(conversation.title || 'Assistant');
      setMessages(conversation.messages || []);
      return conversation;
    } catch (error) {
      if (error?.response?.status === 404) {
        setConversationId(null);
        setConversationTitle('Assistant');
        setMessages([]);
      }
      throw error;
    } finally {
      setIsLoadingConversation(false);
    }
  }, []);

  useEffect(() => {
    const initialConversationId = initialConversationIdRef.current;
    initialConversationIdRef.current = null;
    if (!initialConversationId) return;
    loadConversation(initialConversationId).catch(() => {});
  }, [loadConversation]);

  const ensureConversation = useCallback(async () => {
    if (conversationId) return conversationId;
    const conversation = await createConversation();
    setConversationId(conversation.conversation_id);
    setConversationTitle(conversation.title || 'Assistant');
    setMessages([]);
    return conversation.conversation_id;
  }, [conversationId]);

  const startNewConversation = useCallback(async () => {
    if (abortRef.current) {
      abortRef.current();
    }
    resetDraft();
    const conversation = await createConversation();
    setConversationId(conversation.conversation_id);
    setConversationTitle(conversation.title || 'Assistant');
    setMessages([]);
    return conversation.conversation_id;
  }, [resetDraft]);

  const sendMessage = useCallback(async (content) => {
    const normalizedContent = content.trim();
    if (!normalizedContent || isStreaming) {
      return;
    }

    const targetConversationId = await ensureConversation();
    const userMessage = {
      id: `user-${Date.now()}`,
      role: 'user',
      content: normalizedContent,
      created_at: new Date().toISOString(),
    };

    setMessages((previous) => [...previous, userMessage]);
    setConversationTitle((current) => (
      !current || current === 'Assistant' || current === 'New Conversation'
        ? summarizeTitle(normalizedContent)
        : current
    ));
    setIsStreaming(true);
    updateDraft(buildStreamingDraft());

    abortRef.current = sendMessageStream(
      targetConversationId,
      normalizedContent,
      (chunk) => {
        switch (chunk.type) {
          case 'content':
            updateDraft((previous) => ({
              ...previous,
              content: `${previous.content || ''}${chunk.content || ''}`,
            }));
            break;
          case 'tool_call':
            updateDraft((previous) => ({
              ...previous,
              tool_calls: [
                ...(previous.tool_calls || []),
                {
                  type: 'call',
                  tool: chunk.tool,
                  params: chunk.params || {},
                },
              ],
            }));
            break;
          case 'tool_result':
            updateDraft((previous) => ({
              ...previous,
              tool_calls: [
                ...(previous.tool_calls || []),
                {
                  type: 'result',
                  tool: chunk.tool,
                  status: chunk.status,
                  result: chunk.result,
                },
              ],
            }));
            break;
          case 'done': {
            const finalMessage = chunk.message || {
              ...draftRef.current,
              content: draftRef.current?.content || 'Response complete.',
              isStreaming: false,
            };
            finalMessage.tool_calls = chunk.tool_calls || finalMessage.tool_calls || [];
            finalMessage.source_references = chunk.references || finalMessage.source_references || [];
            finalMessage.isStreaming = false;
            setMessages((previous) => [...previous, finalMessage]);
            if (chunk.message?.created_at) {
              setConversationTitle((currentTitle) => currentTitle || 'Assistant');
            }
            resetDraft();
            break;
          }
          case 'error':
            setMessages((previous) => [
              ...previous,
              {
                id: `assistant-error-${Date.now()}`,
                role: 'assistant',
                content: `Error: ${chunk.error}`,
                created_at: new Date().toISOString(),
                isError: true,
              },
            ]);
            resetDraft();
            break;
          default:
            break;
        }
      },
      (error) => {
        setMessages((previous) => [
          ...previous,
          {
            id: `assistant-error-${Date.now()}`,
            role: 'assistant',
            content: `Connection error: ${error.message}`,
            created_at: new Date().toISOString(),
            isError: true,
          },
        ]);
        resetDraft();
      },
      () => {}
    );
  }, [ensureConversation, isStreaming, resetDraft, updateDraft]);

  const displayedMessages = useMemo(() => {
    if (!draftMessage) return messages;
    return [...messages, draftMessage];
  }, [draftMessage, messages]);

  const value = useMemo(() => ({
    assistantHealth: healthQuery.data || {
      status: 'loading',
      available: false,
      streaming: true,
      popup_enabled: false,
      model: null,
      detail: null,
    },
    assistantHealthQuery: healthQuery,
    conversationId,
    conversationTitle,
    displayedMessages,
    messages,
    isLoadingConversation,
    isStreaming,
    ensureConversation,
    loadConversation,
    sendMessage,
    startNewConversation,
  }), [
    conversationId,
    conversationTitle,
    displayedMessages,
    ensureConversation,
    healthQuery,
    isLoadingConversation,
    isStreaming,
    loadConversation,
    messages,
    sendMessage,
    startNewConversation,
  ]);

  return (
    <AssistantChatContext.Provider value={value}>
      {children}
    </AssistantChatContext.Provider>
  );
}

export function useAssistantChat() {
  const context = useContext(AssistantChatContext);
  if (!context) {
    throw new Error('useAssistantChat must be used within AssistantChatProvider');
  }
  return context;
}

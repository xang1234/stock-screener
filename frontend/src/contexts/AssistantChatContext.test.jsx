import { act, fireEvent, screen, waitFor } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { AssistantChatProvider, useAssistantChat } from './AssistantChatContext';
import { renderWithProviders } from '../test/renderWithProviders';

const checkHealth = vi.fn();
const createConversation = vi.fn();
const getConversation = vi.fn();
const sendMessageStream = vi.fn();

vi.mock('../api/assistant', () => ({
  checkHealth: (...args) => checkHealth(...args),
  createConversation: (...args) => createConversation(...args),
  getConversation: (...args) => getConversation(...args),
  sendMessageStream: (...args) => sendMessageStream(...args),
}));

vi.mock('./RuntimeContext', () => ({
  useRuntime: () => ({
    auth: {
      required: true,
      authenticated: true,
    },
    features: {
      chatbot: true,
    },
  }),
}));

function AssistantConsumer() {
  const {
    assistantHealth,
    conversationId,
    conversationTitle,
    displayedMessages,
    isStreaming,
    sendMessage,
  } = useAssistantChat();
  const latestMessage = displayedMessages[displayedMessages.length - 1];
  const latestToolArgs = latestMessage?.tool_calls?.[0]?.args ?? null;

  return (
    <div>
      <div data-testid="assistant-health">{assistantHealth.available ? 'online' : 'offline'}</div>
      <div data-testid="conversation-id">{conversationId || 'none'}</div>
      <div data-testid="conversation-title">{conversationTitle}</div>
      <div data-testid="message-log">
        {displayedMessages.map((message) => `${message.role}:${message.content}`).join('|')}
      </div>
      <div data-testid="streaming-state">{isStreaming ? 'streaming' : 'idle'}</div>
      <div data-testid="tool-args">{latestToolArgs ? JSON.stringify(latestToolArgs) : 'none'}</div>
      <button type="button" onClick={() => sendMessage('What do you think about NVDA?')}>
        Send
      </button>
    </div>
  );
}

describe('AssistantChatContext', () => {
  beforeEach(() => {
    window.localStorage.clear();
    checkHealth.mockReset();
    createConversation.mockReset();
    getConversation.mockReset();
    sendMessageStream.mockReset();

    checkHealth.mockResolvedValue({
      status: 'ok',
      available: true,
      streaming: true,
      popup_enabled: true,
      model: 'hermes-test',
      detail: null,
    });
    createConversation.mockResolvedValue({
      id: 1,
      conversation_id: 'conv-1',
      title: 'Assistant',
      created_at: '2026-04-09T00:00:00Z',
      updated_at: '2026-04-09T00:00:00Z',
      is_active: true,
      message_count: 0,
    });
    getConversation.mockResolvedValue({
      id: 1,
      conversation_id: 'conv-1',
      title: 'Assistant',
      created_at: '2026-04-09T00:00:00Z',
      updated_at: '2026-04-09T00:00:00Z',
      is_active: true,
      message_count: 0,
      messages: [],
    });
  });

  it('creates a conversation lazily and streams assistant responses', async () => {
    sendMessageStream.mockImplementation((_conversationId, _content, onChunk) => {
      onChunk({ type: 'content', content: 'NVDA still leads the AI cohort [1].' });
      onChunk({
        type: 'done',
        message: {
          id: 2,
          conversation_id: 'conv-1',
          role: 'assistant',
          content: 'NVDA still leads the AI cohort [1].',
          created_at: '2026-04-09T00:01:00Z',
          source_references: [
            {
              reference_number: 1,
              type: 'internal',
              title: 'Feature run snapshot',
              url: '/stocks/NVDA',
              section: 'As of 2026-04-09',
              snippet: 'Latest scan posture.',
            },
          ],
        },
      });
      return vi.fn();
    });

    renderWithProviders(
      <AssistantChatProvider>
        <AssistantConsumer />
      </AssistantChatProvider>,
    );

    await waitFor(() => expect(screen.getByTestId('assistant-health')).toHaveTextContent('online'));

    fireEvent.click(screen.getByRole('button', { name: 'Send' }));

    await waitFor(() => {
      expect(createConversation).toHaveBeenCalledTimes(1);
      expect(sendMessageStream).toHaveBeenCalledWith(
        'conv-1',
        'What do you think about NVDA?',
        expect.any(Function),
        expect.any(Function),
        expect.any(Function),
      );
    });

    await waitFor(() => {
      expect(screen.getByTestId('conversation-id')).toHaveTextContent('conv-1');
      expect(screen.getByTestId('conversation-title')).toHaveTextContent('What do you think about NVDA?');
      expect(screen.getByTestId('message-log')).toHaveTextContent('user:What do you think about NVDA?');
      expect(screen.getByTestId('message-log')).toHaveTextContent('assistant:NVDA still leads the AI cohort [1].');
    });
  });

  it('restores the stored conversation on mount', async () => {
    window.localStorage.setItem('assistant_conversation_id', 'conv-restore');
    getConversation.mockResolvedValueOnce({
      id: 2,
      conversation_id: 'conv-restore',
      title: 'Restored assistant thread',
      created_at: '2026-04-09T00:00:00Z',
      updated_at: '2026-04-09T00:05:00Z',
      is_active: true,
      message_count: 2,
      messages: [
        {
          id: 9,
          conversation_id: 'conv-restore',
          role: 'user',
          content: 'How does breadth look?',
          created_at: '2026-04-09T00:04:00Z',
        },
        {
          id: 10,
          conversation_id: 'conv-restore',
          role: 'assistant',
          content: 'Breadth remains constructive.',
          created_at: '2026-04-09T00:05:00Z',
        },
      ],
    });

    renderWithProviders(
      <AssistantChatProvider>
        <AssistantConsumer />
      </AssistantChatProvider>,
    );

    await waitFor(() => {
      expect(getConversation).toHaveBeenCalledWith('conv-restore');
      expect(screen.getByTestId('conversation-id')).toHaveTextContent('conv-restore');
      expect(screen.getByTestId('conversation-title')).toHaveTextContent('Restored assistant thread');
      expect(screen.getByTestId('message-log')).toHaveTextContent('user:How does breadth look?');
      expect(screen.getByTestId('message-log')).toHaveTextContent('assistant:Breadth remains constructive.');
    });
  });

  it('keeps live tool-call args visible while streaming', async () => {
    sendMessageStream.mockImplementation((_conversationId, _content, onChunk) => {
      onChunk({
        type: 'tool_call',
        tool: 'stock_snapshot',
        params: { symbol: 'NVDA' },
      });
      return vi.fn();
    });

    renderWithProviders(
      <AssistantChatProvider>
        <AssistantConsumer />
      </AssistantChatProvider>,
    );

    await waitFor(() => expect(screen.getByTestId('assistant-health')).toHaveTextContent('online'));

    fireEvent.click(screen.getByRole('button', { name: 'Send' }));

    await waitFor(() => {
      expect(screen.getByTestId('tool-args')).toHaveTextContent('"symbol":"NVDA"');
      expect(screen.getByTestId('streaming-state')).toHaveTextContent('streaming');
    });
  });

  it('resets the draft when the stream closes without a terminal event', async () => {
    let onDoneCallback;
    sendMessageStream.mockImplementation((_conversationId, _content, onChunk, _onError, onDone) => {
      onChunk({ type: 'content', content: 'Partial answer' });
      onDoneCallback = onDone;
      return vi.fn();
    });

    renderWithProviders(
      <AssistantChatProvider>
        <AssistantConsumer />
      </AssistantChatProvider>,
    );

    await waitFor(() => expect(screen.getByTestId('assistant-health')).toHaveTextContent('online'));

    fireEvent.click(screen.getByRole('button', { name: 'Send' }));

    await waitFor(() => expect(screen.getByTestId('streaming-state')).toHaveTextContent('streaming'));

    await act(async () => {
      onDoneCallback();
    });

    await waitFor(() => {
      expect(screen.getByTestId('streaming-state')).toHaveTextContent('idle');
      expect(screen.getByTestId('message-log')).toHaveTextContent('assistant:Partial answer');
      expect(screen.getByTestId('message-log')).toHaveTextContent('Stream ended unexpectedly before the assistant completed its reply.');
    });
  });
});

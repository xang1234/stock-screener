import { fireEvent, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import { renderWithProviders } from '../../test/renderWithProviders';
import AssistantChat from './AssistantChat';

const mockUseAssistantChat = vi.fn();

vi.mock('../../contexts/AssistantChatContext', () => ({
  useAssistantChat: () => mockUseAssistantChat(),
}));

describe('AssistantChat', () => {
  it('does not create a conversation just by mounting', () => {
    const ensureConversation = vi.fn();
    mockUseAssistantChat.mockReturnValue({
      assistantHealth: {
        available: true,
        detail: null,
      },
      conversationTitle: 'Assistant',
      displayedMessages: [],
      ensureConversation,
      isLoadingConversation: false,
      isStreaming: false,
      sendMessage: vi.fn(),
      startNewConversation: vi.fn(),
    });

    renderWithProviders(<AssistantChat />);

    expect(ensureConversation).not.toHaveBeenCalled();
  });

  it('sends the typed message through the assistant context', async () => {
    const sendMessage = vi.fn().mockResolvedValue(undefined);
    mockUseAssistantChat.mockReturnValue({
      assistantHealth: {
        available: true,
        detail: null,
      },
      conversationTitle: 'Assistant',
      displayedMessages: [],
      isLoadingConversation: false,
      isStreaming: false,
      sendMessage,
      startNewConversation: vi.fn(),
    });

    renderWithProviders(<AssistantChat />);

    fireEvent.change(
      screen.getByPlaceholderText(/Ask about scans, themes, breadth/),
      { target: { value: 'Review NVDA' } },
    );
    fireEvent.click(screen.getByRole('button', { name: 'Send' }));

    expect(sendMessage).toHaveBeenCalledWith('Review NVDA');
  });
});

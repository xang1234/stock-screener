import { beforeEach, describe, expect, it, vi } from 'vitest';

const { mockApiClient } = vi.hoisted(() => ({
  mockApiClient: {
    defaults: {
      baseURL: 'https://api.example.com/api',
      withCredentials: true,
    },
    post: vi.fn(),
    get: vi.fn(),
    delete: vi.fn(),
    patch: vi.fn(),
  },
}));

vi.mock('./client', () => ({
  default: mockApiClient,
}));

import { sendMessageStream } from './chatbot';

describe('chatbot streaming api', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('includes credentials on streaming fetch requests', async () => {
    const fetchSpy = vi.fn().mockResolvedValue({
      ok: true,
      body: {
        getReader: () => ({
          read: vi.fn().mockResolvedValue({ done: true, value: undefined }),
        }),
      },
    });
    vi.stubGlobal('fetch', fetchSpy);

    sendMessageStream(
      'conv-123',
      'hello world',
      ['read_url'],
      true,
      vi.fn(),
      vi.fn(),
      vi.fn(),
    );

    await vi.waitFor(() => {
      expect(fetchSpy).toHaveBeenCalledWith(
        'https://api.example.com/api/v1/chatbot/conversations/conv-123/messages',
        expect.objectContaining({
          method: 'POST',
          credentials: 'include',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            content: 'hello world',
            enabled_tools: ['read_url'],
            research_mode: true,
          }),
        }),
      );
    });
  });
});

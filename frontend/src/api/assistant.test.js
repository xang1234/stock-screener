import { beforeEach, describe, expect, it, vi } from 'vitest';

const { mockApiClient, mockNotifyUnauthorizedResponse } = vi.hoisted(() => ({
  mockApiClient: {
    defaults: {
      baseURL: 'https://api.example.com/api',
      withCredentials: true,
    },
    post: vi.fn(),
    get: vi.fn(),
  },
  mockNotifyUnauthorizedResponse: vi.fn(),
}));

vi.mock('./client', () => ({
  default: mockApiClient,
  notifyUnauthorizedResponse: mockNotifyUnauthorizedResponse,
}));

import { sendMessageStream } from './assistant';

describe('assistant streaming api', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('includes credentials on assistant streaming fetch requests', async () => {
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
      vi.fn(),
      vi.fn(),
      vi.fn(),
    );

    await vi.waitFor(() => {
      expect(fetchSpy).toHaveBeenCalledWith(
        'https://api.example.com/api/v1/assistant/conversations/conv-123/messages',
        expect.objectContaining({
          method: 'POST',
          credentials: 'include',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            content: 'hello world',
          }),
        }),
      );
    });
  });

  it('notifies the unauthorized handler when assistant streaming returns 401', async () => {
    const onError = vi.fn();
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({
      ok: false,
      status: 401,
    }));

    sendMessageStream(
      'conv-123',
      'hello world',
      vi.fn(),
      onError,
      vi.fn(),
    );

    await vi.waitFor(() => {
      expect(mockNotifyUnauthorizedResponse).toHaveBeenCalledWith({
        status: 401,
        url: 'https://api.example.com/api/v1/assistant/conversations/conv-123/messages',
        headers: {
          'Content-Type': 'application/json',
        },
        error: expect.objectContaining({
          message: 'HTTP error! status: 401',
          status: 401,
        }),
      });
    });

    expect(onError).toHaveBeenCalledWith(
      expect.objectContaining({
        message: 'HTTP error! status: 401',
        status: 401,
      }),
    );
  });
});

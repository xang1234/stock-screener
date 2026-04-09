/**
 * API client for the Hermes-backed assistant endpoints.
 */
import apiClient, { notifyUnauthorizedResponse } from './client';

export const createConversation = async (title = null) => {
  const response = await apiClient.post('/v1/assistant/conversations', { title });
  return response.data;
};

export const listConversations = async (limit = 20, offset = 0) => {
  const response = await apiClient.get('/v1/assistant/conversations', {
    params: { limit, offset },
  });
  return response.data;
};

export const getConversation = async (conversationId) => {
  const response = await apiClient.get(`/v1/assistant/conversations/${conversationId}`);
  return response.data;
};

export const checkHealth = async () => {
  const response = await apiClient.get('/v1/assistant/health');
  return response.data;
};

export const previewWatchlistAdd = async ({ watchlist, symbols, reason = null }) => {
  const response = await apiClient.post('/v1/assistant/watchlist-add-preview', {
    watchlist,
    symbols,
    reason,
  });
  return response.data;
};

export const sendMessageStream = (conversationId, content, onChunk, onError, onDone) => {
  const controller = new AbortController();
  const baseUrl = apiClient.defaults.baseURL || '';
  const url = `${baseUrl}/v1/assistant/conversations/${conversationId}/messages`;
  const requestHeaders = {
    'Content-Type': 'application/json',
  };

  fetch(url, {
    method: 'POST',
    credentials: apiClient.defaults.withCredentials ? 'include' : 'same-origin',
    headers: requestHeaders,
    body: JSON.stringify({ content }),
    signal: controller.signal,
  })
    .then(async (response) => {
      if (!response.ok) {
        const error = new Error(`HTTP error! status: ${response.status}`);
        error.status = response.status;

        notifyUnauthorizedResponse({
          status: response.status,
          url,
          headers: requestHeaders,
          error,
        });
        throw error;
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) {
          if (onDone) onDone();
          break;
        }

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue;
          try {
            const data = JSON.parse(line.slice(6));
            if (onChunk) onChunk(data);
          } catch (error) {
            console.error('Failed to parse assistant SSE data', error);
          }
        }
      }
    })
    .catch((error) => {
      if (error.name !== 'AbortError' && onError) {
        onError(error);
      }
    });

  return () => controller.abort();
};

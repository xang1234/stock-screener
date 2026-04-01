/**
 * API client for Chatbot endpoints.
 */
import apiClient from './client';

/**
 * Create a new conversation.
 *
 * @param {string|null} title - Optional title for the conversation
 * @returns {Promise<Object>} Created conversation with conversation_id
 */
export const createConversation = async (title = null) => {
  const response = await apiClient.post('/v1/chatbot/conversations', { title });
  return response.data;
};

/**
 * List all conversations.
 *
 * @param {number} limit - Maximum conversations to return (default: 20)
 * @param {number} offset - Offset for pagination (default: 0)
 * @returns {Promise<Object>} List of conversations with total count
 */
export const listConversations = async (limit = 20, offset = 0) => {
  const response = await apiClient.get('/v1/chatbot/conversations', {
    params: { limit, offset }
  });
  return response.data;
};

/**
 * Get a conversation with its messages.
 *
 * @param {string} conversationId - Conversation UUID
 * @returns {Promise<Object>} Conversation with messages
 */
export const getConversation = async (conversationId) => {
  const response = await apiClient.get(`/v1/chatbot/conversations/${conversationId}`);
  return response.data;
};

/**
 * Delete a conversation.
 *
 * @param {string} conversationId - Conversation UUID
 * @returns {Promise<Object>} Deletion confirmation
 */
export const deleteConversation = async (conversationId) => {
  const response = await apiClient.delete(`/v1/chatbot/conversations/${conversationId}`);
  return response.data;
};

/**
 * Send a message and get a streaming response via SSE.
 *
 * @param {string} conversationId - Conversation UUID
 * @param {string} content - Message content
 * @param {string[]|null} enabledTools - Array of enabled tool names, or null for all tools
 * @param {boolean} researchMode - Enable deep research mode
 * @param {function} onChunk - Callback for each chunk: (chunk) => void
 * @param {function} onError - Callback for errors: (error) => void
 * @param {function} onDone - Callback when complete: () => void
 * @returns {function} Abort function to cancel the stream
 */
export const sendMessageStream = (conversationId, content, enabledTools, researchMode, onChunk, onError, onDone) => {
  const controller = new AbortController();

  // Make POST request with fetch for SSE support
  const baseUrl = apiClient.defaults.baseURL || '';
  const url = `${baseUrl}/v1/chatbot/conversations/${conversationId}/messages`;

  // Build request body
  const requestBody = { content };
  if (enabledTools !== null) {
    requestBody.enabled_tools = enabledTools;
  }
  if (researchMode) {
    requestBody.research_mode = true;
  }

  fetch(url, {
    method: 'POST',
    credentials: apiClient.defaults.withCredentials ? 'include' : 'same-origin',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(requestBody),
    signal: controller.signal,
  })
    .then(async (response) => {
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
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

        // Process complete SSE messages
        const lines = buffer.split('\n');
        buffer = lines.pop() || ''; // Keep incomplete line in buffer

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const data = JSON.parse(line.slice(6));
              if (onChunk) onChunk(data);
            } catch (e) {
              console.error('Failed to parse SSE data:', e);
            }
          }
        }
      }
    })
    .catch((error) => {
      if (error.name !== 'AbortError') {
        if (onError) onError(error);
      }
    });

  // Return abort function
  return () => controller.abort();
};

/**
 * Send a message and get a non-streaming response.
 *
 * @param {string} conversationId - Conversation UUID
 * @param {string} content - Message content
 * @returns {Promise<Object>} Complete response with message and metadata
 */
export const sendMessageSync = async (conversationId, content) => {
  const response = await apiClient.post(
    `/v1/chatbot/conversations/${conversationId}/messages/sync`,
    { content }
  );
  return response.data;
};

/**
 * Get message history for a conversation.
 *
 * @param {string} conversationId - Conversation UUID
 * @param {number} limit - Maximum messages to return (default: 50)
 * @param {number} offset - Offset for pagination (default: 0)
 * @returns {Promise<Array>} Array of messages
 */
export const getMessages = async (conversationId, limit = 50, offset = 0) => {
  const response = await apiClient.get(`/v1/chatbot/conversations/${conversationId}/messages`, {
    params: { limit, offset }
  });
  return response.data;
};

/**
 * Get available chatbot tools.
 *
 * @returns {Promise<Object>} List of available tools
 */
export const getTools = async () => {
  const response = await apiClient.get('/v1/chatbot/tools');
  return response.data;
};

/**
 * Check chatbot health status.
 *
 * @returns {Promise<Object>} Health check response
 */
export const checkHealth = async () => {
  const response = await apiClient.get('/v1/chatbot/health');
  return response.data;
};

/**
 * Update a conversation (rename, move to folder).
 *
 * @param {string} conversationId - Conversation UUID
 * @param {Object} updates - Updates to apply
 * @param {string} [updates.title] - New title
 * @param {number|null} [updates.folder_id] - Folder ID to move to (null to remove from folder)
 * @returns {Promise<Object>} Updated conversation
 */
export const updateConversation = async (conversationId, updates) => {
  const response = await apiClient.patch(`/v1/chatbot/conversations/${conversationId}`, updates);
  return response.data;
};

// ==================== Folder API ====================

/**
 * List all folders.
 *
 * @returns {Promise<Object>} List of folders with total count
 */
export const listFolders = async () => {
  const response = await apiClient.get('/v1/chatbot/folders');
  return response.data;
};

/**
 * Create a new folder.
 *
 * @param {string} name - Folder name
 * @param {number} [position] - Optional position for ordering
 * @returns {Promise<Object>} Created folder
 */
export const createFolder = async (name, position = null) => {
  const response = await apiClient.post('/v1/chatbot/folders', { name, position });
  return response.data;
};

/**
 * Update a folder.
 *
 * @param {number} folderId - Folder ID
 * @param {Object} updates - Updates to apply
 * @param {string} [updates.name] - New name
 * @param {number} [updates.position] - New position
 * @param {boolean} [updates.is_collapsed] - Collapsed state
 * @returns {Promise<Object>} Updated folder
 */
export const updateFolder = async (folderId, updates) => {
  const response = await apiClient.patch(`/v1/chatbot/folders/${folderId}`, updates);
  return response.data;
};

/**
 * Delete a folder.
 *
 * @param {number} folderId - Folder ID
 * @returns {Promise<Object>} Deletion confirmation
 */
export const deleteFolder = async (folderId) => {
  const response = await apiClient.delete(`/v1/chatbot/folders/${folderId}`);
  return response.data;
};

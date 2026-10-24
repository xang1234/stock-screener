/**
 * API client for User-defined Watchlists feature.
 * Handles CRUD operations for watchlists and items.
 */
import apiClient from './client';

const BASE_PATH = '/v1/user-watchlists';

// ================= Watchlists =================

/**
 * Get all user watchlists.
 * @returns {Promise<Object>} List of watchlists
 */
export const getWatchlists = async () => {
  const response = await apiClient.get(BASE_PATH);
  return response.data;
};

/**
 * Get watchlist memberships for one or more symbols.
 * @param {string[]} symbols - Symbols whose memberships should be returned
 * @returns {Promise<Object>} Map of normalized symbols to watchlist IDs
 */
export const getWatchlistMemberships = async (symbols) => {
  const response = await apiClient.get(`${BASE_PATH}/memberships`, {
    params: { symbols: symbols.join(',') },
  });
  return response.data;
};

/**
 * Create a new watchlist.
 * @param {Object} watchlistData - Watchlist data { name, description?, color? }
 * @returns {Promise<Object>} Created watchlist
 */
export const createWatchlist = async (watchlistData) => {
  const response = await apiClient.post(BASE_PATH, watchlistData);
  return response.data;
};

/**
 * Update a watchlist.
 * @param {number} watchlistId - The watchlist ID
 * @param {Object} updates - Fields to update
 * @returns {Promise<Object>} Updated watchlist
 */
export const updateWatchlist = async (watchlistId, updates) => {
  const response = await apiClient.put(`${BASE_PATH}/${watchlistId}`, updates);
  return response.data;
};

/**
 * Delete a watchlist (cascades to items).
 * @param {number} watchlistId - The watchlist ID
 * @returns {Promise<Object>} Deletion confirmation
 */
export const deleteWatchlist = async (watchlistId) => {
  const response = await apiClient.delete(`${BASE_PATH}/${watchlistId}`);
  return response.data;
};

/**
 * Reorder watchlists.
 * @param {number[]} watchlistIds - Array of watchlist IDs in new order
 * @returns {Promise<Object>} Reorder confirmation
 */
export const reorderWatchlists = async (watchlistIds) => {
  const response = await apiClient.put(`${BASE_PATH}/reorder`, {
    watchlist_ids: watchlistIds,
  });
  return response.data;
};

/**
 * Get complete watchlist data with items, sparklines, and price changes.
 * @param {number} watchlistId - The watchlist ID
 * @returns {Promise<Object>} Watchlist data with market info
 */
export const getWatchlistData = async (watchlistId) => {
  const response = await apiClient.get(`${BASE_PATH}/${watchlistId}/data`);
  return response.data;
};

/**
 * Get a watchlist's symbols in display order, without market data.
 * @param {number} watchlistId - The watchlist ID
 * @returns {Promise<Object>} { symbols }
 */
export const getWatchlistSymbols = async (watchlistId) => {
  const response = await apiClient.get(`${BASE_PATH}/${watchlistId}/symbols`);
  return response.data;
};

export const getWatchlistStewardship = async (watchlistId, profile, asOfDate) => {
  const response = await apiClient.get(`${BASE_PATH}/${watchlistId}/stewardship`, {
    params: {
      ...(profile ? { profile } : {}),
      ...(asOfDate ? { as_of_date: asOfDate } : {}),
    },
  });
  return response.data;
};

// ================= Items =================

/**
 * Add a stock to a watchlist.
 * @param {number} watchlistId - The watchlist ID
 * @param {Object} itemData - Item data { symbol, display_name?, notes? }
 * @returns {Promise<Object>} Created item
 */
export const addItem = async (watchlistId, itemData) => {
  const response = await apiClient.post(
    `${BASE_PATH}/${watchlistId}/items`,
    itemData
  );
  return response.data;
};

/**
 * Add multiple stocks to a watchlist at once.
 * @param {number} watchlistId - The watchlist ID
 * @param {string[]} symbols - Array of stock symbols
 * @returns {Promise<Object[]>} Array of created items
 */
export const bulkAddItems = async (watchlistId, symbols) => {
  const response = await apiClient.post(
    `${BASE_PATH}/${watchlistId}/items/bulk`,
    { symbols }
  );
  return response.data;
};

/**
 * Import pasted text or CSV content into a watchlist.
 * @param {number} watchlistId - The watchlist ID
 * @param {Object} payload - Import payload { content, format? }
 * @returns {Promise<Object>} Partial-success import result
 */
export const importItems = async (watchlistId, payload) => {
  const response = await apiClient.post(
    `${BASE_PATH}/${watchlistId}/items/import`,
    payload
  );
  return response.data;
};

/**
 * Remove an item from a watchlist.
 * @param {number} itemId - The item ID
 * @returns {Promise<Object>} Deletion confirmation
 */
export const removeItem = async (itemId) => {
  const response = await apiClient.delete(`${BASE_PATH}/items/${itemId}`);
  return response.data;
};

/**
 * Reorder items within a watchlist.
 * @param {number} watchlistId - The watchlist ID
 * @param {number[]} itemIds - Array of item IDs in new order
 * @returns {Promise<Object>} Reorder confirmation
 */
export const reorderItems = async (watchlistId, itemIds) => {
  const response = await apiClient.put(
    `${BASE_PATH}/${watchlistId}/items/reorder`,
    { item_ids: itemIds }
  );
  return response.data;
};

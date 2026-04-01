/**
 * Axios API client for Stock Scanner backend
 */
import axios from 'axios';

// API base URL defaults to '/api' so the packaged desktop app can be served by FastAPI.
// Vite development uses a local proxy to forward '/api' to the backend.
const API_BASE_URL = import.meta.env.VITE_API_URL || '/api';
const DEFAULT_API_TIMEOUT_MS = 30000;
const THEMES_API_TIMEOUT_MS = 300000;
let unauthorizedResponseHandler = null;

const isThemesApiUrl = (url) => {
  if (!url || typeof url !== 'string') return false;
  return /(^|\/)v1\/themes(?:\/|$)/.test(url);
};

const isAuthUrl = (url) => {
  if (!url || typeof url !== 'string') return false;
  return /(^|\/)v1\/auth(?:\/|$)/.test(url);
};

const getHeaderValue = (headers, name) => {
  if (!headers) return undefined;
  if (typeof headers.get === 'function') {
    return headers.get(name);
  }
  const target = String(name).toLowerCase();
  return Object.entries(headers).find(([key]) => key.toLowerCase() === target)?.[1];
};

export const setUnauthorizedResponseHandler = (handler) => {
  unauthorizedResponseHandler = typeof handler === 'function' ? handler : null;
};

// Create axios instance with default config
const apiClient = axios.create({
  baseURL: API_BASE_URL,
  withCredentials: true,
  timeout: DEFAULT_API_TIMEOUT_MS,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor for logging (development only)
apiClient.interceptors.request.use(
  (config) => {
    if (isThemesApiUrl(config?.url)) {
      const currentTimeout = Number(config?.timeout);
      if (!Number.isFinite(currentTimeout) || currentTimeout < THEMES_API_TIMEOUT_MS) {
        config.timeout = THEMES_API_TIMEOUT_MS;
      }
    }

    if (import.meta.env.DEV) {
      console.log(`API Request: ${config.method.toUpperCase()} ${config.url}`);
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Response interceptor for error handling
apiClient.interceptors.response.use(
  (response) => {
    return response;
  },
  (error) => {
    if (error.response) {
      // Server responded with error status
      console.error('API Error Response:', {
        status: error.response.status,
        data: error.response.data,
        url: error.config.url,
      });
      if (
        error.response.status === 401
        && unauthorizedResponseHandler
        && !isAuthUrl(error.config?.url)
        && !getHeaderValue(error.config?.headers, 'X-Admin-Key')
      ) {
        unauthorizedResponseHandler(error);
      }
    } else if (error.request) {
      // Request made but no response
      console.error('API No Response:', error.request);
    } else {
      // Error in request setup
      console.error('API Request Error:', error.message);
    }
    return Promise.reject(error);
  }
);

export default apiClient;

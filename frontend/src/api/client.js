/**
 * Axios API client for Stock Scanner backend
 */
import axios from 'axios';

// API base URL defaults to '/api' so the packaged desktop app can be served by FastAPI.
// Vite development uses a local proxy to forward '/api' to the backend.
const API_BASE_URL = import.meta.env.VITE_API_URL || '/api';
const DEFAULT_API_TIMEOUT_MS = 30000;
const THEMES_API_TIMEOUT_MS = 300000;

const isThemesApiUrl = (url) => {
  if (!url || typeof url !== 'string') return false;
  return /(^|\/)v1\/themes(?:\/|$)/.test(url);
};

// Create axios instance with default config
const apiClient = axios.create({
  baseURL: API_BASE_URL,
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

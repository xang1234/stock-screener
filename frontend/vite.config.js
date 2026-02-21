import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  test: {
    environment: 'jsdom',
    globals: true,
    setupFiles: './src/test/setup.js',
    css: false,
    include: ['src/**/*.test.{js,jsx}'],
  },
  server: {
    port: 5173,
    open: true,
  },
  build: {
    // Manual chunks for better caching and smaller initial bundle
    rollupOptions: {
      output: {
        manualChunks: {
          // Core React ecosystem
          'react-vendor': ['react', 'react-dom', 'react-router-dom'],
          // MUI components (large library)
          'mui-vendor': ['@mui/material', '@mui/icons-material'],
          // Data visualization
          'charts-vendor': ['recharts', 'lightweight-charts'],
          // React Query for data fetching
          'query-vendor': ['@tanstack/react-query', '@tanstack/react-virtual'],
        },
      },
    },
    // Increase chunk size warning limit
    chunkSizeWarningLimit: 1000,
    // Enable source maps for production debugging
    sourcemap: false,
  },
  // Optimize dependencies
  optimizeDeps: {
    include: [
      'react',
      'react-dom',
      'react-router-dom',
      '@mui/material',
      '@tanstack/react-query',
      '@tanstack/react-virtual',
      'recharts',
    ],
  },
});

import { defineConfig, loadEnv } from 'vite';
import react from '@vitejs/plugin-react';
import { fileURLToPath } from 'node:url';

import { optionsValidatorsPlugin } from './build/optionsValidatorsPlugin.js';

import { SITE_DESCRIPTION as STATIC_DESCRIPTION, SITE_NAME, SITE_TITLE as STATIC_TITLE } from './src/static/siteMeta.js';

// Search and link-preview tags for the public static demo only; the self-hosted
// app keeps its plain index.html. og:image must be absolute, so it needs VITE_SITE_URL.
const staticSiteMetaPlugin = (siteUrl) => ({
  name: 'static-site-meta',
  transformIndexHtml(html) {
    const meta = [
      ['name', 'description', STATIC_DESCRIPTION],
      ['property', 'og:type', 'website'],
      ['property', 'og:site_name', SITE_NAME],
      ['property', 'og:title', STATIC_TITLE],
      ['property', 'og:description', STATIC_DESCRIPTION],
      ['name', 'twitter:card', 'summary_large_image'],
      ...(siteUrl ? [
        ['property', 'og:url', siteUrl],
        ['property', 'og:image', `${siteUrl}og-image.png`],
        ['property', 'og:image:width', '1200'],
        ['property', 'og:image:height', '630'],
      ] : []),
    ];
    return {
      html: html.replace(/<title>.*<\/title>/, `<title>${STATIC_TITLE}</title>`),
      tags: meta.map(([key, value, content]) => ({
        tag: 'meta', attrs: { [key]: value, content }, injectTo: 'head',
      })),
    };
  },
});

// https://vitejs.dev/config/
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, fileURLToPath(new URL('.', import.meta.url)), '');
  const configuredBase = env.VITE_BASE_PATH || '/';
  const base = configuredBase.endsWith('/') ? configuredBase : `${configuredBase}/`;
  const siteUrl = env.VITE_SITE_URL && (env.VITE_SITE_URL.endsWith('/') ? env.VITE_SITE_URL : `${env.VITE_SITE_URL}/`);

  return {
    base,
    plugins: [
      react(),
      optionsValidatorsPlugin(),
      ...(env.VITE_STATIC_SITE === 'true' ? [staticSiteMetaPlugin(siteUrl)] : []),
    ],
    test: {
      environment: 'jsdom',
      globals: true,
      setupFiles: './src/test/setup.js',
      css: false,
      testTimeout: 20000,
      hookTimeout: 20000,
      include: ['src/**/*.test.{js,jsx}'],
    },
    server: {
      port: 5173,
      open: true,
      proxy: {
        '/api': {
          target: 'http://127.0.0.1:8000',
          changeOrigin: true,
        },
      },
    },
    build: {
      rollupOptions: {
        output: {
          // MUI and recharts are left to Rollup: a named vendor chunk holds
          // every component any page imports and loads eagerly, while
          // automatic splitting ships only what the entry needs at boot.
          manualChunks: {
            // Core React ecosystem
            // Query's React adapters share React's initialization boundary.
            // Splitting these creates a circular vendor chunk in production.
            'react-vendor': [
              'react', 'react-dom', 'react-router-dom',
              '@tanstack/react-query', '@tanstack/react-virtual',
              '@tanstack/react-query-persist-client', '@tanstack/query-sync-storage-persister',
            ],
            // Only loads with the (lazy) candlestick chart modals.
            'price-charts-vendor': ['lightweight-charts'],
          },
          // Fold the many small chunks automatic splitting leaves behind into
          // chunks that always load with them; the boot set is unchanged and
          // a route needs about a third fewer requests (matters on HTTP/1.1).
          experimentalMinChunkSize: 20_000,
        },
      },
      // Increase chunk size warning limit
      chunkSizeWarningLimit: 1000,
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
  };
});

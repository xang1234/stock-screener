export const STATIC_SITE_MODE = String(import.meta.env.VITE_STATIC_SITE || '').toLowerCase() === 'true';

export const getStaticDataUrl = (relativePath = 'manifest.json') => {
  const normalizedPath = String(relativePath).replace(/^\/+/, '');
  return `${import.meta.env.BASE_URL}static-data/${normalizedPath}`;
};

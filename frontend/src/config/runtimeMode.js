export const STATIC_SITE_MODE = String(import.meta.env.VITE_STATIC_SITE || '').toLowerCase() === 'true';

// ``dataRoot`` is the manifest's ``data_root`` (``g/<generation>/``) for data
// files; the root manifest itself is always read with no data root.
export const getStaticDataUrl = (relativePath = 'manifest.json', dataRoot = '') => {
  const normalizedPath = String(relativePath).replace(/^\/+/, '');
  return `${import.meta.env.BASE_URL}static-data/${dataRoot}${normalizedPath}`;
};

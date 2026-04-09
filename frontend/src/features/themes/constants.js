export const SOURCE_TYPES = [
  { value: 'substack', label: 'Substack' },
  { value: 'twitter', label: 'Twitter' },
  { value: 'news', label: 'News' },
  { value: 'reddit', label: 'Reddit' },
];

export const DEFAULT_SOURCE_TYPES = SOURCE_TYPES.map((source) => source.value);

export const THEME_STATUS_COLORS = {
  trending: 'success',
  emerging: 'warning',
  active: 'info',
  fading: 'error',
  dormant: 'default',
};

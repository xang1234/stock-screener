export const MARKET_FLAGS = {
  US: '🇺🇸',
  HK: '🇭🇰',
  IN: '🇮🇳',
  JP: '🇯🇵',
  KR: '🇰🇷',
  TW: '🇹🇼',
};

export function marketFlag(code) {
  if (!code) return '';
  return MARKET_FLAGS[code.toUpperCase()] || '';
}

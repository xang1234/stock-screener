import { describe, it, expect } from 'vitest';
import { getCurrencyPrefix, formatLocalCurrency } from './formatUtils';

describe('getCurrencyPrefix', () => {
  it('returns the short symbol for each supported market currency', () => {
    expect(getCurrencyPrefix('USD')).toBe('$');
    expect(getCurrencyPrefix('HKD')).toBe('HK$');
    expect(getCurrencyPrefix('JPY')).toBe('¥');
    expect(getCurrencyPrefix('TWD')).toBe('NT$');
  });

  it('falls back to $ for null/undefined/unknown codes', () => {
    expect(getCurrencyPrefix(null)).toBe('$');
    expect(getCurrencyPrefix(undefined)).toBe('$');
    expect(getCurrencyPrefix('')).toBe('$');
    expect(getCurrencyPrefix('XYZ')).toBe('$');
  });
});

describe('formatLocalCurrency', () => {
  it('prefixes the value with the market currency symbol', () => {
    expect(formatLocalCurrency(410.5, 'HKD')).toBe('HK$410.50');
    expect(formatLocalCurrency(14200, 'JPY')).toBe('¥14200.00');
    expect(formatLocalCurrency(875, 'TWD')).toBe('NT$875.00');
    expect(formatLocalCurrency(189.5, 'USD')).toBe('$189.50');
  });

  it('returns "-" for null values so empty cells stay consistent', () => {
    expect(formatLocalCurrency(null, 'HKD')).toBe('-');
    expect(formatLocalCurrency(undefined, 'USD')).toBe('-');
  });

  it('respects the decimals arg', () => {
    expect(formatLocalCurrency(123.456, 'USD', 0)).toBe('$123');
    expect(formatLocalCurrency(123.456, 'USD', 3)).toBe('$123.456');
  });
});

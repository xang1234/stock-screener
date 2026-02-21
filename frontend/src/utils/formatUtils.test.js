import {
  formatPatternName,
  getScoreColor,
  formatLargeNumber,
  formatPercent,
  formatMarketCap,
  formatRatio,
} from './formatUtils';

describe('formatPatternName', () => {
  it('converts snake_case to Title Case', () => {
    expect(formatPatternName('three_weeks_tight')).toBe('Three Weeks Tight');
  });

  it('handles single word', () => {
    expect(formatPatternName('breakout')).toBe('Breakout');
  });

  it('returns "-" for null', () => {
    expect(formatPatternName(null)).toBe('-');
  });

  it('returns "-" for empty string', () => {
    expect(formatPatternName('')).toBe('-');
  });

  it('returns "-" for undefined', () => {
    expect(formatPatternName(undefined)).toBe('-');
  });
});

describe('getScoreColor', () => {
  it('returns green (#4caf50) for score >= 70', () => {
    expect(getScoreColor(70)).toBe('#4caf50');
    expect(getScoreColor(100)).toBe('#4caf50');
  });

  it('returns amber (#ff9800) for score >= 40 and < 70', () => {
    expect(getScoreColor(40)).toBe('#ff9800');
    expect(getScoreColor(69)).toBe('#ff9800');
  });

  it('returns red (#f44336) for score < 40', () => {
    expect(getScoreColor(39)).toBe('#f44336');
    expect(getScoreColor(0)).toBe('#f44336');
  });

  it('returns undefined for null', () => {
    expect(getScoreColor(null)).toBeUndefined();
  });

  it('returns undefined for undefined', () => {
    expect(getScoreColor(undefined)).toBeUndefined();
  });

  it('handles exact boundary at 70', () => {
    expect(getScoreColor(70)).toBe('#4caf50');
    expect(getScoreColor(69.9)).toBe('#ff9800');
  });
});

describe('formatLargeNumber', () => {
  it('formats trillions', () => {
    expect(formatLargeNumber(2.5e12)).toBe('2.5T');
  });

  it('formats billions', () => {
    expect(formatLargeNumber(1.23e9)).toBe('1.2B');
  });

  it('formats millions', () => {
    expect(formatLargeNumber(5.67e6)).toBe('5.7M');
  });

  it('formats thousands (0 decimal places)', () => {
    expect(formatLargeNumber(8500)).toBe('9K');
  });

  it('returns raw value below 1K', () => {
    expect(formatLargeNumber(500)).toBe('500');
  });

  it('returns "-" for null', () => {
    expect(formatLargeNumber(null)).toBe('-');
  });

  it('prepends prefix when provided', () => {
    expect(formatLargeNumber(1e9, '$')).toBe('$1.0B');
  });
});

describe('formatPercent', () => {
  it('formats positive with + sign', () => {
    expect(formatPercent(12.34)).toBe('+12.3%');
  });

  it('formats negative without extra sign', () => {
    expect(formatPercent(-5.67)).toBe('-5.7%');
  });

  it('formats zero with + sign', () => {
    expect(formatPercent(0)).toBe('+0.0%');
  });

  it('returns "-" for null', () => {
    expect(formatPercent(null)).toBe('-');
  });

  it('respects custom decimals', () => {
    expect(formatPercent(12.345, 2)).toBe('+12.35%');
  });
});

describe('formatMarketCap', () => {
  it('formats billions with 2 decimal places and $ prefix', () => {
    expect(formatMarketCap(3.456e9)).toBe('$3.46B');
  });

  it('formats trillions with 2 decimal places', () => {
    expect(formatMarketCap(1.1e12)).toBe('$1.10T');
  });

  it('returns "-" for null', () => {
    expect(formatMarketCap(null)).toBe('-');
  });

  it('formats sub-thousand with 2 decimal places', () => {
    expect(formatMarketCap(999)).toBe('$999.00');
  });
});

describe('formatRatio', () => {
  it('formats with 2 decimal places by default', () => {
    expect(formatRatio(3.14159)).toBe('3.14');
  });

  it('returns "-" for null', () => {
    expect(formatRatio(null)).toBe('-');
  });

  it('respects custom decimals', () => {
    expect(formatRatio(3.14159, 4)).toBe('3.1416');
  });
});

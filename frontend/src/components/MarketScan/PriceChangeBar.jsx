/**
 * Price Change Bar Component
 *
 * Renders a horizontal bar showing relative price change.
 * Mimics Google Sheets SPARKLINE bar chart formula:
 * - Center of bar = 0%
 * - Green bar extends RIGHT for positive values
 * - Red bar extends LEFT for negative values
 * - Scale based on min/max across all stocks in theme
 *
 * Displays:
 * - Percentage text (colored green/red)
 * - Relative bar below the text
 */
import { useMemo } from 'react';
import { Box, Typography, useTheme } from '@mui/material';

function PriceChangeBar({ value, min, max, width = 60, height = 24 }) {
  const theme = useTheme();

  // Background color - use theme's paper background to blend in perfectly
  const bgColor = theme.palette.background.paper;

  // Calculate bar properties based on Google Sheets formula logic
  const barData = useMemo(() => {
    if (value === null || value === undefined) {
      return { barLeft: 0, barWidth: 0, centerPercent: 50, color: 'transparent' };
    }

    // Total range spans from min (negative or 0) to max (positive or 0)
    const absMin = Math.abs(min);
    const absMax = Math.abs(max);
    const totalRange = absMin + absMax;

    if (totalRange === 0) {
      return { barLeft: 50, barWidth: 0, centerPercent: 50, color: 'transparent' };
    }

    // Center point is where 0 sits in the bar
    // Center position = |min| / totalRange (as percentage from left)
    const centerPercent = (absMin / totalRange) * 100;

    if (value >= 0) {
      // Green bar extends right from center
      const barPercent = (value / totalRange) * 100;
      return {
        barLeft: centerPercent,
        barWidth: barPercent,
        centerPercent,
        color: '#4caf50', // green
      };
    } else {
      // Red bar extends left from center
      const absValue = Math.abs(value);
      const barPercent = (absValue / totalRange) * 100;
      return {
        barLeft: centerPercent - barPercent,
        barWidth: barPercent,
        centerPercent,
        color: '#f44336', // red
      };
    }
  }, [value, min, max]);

  // Format percentage text
  const percentText = useMemo(() => {
    if (value === null || value === undefined) return '-';
    const sign = value >= 0 ? '+' : '';
    return `${sign}${value.toFixed(1)}%`;
  }, [value]);

  const textColor = useMemo(() => {
    if (value === null || value === undefined) return 'text.disabled';
    return value >= 0 ? '#4caf50' : '#f44336';
  }, [value]);

  return (
    <Box
      sx={{
        width,
        height,
        display: 'flex',
        flexDirection: 'row',
        alignItems: 'center',
        justifyContent: 'flex-start',
        gap: 0.5,
      }}
    >
      {/* Percentage text on the left */}
      <Typography
        sx={{
          fontSize: '0.65rem',
          fontWeight: 600,
          fontFamily: 'monospace',
          color: textColor,
          lineHeight: 1,
          minWidth: 38,
          textAlign: 'right',
        }}
      >
        {percentText}
      </Typography>

      {/* Bar on the right */}
      <Box
        sx={{
          flex: 1,
          height: 12,
          bgcolor: bgColor,
          borderRadius: 0.5,
          position: 'relative',
          overflow: 'hidden',
        }}
      >
        {value !== null && value !== undefined && (
          /* The colored bar - only red or green, no other colors */
          <Box
            sx={{
              position: 'absolute',
              left: `${barData.barLeft}%`,
              width: `${barData.barWidth}%`,
              height: '100%',
              bgcolor: barData.color,
              borderRadius: 0.5,
              transition: 'all 0.2s ease-in-out',
            }}
          />
        )}
      </Box>
    </Box>
  );
}

/**
 * Compact version showing just the value text with color
 */
export function PriceChangeText({ value, showSign = true }) {
  if (value === null || value === undefined) {
    return (
      <Typography variant="caption" color="text.disabled">
        -
      </Typography>
    );
  }

  const color = value >= 0 ? '#4caf50' : '#f44336';
  const sign = value >= 0 && showSign ? '+' : '';

  return (
    <Typography
      variant="caption"
      sx={{
        color,
        fontWeight: 600,
        fontSize: '0.7rem',
      }}
    >
      {sign}{value.toFixed(1)}%
    </Typography>
  );
}

export default PriceChangeBar;

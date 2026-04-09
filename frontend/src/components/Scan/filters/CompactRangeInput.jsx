import { useState, useEffect, useRef } from 'react';
import { Box, TextField, Typography } from '@mui/material';

/**
 * Compact min/max number input for filtering
 * Two small text fields with a dash separator
 */
function CompactRangeInput({
  label,
  minValue,
  maxValue,
  onChange,
  step = 1,
  minLimit,
  maxLimit,
  prefix = '',
  suffix = '',
  minOnly = false,
}) {
  const [localMin, setLocalMin] = useState(minValue ?? '');
  const [localMax, setLocalMax] = useState(maxValue ?? '');
  const debounceTimeoutRef = useRef();

  // Sync local state with props
  useEffect(() => {
    setLocalMin(minValue ?? '');
    setLocalMax(maxValue ?? '');
  }, [minValue, maxValue]);

  useEffect(() => () => {
    clearTimeout(debounceTimeoutRef.current);
  }, []);

  // Debounced onChange handler
  const debouncedOnChange = (min, max) => {
    clearTimeout(debounceTimeoutRef.current);
    debounceTimeoutRef.current = setTimeout(() => {
      const parsedMin = min === '' ? null : parseFloat(min);
      const parsedMax = max === '' ? null : parseFloat(max);
      // Convert NaN to null to prevent invalid filter values
      const validMin = (parsedMin !== null && !Number.isNaN(parsedMin)) ? parsedMin : null;
      const validMax = (parsedMax !== null && !Number.isNaN(parsedMax)) ? parsedMax : null;
      onChange({ min: validMin, max: validMax });
    }, 300);
  };

  const handleMinChange = (e) => {
    const value = e.target.value;
    setLocalMin(value);
    debouncedOnChange(value, localMax);
  };

  const handleMaxChange = (e) => {
    const value = e.target.value;
    setLocalMax(value);
    debouncedOnChange(localMin, value);
  };

  return (
    <Box sx={{ minWidth: 100 }}>
      <Typography
        variant="caption"
        color="text.secondary"
        sx={{ display: 'block', mb: 0.5, fontSize: '0.7rem' }}
      >
        {label}
      </Typography>
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
        <TextField
          size="small"
          type="number"
          value={localMin}
          onChange={handleMinChange}
          placeholder={minOnly ? '≥' : 'Min'}
          inputProps={{
            step,
            min: minLimit,
            max: maxLimit,
            style: { padding: '4px 6px', fontSize: '0.75rem' },
          }}
          sx={{
            width: minOnly ? 70 : 55,
            '& .MuiOutlinedInput-root': {
              height: 28,
            },
            '& input[type=number]': {
              MozAppearance: 'textfield',
            },
            '& input[type=number]::-webkit-outer-spin-button, & input[type=number]::-webkit-inner-spin-button': {
              WebkitAppearance: 'none',
              margin: 0,
            },
          }}
          InputProps={{
            startAdornment: prefix ? (
              <Typography variant="caption" sx={{ mr: 0.25, color: 'text.secondary' }}>
                {prefix}
              </Typography>
            ) : null,
            endAdornment: minOnly && suffix ? (
              <Typography variant="caption" sx={{ ml: 0.25, color: 'text.secondary' }}>
                {suffix}
              </Typography>
            ) : null,
          }}
        />
        {!minOnly && (
          <>
            <Typography variant="caption" color="text.secondary">
              -
            </Typography>
            <TextField
              size="small"
              type="number"
              value={localMax}
              onChange={handleMaxChange}
              placeholder="Max"
              inputProps={{
                step,
                min: minLimit,
                max: maxLimit,
                style: { padding: '4px 6px', fontSize: '0.75rem' },
              }}
              sx={{
                width: 55,
                '& .MuiOutlinedInput-root': {
                  height: 28,
                },
                '& input[type=number]': {
                  MozAppearance: 'textfield',
                },
                '& input[type=number]::-webkit-outer-spin-button, & input[type=number]::-webkit-inner-spin-button': {
                  WebkitAppearance: 'none',
                  margin: 0,
                },
              }}
              InputProps={{
                endAdornment: suffix ? (
                  <Typography variant="caption" sx={{ ml: 0.25, color: 'text.secondary' }}>
                    {suffix}
                  </Typography>
                ) : null,
              }}
            />
          </>
        )}
      </Box>
    </Box>
  );
}

export default CompactRangeInput;

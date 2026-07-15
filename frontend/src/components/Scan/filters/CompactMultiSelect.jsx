import { Box, Autocomplete, TextField, Typography, Chip, IconButton, Tooltip } from '@mui/material';
import AddIcon from '@mui/icons-material/Add';
import RemoveIcon from '@mui/icons-material/Remove';

/**
 * Compact multi-select autocomplete for Industry/Sector filters
 * Supports include/exclude mode toggle
 */
function CompactMultiSelect({
  label,
  values,
  options,
  onChange,
  placeholder = 'Select...',
  mode = 'include',
  onModeChange,
  showModeToggle = false,
  maxValues = null,
}) {
  const isExcludeMode = mode === 'exclude';
  const selectedValues = values || [];
  const hasSelectionLimit = Number.isInteger(maxValues) && maxValues > 0;
  const selectionLimitReached = hasSelectionLimit && selectedValues.length >= maxValues;
  const showSelectionLimit = hasSelectionLimit && (
    (options?.length ?? 0) > maxValues || selectionLimitReached
  );

  const handleModeToggle = (e) => {
    e.stopPropagation();
    if (onModeChange) {
      onModeChange(isExcludeMode ? 'include' : 'exclude');
    }
  };

  return (
    <Box sx={{ minWidth: 140 }}>
      <Box sx={{ display: 'flex', alignItems: 'center', mb: 0.5 }}>
        <Typography
          variant="caption"
          color="text.secondary"
          sx={{ fontSize: '0.7rem', flexGrow: 1 }}
        >
          {label}
        </Typography>
        {showModeToggle && (
          <Tooltip title={isExcludeMode ? 'Excluding selected (click to include)' : 'Including selected (click to exclude)'}>
            <IconButton
              size="small"
              onClick={handleModeToggle}
              sx={{
                p: 0,
                ml: 0.5,
                width: 18,
                height: 18,
                bgcolor: isExcludeMode ? 'error.main' : 'primary.main',
                color: 'white',
                '&:hover': {
                  bgcolor: isExcludeMode ? 'error.dark' : 'primary.dark',
                },
              }}
            >
              {isExcludeMode ? (
                <RemoveIcon sx={{ fontSize: 14 }} />
              ) : (
                <AddIcon sx={{ fontSize: 14 }} />
              )}
            </IconButton>
          </Tooltip>
        )}
      </Box>
      <Autocomplete
        multiple
        size="small"
        value={selectedValues}
        onChange={(event, newValue) => onChange(
          hasSelectionLimit ? newValue.slice(0, maxValues) : newValue,
        )}
        options={options || []}
        disableCloseOnSelect
        getOptionDisabled={(option) => (
          selectionLimitReached && !selectedValues.includes(option)
        )}
        renderInput={(params) => (
          <TextField
            {...params}
            placeholder={values?.length ? '' : placeholder}
            sx={{
              '& .MuiOutlinedInput-root': {
                minHeight: 28,
                padding: '2px 6px',
                fontSize: '0.75rem',
                ...(isExcludeMode && values?.length > 0 && {
                  borderColor: 'error.main',
                  '& fieldset': {
                    borderColor: 'error.light',
                  },
                  '&:hover fieldset': {
                    borderColor: 'error.main',
                  },
                  '&.Mui-focused fieldset': {
                    borderColor: 'error.main',
                  },
                }),
              },
              '& .MuiAutocomplete-input': {
                padding: '2px 4px !important',
                fontSize: '0.75rem',
              },
            }}
          />
        )}
        renderTags={(value, getTagProps) =>
          value.map((option, index) => (
            <Chip
              {...getTagProps({ index })}
              key={option}
              label={option}
              size="small"
              color={isExcludeMode ? 'error' : 'default'}
              sx={{
                height: 20,
                fontSize: '0.65rem',
                '& .MuiChip-label': {
                  px: 0.75,
                },
                '& .MuiChip-deleteIcon': {
                  fontSize: '0.8rem',
                },
              }}
            />
          ))
        }
        sx={{
          '& .MuiAutocomplete-tag': {
            margin: '1px',
          },
        }}
      />
      {showSelectionLimit && (
        <Typography
          variant="caption"
          color={selectionLimitReached ? 'primary.main' : 'text.secondary'}
          sx={{ display: 'block', mt: 0.25, fontSize: '0.65rem' }}
        >
          Up to {maxValues} values
        </Typography>
      )}
    </Box>
  );
}

export default CompactMultiSelect;

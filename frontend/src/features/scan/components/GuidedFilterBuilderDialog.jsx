import { useEffect, useMemo, useState } from 'react';
import {
  Alert,
  Autocomplete,
  Box,
  Button,
  Chip,
  Dialog,
  DialogActions,
  DialogContent,
  DialogTitle,
  Divider,
  FormControl,
  FormControlLabel,
  IconButton,
  InputLabel,
  MenuItem,
  Paper,
  Radio,
  RadioGroup,
  Select,
  Stack,
  Switch,
  TextField,
  Tooltip,
  Typography,
} from '@mui/material';
import AddIcon from '@mui/icons-material/Add';
import ContentCopyIcon from '@mui/icons-material/ContentCopy';
import DeleteOutlineIcon from '@mui/icons-material/DeleteOutline';
import {
  FILTER_FIELD_CATALOG,
  conditionLabel,
  fieldMeta,
  newCondition,
  validateExpression,
} from '../filterExpression';

const groupId = () => `setup-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`;

function clone(value) {
  return structuredClone(value);
}

function ConditionEditor({ condition, onChange, onDelete, valueOptions = [], fieldCatalog }) {
  const meta = fieldMeta(condition.field, fieldCatalog);
  const handleFieldChange = (field) => onChange(newCondition(field, fieldCatalog));

  return (
    <Paper variant="outlined" sx={{ p: 1, bgcolor: 'background.default' }}>
      <Stack direction={{ xs: 'column', md: 'row' }} spacing={1} alignItems={{ md: 'center' }}>
        <FormControl size="small" sx={{ minWidth: 210 }}>
          <InputLabel>Field</InputLabel>
          <Select
            label="Field"
            value={condition.field}
            onChange={(event) => handleFieldChange(event.target.value)}
          >
            {fieldCatalog.map((item) => (
              <MenuItem key={item.field} value={item.field}>
                <Box>
                  <Typography variant="body2">{item.label}</Typography>
                  <Typography variant="caption" color="text.secondary">{item.category}</Typography>
                </Box>
              </MenuItem>
            ))}
          </Select>
        </FormControl>

        {meta.type === 'range' && (
          <>
            <TextField
              size="small"
              label="Minimum"
              type="number"
              value={condition.min ?? ''}
              onChange={(event) => onChange({
                ...condition,
                min: event.target.value === '' ? null : Number(event.target.value),
              })}
              sx={{ width: 135 }}
            />
            <TextField
              size="small"
              label="Maximum"
              type="number"
              value={condition.max ?? ''}
              onChange={(event) => onChange({
                ...condition,
                max: event.target.value === '' ? null : Number(event.target.value),
              })}
              sx={{ width: 135 }}
            />
          </>
        )}

        {meta.type === 'categorical' && (
          <>
            <FormControl size="small" sx={{ minWidth: 115 }}>
              <InputLabel>Mode</InputLabel>
              <Select
                label="Mode"
                value={condition.mode}
                onChange={(event) => onChange({ ...condition, mode: event.target.value })}
              >
                <MenuItem value="include">Include</MenuItem>
                <MenuItem value="exclude">Exclude</MenuItem>
              </Select>
            </FormControl>
            <Autocomplete
              multiple
              freeSolo
              size="small"
              options={valueOptions}
              value={condition.values || []}
              onChange={(_event, values) => onChange({
                ...condition,
                values: values.map((value) => String(value).trim()).filter(Boolean),
              })}
              renderInput={(params) => (
                <TextField
                  {...params}
                  label="Values"
                  placeholder={valueOptions.length ? 'Choose or type a value' : 'Type a value, then press Enter'}
                />
              )}
              sx={{ minWidth: 260, flex: 1 }}
            />
          </>
        )}

        {meta.type === 'boolean' && (
          <FormControl size="small" sx={{ minWidth: 130 }}>
            <InputLabel>Value</InputLabel>
            <Select
              label="Value"
              value={condition.value ? 'yes' : 'no'}
              onChange={(event) => onChange({ ...condition, value: event.target.value === 'yes' })}
            >
              <MenuItem value="yes">Yes</MenuItem>
              <MenuItem value="no">No</MenuItem>
            </Select>
          </FormControl>
        )}

        {meta.type === 'text' && (
          <TextField
            size="small"
            fullWidth
            label="Contains"
            value={condition.pattern || ''}
            onChange={(event) => onChange({ ...condition, pattern: event.target.value })}
          />
        )}

        <Tooltip title="Remove rule">
          <IconButton size="small" onClick={onDelete} aria-label="Remove rule">
            <DeleteOutlineIcon fontSize="small" />
          </IconButton>
        </Tooltip>
      </Stack>
    </Paper>
  );
}

function SetupGroupCard({
  group,
  index,
  onChange,
  onDelete,
  onDuplicate,
  filterOptions,
  fieldCatalog,
}) {
  const updateCondition = (conditionIndex, condition) => {
    const conditions = [...group.conditions];
    conditions[conditionIndex] = condition;
    onChange({ ...group, conditions });
  };

  return (
    <Paper
      variant="outlined"
      sx={{ p: 1.5, opacity: group.enabled ? 1 : 0.68, borderColor: group.enabled ? 'primary.light' : 'divider' }}
    >
      <Stack direction={{ xs: 'column', sm: 'row' }} spacing={1} alignItems={{ sm: 'center' }}>
        <TextField
          size="small"
          label={`Setup ${index + 1} name`}
          value={group.name}
          inputProps={{ maxLength: 60 }}
          onChange={(event) => onChange({ ...group, name: event.target.value })}
          sx={{ minWidth: 230 }}
        />
        <FormControl size="small" sx={{ minWidth: 175 }}>
          <InputLabel>Rules in this setup</InputLabel>
          <Select
            label="Rules in this setup"
            value={group.match}
            onChange={(event) => onChange({ ...group, match: event.target.value })}
          >
            <MenuItem value="all">Match all rules</MenuItem>
            <MenuItem value="any">Match any rule</MenuItem>
          </Select>
        </FormControl>
        <FormControlLabel
          sx={{ ml: { sm: 'auto' } }}
          control={(
            <Switch
              checked={group.enabled}
              onChange={(event) => onChange({ ...group, enabled: event.target.checked })}
            />
          )}
          label={group.enabled ? 'Enabled' : 'Disabled'}
        />
        <Tooltip title="Duplicate setup">
          <IconButton size="small" onClick={onDuplicate} aria-label="Duplicate setup">
            <ContentCopyIcon fontSize="small" />
          </IconButton>
        </Tooltip>
        <Tooltip title="Delete setup">
          <IconButton size="small" color="error" onClick={onDelete} aria-label="Delete setup">
            <DeleteOutlineIcon fontSize="small" />
          </IconButton>
        </Tooltip>
      </Stack>

      <Stack spacing={1} sx={{ mt: 1.25 }}>
        {group.conditions.map((condition, conditionIndex) => (
          <ConditionEditor
            key={`${group.id}-${conditionIndex}`}
            condition={condition}
            onChange={(next) => updateCondition(conditionIndex, next)}
            onDelete={() => onChange({
              ...group,
              conditions: group.conditions.filter((_, itemIndex) => itemIndex !== conditionIndex),
            })}
            valueOptions={(
              condition.field === 'rating' ? filterOptions.ratings
                : condition.field === 'ibd_industry_group' ? filterOptions.ibdIndustries
                  : condition.field === 'gics_sector' ? filterOptions.gicsSectors
                    : condition.field === 'market' ? filterOptions.markets
                      : condition.field === 'se_pattern_primary' ? filterOptions.patterns
                        : []
            ) || []}
            fieldCatalog={fieldCatalog}
          />
        ))}
      </Stack>

      <Button
        size="small"
        startIcon={<AddIcon />}
        sx={{ mt: 1 }}
        disabled={group.conditions.length >= 20}
        onClick={() => onChange({
          ...group,
          conditions: [...group.conditions, newCondition('composite_score', fieldCatalog)],
        })}
      >
        Add rule
      </Button>

      {group.conditions.length > 0 && (
        <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 0.5 }}>
          Reads as: {group.conditions.map((condition) => conditionLabel(condition, fieldCatalog)).join(group.match === 'all' ? ' AND ' : ' OR ')}
        </Typography>
      )}
    </Paper>
  );
}

export default function GuidedFilterBuilderDialog({
  open,
  expression,
  onClose,
  onApply,
  filterOptions = {},
}) {
  const [draft, setDraft] = useState(() => clone(expression));
  const [showErrors, setShowErrors] = useState(false);

  useEffect(() => {
    if (open) {
      setDraft(clone(expression));
      setShowErrors(false);
    }
  }, [expression, open]);

  const fieldCatalog = filterOptions.filterCatalog?.length
    ? filterOptions.filterCatalog
    : FILTER_FIELD_CATALOG;
  const errors = useMemo(
    () => validateExpression(draft, fieldCatalog),
    [draft, fieldCatalog],
  );
  const guidedOptions = useMemo(() => ({
    ratings: filterOptions.ratings || [],
    ibdIndustries: filterOptions.ibdIndustries || [],
    gicsSectors: filterOptions.gicsSectors || [],
    markets: filterOptions.markets || ['US', 'HK', 'IN', 'JP', 'KR', 'TW', 'CN', 'CA', 'DE', 'SG', 'AU', 'MY'],
    patterns: filterOptions.patterns || [
      'cup_with_handle',
      'double_bottom',
      'high_tight_flag',
      'first_pullback',
      'vcp',
      'three_weeks_tight',
      'nr7_inside_day',
    ],
  }), [filterOptions]);
  const updateGroup = (index, group) => {
    const groups = [...draft.groups];
    groups[index] = group;
    setDraft({ ...draft, groups });
  };

  const addGroup = () => {
    const number = draft.groups.length + 1;
    setDraft({
      ...draft,
      groups: [
        ...draft.groups,
        {
          id: groupId(),
          name: `Setup ${number}`,
          match: 'all',
          enabled: true,
          conditions: [newCondition('composite_score', fieldCatalog)],
        },
      ],
    });
  };

  const handleApply = () => {
    setShowErrors(true);
    if (errors.length) return;
    onApply(draft);
  };

  return (
    <Dialog open={open} onClose={onClose} fullWidth maxWidth="lg">
      <DialogTitle>Build scan logic</DialogTitle>
      <DialogContent dividers>
        <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
          Keep non-negotiable rules in “Always require,” then describe alternate named setups.
          This makes the result list explainable without exposing raw boolean syntax.
        </Typography>

        <Paper variant="outlined" sx={{ p: 1.5, mb: 2, bgcolor: 'action.hover' }}>
          <Stack direction="row" spacing={1} alignItems="center" flexWrap="wrap">
            <Typography variant="subtitle2">Always require</Typography>
            <Chip size="small" label={`${draft.required.conditions.length} rules`} />
            <Typography variant="caption" color="text.secondary">
              Managed by the quick filters on the results page. Every matching stock must pass these rules.
            </Typography>
          </Stack>
          {draft.required.conditions.length > 0 && (
            <Typography variant="caption" sx={{ display: 'block', mt: 1 }}>
              {draft.required.conditions
                .map((condition) => conditionLabel(condition, fieldCatalog))
                .join(' AND ')}
            </Typography>
          )}
        </Paper>

        <Box sx={{ mb: 2 }}>
          <Typography variant="subtitle2">How should named setups combine?</Typography>
          <RadioGroup
            row
            value={draft.group_join}
            onChange={(event) => setDraft({ ...draft, group_join: event.target.value })}
          >
            <FormControlLabel value="any" control={<Radio size="small" />} label="Match any setup (broader)" />
            <FormControlLabel value="all" control={<Radio size="small" />} label="Match all setups (narrower)" />
          </RadioGroup>
        </Box>

        <Divider sx={{ mb: 2 }} />
        <Stack spacing={1.5}>
          {draft.groups.map((group, index) => (
            <SetupGroupCard
              key={group.id}
              group={group}
              index={index}
              onChange={(next) => updateGroup(index, next)}
              onDelete={() => setDraft({
                ...draft,
                groups: draft.groups.filter((_, itemIndex) => itemIndex !== index),
              })}
              onDuplicate={() => {
                if (draft.groups.length >= 8) return;
                const duplicate = clone(group);
                duplicate.id = groupId();
                duplicate.name = `${group.name} copy`.slice(0, 60);
                setDraft({ ...draft, groups: [...draft.groups, duplicate] });
              }}
              filterOptions={guidedOptions}
              fieldCatalog={fieldCatalog}
            />
          ))}
        </Stack>

        {!draft.groups.length && (
          <Paper variant="outlined" sx={{ p: 3, textAlign: 'center', borderStyle: 'dashed' }}>
            <Typography variant="body2" color="text.secondary">
              Add a named setup such as “Breakout ready” or “Pullback candidate.”
            </Typography>
          </Paper>
        )}

        <Button
          startIcon={<AddIcon />}
          onClick={addGroup}
          disabled={draft.groups.length >= 8}
          sx={{ mt: 1.5 }}
        >
          Add named setup
        </Button>

        {showErrors && errors.length > 0 && (
          <Alert severity="warning" sx={{ mt: 2 }}>
            {errors.map((error) => <div key={error}>{error}</div>)}
          </Alert>
        )}
      </DialogContent>
      <DialogActions sx={{ px: 3, py: 1.5 }}>
        <Button onClick={onClose}>Cancel</Button>
        <Button variant="contained" onClick={handleApply}>Apply logic</Button>
      </DialogActions>
    </Dialog>
  );
}

import {
  Alert, Box, Chip, Divider, List, ListItem, Paper, Stack, Typography,
} from '@mui/material';

// Presentational view of one research job and, when ready, its shadow
// preview. The preview is an unaccepted research result: it never claims
// live membership, a company-wide "verified" badge or a confidence score.

const UNKNOWN_MATERIALITY = 'Not separately disclosed in reviewed evidence';

const SUPPORT_LABELS = {
  primary_explicit: 'Primary source — explicit',
  primary_synthesis: 'Primary source — bounded synthesis',
  secondary_reported: 'Secondary report only',
  inferred_unverified: 'Inferred, not verified',
  unresolved: 'Unresolved',
};

const STATUS_LABELS = {
  research: 'Research', announced: 'Announced', qualification: 'Qualification',
  commercially_available: 'Commercially available',
  shipping_or_operating: 'Shipping / operating', discontinued: 'Discontinued',
  unknown: 'Stage unknown',
};

const CONDITION_TEXT = {
  review_required: 'Issuer identity needs administrator review before research continues.',
  multiple_ciks: 'More than one SEC CIK matches this ticker.',
  issuer_link_required: 'No accepted issuer link for this listing.',
  sec_user_agent_not_configured: 'SEC access is not configured (contact User-Agent missing).',
  route_not_approved: 'The subscription text route is not enabled.',
  subscription_credentials_missing: 'Subscription credentials are not configured.',
  allocation_not_configured: 'No daily research allocation is configured.',
  theme_definition_unavailable: 'The theme has no sealed definition to research against.',
};

const shortDate = (value) => (value ? String(value).slice(0, 10) : 'undated');

function materialityText(materiality) {
  if (!materiality || materiality.display || materiality.basis === 'unknown') {
    return materiality?.display || UNKNOWN_MATERIALITY;
  }
  if (materiality.qualitative_label) return `Qualitative: ${materiality.qualitative_label}`;
  const scope = materiality.scope_label ? ` of ${materiality.scope_label}` : '';
  const unit = materiality.unit ? ` ${materiality.unit}` : '';
  const denominator = materiality.denominator_definition
    ? ` (denominator: ${materiality.denominator_definition})` : '';
  return `${materiality.metric} ${materiality.value}${unit}${scope}, ${materiality.period}${denominator}`;
}

function ClaimCard({ claim }) {
  const held = claim.active_holds?.length > 0 || claim.freshness_state !== 'current';
  return (
    <Paper variant="outlined" sx={{ p: 1.5 }} data-testid="exposure-claim">
      <Stack direction="row" spacing={1} useFlexGap flexWrap="wrap" sx={{ mb: 0.75 }}>
        <Chip size="small" label={claim.claim_kind.replaceAll('_', ' ')} />
        <Chip
          size="small"
          variant="outlined"
          color={claim.support_basis?.startsWith('primary') ? 'success' : 'default'}
          label={SUPPORT_LABELS[claim.support_basis] || claim.support_basis}
        />
        <Chip size="small" variant="outlined" label={STATUS_LABELS[claim.commercial_status] || claim.commercial_status} />
        {claim.conclusion !== 'supported' && (
          <Chip size="small" color="warning" label={`Conclusion: ${claim.conclusion}`} />
        )}
        {claim.carried_forward && <Chip size="small" variant="outlined" label="Carried forward" />}
        {(claim.active_holds || []).map((hold) => (
          <Chip key={hold} size="small" color="warning" label={`Hold: ${hold}`} />
        ))}
      </Stack>
      <Typography>{claim.statement}</Typography>
      <Typography variant="body2" color="text.secondary" sx={{ mt: 0.5 }}>
        Supported as of {shortDate(claim.supported_as_of)}
        {claim.fresh_until ? ` · usable until ${shortDate(claim.fresh_until)}` : ''}
        {' · '}freshness {claim.freshness_state}
        {claim.reporting_scope === 'segment_or_subsidiary' && claim.scope_label
          ? ` · scope: ${claim.scope_label}` : ''}
      </Typography>
      <Typography variant="body2" sx={{ mt: 0.5 }}>
        Materiality: {materialityText(claim.materiality)}
      </Typography>
      {held && (
        <Typography variant="body2" color="warning.main" sx={{ mt: 0.5 }}>
          Held from new automated use until reviewed or refreshed.
        </Typography>
      )}
      <List dense disablePadding sx={{ mt: 0.5 }}>
        {(claim.evidence || []).map((item) => (
          <ListItem key={item.link_id} disableGutters sx={{ display: 'block' }}>
            <Typography variant="caption" color="text.secondary">
              {item.direction === 'conflicting' ? 'Conflicting' : 'Supporting'}
              {' · '}{item.evidence_role.replaceAll('_', ' ')}
              {item.language && item.language !== 'en' ? ` · ${item.language}` : ''}
            </Typography>
            {/* Original passages render as plain text, never as HTML. */}
            <Typography variant="body2" component="blockquote" sx={{ m: 0, pl: 1, borderLeft: 2, borderColor: 'divider' }}>
              {item.quote}
            </Typography>
          </ListItem>
        ))}
      </List>
    </Paper>
  );
}

export default function ExposureResearchPanel({ job, preview }) {
  if (!job) return null;
  const condition = job.condition ? CONDITION_TEXT[job.condition] || job.condition : null;
  const gaps = (preview?.coverage || []).filter(
    (item) => item.outcome !== 'complete_for_requested_scope',
  );
  return (
    <Box data-testid="exposure-research-panel">
      <Stack direction="row" spacing={1} useFlexGap flexWrap="wrap" alignItems="center">
        <Typography variant="subtitle1" fontWeight={700}>Research job</Typography>
        <Chip size="small" label={job.state || 'queued'} />
        <Chip size="small" variant="outlined" label="Research progress — not accepted" />
      </Stack>
      <Stack direction="row" spacing={1} useFlexGap flexWrap="wrap" sx={{ mt: 1 }}>
        {(job.stages || []).map((stage) => (
          <Chip
            key={stage.stage}
            size="small"
            variant="outlined"
            label={`${stage.stage.replaceAll('_', ' ')}: ${stage.pause_reason || stage.status}`}
          />
        ))}
      </Stack>
      {condition && (
        <Alert severity={job.state === 'terminal_failure' ? 'error' : 'warning'} sx={{ mt: 1 }}>
          {condition}
        </Alert>
      )}

      {preview && (
        <Box sx={{ mt: 2 }}>
          <Divider sx={{ mb: 1.5 }} />
          <Alert severity="info">
            Shadow preview — research result for review only. It does not change theme membership
            or any live basket.
          </Alert>
          <Typography variant="caption" color="text.secondary" component="div" sx={{ mt: 1 }}>
            Dossier revision {preview.revision_number} · {preview.assessment_revision_id}
          </Typography>
          <Stack spacing={1} sx={{ mt: 1 }}>
            {preview.claims.map((claim) => <ClaimCard key={claim.claim_revision_id} claim={claim} />)}
            {!preview.claims.length && (
              <Typography color="text.secondary">
                No supported claims in the reviewed evidence. This is not evidence of no exposure.
              </Typography>
            )}
          </Stack>
          {gaps.length > 0 && (
            <Box sx={{ mt: 1.5 }}>
              <Typography variant="subtitle2">Coverage gaps</Typography>
              {gaps.map((item, index) => (
                <Typography key={`${item.route}:${item.reason}:${index}`} variant="body2" color="text.secondary">
                  {item.route} · {item.outcome.replaceAll('_', ' ')}{item.reason ? ` · ${item.reason}` : ''}
                </Typography>
              ))}
            </Box>
          )}
        </Box>
      )}
    </Box>
  );
}
